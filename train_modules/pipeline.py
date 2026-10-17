from architectures.HybridDWTGLMambaUNet import HybridDWTGLMambaUNet
from fine_tuning.peft_wrapper import CONV_TARGET_MODULES
import json
import os
import gc
import torch
import torch.optim as optim
from torch.optim.lr_scheduler import CosineAnnealingLR, LinearLR, SequentialLR
from train_modules.configs import CONFIG
from train_modules.loss import build_loss_criterion, compute_class_weights
from train_modules.trainer import evaluate, train_one_epoch
from train_modules.utils import safe_atomic_save
from fine_tuning.peft_wrapper import (
    CONV_TARGET_MODULES,
    get_dora_magnitude_params,
)


def configure_hybrid_fine_tuning(model: HybridDWTGLMambaUNet, config):
    """Mengonfigurasi strategi fine-tuning (FROZEN, FULL, SELECTIVE, PEFT) pada branch DWT-Mamba."""
    strategy = config["finetune_strategy"].lower()
    peft_method = config.get("peft_method", "none").lower()

    print(
        "\n⚙️ [Fine-Tuning Strategy] Mengaplikasikan Strategi pada DWT"
        f" Encoder: [{strategy.upper()}]"
    )

    # 1. MUTASI REQUIRES_GRAD & PEFT INJECTION
    if strategy == "frozen":
        for param in model.dwt_encoder.parameters():
            param.requires_grad = False
        print(" 🔒 [DWT Encoder] Seluruh parameter DWT-Mamba DIBEKUKAN (Frozen).")

    elif strategy == "full":
        for param in model.dwt_encoder.parameters():
            param.requires_grad = True
        print(
            " 🔥 [DWT Encoder] Seluruh parameter DWT-Mamba DIBUKA (Full"
            " Fine-Tuning)."
        )

    elif strategy == "selective":
        for param in model.dwt_encoder.parameters():
            param.requires_grad = False

        # Aktifkan module spesifik tanpa adapter PEFT
        active_modules = [
            model.dwt_encoder.pe_branches,
            model.dwt_encoder.spase_branches,
            model.dwt_encoder.pm_branches,
            model.dwt_encoder.mb_gsf,
        ]
        for module in active_modules:
            for param in module.parameters():
                param.requires_grad = True
        print(
            " 🎯 [DWT Encoder] Fine-tuning Selektif aktif pada (pe_branches,"
            " spase_branches, pm_branches, mb_gsf)."
        )

    elif strategy == "peft":
        for param in model.dwt_encoder.parameters():
            param.requires_grad = False

        active_r = (
            config["prodial_r_eps"] if peft_method == "prodial" else config["lora_r"]
        )

        model.dwt_encoder = apply_peft(  # type: ignore
            model.dwt_encoder,
            method=peft_method,
            r=active_r,
            alpha=config["lora_alpha"],
            dropout=config["lora_dropout"],
            r_b=config["prodial_r_b"],
            target_modules=config.get("target_modules", None),
        )
        print(
            f" 🧬 [DWT Encoder] PEFT Adapter ({peft_method.upper()}) berhasil"
            " disuntikkan ke LINEAR_TARGET_MODULES & CONV_TARGET_MODULES."
        )

    else:
        raise ValueError(f"🚨 Strategy fine-tuning tidak dikenal: {strategy}")

    # 2. EKSTRAKSI DORA MAGNITUDE PARAMS (KHUSUS DORA)
    dora_magnitude_params = (
        get_dora_magnitude_params(model)
        if peft_method == "dora" and strategy == "peft"
        else []
    )

    # 3. REKAPITULASI PRESISI PARAMETER
    dwt_trainable = sum(
        p.numel() for p in model.dwt_encoder.parameters() if p.requires_grad
    )
    dwt_total = sum(p.numel() for p in model.dwt_encoder.parameters())

    scratch_modules = [
        model.stem,
        model.enc1,
        model.down1,
        model.enc2,
        model.down2,
        model.enc3,
        model.down3,
        model.fuse_sk1,
        model.fuse_sk2,
        model.fuse_sk3,
        model.bottleneck,
        model.up3,
        model.dec3,
        model.up2,
        model.dec2,
        model.up1,
        model.dec1,
        model.final_proj,
    ]
    scratch_trainable = sum(
        p.numel() for m in scratch_modules for p in m.parameters() if p.requires_grad
    )
    scratch_total = sum(p.numel() for m in scratch_modules for p in m.parameters())

    total_trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    total_params = sum(p.numel() for p in model.parameters())

    pct_dwt = (dwt_trainable / dwt_total * 100) if dwt_total > 0 else 0.0
    pct_model = (total_trainable / total_params * 100) if total_params > 0 else 0.0

    print(" 📊 [Parameter Breakdown Summary]")
    print(
        f"  ├─ DWT Encoder (Pretrained) : {dwt_trainable:>10,} / {dwt_total:>10,}"
        f" params trainable ({pct_dwt:.2f}%)"
    )
    print(
        f"  ├─ Scratch Modules (GL+UNet): {scratch_trainable:>10,} /"
        f" {scratch_total:>10,} params trainable (100.00%)"
    )
    print(
        f"  └─ TOTAL MODEL STAGE 2      : {total_trainable:>10,} /"
        f" {total_params:>10,} params trainable ({pct_model:.2f}%)\n"
    )

    return model, dora_magnitude_params


def build_optimizer_and_scheduler(
    model: HybridDWTGLMambaUNet, config, dora_magnitude_params=None
):
    """Membentuk AdamW Optimizer dengan 5-Tier Differential Learning Rate khusus Stage 2 Hybrid DWT-GLMamba UNet."""
    base_lr = config["effective_lr"]
    encoder_lr_ratio = config["encoder_lr_ratio"]
    lr_conv_ratio = config["lr_conv_ratio"]
    weight_decay = config["weight_decay"]
    strategy = config.get("finetune_strategy", "peft").lower()
    peft_method = config.get("peft_method", "none").upper()

    dwt_lr = base_lr * encoder_lr_ratio
    conv_lr = dwt_lr * lr_conv_ratio

    dora_mag_set = (
        set(dora_magnitude_params) if dora_magnitude_params is not None else set()
    )

    # 1. SELEKSI EKSLUSIF PARAMETER AKTIF DWT ENCODER
    dwt_conv_params = []
    dwt_linear_params = []

    for name, param in model.dwt_encoder.named_parameters():
        if not param.requires_grad or param in dora_mag_set:
            continue

        # Jika mengandung keyword Conv, masuk grup Conv (0.02x LR)
        if any(k in name for k in CONV_TARGET_MODULES):
            dwt_conv_params.append(param)
        # Sisanya (Linear, SSM, Norm) masuk grup Linear/SSM (0.1x LR)
        else:
            dwt_linear_params.append(param)

    # 2. PARAMETER SCRATCH (GL-MAMBA & UNET DECODER)
    scratch_modules = [
        model.stem,
        model.enc1,
        model.down1,
        model.enc2,
        model.down2,
        model.enc3,
        model.down3,
        model.fuse_sk1,
        model.fuse_sk2,
        model.fuse_sk3,
        model.bottleneck,
        model.up3,
        model.dec3,
        model.up2,
        model.dec2,
        model.up1,
        model.dec1,
        model.final_proj,
    ]
    scratch_params = [
        p for m in scratch_modules for p in m.parameters() if p.requires_grad
    ]

    # 3. LABELING DINAMIS SETIAP GROUP
    if strategy == "full":
        g2_label = "DWT Linear/SSM (Full FT)"
        g3_label = "DWT Conv2d Modules (Full FT)"
    elif strategy == "selective":
        g2_label = "DWT Active Linear/SSM"
        g3_label = "DWT Active Conv2d Modules"
    elif strategy == "peft":
        g2_label = f"DWT PEFT Linear ({peft_method})"
        g3_label = f"DWT CoLoRA Conv2d ({peft_method})"
    else:
        g2_label = "DWT Base Linear (Frozen)"
        g3_label = "DWT Base Conv2d (Frozen)"

    # 4. PEMBENTUKAN PARAMETER GROUPS ADAMW
    optimizer_grouped_parameters = []
    group_summary = []

    # Group 1: Scratch Components (1.00x LR)
    if scratch_params:
        scratch_count = sum(p.numel() for p in scratch_params)
        optimizer_grouped_parameters.append(
            {
                "params": scratch_params,
                "lr": base_lr,
                "weight_decay": weight_decay,
                "name": "scratch_components",
            }
        )
        group_summary.append(
            (
                "1. Scratch Modules (GL-Mamba + UNet)",
                len(scratch_params),
                scratch_count,
                base_lr,
                "1.00x",
            )
        )

    # Group 2: DWT Linear / SSM / PEFT (0.10x LR)
    if dwt_linear_params:
        g2_count = sum(p.numel() for p in dwt_linear_params)
        optimizer_grouped_parameters.append(
            {
                "params": dwt_linear_params,
                "lr": dwt_lr,
                "weight_decay": weight_decay,
                "name": "dwt_linear_peft",
            }
        )
        group_summary.append(
            (
                f"2. {g2_label}",
                len(dwt_linear_params),
                g2_count,
                dwt_lr,
                f"{encoder_lr_ratio}x",
            )
        )

    # Group 3: DWT Conv2d / CoLoRA (0.02x LR)
    if dwt_conv_params:
        g3_count = sum(p.numel() for p in dwt_conv_params)
        optimizer_grouped_parameters.append(
            {
                "params": dwt_conv_params,
                "lr": conv_lr,
                "weight_decay": weight_decay,
                "name": "dwt_conv",
            }
        )
        group_summary.append(
            (
                f"3. {g3_label}",
                len(dwt_conv_params),
                g3_count,
                conv_lr,
                f"{(encoder_lr_ratio * lr_conv_ratio):.2f}x",
            )
        )

    # Group 4: DoRA Magnitude Vector (0.10x LR, WD = 0.0)
    if dora_magnitude_params:
        dora_count = sum(p.numel() for p in dora_magnitude_params)
        optimizer_grouped_parameters.append(
            {
                "params": dora_magnitude_params,
                "lr": dwt_lr,
                "weight_decay": 0.0,
                "name": "dora_magnitude",
            }
        )
        group_summary.append(
            (
                "4. DoRA Magnitude Vector (WD = 0.0)",
                len(dora_magnitude_params),
                dora_count,
                dwt_lr,
                f"{encoder_lr_ratio}x",
            )
        )

    optimizer = optim.AdamW(optimizer_grouped_parameters, eps=1e-6)

    opt_total_params = sum(
        sum(p.numel() for p in g["params"]) for g in optimizer_grouped_parameters
    )

    print("⚙️ [Stage 2 AdamW Optimizer Grouping Setup]")
    print(
        " 🎯 Total Parameter Dilatih (Optimized):"
        f" {opt_total_params:,} params across"
        f" {len(optimizer_grouped_parameters)} groups"
    )

    for idx, (label, tensor_len, param_count, lr_val, lr_ratio) in enumerate(
        group_summary
    ):
        prefix = " └─" if idx == len(group_summary) - 1 else " ├─"
        print(
            f"{prefix} {label:<36s} : {tensor_len:>3d} tensors |"
            f" {param_count:>10,} params | LR: {lr_val:.6f} ({lr_ratio})"
        )

    # 5. SCHEDULER (WARMUP + COSINE ANNEALING)
    warmup_epochs = config["warmup_epochs"]
    max_epochs = config["max_epochs"]

    scheduler_warmup = LinearLR(
        optimizer, start_factor=0.1, end_factor=1.0, total_iters=warmup_epochs
    )
    scheduler_decay = CosineAnnealingLR(
        optimizer, T_max=max_epochs - warmup_epochs, eta_min=base_lr * 0.01
    )
    scheduler = SequentialLR(
        optimizer,
        schedulers=[scheduler_warmup, scheduler_decay],
        milestones=[warmup_epochs],
    )

    return optimizer, scheduler


def evaluate_best_checkpoints(
    model,
    logger,
    device,
    criterion,
    pixel_counts,
    class_weights,
    best_dsc_epoch,
    best_mcc_epoch,
    best_loss_epoch,
    best_val_dsc,
    best_val_mcc,
    best_val_loss,
    path_best_val_dsc,
    path_best_val_mcc,
    path_best_val_loss,
    train_eval_loader,
    val_loader,
    test_loader,
    gpu_eval_transform,
    evaluate_fn=None,
):
    print(
        "\n🧪 Mengevaluasi Performance (Train, Val, Test Set) untuk Ketiga"
        " Model Best Checkpoint (DSC, MCC, Loss)..."
    )

    eval_func = evaluate_fn if evaluate_fn is not None else evaluate

    def wrap_eval_results(eval_tuple):
        if eval_tuple is None or eval_tuple[0] is None:
            return None, None
        metrics_dict, cm = logger.compute_metrics(*eval_tuple, compute_hd95=True)
        res_dict = {
            "confusion_matrix": cm.tolist() if hasattr(cm, "tolist") else cm,
            "global_metrics": metrics_dict["global_metrics"],
            "per_class_metrics": metrics_dict["per_class_metrics"],
        }
        return res_dict, cm

    def eval_single_checkpoint(ckpt_path, desc_tag):
        if not os.path.exists(ckpt_path):
            print(f"⚠️ Checkpoint tidak ditemukan: {ckpt_path}")
            return None, None, None, None, None

        print(f"   --> Evaluasi Model '{desc_tag}'...")
        gc.collect()
        torch.cuda.empty_cache()

        checkpoint = torch.load(ckpt_path, map_location=device)
        if isinstance(checkpoint, dict) and "model_state" in checkpoint:
            model_state = checkpoint["model_state"]
            ckpt_weights = checkpoint.get("class_weights", None)
        else:
            model_state = checkpoint
            ckpt_weights = None

        model.load_state_dict(model_state)

        if ckpt_weights is not None:
            if hasattr(criterion, "weight"):
                setattr(criterion, "weight", ckpt_weights.to(device))
            elif hasattr(criterion, "loss_fn") and hasattr(
                getattr(criterion, "loss_fn"), "weight"
            ):
                setattr(criterion.loss_fn, "weight", ckpt_weights.to(device))

        model.eval()

        with torch.no_grad():
            tr_eval = eval_func(
                model,
                train_eval_loader,
                criterion,
                device,
                gpu_eval_transform,
                desc=f"Eval Train ({desc_tag})",
            )
            va_eval = eval_func(
                model,
                val_loader,
                criterion,
                device,
                gpu_eval_transform,
                desc=f"Eval Val ({desc_tag})",
            )

            te_eval = None
            if test_loader is not None:
                te_eval = eval_func(
                    model,
                    test_loader,
                    criterion,
                    device,
                    gpu_eval_transform,
                    desc=f"Testing ({desc_tag})",
                )

        def extract_raw_tuple(res):
            if res is None:
                return None
            if isinstance(res, tuple) and len(res) == 2:
                return res[1]
            return res[2] if len(res) == 3 else res

        tr_raw = extract_raw_tuple(tr_eval)
        va_raw = extract_raw_tuple(va_eval)
        te_raw = extract_raw_tuple(te_eval)

        tr_res, _ = wrap_eval_results(tr_raw)
        va_res, _ = wrap_eval_results(va_raw)
        te_res, cm_te = wrap_eval_results(te_raw)

        del tr_raw, va_raw, tr_eval, va_eval, checkpoint, model_state
        gc.collect()

        return tr_res, va_res, te_res, cm_te, te_raw

    # 1. Evaluasi Best DSC Checkpoint
    tr_dsc_res, va_dsc_res, te_dsc_res, cm_dsc, te_raw_dsc = eval_single_checkpoint(
        path_best_val_dsc, f"Best DSC (Ep {best_dsc_epoch})"
    )

    gc.collect()
    torch.cuda.empty_cache()

    # 2. Evaluasi Best MCC Checkpoint (Reuse jika epoch sama dengan Best DSC)
    if best_mcc_epoch == best_dsc_epoch:
        print(
            f"   ℹ️ Best MCC berada pada epoch yang sama dengan Best DSC (Ep {best_mcc_epoch}), mereuse hasil evaluasi..."
        )
        tr_mcc_res, va_mcc_res, te_mcc_res, cm_mcc, te_raw_mcc = (
            tr_dsc_res,
            va_dsc_res,
            te_dsc_res,
            cm_dsc,
            te_raw_dsc,
        )
    else:
        tr_mcc_res, va_mcc_res, te_mcc_res, cm_mcc, te_raw_mcc = eval_single_checkpoint(
            path_best_val_mcc, f"Best MCC (Ep {best_mcc_epoch})"
        )

    gc.collect()
    torch.cuda.empty_cache()

    # 3. Evaluasi Best Loss Checkpoint (Reuse jika epoch sama dengan Best DSC/MCC)
    if best_loss_epoch == best_dsc_epoch:
        print(
            f"   ℹ️ Best Loss berada pada epoch yang sama dengan Best DSC (Ep {best_loss_epoch}), mereuse hasil evaluasi..."
        )
        tr_loss_res, va_loss_res, te_loss_res, cm_loss, te_raw_loss = (
            tr_dsc_res,
            va_dsc_res,
            te_dsc_res,
            cm_dsc,
            te_raw_dsc,
        )
    elif best_loss_epoch == best_mcc_epoch:
        print(
            f"   ℹ️ Best Loss berada pada epoch yang sama dengan Best MCC (Ep {best_loss_epoch}), mereuse hasil evaluasi..."
        )
        tr_loss_res, va_loss_res, te_loss_res, cm_loss, te_raw_loss = (
            tr_mcc_res,
            va_mcc_res,
            te_mcc_res,
            cm_mcc,
            te_raw_mcc,
        )
    else:
        tr_loss_res, va_loss_res, te_loss_res, cm_loss, te_raw_loss = (
            eval_single_checkpoint(
                path_best_val_loss, f"Best Loss (Ep {best_loss_epoch})"
            )
        )

    # Export Logging & Visualisasi Dashboard
    logger.export_csv()
    if te_raw_dsc is not None:
        logger.plot_learning_curves(
            test_res=te_dsc_res, filename="training_dashboard.png"
        )

    summary_json_path = os.path.join(logger.save_dir, "best_model_metrics.json")

    pc_list = (
        pixel_counts.cpu().tolist()
        if hasattr(pixel_counts, "cpu")
        else list(pixel_counts)
    )
    cw_list = (
        class_weights.cpu().tolist()
        if hasattr(class_weights, "cpu")
        else list(class_weights)
    )

    # Simpan Ringkasan Evaluasi 3 Model di JSON
    with open(summary_json_path, "w", encoding="utf-8") as f:
        json.dump(
            {
                "dataset_info": {
                    "num_channels": 2,
                    "channel_labels": ["Ischemia", "Bleeding"],
                    "train_pixel_counts_matrix": pc_list,
                    "pos_weights": cw_list,
                },
                "hyperparameters": getattr(logger, "hparams", {}),
                "model_by_val_dsc": {
                    "criterion": "best_val_dsc",
                    "best_epoch": int(best_dsc_epoch),
                    "best_metric_value": round(float(best_val_dsc), 4),
                    "train_results": tr_dsc_res,
                    "val_results": va_dsc_res,
                    "test_results": te_dsc_res,
                },
                "model_by_val_mcc": {
                    "criterion": "best_val_mcc",
                    "best_epoch": int(best_mcc_epoch),
                    "best_metric_value": round(float(best_val_mcc), 4),
                    "train_results": tr_mcc_res,
                    "val_results": va_mcc_res,
                    "test_results": te_mcc_res,
                },
                "model_by_val_loss": {
                    "criterion": "best_val_loss",
                    "best_epoch": int(best_loss_epoch),
                    "best_metric_value": round(float(best_val_loss), 4),
                    "train_results": tr_loss_res,
                    "val_results": va_loss_res,
                    "test_results": te_loss_res,
                },
            },
            f,
            indent=4,
        )

    # Plot Confusion Matrix Masing-Masing Checkpoint di Test Set
    if cm_dsc is not None:
        logger.plot_confusion_matrix(
            cm_dsc, filename="test_confusion_matrix_by_dsc.png"
        )
    if cm_mcc is not None:
        logger.plot_confusion_matrix(
            cm_mcc, filename="test_confusion_matrix_by_mcc.png"
        )
    if cm_loss is not None:
        logger.plot_confusion_matrix(
            cm_loss, filename="test_confusion_matrix_by_loss.png"
        )

    print(
        f"✨ Eksperimen Selesai! Seluruh metrik dari 3 model disimpan di: {summary_json_path}"
    )


def register_nan_hooks(model):
    """Memasang forward hook dengan Full Module Path untuk pelacakan spesifik 100% presisi."""
    nan_detected = [False]

    def create_hook(full_module_path):
        def nan_hook(module, input, output):
            if nan_detected[0]:
                return

            outputs = (
                [output]
                if isinstance(output, torch.Tensor)
                else (output if isinstance(output, (tuple, list)) else [])
            )

            for out in outputs:
                if isinstance(out, torch.Tensor) and torch.isnan(out).any():
                    nan_detected[0] = True
                    print("\n💥 [NaN DETECTED AT EXACT MODULE PATH]")
                    print(f" ├─ Full Path   : {full_module_path}")
                    print(f" ├─ Class Type  : {module.__class__.__name__}")
                    print(f" └─ Output Shape: {out.shape}")
                    break

        return nan_hook

    handles = []
    for name, module in model.named_modules():
        # Hanya pasang pada leaf-modules dan kirim full path (misal: 'bottleneck.bim_local.out_proj')
        if len(list(module.children())) == 0:
            handles.append(module.register_forward_hook(create_hook(name)))

    print(
        f"🛡️ [NaN Guard] High-Precision Hooks aktif pada {len(handles)}"
        " leaf-modules."
    )
    return handles


def run_training_pipeline(
    model,
    raw_model,
    loaders,
    transforms,
    amp_params,
    logger,
    device,
    train_eval_loader,
    dora_magnitude_params,
):
    train_loader, val_loader, test_loader = loaders
    gpu_train_transform, gpu_eval_transform = transforms
    amp_dtype, use_bf16, scaler = amp_params

    # 1. Hitung Class Weights
    class_weights, pixel_counts = compute_class_weights(
        train_loader=train_loader,
        num_classes=CONFIG["num_classes"],
        device=device,
        use_class_weights=CONFIG["use_class_weights"],
        weight_mode=CONFIG["weight_mode"],
        beta=CONFIG["weight_beta"],
    )
    criterion = build_loss_criterion(CONFIG, class_weights=class_weights)

    # 2. Setup Optimizer & Scheduler
    optimizer, scheduler = build_optimizer_and_scheduler(
        model, CONFIG, dora_magnitude_params
    )

    # 3. Setup Checkpoint Paths & Metric State Variables
    start_epoch = 1
    best_val_dsc = -1.0
    best_val_mcc = -1.0
    best_val_loss = float("inf")

    best_dsc_epoch = -1
    best_mcc_epoch = -1
    best_loss_epoch = -1
    patience_counter = 0

    checkpoint_path = os.path.join(logger.save_dir, "last_checkpoint.pth")
    path_best_val_dsc = os.path.join(logger.save_dir, "best_model_val_dsc.pth")
    path_best_val_mcc = os.path.join(logger.save_dir, "best_model_val_mcc.pth")
    path_best_val_loss = os.path.join(logger.save_dir, "best_model_val_loss.pth")

    # 4. Auto-Resume Logic
    if os.path.exists(checkpoint_path):
        print(
            f"🔄 Checkpoint terputus ditemukan! Memuat state dari:"
            f" {checkpoint_path}"
        )
        checkpoint = torch.load(checkpoint_path, map_location=device)
        missing, unexpected = raw_model.load_state_dict(
            checkpoint["model_state"], strict=False
        )
        print(
            f"  ├─ State dict loaded (Missing: {len(missing)} [LayerNorm baru] |"
            f" Unexpected: {len(unexpected)})"
        )

        # Warn jika ada unexpected keys (biar langsung ngeh kalau ada key nyangkut)
        if unexpected:
            print(f"  ⚠️ Warning! Key tidak terduga ditemukan: {unexpected}")

        # 🌟 SAFE LOAD OPTIMIZER (Mencegah KeyError jika restored dari Best Model)
        if "optimizer_state" in checkpoint and checkpoint["optimizer_state"]:
            try:
                optimizer.load_state_dict(checkpoint["optimizer_state"])
                print("  ├─ Optimizer state berhasil dimuat.")
            except Exception as opt_err:
                print(
                    "  ├─ ⚠️ Optimizer state di-reset (Perubahan struktur"
                    f" parameter): {opt_err}"
                )

        if "scheduler_state" in checkpoint and checkpoint["scheduler_state"]:
            scheduler.load_state_dict(checkpoint["scheduler_state"])

        if scaler and checkpoint.get("scaler_state"):
            scaler.load_state_dict(checkpoint["scaler_state"])

        start_epoch = checkpoint.get("epoch", 1) + 1
        best_val_dsc = checkpoint.get("best_val_dsc", -1.0)
        best_val_mcc = checkpoint.get("best_val_mcc", -1.0)
        best_val_loss = checkpoint.get("best_val_loss", float("inf"))
        best_dsc_epoch = checkpoint.get("best_dsc_epoch", -1)
        best_mcc_epoch = checkpoint.get("best_mcc_epoch", -1)
        best_loss_epoch = checkpoint.get("best_loss_epoch", -1)
        patience_counter = checkpoint.get("patience_counter", 0)

        logger.load_history_from_csv()
        print(f"✅ Auto-resume sukses! Melanjutkan dari Epoch {start_epoch}")

    target_base_lr = 0.001
    lr_ratios = [1.0, 0.1, 0.02]  # Ratio per group: [Scratch, DWT Linear, DWT Conv]

    for group, ratio in zip(optimizer.param_groups, lr_ratios):
        group["lr"] = target_base_lr * ratio

    scheduler.last_epoch = start_epoch - 1
    print(
        f"⚡ [Hardcoded LR Applied] Base LR dipaksa melompat ke {target_base_lr}"
        f" (Scheduler Last Epoch: {scheduler.last_epoch})"
    )

    nan_hook_handles = register_nan_hooks(raw_model)
    # 5. Training Loop Utama
    try:
        print(
            "🔥 Memulai Pelatihan Aktif Segmentasi Hybrid DWT GL-Mamba UNet (2-Channel)..."
        )
        for epoch in range(start_epoch, CONFIG["max_epochs"] + 1):
            train_loss, train_dsc = train_one_epoch(
                model,
                train_loader,
                criterion,
                optimizer,
                device,
                gpu_train_transform,
                amp_dtype,
                use_bf16,
                scaler,
                dora_magnitude_params,
                accumulation_steps=CONFIG["accumulation_steps"],
            )

            val_loss, val_eval = evaluate(
                model,
                val_loader,
                criterion,
                device,
                gpu_eval_transform,
                amp_dtype=amp_dtype,  # Pass amp_dtype konsisten
                desc="Validation",
            )

            scheduler.step()

            val_metrics, _ = logger.compute_metrics(*val_eval, compute_hd95=False)
            g_metrics = val_metrics["global_metrics"]

            val_dsc = g_metrics["mean_lesion_dsc"]
            val_iou = g_metrics["mean_lesion_iou"]
            val_cls_f1 = g_metrics["slice_cls_f1"]
            val_cls_mcc = g_metrics["slice_cls_mcc"]

            current_lr = optimizer.param_groups[0]["lr"]

            # Evaluasi Peningkatan Metrik
            is_dsc_improved = val_dsc > (best_val_dsc + CONFIG["early_stop_delta"])
            is_mcc_improved = val_cls_mcc > (best_val_mcc + CONFIG["early_stop_delta"])
            is_loss_improved = val_loss < (best_val_loss - CONFIG["early_stop_delta"])

            status_notes = []

            # 1) Save Best Val DSC Model (Primary Metric)
            if is_dsc_improved:
                best_val_dsc, best_dsc_epoch = val_dsc, epoch
                safe_atomic_save(
                    {
                        "model_state": raw_model.state_dict(),
                        "class_weights": class_weights,
                        "epoch": epoch,
                        "val_dsc": val_dsc,
                    },
                    path_best_val_dsc,
                )
                status_notes.append(f"🎯 Best Val DSC Saved ({val_dsc:.4f})")

            # 2) Save Best Val MCC Model
            if is_mcc_improved:
                best_val_mcc, best_mcc_epoch = val_cls_mcc, epoch
                safe_atomic_save(
                    {
                        "model_state": raw_model.state_dict(),
                        "class_weights": class_weights,
                        "epoch": epoch,
                        "val_mcc": val_cls_mcc,
                    },
                    path_best_val_mcc,
                )
                status_notes.append(f"🧩 Best Val MCC Saved ({val_cls_mcc:.4f})")

            # 3) Save Best Val Loss Model
            if is_loss_improved:
                best_val_loss, best_loss_epoch = val_loss, epoch
                safe_atomic_save(
                    {
                        "model_state": raw_model.state_dict(),
                        "class_weights": class_weights,
                        "epoch": epoch,
                        "val_loss": val_loss,
                    },
                    path_best_val_loss,
                )
                status_notes.append(f"📉 Best Val Loss Saved ({val_loss:.4f})")

            if is_dsc_improved or is_loss_improved or is_mcc_improved:
                patience_counter = 0
            else:
                patience_counter += 1

            if status_notes:
                status_msg = "  --> " + " | ".join(status_notes)
            else:
                status_msg = (
                    "  --> ⏳ Patience:"
                    f" [{patience_counter}/{CONFIG['early_stop_patience']}]"
                )

            logger.log_epoch(
                epoch,
                train_loss,
                val_loss,
                train_dsc,
                val_dsc,
                val_iou,
                val_cls_f1,
                val_cls_mcc,
                current_lr,
                patience_counter,
            )
            logger.export_csv()

            print(
                f"Epoch [{epoch:03d}/{CONFIG['max_epochs']}] | LR:"
                f" {current_lr:.6f} | Train Loss: {train_loss:.4f} - DSC:"
                f" {train_dsc:.4f} | Val Loss: {val_loss:.4f} - DSC:"
                f" {val_dsc:.4f} - IoU: {val_iou:.4f} - Slice MCC:"
                f" {val_cls_mcc:.4f}{status_msg}"
            )

            # Save Last Checkpoint for Auto-Resume
            safe_atomic_save(
                {
                    "epoch": epoch,
                    "model_state": raw_model.state_dict(),
                    "optimizer_state": optimizer.state_dict(),
                    "scheduler_state": scheduler.state_dict(),
                    "scaler_state": scaler.state_dict() if scaler else None,
                    "class_weights": class_weights,
                    "best_val_dsc": best_val_dsc,
                    "best_val_mcc": best_val_mcc,
                    "best_val_loss": best_val_loss,
                    "best_dsc_epoch": best_dsc_epoch,
                    "best_mcc_epoch": best_mcc_epoch,
                    "best_loss_epoch": best_loss_epoch,
                    "patience_counter": patience_counter,
                },
                checkpoint_path,
            )

            if patience_counter >= CONFIG["early_stop_patience"]:
                print(f"\n🛑 Early stopping dipicu pada epoch {epoch}.")
                break

    except KeyboardInterrupt:
        print("\n⚠️ Pelatihan dihentikan secara manual (Ctrl + C)!")
        logger.log_error(
            "Pelatihan dihentikan paksa oleh user (KeyboardInterrupt).",
            context="USER_INTERRUPT",
        )
        logger.export_csv()
        logger.plot_learning_curves()
        return

    except Exception as e:
        print(f"\n🚨 Pelatihan mengalami crash/error: {e}")
        logger.log_error(e, context="TRAINING_CRASH")
        logger.export_csv()
        logger.plot_learning_curves()
        raise e
    finally:
        for handle in nan_hook_handles:
            handle.remove()
        print("🧹 [NaN Guard] Forward hooks berhasil dibersihkan.")

    gc.collect()
    torch.cuda.empty_cache()

    # Evaluasi Akhir untuk Ketiga Best Checkpoints
    evaluate_best_checkpoints(
        model=model,
        logger=logger,
        device=device,
        criterion=criterion,
        pixel_counts=pixel_counts,
        class_weights=class_weights,
        best_dsc_epoch=best_dsc_epoch,
        best_mcc_epoch=best_mcc_epoch,
        best_loss_epoch=best_loss_epoch,
        best_val_dsc=best_val_dsc,
        best_val_mcc=best_val_mcc,
        best_val_loss=best_val_loss,
        path_best_val_dsc=path_best_val_dsc,
        path_best_val_mcc=path_best_val_mcc,
        path_best_val_loss=path_best_val_loss,
        train_eval_loader=train_eval_loader,
        val_loader=val_loader,
        test_loader=test_loader,
        gpu_eval_transform=gpu_eval_transform,
        evaluate_fn=evaluate,
    )
