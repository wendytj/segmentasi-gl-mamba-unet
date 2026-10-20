import gc
import os
import random
import warnings
import numpy as np
import torch
from torch.utils.data import DataLoader

warnings.filterwarnings("ignore")

torch.backends.cuda.matmul.allow_tf32 = True
torch.backends.cudnn.allow_tf32 = True
torch.backends.cudnn.benchmark = True
torch._dynamo.config.suppress_errors = True  # type: ignore

# Patch Mamba / Triton
from architectures.patcher import apply_mamba_patch

apply_mamba_patch()

# Import Arsitektur & Loaders
from architectures.HybridDWTGLMambaUNet import HybridDWTGLMambaUNet
from loaders.LessionSegLoader import get_stroke_segmentation_dataloaders
from loggers.experiment_logger import ExperimentLogger

# Import Modul Pipeline Internal
from train_modules.builder import configure_hybrid_fine_tuning
from train_modules.configs import CONFIG, parse_args
from train_modules.pipeline import run_training_pipeline
from train_modules.utils import run_mock_test, sanitize_all_hparams


def main():
    args = parse_args()

    # Override Config dari Argparse CLI
    for key, value in vars(args).items():
        if value is not None:
            CONFIG[key] = value

    if args.target_modules:
        CONFIG["target_modules"] = args.target_modules

    if "3channel" in CONFIG["npz_path"] and CONFIG["in_channels"] == 1:
        CONFIG["in_channels"] = 3
        print("💡 Auto-detected 3-channel dataset! Mengubah in_channels=3.")

    # Hitung Effective LR berdasarkan Batch Size

    effective_bs = CONFIG["batch_size"] * CONFIG["accumulation_steps"]

    CONFIG["effective_lr"] = CONFIG["learning_rate"] * (effective_bs / 16) ** 0.5

    # Seed Reproducibility
    torch.manual_seed(CONFIG["seed"])
    torch.cuda.manual_seed_all(CONFIG["seed"])
    np.random.seed(CONFIG["seed"])
    random.seed(CONFIG["seed"])

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"🚀 Memulai Eksekusi pada Perangkat: {device}")
    print(
        f"💡 Base LR: {CONFIG['learning_rate']} | Effective LR (Effective Batch Size"
        f" {effective_bs}): {CONFIG['effective_lr']:.6f} | Encoder LR"
        f" Ratio: {CONFIG['encoder_lr_ratio']}x"
    )

    use_bf16 = torch.cuda.is_bf16_supported()
    amp_dtype = torch.bfloat16 if use_bf16 else torch.float16
    scaler = torch.amp.GradScaler("cuda", enabled=not use_bf16)  # type: ignore

    # 1. Load Data Loaders
    loader_output = get_stroke_segmentation_dataloaders(
        npz_path=CONFIG["npz_path"],
        config=CONFIG.get("split_config", "split_80_10_10"),
        batch_size=CONFIG["batch_size"],
        eval_batch_size=CONFIG["eval_batch_size"],
        num_workers=CONFIG["num_workers"],
        loss_type=CONFIG["loss_type"],
        aug_trial=CONFIG["aug_trial"],
    )

    (
        train_loader,
        val_loader,
        test_loader,
        gpu_train_transform,
        gpu_eval_transform,
    ) = loader_output

    train_eval_loader = DataLoader(
        train_loader.dataset,
        batch_size=CONFIG["eval_batch_size"],
        shuffle=False,
        num_workers=CONFIG["num_workers"],
        pin_memory=True,
    )

    class_names = ["Ischemia", "Bleeding"]

    # 2. Inisialisasi Arsitektur Hybrid DWT-GLMamba UNet
    model = HybridDWTGLMambaUNet(
        in_channels=CONFIG["in_channels"],
        num_classes=CONFIG["num_classes"],
        dwt_embed_dim=CONFIG["embed_dim"],
    ).to(device)

    # 3. Load Pretrained Weights DWT-Mamba dari Stage 1
    model.load_dwt_pretrained(CONFIG["pretrained_path"])

    # 4. Konfigurasi Strategi Fine-Tuning & Differential LR Parameter Groups
    model, dora_magnitude_params = configure_hybrid_fine_tuning(model, CONFIG)
    model = model.to(device).to(memory_format=torch.channels_last)  # type: ignore

    raw_model = model

    # 5. Inisialisasi Experiment Logger
    clean_hparams = sanitize_all_hparams(CONFIG)
    logger = ExperimentLogger(
        save_dir=f"./logs/{CONFIG['experiment_code']}",
        class_names=class_names,
        hparams=clean_hparams,
    )

    # 6. Sanity Check / Mock Test Forward Pass
    run_mock_test(model, train_loader, device, gpu_eval_transform)

    if args.dry_run:
        print("💡 Mode --dry_run selesai. Program keluar tanpa melakukan pelatihan.")
        return

    gc.collect()
    torch.cuda.empty_cache()

    loaders = (train_loader, val_loader, test_loader)
    transforms = (gpu_train_transform, gpu_eval_transform)
    amp_params = (amp_dtype, use_bf16, scaler)

    # 7. Eksekusi Training Pipeline Stage 2
    run_training_pipeline(
        model=model,
        raw_model=raw_model,
        loaders=loaders,
        transforms=transforms,
        amp_params=amp_params,
        logger=logger,
        device=device,
        train_eval_loader=train_eval_loader,  # Pass parameter groups langsung ke optimizer builder
        dora_magnitude_params=dora_magnitude_params,
    )


if __name__ == "__main__":
    main()
