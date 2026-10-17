from architectures.HybridDWTGLMambaUNet import HybridDWTGLMambaUNet
from fine_tuning.peft_wrapper import apply_peft, get_dora_magnitude_params


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
