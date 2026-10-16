from architectures.HybridDWTGLMambaUNet import HybridDWTGLMambaUNet
from fine_tuning.peft_wrapper import apply_peft, get_dora_magnitude_params


def configure_hybrid_fine_tuning(model: HybridDWTGLMambaUNet, config):
    """Mengonfigurasi 4 strategi fine-tuning pada branch DWT-Mamba khusus HybridDWTGLMambaUNet.

    Fungsi ini murni melakukan mutasi model state (requires_grad & PEFT injection).
    """
    strategy = config["finetune_strategy"].lower()
    peft_method = config["peft_method"].lower()

    print(
        "\n⚙️ Mengaplikasikan Strategi Fine-Tuning Encoder DWT:"
        f" [{strategy.upper()}]"
    )

    # 1. SETUP STRATEGI FINETUNING PADA DWT ENCODER
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
            " disuntikkan ke target modules."
        )

    else:
        raise ValueError(f"🚨 Strategy fine-tuning tidak dikenal: {strategy}")

    # 2. EKSTRAKSI DORA MAGNITUDE PARAMS JIKA MENGGUNAKAN DORA
    dora_magnitude_params = (
        get_dora_magnitude_params(model) if peft_method == "dora" else []
    )

    return model, dora_magnitude_params
