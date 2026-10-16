import os
import sys

PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

import torch.nn as nn
from peft import LoraConfig, get_peft_model, PeftModel

try:
    from fine_tuning.ProDiaL import (
        apply_prodial_to_model,
        merge_prodial_weights,
    )
    from fine_tuning.CoLoRA import (
        CoLoRAConv2d,
        apply_colora_to_model,
        merge_colora_weights,
    )
except ImportError:
    from ProDiaL import apply_prodial_to_model, merge_prodial_weights
    from CoLoRA import apply_colora_to_model, merge_colora_weights, CoLoRAConv2d

# ==============================================================================
# MASTER LIST SELURUH MODUL BERBOBOT DI DWT-MAMBA
# ==============================================================================

# 1. Modul Linear Target untuk LoRA / DoRA / ProDiaL
LINEAR_TARGET_MODULES = [
    "in_proj",  # Proyeksi input Mamba (row_mamba & col_mamba)
    "out_proj",  # Proyeksi output Mamba (row_mamba & col_mamba)
]

# 2. Modul Conv2D Target murni untuk CoLoRA
CONV_TARGET_MODULES = [
    "pe_branches",  # Conv2d Patch Embedding (4 cabang subband)
    "pm_branches",  # Conv2d Patch Match (4 cabang subband)
    "conv_branch",  # Conv2d Spasial 3x3 pada SpaSE_SSM
    "mb_gsf.proj",  # Conv2d 1x1 Fusi Subband pada MB_GSF
]
# Master List Layer Normalisasi (TIDAK DILATIH)
# NORM_TARGET_MODULES = [
#     "norm",  # LayerNorm pada SpaSE_SSM
#     "ln",  # LayerNorm pada MB_GSF
# ]


def apply_lora(
    model: nn.Module,
    r: int = 8,
    alpha: int = 16,
    dropout: float = 0.05,
    target_modules: list = LINEAR_TARGET_MODULES,
) -> nn.Module:
    config = LoraConfig(
        r=r,
        lora_alpha=alpha,
        target_modules=target_modules,
        lora_dropout=dropout,
        bias="none",
        use_dora=False,
    )  # type: ignore
    peft_model = get_peft_model(model, config)
    print(f"✅ [LoRA Injected] r={r}, alpha={alpha} pada {target_modules}")
    return peft_model


def apply_dora(
    model: nn.Module,
    r: int = 8,
    alpha: int = 16,
    dropout: float = 0.05,
    target_modules: list = LINEAR_TARGET_MODULES,
) -> nn.Module:
    config = LoraConfig(
        r=r,
        lora_alpha=alpha,
        target_modules=target_modules,
        lora_dropout=dropout,
        bias="none",
        use_dora=True,
    )  # type: ignore
    peft_model = get_peft_model(model, config)

    # 🌟 Proteksi ganda: Konversi lora_magnitude_vector ke FP32 dan clamping
    for name, param in peft_model.named_parameters():
        if "lora_magnitude_vector" in name:
            param.data = param.data.float()
            param.data.clamp_(min=1e-6)

    print(
        f"✅ [DoRA Injected Safely with FP32 Magnitude] r={r}, alpha={alpha} pada {target_modules}"
    )
    return peft_model


def apply_prodial(
    model: nn.Module,
    r_eps: int = 8,
    r_b: int = 16,
    target_modules: list = LINEAR_TARGET_MODULES,
) -> nn.Module:
    peft_model = apply_prodial_to_model(
        model, target_modules=target_modules, r_b=r_b, r_eps=r_eps
    )
    print(f"⚡ [ProDiaL Injected] r_b={r_b}, r_eps={r_eps} pada {target_modules}")
    return peft_model


def apply_peft(
    model: nn.Module,
    method: str = "dora",
    r: int = 8,
    alpha: int = 16,
    dropout: float = 0.05,
    r_b: int = 16,
    target_modules: list = None,  # type: ignore
    use_colora_for_conv: bool = True,
) -> nn.Module:
    if method is None or str(method).lower() in ["full", "none"]:
        print("🔓 [PEFT Config] Mode Full Fine-Tuning.")
        return model

    method = method.lower()

    # 1. Bekukan seluruh parameter dasar model
    for param in model.parameters():
        param.requires_grad = False

    # 2. Track 1: Terapkan CoLoRA pada Layer Konvolusi
    if use_colora_for_conv:
        print(
            "🎨 [Dual-Track PEFT] Mengaplikasikan CoLoRA pada layer Konvolusi (2D Spasial)..."
        )
        model = apply_colora_to_model(model, target_modules=CONV_TARGET_MODULES)

    # 3. Track 2: Terapkan Adaptor Linear Pilihan
    raw_targets = LINEAR_TARGET_MODULES if target_modules is None else target_modules

    skipped = [m for m in raw_targets if m in ["dt_proj", "x_proj"]]
    linear_targets = [m for m in raw_targets if m not in ["dt_proj", "x_proj"]]

    if skipped:
        print(
            f"🛡️ [Safety Notice] Modul {skipped} dilewati secara otomatis dari adaptasi PEFT demi stabilitas Mamba SSM."
        )

    if method == "lora":
        model = apply_lora(
            model, r=r, alpha=alpha, dropout=dropout, target_modules=linear_targets
        )
    elif method == "dora":
        model = apply_dora(
            model, r=r, alpha=alpha, dropout=dropout, target_modules=linear_targets
        )
    elif method == "prodial":
        model = apply_prodial(model, r_eps=r, r_b=r_b, target_modules=linear_targets)

    # 4. 🚨 RE-ENABLE CoLoRA (Mencegah ter-freeze ulang oleh get_peft_model / ProDiaL)
    if use_colora_for_conv:
        for m in model.modules():
            if isinstance(m, CoLoRAConv2d):
                for p in m.depthwise.parameters():
                    p.requires_grad = True
                for p in m.pointwise.parameters():
                    p.requires_grad = True

    return model


def get_dora_magnitude_params(model: nn.Module) -> list:
    """Mengambil seluruh tensor lora_magnitude_vector DoRA untuk di-clamp di training loop."""
    return [
        param
        for name, param in model.named_parameters()
        if "lora_magnitude_vector" in name and param.requires_grad
    ]


def merge_peft_and_unload(model: nn.Module) -> nn.Module:
    # 🌟 1. Paksa model ke FP32 sebelum merge agar pembagian norma DoRA aman dari underflow
    if isinstance(model, PeftModel):
        model = model.float()
        model = model.merge_and_unload()

    # 2. Unload CoLoRA
    if any(isinstance(m, CoLoRAConv2d) for m in model.modules()):
        model = merge_colora_weights(model)

    # 3. Unload ProDiaL
    if hasattr(model, "is_prodial") or any(
        "ProDiaLLinear" in m.__class__.__name__ for m in model.modules()
    ):
        model = merge_prodial_weights(model)

    return model
