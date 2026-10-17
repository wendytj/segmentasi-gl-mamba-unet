import argparse

# ==============================================================================
# DEFAULT CONFIGURATION DICTIONARY
# ==============================================================================
CONFIG = {
    # --------------------------------------------------------------------------
    # 1. Eksperimen & System Config
    # --------------------------------------------------------------------------
    "experiment_code": "mock-run-seg-1",
    "split_config": "split_80_10_10",
    "seed": 42,
    "num_classes": 2,
    "npz_path": "data/dataset_seg_2ch_224x224_nocrop_ratio_0_1.npz",
    "pretrained_path": "weights/classification_merged_best.pth",
    "batch_size": 16,
    "eval_batch_size": 16,
    "accumulation_steps": 4,
    "num_workers": 4,
    "img_size": 224,
    "aug_trial": "trial1",  # 'trial1', 'trial2', 'trial3', atau 'trial4'
    # --------------------------------------------------------------------------
    # 2. Saklar Model & Fine-Tuning Stage 2
    # --------------------------------------------------------------------------
    "finetune_strategy": "peft",  # Opsi 4 strategi: 'full', 'selective', 'peft', 'frozen'
    "encoder_lr_ratio": 0.1,  # Rasio LR untuk backbone pretrained DWT (misal 0.1x dari Effective LR)
    "min_lesion_pixels": 10,  # Threshold piksel lesi minimum untuk klasifikasi slice 3-kelas
    # --------------------------------------------------------------------------
    # 3. Arsitektur DWT-Mamba Backbone
    # --------------------------------------------------------------------------
    "in_channels": 1,
    "embed_dim": 256,
    "depth": 3,
    "mamba_d_state": 16,
    "mamba_d_conv": 4,
    "mamba_expand": 2,
    "se_reduction": 16,
    "mb_gsf_reduction": 4,
    "latent_dim": 128,
    "proj_dim": 256,
    # --------------------------------------------------------------------------
    # 4. Konfigurasi PEFT (ProDiaL / LoRA / DoRA / CoLoRA)
    # --------------------------------------------------------------------------
    "peft_method": "prodial",  # 'lora', 'dora', 'prodial', 'colora', 'full', atau 'none'
    "target_modules": ["in_proj", "out_proj"],
    "lora_r": 32,
    "lora_alpha": 64,
    "lora_dropout": 0.0,
    "prodial_r_eps": 32,
    "prodial_r_b": 128,
    # --------------------------------------------------------------------------
    # 5. Konfigurasi Loss Function (Dinamis)
    # --------------------------------------------------------------------------
    "loss_type": "dice+ce",  # 'dice+ce', 'tversky+ce', 'tversky+focal', 'dice+gsl', 'tversky+gsl'
    "is_generalized": False,
    "use_class_weights": True,  # Pembobotan kelas piksel statis
    "weight_mode": "median",  # Pilihan: ['dbce', 'cdw', 'raw', 'median', 'cui_cb']
    "weight_beta": 0.9999,  # Effective Number of Samples (Cui et al., 2019)
    "tversky_alpha": 0.3,  # Hyperparameter Tversky Loss
    "tversky_beta": 0.7,
    "focal_gamma": 2.0,  # Hyperparameter Focal Loss (Lin et al., 2017)
    "compound_weight_a": 1.0,  # Bobot komponen loss pertama
    "compound_weight_b": 1.0,  # Bobot komponen loss kedua
    # --------------------------------------------------------------------------
    # 6. Konfigurasi Optimizer & Pelatihan
    # --------------------------------------------------------------------------
    "optimizer": "AdamW",
    "learning_rate": 5e-4,
    "effective_lr": 5e-4,
    "lr_conv_ratio": 0.2,  # Rasio LR layer Conv CoLoRA terhadap Effective LR
    "weight_decay": 1e-3,
    "max_epochs": 150,
    "warmup_epochs": 15,
    "early_stop_patience": 25,
    "early_stop_delta": 0.000,
}


# ==============================================================================
# ARGPARSE CLI PARSER
# ==============================================================================
def parse_args():
    parser = argparse.ArgumentParser(
        description=(
            "Stage 2 Hybrid DWT-GLMamba UNet Segmentation & Fine-Tuning Pipeline"
        )
    )

    # 1. Dataset & System Arguments
    parser.add_argument(
        "--experiment_code", type=str, default=CONFIG["experiment_code"]
    )
    parser.add_argument(
        "--split_config",
        type=str,
        default=CONFIG["split_config"],
        help="Skema dataset split (misal: split_80_10_10, split_70_15_15)",
    )
    parser.add_argument("--npz_path", type=str, default=CONFIG["npz_path"])
    parser.add_argument(
        "--pretrained_path", type=str, default=CONFIG["pretrained_path"]
    )
    parser.add_argument(
        "--in_channels",
        type=int,
        default=CONFIG["in_channels"],
        choices=[1, 3],
    )
    parser.add_argument(
        "--img_size",
        type=int,
        default=CONFIG["img_size"],
        help="Ukuran resolusi citra input (224, 384, 512)",
    )
    parser.add_argument(
        "--aug_trial",
        type=str,
        default=CONFIG["aug_trial"],
        choices=["trial1", "trial2", "trial3", "trial4"],
    )

    parser.add_argument(
        "--finetune_strategy",
        type=str,
        default=CONFIG["finetune_strategy"],
        choices=["full", "selective", "peft", "frozen"],
        help="4 Saklar strategi fine-tuning pada DWT Encoder",
    )
    parser.add_argument(
        "--encoder_lr_ratio",
        type=float,
        default=CONFIG["encoder_lr_ratio"],
        help="Rasio Learning Rate DWT Encoder terhadap Effective LR",
    )
    parser.add_argument(
        "--min_lesion_pixels",
        type=int,
        default=CONFIG["min_lesion_pixels"],
        help="Threshold piksel lesi minimum untuk klaim slice positif lesi",
    )

    # 3. Training & Batch Size Arguments
    parser.add_argument("--batch_size", type=int, default=CONFIG["batch_size"])
    parser.add_argument(
        "--eval_batch_size", type=int, default=CONFIG["eval_batch_size"]
    )
    parser.add_argument(
        "--accumulation_steps", type=int, default=CONFIG["accumulation_steps"]
    )
    parser.add_argument("--num_workers", type=int, default=CONFIG["num_workers"])
    parser.add_argument("--max_epochs", type=int, default=CONFIG["max_epochs"])
    parser.add_argument("--warmup_epochs", type=int, default=CONFIG["warmup_epochs"])
    parser.add_argument(
        "--early_stop_patience",
        type=int,
        default=CONFIG["early_stop_patience"],
        help="Patience counter untuk early stopping berbasis val_mean_dsc",
    )
    parser.add_argument(
        "--early_stop_delta",
        type=float,
        default=CONFIG["early_stop_delta"],
        help="Ambang batas minimum perbaikan val_mean_dsc",
    )

    # 4. Hyperparameters Optimization
    parser.add_argument("--learning_rate", type=float, default=CONFIG["learning_rate"])
    parser.add_argument(
        "--lr_conv_ratio",
        type=float,
        default=CONFIG["lr_conv_ratio"],
        help="Rasio Learning Rate layer Conv CoLoRA terhadap Effective LR",
    )
    parser.add_argument("--weight_decay", type=float, default=CONFIG["weight_decay"])

    # 5. PEFT Arguments
    parser.add_argument(
        "--peft_method",
        type=str,
        default=CONFIG["peft_method"],
        choices=["lora", "dora", "prodial", "colora", "full", "none"],
    )
    parser.add_argument("--lora_r", type=int, default=CONFIG["lora_r"])
    parser.add_argument("--lora_alpha", type=int, default=CONFIG["lora_alpha"])
    parser.add_argument("--lora_dropout", type=float, default=CONFIG["lora_dropout"])
    parser.add_argument("--prodial_r_eps", type=int, default=CONFIG["prodial_r_eps"])
    parser.add_argument("--prodial_r_b", type=int, default=CONFIG["prodial_r_b"])
    parser.add_argument("--target_modules", nargs="+", default=CONFIG["target_modules"])

    # 6. Loss Function Arguments
    parser.add_argument(
        "--loss_type",
        type=str,
        default=CONFIG["loss_type"],
        help=(
            "Fungsi loss (misal: dice+ce, dice+wce, tversky+ce, tversky+focal,"
            " dice+gsl)"
        ),
    )
    parser.add_argument(
        "--use_class_weights",
        action=argparse.BooleanOptionalAction,
        default=CONFIG["use_class_weights"],
        help="Gunakan pembobotan kelas piksel statis",
    )
    parser.add_argument(
        "--is_generalized",
        action=argparse.BooleanOptionalAction,
        default=CONFIG.get("is_generalized", False),
        help="Gunakan Generalized Dice/Tversky Loss berbasis volume",
    )
    parser.add_argument(
        "--weight_mode",
        type=str,
        default=CONFIG["weight_mode"],
        choices=["median", "cui_cb", "cdw", "dbce"],
        help="Skema pembobotan piksel berdasarkan 4 literatur utama",
    )
    parser.add_argument("--weight_beta", type=float, default=CONFIG["weight_beta"])
    parser.add_argument("--tversky_alpha", type=float, default=CONFIG["tversky_alpha"])
    parser.add_argument("--tversky_beta", type=float, default=CONFIG["tversky_beta"])
    parser.add_argument("--focal_gamma", type=float, default=CONFIG["focal_gamma"])
    parser.add_argument(
        "--compound_weight_a", type=float, default=CONFIG["compound_weight_a"]
    )
    parser.add_argument(
        "--compound_weight_b", type=float, default=CONFIG["compound_weight_b"]
    )

    # Mode Sanity Check
    parser.add_argument("--dry_run", action="store_true")

    return parser.parse_args()
