import os
import torch
from .configs import CONFIG


def safe_atomic_save(obj, path):
    tmp_path = path + ".tmp"
    torch.save(obj, tmp_path)
    os.replace(tmp_path, path)


def sanitize_peft_hparams(config: dict) -> dict:
    """Membersihkan parameter PEFT yang tidak relevan agar JSON log tetap rapi."""
    clean_cfg = config.copy()
    method = str(clean_cfg.get("peft_method", "full")).lower()

    if method in ["full", "none"]:
        clean_cfg["peft_method"] = "full"
        # 🌟 Hapus seluruh parameter spesifik PEFT (termasuk target_modules)
        for key in [
            "lora_r",
            "lora_alpha",
            "lora_dropout",
            "prodial_r_eps",
            "prodial_r_b",
            "target_modules",
            "lr_conv_ratio",
        ]:
            clean_cfg.pop(key, None)

    elif method in ["lora", "dora"]:
        # Hapus parameter spesifik ProDiAL
        clean_cfg.pop("prodial_r_eps", None)
        clean_cfg.pop("prodial_r_b", None)

    elif method == "prodial":
        # Hapus parameter spesifik LoRA/DoRA (Tetap simpan lora_dropout jika dipakai oleh ProDiAL)
        clean_cfg.pop("lora_r", None)
        clean_cfg.pop("lora_alpha", None)
        clean_cfg.pop("lora_dropout", None)

    return clean_cfg


def sanitize_loss_hparams(config: dict) -> dict:
    clean_cfg = config.copy()
    loss_type_raw = str(clean_cfg.get("loss_type", "dice+ce")).lower()
    components = [c.strip() for c in loss_type_raw.split("+")]

    # Periksa komponen apa saja yang aktif dalam string compound loss
    has_tversky = "tversky" in components
    has_focal = "focal" in components

    if not has_tversky:
        clean_cfg.pop("tversky_alpha", None)
        clean_cfg.pop("tversky_beta", None)
    if not has_focal:
        clean_cfg.pop("focal_gamma", None)

    return clean_cfg


def sanitize_class_weight_hparams(config: dict) -> dict:
    """Membersihkan parameter class weight yang tidak relevan agar JSON log tetap rapi."""
    clean_cfg = config.copy()
    if not clean_cfg.get("use_class_weights", False):
        for key in ["weight_mode", "weight_beta"]:
            clean_cfg.pop(key, None)
    else:
        mode = clean_cfg.get("weight_mode", "median")
        # weight_beta hanya dibutuhkan jika mode 'cui_cb' yang aktif
        if mode != "cui_cb":
            clean_cfg.pop("weight_beta", None)
    return clean_cfg


def sanitize_all_hparams(config: dict) -> dict:
    """Master function untuk membersihkan hparams PEFT, Loss, dan Class Weights secara sekaligus."""
    clean_cfg = sanitize_peft_hparams(config)
    clean_cfg = sanitize_loss_hparams(clean_cfg)
    clean_cfg = sanitize_class_weight_hparams(clean_cfg)
    return clean_cfg


def run_mock_test(model, train_loader, device, gpu_transform=None):
    """Fungsi Dry-Run 1 Batch untuk memvalidasi aliran tensor, shape output,

    dan alokasi VRAM sebelum pelatihan skala penuh dimulai.
    """
    print("\n🔍 Memulai Mock Test (Dry-Run 1 Batch)...")
    model.eval()

    try:
        batch = next(iter(train_loader))

        images = None
        masks = None
        dtms = None
        labels = None

        # ======================================================================
        # 1. UNPACK BATCH DENGAN AMAN (BISA UNTUK TUPLE/LIST DENGAN 2, 3, ATAU 4 ITEM)
        # ======================================================================
        if isinstance(batch, (tuple, list)):
            images = batch[0]
            masks = batch[1]

            # Jika batch punya lebih dari 2 elemen (misal: DTM dan/atau Label)
            if len(batch) >= 3:
                elem2 = batch[2]
                # DTM adalah tensor spasial 3D/4D ([B, H, W, C] atau [B, C, H, W])
                if isinstance(elem2, torch.Tensor) and elem2.ndim >= 3:
                    dtms = elem2
                    if len(batch) >= 4:
                        labels = batch[3]
                else:
                    labels = elem2

        elif isinstance(batch, dict):
            images = batch.get("image", batch.get("images"))
            masks = batch.get("mask", batch.get("masks"))
            dtms = batch.get("dtm", batch.get("dtms"), None)  # type: ignore
            labels = batch.get("label", batch.get("labels"), None)  # type: ignore
        else:
            raise TypeError(f"🚨 Tipe data batch tidak didukung: {type(batch)}")

        # Transfer Tensor ke Device (GPU/CPU)
        images = images.to(device, non_blocking=True)  # type: ignore

        if masks is not None:
            masks = masks.to(device, non_blocking=True).long()

        if dtms is not None:
            dtms = dtms.to(device, non_blocking=True)

        if labels is not None:
            labels = labels.to(device, non_blocking=True)

        # ======================================================================
        # 2. AUTO-CAST & GPU TRANSFORM EXECUTION
        # ======================================================================
        use_bf16 = torch.cuda.is_bf16_supported() if device.type == "cuda" else False
        amp_dtype = torch.bfloat16 if use_bf16 else torch.float16

        with (
            torch.no_grad(),
            torch.autocast(
                device_type=device.type,
                dtype=amp_dtype,
                enabled=(device.type == "cuda"),
            ),
        ):
            # Pengaplikasian Augmentasi GPU jika diaktifkan
            if gpu_transform is not None:
                if dtms is not None:
                    trans_out = gpu_transform(images, masks, dtms)
                else:
                    trans_out = gpu_transform(images, masks)

                if isinstance(trans_out, (tuple, list)):
                    if len(trans_out) >= 3:
                        images, masks, dtms = trans_out[:3]
                    elif len(trans_out) == 2:
                        images, masks = trans_out
                        dtms = None
                else:
                    images = trans_out

            # Executing Model Forward Pass
            try:
                outputs = model(images)
            except RuntimeError as e:
                if "out of memory" in str(e).lower():
                    print("\n🚨 CUDA OOM TERDETEKSI SAAT MOCK TEST!")
                    if torch.cuda.is_available():
                        print(
                            "Alokasi VRAM saat ini :"
                            f" {torch.cuda.memory_allocated() / 1e9:.2f} GB"
                        )
                        print(
                            "VRAM yang direservasi :"
                            f" {torch.cuda.memory_reserved() / 1e9:.2f} GB"
                        )
                        print(
                            "Maksimum VRAM terpakai :"
                            f" {torch.cuda.max_memory_allocated() / 1e9:.2f} GB"
                        )
                    print(f"Detail Error: {e}")
                raise e

        # ======================================================================
        # 3. VALIDASI BENTUK TENSOR (ASSERTIONS)
        # ======================================================================
        assert outputs.dim() == 4, (
            "Output segmentasi harus berdimensi 4 (B, C, H, W)! Got shape:"
            f" {outputs.shape}"
        )
        assert outputs.shape[0] == images.size(0), (
            f"Batch size output ({outputs.shape[0]}) tidak sesuai input"
            f" ({images.size(0)})!"
        )

        expected_classes = CONFIG.get("num_classes", 2)
        assert outputs.shape[1] == expected_classes, (
            f"Jumlah kanal output tidak sesuai! (Expected {expected_classes},"
            f" got {outputs.shape[1]})"
        )

        # Print Laporan Log
        print("✅ Mock Test Berhasil!")
        print(f"   - Input Shape  : {images.shape}")
        print(f"   - Mask Shape   : {masks.shape if masks is not None else 'N/A'}")
        if dtms is not None:
            print(f"   - DTM Shape    : {dtms.shape}")
        print(f"   - Output Shape : {outputs.shape}")

        if device.type == "cuda":
            print(
                "   - Memory VRAM  :" f" {torch.cuda.memory_allocated() / 1e6:.2f} MB\n"
            )

        return True

    except Exception as e:
        print(f"❌ Mock Test Gagal: {e}")
        raise e
