import os, sys
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset
from torchvision import tv_tensors
import torchvision.transforms.v2 as v2
from .debugger import unpack_split_info


class StrokeSegmentationNPZDataset(Dataset):
    """Dataset class khusus segmentasi untuk memuat Citra CT-Scan, Masker Lesi, dan DTM (Opsional) dari file NPZ."""

    def __init__(
        self,
        images_array,
        masks_array,
        labels_array,
        dtms_array=None,
        use_dtm: bool = False,
    ):
        self.use_dtm = use_dtm

        # 1. Standardisasi Dimensi Citra -> [N, C, H, W]
        if images_array.ndim == 3:  # [N, H, W] -> [N, 1, H, W]
            images_array = np.expand_dims(images_array, axis=1)
        elif images_array.ndim == 4 and images_array.shape[-1] in [1, 3]:
            images_array = np.transpose(images_array, (0, 3, 1, 2))

        # 2. Standardisasi Dimensi Masker -> [N, 1, H, W]
        if masks_array.ndim == 3:  # [N, H, W] -> [N, 1, H, W]
            masks_array = np.expand_dims(masks_array, axis=1)
        elif masks_array.ndim == 4 and masks_array.shape[-1] == 1:
            masks_array = np.transpose(masks_array, (0, 3, 1, 2))

        # 3. Standardisasi Dimensi DTM -> [N, C, H, W] (Jika diaktifkan)
        if self.use_dtm:
            if dtms_array is None:
                raise ValueError(
                    "🚨 Parameter `use_dtm=True`, tetapi `dtms_array` bernilai None / tidak tersedia di NPZ!"
                )
            if (
                dtms_array.ndim == 4 and dtms_array.shape[-1] == 3
            ):  # [N, H, W, C] -> [N, C, H, W]
                dtms_array = np.transpose(dtms_array, (0, 3, 1, 2))
            dtms_array = np.ascontiguousarray(dtms_array)
            self.dtms = torch.from_numpy(dtms_array).float()

        # 💡 KUNCI BEBAS LAG: Memori C-Contiguous untuk zero-copy loading
        images_array = np.ascontiguousarray(images_array)
        masks_array = np.ascontiguousarray(masks_array)
        labels_array = np.ascontiguousarray(labels_array)

        self.images = torch.from_numpy(images_array).float()
        # Masker dikonversi ke Long Tensor untuk target Loss Segmentasi (0: Background, 1: Ischemia, 2: Bleeding)
        self.masks = torch.from_numpy(masks_array).long()
        self.labels = torch.from_numpy(labels_array).long().reshape(-1)

    def __len__(self):
        return len(self.images)

    def __getitem__(self, idx):
        if self.use_dtm:
            return self.images[idx], self.masks[idx], self.labels[idx], self.dtms[idx]
        return self.images[idx], self.masks[idx], self.labels[idx]


# ==============================================================================
# VARIASI SKEMA AUGMENTASI GPU PASANGAN (IMAGE & MASK) UNTUK ABLASI
# ==============================================================================
class GPUSegmentationAugmenter(nn.Module):
    """Wrapper augmentasi GPU pasangan (Image, Mask, DTM) yang diselaraskan 100%
    dengan urutan matriks ablasi tahap klasifikasi DWT-Mamba.
    """

    def __init__(self, trial_mode="trial1", use_dtm: bool = False):
        super().__init__()
        self.trial_mode = trial_mode
        self.use_dtm = use_dtm

        # 🌟 Trial 1: H. Flip + Gauss. Blur + Affine (Sesuai gpu_train_transform utama)
        self.transform_trial1 = v2.Compose(
            [
                v2.RandomHorizontalFlip(p=0.5),
                v2.RandomAffine(
                    degrees=(-5, 5),  # <- Diperbaiki jadi (-5, 5)
                    translate=(0.03, 0.03),
                    scale=(0.95, 1.05),
                    interpolation=v2.InterpolationMode.BILINEAR,
                    fill=0,
                ),
            ]
        )

        self.blur_trial1 = v2.RandomApply(
            [v2.GaussianBlur(kernel_size=3, sigma=(0.1, 0.8))], p=0.3
        )

        # 🌟 Trial 2: Gauss. Blur + Affine (Tanpa H. Flip)
        self.transform_trial2 = v2.Compose(
            [
                v2.RandomAffine(
                    degrees=(-5, 5),  # <- Diperbaiki jadi (-5, 5)
                    translate=(0.03, 0.03),
                    scale=(0.95, 1.05),
                    interpolation=v2.InterpolationMode.BILINEAR,
                    fill=0,
                ),
            ]
        )

        self.blur_trial2 = v2.RandomApply(
            [v2.GaussianBlur(kernel_size=3, sigma=(0.1, 0.8))], p=0.3
        )

        # 🌟 Trial 3: Affine Only (Tanpa H. Flip & Tanpa Blur - Performa terbaik di klasifikasi)
        self.transform_trial3 = v2.Compose(
            [
                v2.RandomAffine(
                    degrees=(-5, 5),  # <- Diperbaiki jadi (-5, 5)
                    translate=(0.03, 0.03),
                    scale=(0.95, 1.05),
                    interpolation=v2.InterpolationMode.BILINEAR,
                    fill=0,
                ),
            ]
        )

        # Cadangan Trial 4: Elastic Transform + Affine (Untuk uji deformasi jaringan lunak)
        self.transform_trial4 = v2.Compose(
            [
                v2.ElasticTransform(alpha=20.0, sigma=5.0, fill=0),
                v2.RandomAffine(
                    degrees=(-5, 5),  # <- Diperbaiki jadi (-5, 5)
                    translate=(0.03, 0.03),
                    scale=(0.95, 1.05),
                    interpolation=v2.InterpolationMode.BILINEAR,
                    fill=0,
                ),
            ]
        )

    def forward(self, images, masks, dtms=None):
        # Mode Passthrough / Eval (Tanpa Augmentasi)
        if self.trial_mode in ["trial0", "identity", "eval"]:
            mask_out = masks.squeeze(1) if masks.ndim == 4 else masks
            if self.use_dtm and dtms is not None:
                dtm_out = (
                    dtms.squeeze(1) if dtms.ndim == 4 and dtms.shape[1] == 1 else dtms
                )
                return images, mask_out, dtm_out
            return images, mask_out

        # Bungkus ke tv_tensors agar transformasi spasial tereksekusi secara sinkron
        images_tv = tv_tensors.Image(images)
        masks_tv = tv_tensors.Mask(masks)

        if self.use_dtm and dtms is not None:
            dtms_tv = tv_tensors.Image(dtms)

        # Seleksi Trial Berdasarkan Urutan Klasifikasi
        if self.trial_mode == "trial1":  # H. Flip + Gauss. Blur + Affine
            if self.use_dtm and dtms is not None:
                img_aug, mask_aug, dtm_aug = self.transform_trial1(
                    images_tv, masks_tv, dtms_tv
                )
            else:
                img_aug, mask_aug = self.transform_trial1(images_tv, masks_tv)
            # Blur HANYA pada Image (Steril untuk Mask dan DTM)
            img_aug = self.blur_trial1(img_aug)

        elif self.trial_mode == "trial2":  # Gauss. Blur + Affine
            if self.use_dtm and dtms is not None:
                img_aug, mask_aug, dtm_aug = self.transform_trial2(
                    images_tv, masks_tv, dtms_tv
                )
            else:
                img_aug, mask_aug = self.transform_trial2(images_tv, masks_tv)
            # Blur HANYA pada Image
            img_aug = self.blur_trial2(img_aug)

        elif self.trial_mode == "trial3":  # Affine Only
            if self.use_dtm and dtms is not None:
                img_aug, mask_aug, dtm_aug = self.transform_trial3(
                    images_tv, masks_tv, dtms_tv
                )
            else:
                img_aug, mask_aug = self.transform_trial3(images_tv, masks_tv)

        elif self.trial_mode == "trial4":  # Elastic Transform + Affine
            if self.use_dtm and dtms is not None:
                img_aug, mask_aug, dtm_aug = self.transform_trial4(
                    images_tv, masks_tv, dtms_tv
                )
            else:
                img_aug, mask_aug = self.transform_trial4(images_tv, masks_tv)
        else:
            img_aug, mask_aug = images_tv, masks_tv
            dtm_aug = dtms_tv if (self.use_dtm and dtms is not None) else None

        # Konversi Masker kembali ke Long Tensor 3D [B, H, W]
        mask_out = mask_aug.as_subclass(torch.Tensor).long()
        if mask_out.ndim == 4 and mask_out.shape[1] == 1:
            mask_out = mask_out.squeeze(1)

        img_out = img_aug.as_subclass(torch.Tensor)

        # Kembalikan output sesuai status penggunaan DTM
        if self.use_dtm and dtms is not None:
            dtm_out = dtm_aug.as_subclass(torch.Tensor).float()  # type: ignore
            return img_out, mask_out, dtm_out

        return img_out, mask_out


def get_stroke_segmentation_dataloaders(
    npz_path="data/dataset_seg_1ch_224x224_nocrop.npz",
    config="splits",
    fold_idx=0,
    batch_size=16,
    eval_batch_size=None,
    num_workers=2,
    aug_trial="trial1",
    loss_type="dice+ce",  # 🌟 Parameter string loss type untuk deteksi otomatis DTM
    use_dtm_override=None,  # 🌟 Opsi manual jika ingin memaksa True/False
):
    """Fungsi pembuat DataLoader Segmentasi yang mendeteksi kebutuhan DTM secara otomatis dari loss_type."""
    if not os.path.exists(npz_path):
        raise FileNotFoundError(
            f"❌ File NPZ Segmentasi tidak ditemukan di path: {npz_path}"
        )

    if eval_batch_size is None:
        eval_batch_size = batch_size * 2

    data = np.load(npz_path, allow_pickle=True)
    images = data["images"]
    masks = data["masks"]
    labels = data["labels"]

    # 🌟 Deteksi otomatis apakah DTM dibutuhkan berdasarkan 'gsl' di loss_type
    components = [c.strip() for c in loss_type.lower().split("+")]
    auto_use_dtm = any("gsl" in c or "surface" in c for c in components)

    # Override jika parameter manual diberikan
    use_dtm = auto_use_dtm if use_dtm_override is None else use_dtm_override

    dtms = None
    if use_dtm:
        if "dtms" in data:
            dtms = data["dtms"]
            print("🗺️ [DataLoader] DTM Terdeteksi & Diaktifkan untuk GSL Loss.")
        else:
            raise KeyError(
                f"❌ Loss Type '{loss_type}' memerlukan DTM, tetapi key 'dtms' tidak ditemukan di file NPZ!"
            )
    else:
        print("⚡ [DataLoader] DTM Dinonaktifkan (Standar Image + Mask).")

    if config not in data and "splits" in data:
        config = "splits"

    if config not in data:
        raise KeyError(f"❌ Key config '{config}' tidak ditemukan dalam NPZ.")

    split_info = unpack_split_info(data[config], fold_idx=fold_idx)
    train_idx = split_info["train"]  # type: ignore
    val_idx = split_info["val"]  # type: ignore
    test_idx = split_info["test"]  # type: ignore

    # Inisialisasi Dataset dengan parameter use_dtm
    train_dataset = StrokeSegmentationNPZDataset(
        images[train_idx],
        masks[train_idx],
        labels[train_idx],
        dtms_array=dtms[train_idx] if use_dtm else None,  # type: ignore
        use_dtm=use_dtm,  # type: ignore
    )
    val_dataset = StrokeSegmentationNPZDataset(
        images[val_idx],
        masks[val_idx],
        labels[val_idx],
        dtms_array=dtms[val_idx] if use_dtm else None,  # type: ignore
        use_dtm=use_dtm,  # type: ignore
    )

    test_dataset = None
    if len(test_idx) > 0:
        test_dataset = StrokeSegmentationNPZDataset(
            images[test_idx],
            masks[test_idx],
            labels[test_idx],
            dtms_array=dtms[test_idx] if use_dtm else None,  # type: ignore
            use_dtm=use_dtm,  # type: ignore
        )

    use_persistent = num_workers > 0
    base_kwargs = {
        "num_workers": num_workers,
        "pin_memory": True,
        "persistent_workers": use_persistent,
    }
    if use_persistent:
        base_kwargs["prefetch_factor"] = 1

    train_loader = DataLoader(
        train_dataset, batch_size=batch_size, shuffle=True, **base_kwargs
    )
    val_loader = DataLoader(
        val_dataset, batch_size=eval_batch_size, shuffle=False, **base_kwargs
    )

    test_loader = None
    if test_dataset is not None:
        test_loader = DataLoader(
            test_dataset, batch_size=eval_batch_size, shuffle=False, **base_kwargs
        )

    gpu_train_augmenter = GPUSegmentationAugmenter(
        trial_mode=aug_trial, use_dtm=use_dtm
    )
    gpu_eval_augmenter = GPUSegmentationAugmenter(trial_mode="eval", use_dtm=use_dtm)

    # Kembalikan 6 nilai persis seperti struktur lama tahap klasifikasi
    return (
        train_loader,
        val_loader,
        test_loader,
        gpu_train_augmenter,
        gpu_eval_augmenter,
    )


@torch.no_grad()
def predict_segmentation_with_tta(model, images):
    """5-Pass Multi-Scale & Rotational TTA khusus masker logits segmentasi."""
    model.eval()

    # Pass 1: Original Image
    out_orig = model(images)

    # Pass 2 & 3: Small Rotations
    img_rot_p = v2.functional.rotate(
        images,
        angle=3,
        interpolation=v2.InterpolationMode.BILINEAR,
        fill=0,  # type: ignore
    )
    out_rot_p = v2.functional.rotate(
        model(img_rot_p),
        angle=-3,
        interpolation=v2.InterpolationMode.BILINEAR,
        fill=0,  # type: ignore
    )

    img_rot_m = v2.functional.rotate(
        images,
        angle=-3,
        interpolation=v2.InterpolationMode.BILINEAR,
        fill=0,  # type: ignore
    )
    out_rot_m = v2.functional.rotate(
        model(img_rot_m),
        angle=3,
        interpolation=v2.InterpolationMode.BILINEAR,
        fill=0,  # type: ignore
    )

    # Average Probabilities
    prob_orig = F.softmax(out_orig, dim=1)
    prob_rot_p = F.softmax(out_rot_p, dim=1)
    prob_rot_m = F.softmax(out_rot_m, dim=1)

    final_prob = (prob_orig + prob_rot_p + prob_rot_m) / 3.0
    return final_prob


if __name__ == "__main__":
    npz_path = "data/dataset_seg_1ch_224x224_nocrop.npz"

    print("=" * 65)
    print("🧪 TESTING SEGMENTATION DATALOADER & AUTO-DTM DETECTION")
    print("=" * 65)

    # Simulasi uji coba dengan loss_type yang memuat GSL (akan otomatis mengaktifkan DTM jika key ada)
    tr_loader, val_loader, ts_loader, augmenter1, augmenter2 = (
        get_stroke_segmentation_dataloaders(
            npz_path=npz_path,
            config="splits",
            batch_size=8,
            aug_trial="trial1",
            loss_type="dice+gsl",
        )
    )

    batch_data = next(iter(tr_loader))
    if len(batch_data) == 4:
        imgs, msks, lbls, dtms = batch_data
        print(f"✅ Data Batch (WITH DTM) Loaded Successfully!")
        print(f"    ├─ DTM Shape : {dtms.shape} (Type: {dtms.dtype})")
    else:
        imgs, msks, lbls = batch_data
        print(f"✅ Data Batch (STANDARD) Loaded Successfully!")

    print(f"    ├─ Raw Image Shape : {imgs.shape} (Type: {imgs.dtype})")
    print(f"    ├─ Raw Mask Shape  : {msks.shape} (Type: {msks.dtype})")
    print(f"    └─ Label Batch     : {lbls.tolist()}")
