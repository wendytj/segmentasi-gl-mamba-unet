import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, TensorDataset

# ============================================================================
# 1. HELPER & UTILITIES FOR CLASS WEIGHTS & DTM
# ============================================================================


def compute_class_weights(
    train_loader,
    num_classes: int = 2,
    device: torch.device = torch.device("cpu"),
    use_class_weights: bool = True,
    weight_mode: str = "dbce",  # Pilihan: ['dbce', 'cdw', 'raw', 'median', 'cui_cb']
    beta: float = 0.999999,
) -> tuple[torch.Tensor, torch.Tensor]:

    if train_loader is None:
        raise ValueError("🚨 Sediakan `train_loader` untuk menghitung bobot piksel!")

    pos_counts: torch.Tensor | None = None
    total_pixels_per_ch: torch.Tensor | None = None

    print(
        "⏳ [Class Weights] Menghitung distribusi piksel 2-kanal biner"
        " dataset (Train Set) secara aktual..."
    )

    for batch in train_loader:
        targets = batch["mask"] if isinstance(batch, dict) else batch[1]
        if hasattr(targets, "cpu"):
            targets = targets.cpu().float()

        # Normalisasi dimensi target ke [B, C, H, W]
        if (
            targets.ndim == 4
            and targets.shape[1] != num_classes
            and targets.shape[-1] in [1, 2, 3]
        ):
            targets = targets.permute(0, 3, 1, 2)
        elif targets.ndim == 3:
            targets = targets.unsqueeze(1)

        actual_channels = targets.shape[1]
        if pos_counts is None or total_pixels_per_ch is None:
            pos_counts = torch.zeros(actual_channels, dtype=torch.float32)
            total_pixels_per_ch = torch.zeros(actual_channels, dtype=torch.float32)

        num_voxels = targets.shape[0] * targets.shape[2] * targets.shape[3]
        pos_counts += targets.sum(dim=[0, 2, 3])
        total_pixels_per_ch += num_voxels

    # 🌟 Guard Check untuk Pyright Type-Checker & Empty DataLoader Safety
    if pos_counts is None or total_pixels_per_ch is None:
        raise ValueError("🚨 `train_loader` kosong atau tidak memiliki data mask!")

    neg_counts = total_pixels_per_ch - pos_counts
    pixel_counts = torch.stack([pos_counts, neg_counts], dim=1)  # Shape [C, 2]

    # Total piksel global dihitung otomatis dari akumulasi piksel aktual train set
    total_pixels = float(total_pixels_per_ch[0].item())

    # Jika user set use_class_weights=False, langsung return 1.0 (dummy weights)
    if not use_class_weights:
        return torch.ones(num_classes, device=device), pixel_counts.to(device)

    # ==========================================================================
    # LOGIKA DI BAWAH INI TETAP UTUH SEPERTI SEMULA
    # ==========================================================================

    # 2. Extract pos_c dan neg_c secara aman
    if pixel_counts.ndim == 1:
        pos_c = pixel_counts.float().to(device) + 1e-6
        neg_c = torch.clamp(total_pixels - pos_c, min=1e-6)
        pixel_counts = torch.stack([pos_c, neg_c], dim=1)
    else:
        pos_c = pixel_counts[:, 0].float().to(device) + 1e-6
        neg_c = pixel_counts[:, 1].float().to(device) + 1e-6

    # 3. Kalkulasi Rasio Murni N_neg / N_pos per Kanal Secara Independen
    raw_ratio = neg_c / pos_c

    if weight_mode == "raw":
        raw_weights = raw_ratio
        max_clamp = 300.0
    elif weight_mode == "median":
        # 1. Total piksel background murni
        bg_pixels = (total_pixels - pos_c.sum()).clamp(min=1.0)

        # 2. Distribusi frekuensi: f_bg [1] dan f_fg [C]
        f_bg = (bg_pixels / total_pixels).clamp(min=1e-6)
        f_fg = (pos_c / total_pixels).clamp(min=1e-6)

        # 3. Gabungkan seluruh frekuensi [1 + C] untuk mencari median global
        all_freqs = torch.cat([f_bg.unsqueeze(0), f_fg])
        median_freq = torch.median(all_freqs)

        # 4. Bobot Median Frequency Balancing Murni (Eigen & Fergus, 2015)
        # Kelas median akan bernilai 1.0, kelas yang lebih langka bernilai > 1.0
        raw_weights = median_freq / f_fg
        max_clamp = 20.0
    elif weight_mode == "cdw":
        raw_weights = torch.sqrt(raw_ratio)
        max_clamp = 50.0
    elif weight_mode == "dbce":
        raw_weights = torch.log(raw_ratio + 1.0)
        max_clamp = 20.0
    elif weight_mode == "cui_cb":
        n_samples = (
            len(train_loader.dataset)
            if hasattr(train_loader, "dataset") and len(train_loader.dataset) > 0
            else float(total_pixels / (224 * 224))
        )

        n_pos = (pos_c / total_pixels) * n_samples
        n_neg = (neg_c / total_pixels) * n_samples

        # 1. Hitung bobot Cui et al. persis sesuai rumus asli: W_y = (1 - beta) / (1 - beta^n)
        w_pos = (1.0 - beta) / (1.0 - torch.pow(beta, n_pos) + 1e-7)
        w_neg = (1.0 - beta) / (1.0 - torch.pow(beta, n_neg) + 1e-7)

        # 2. Relatifkan ke background untuk pos_weight BCE
        raw_weights = w_pos / w_neg
        max_clamp = 50.0
    else:
        raise ValueError(f"🚨 Unknown weight_mode: '{weight_mode}'")

    class_weights = torch.clamp(raw_weights, min=0.1, max=max_clamp).to(device)
    return class_weights, pixel_counts


# ============================================================================
# 2. BASE LOSS COMPONENTS (DENGAN SAFE DIMENSION ALIGNMENT)
# ============================================================================


def _align_shapes(logits: torch.Tensor, targets: torch.Tensor):
    """Helper privat untuk memastikan logits dan targets memiliki shape [B, C, H, W] dan tipe float32."""
    if targets.ndim == 4 and targets.shape[-1] == logits.shape[1]:
        targets = targets.permute(0, 3, 1, 2)
    elif targets.ndim == 3:
        targets = targets.unsqueeze(1)

    if targets.shape[1] != logits.shape[1]:
        C = logits.shape[1]
        if targets.shape[1] > C:
            targets = targets[:, :C, :, :]

    # 🌟 PAKSA KEDUA TENSOR KE FLOAT32 (Mencegah NaN saat akumulasi spasial di Mixed Precision)
    return logits.float(), targets.float()


class DiceLoss(nn.Module):
    """Multi-Channel Binary Soft Dice Loss (Global Batch-Wise Accumulation)."""

    def __init__(
        self,
        smooth: float = 1e-5,
        is_generalized: bool = False,
        include_background: bool = True,
    ):
        super().__init__()
        self.smooth = smooth
        self.is_generalized = is_generalized
        self.include_background = include_background

    def forward(self, logits: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        logits, targets = _align_shapes(logits, targets)

        probs = torch.sigmoid(logits)

        if self.is_generalized:
            # GDL dihitung per sampel lalu dirata-rata
            probs_flat = probs.contiguous().view(probs.size(0), probs.size(1), -1)
            targets_flat = targets.contiguous().view(
                targets.size(0), targets.size(1), -1
            )
            intersection = torch.sum(probs_flat * targets_flat, dim=-1)
            cardinality = torch.sum(probs_flat + targets_flat, dim=-1)

            volumes = torch.sum(targets_flat, dim=-1)
            weights = 1.0 / (volumes**2 + self.smooth)
            weights = weights / (torch.sum(weights, dim=-1, keepdim=True) + self.smooth)

            num = 2.0 * torch.sum(weights * intersection, dim=-1) + self.smooth
            den = torch.sum(weights * cardinality, dim=-1) + self.smooth
            dice_score = num / den
            return torch.mean(1.0 - dice_score)
        else:
            # 🌟 Standar Global Batch-Wise Sum (Sangat Stabil untuk Imbalance)
            intersection = torch.sum(probs * targets, dim=(0, 2, 3))
            cardinality = torch.sum(probs + targets, dim=(0, 2, 3))

            dice_score = (2.0 * intersection + self.smooth) / (
                cardinality + self.smooth
            )
            return torch.mean(1.0 - dice_score)


class TverskyLoss(nn.Module):
    """Multi-Channel Binary Tversky Loss (Global Batch-Wise Accumulation)."""

    def __init__(
        self,
        alpha: float = 0.3,
        beta: float = 0.7,
        smooth: float = 1e-5,
        is_generalized: bool = False,
        include_background: bool = True,
    ):
        super().__init__()
        self.alpha = alpha
        self.beta = beta
        self.smooth = smooth
        self.is_generalized = is_generalized
        self.include_background = include_background

    def forward(self, logits: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        logits, targets = _align_shapes(logits, targets)

        probs = torch.sigmoid(logits)

        if self.is_generalized:
            probs_flat = probs.contiguous().view(probs.size(0), probs.size(1), -1)
            targets_flat = targets.contiguous().view(
                targets.size(0), targets.size(1), -1
            )
            tp = torch.sum(probs_flat * targets_flat, dim=-1)
            fp = torch.sum(probs_flat * (1.0 - targets_flat), dim=-1)
            fn = torch.sum((1.0 - probs_flat) * targets_flat, dim=-1)

            volumes = torch.sum(targets_flat, dim=-1)
            weights = 1.0 / (volumes**2 + self.smooth)
            weights = weights / (torch.sum(weights, dim=-1, keepdim=True) + self.smooth)

            num = torch.sum(weights * tp, dim=-1) + self.smooth
            den = (
                torch.sum(weights * (tp + self.alpha * fp + self.beta * fn), dim=-1)
                + self.smooth
            )
            tversky_score = num / den
            return torch.mean(1.0 - tversky_score)
        else:
            # 🌟 Standar Global Batch-Wise Sum
            tp = torch.sum(probs * targets, dim=(0, 2, 3))
            fp = torch.sum(probs * (1.0 - targets), dim=(0, 2, 3))
            fn = torch.sum((1.0 - probs) * targets, dim=(0, 2, 3))

            tversky_score = (tp + self.smooth) / (
                tp + self.alpha * fp + self.beta * fn + self.smooth
            )
            return torch.mean(1.0 - tversky_score)


class FocalLoss2D(nn.Module):
    """Binary Multi-Channel Focal Loss 2D untuk piksel segmentasi."""

    def __init__(
        self,
        gamma: float = 2.0,
        weight: torch.Tensor = None,  # type: ignore
        reduction: str = "mean",
    ):
        super().__init__()
        self.gamma = gamma
        self.reduction = reduction
        if weight is not None and weight.ndim == 1:
            weight = weight.view(1, -1, 1, 1).float()
        self.register_buffer("weight", weight)

    def forward(self, logits: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        logits, targets = _align_shapes(logits, targets)

        targets = targets.float()
        logits = logits.float()

        weight_dev = self.weight.to(logits.device) if self.weight is not None else None

        bce_loss = F.binary_cross_entropy_with_logits(
            logits, targets, pos_weight=weight_dev, reduction="none"  # type: ignore
        )
        p = torch.sigmoid(logits)
        p_t = p * targets + (1.0 - p) * (1.0 - targets)

        focal_weight = (1.0 - p_t) ** self.gamma
        loss = focal_weight * bce_loss

        if self.reduction == "mean":
            return loss.mean()
        elif self.reduction == "sum":
            return loss.sum()
        return loss


class GeneralizedSurfaceLoss(nn.Module):
    """Generalized Surface Loss (GSL) berbasis DTM sesuai Celaya et al.
    """

    def __init__(self, smooth: float = 1e-5):
        super().__init__()
        self.smooth = smooth

    def forward(
        self,
        logits: torch.Tensor,
        targets: torch.Tensor,
        dtm_targets: torch.Tensor = None,
    ) -> torch.Tensor:
        if dtm_targets is None:
            raise ValueError(
                "🚨 GeneralizedSurfaceLoss membutuhkan `dtm_targets` (DTM Map)!"
            )

        logits, targets = _align_shapes(logits, targets)
        _, dtm_targets = _align_shapes(logits, dtm_targets)

        probs = torch.sigmoid(logits)

        # Suku residual Celaya et al.: (1 - (T + P))
        surface_term = dtm_targets * (1.0 - (targets + probs))
        numerator = torch.sum(surface_term**2, dim=(2, 3))
        denominator = torch.sum(dtm_targets**2, dim=(2, 3)) + self.smooth

        # Eq. 12: L_gsl = 1 - (Numerator / Denominator)
        # Langsung return tanpa pengurangan ganda
        loss_gsl = 1.0 - (numerator / denominator)
        return torch.mean(loss_gsl)


# ============================================================================
# 3. COMPOUND LOSS WRAPPER
# ============================================================================


class CompoundLoss(nn.Module):
    """Wrapper untuk menggabungkan dua atau lebih Loss Component."""

    def __init__(
        self,
        loss_a: nn.Module,
        loss_b: nn.Module,
        weight_a: float = 1.0,
        weight_b: float = 1.0,
    ):
        super().__init__()
        self.loss_a = loss_a
        self.loss_b = loss_b
        self.weight_a = weight_a
        self.weight_b = weight_b

    def forward(
        self,
        logits: torch.Tensor,
        targets: torch.Tensor,
        dtm_targets: torch.Tensor = None,  # type: ignore
    ) -> torch.Tensor:
        if isinstance(self.loss_a, GeneralizedSurfaceLoss):
            val_a = self.loss_a(logits, targets, dtm_targets)
        else:
            val_a = self.loss_a(logits, targets)

        if isinstance(self.loss_b, GeneralizedSurfaceLoss):
            val_b = self.loss_b(logits, targets, dtm_targets)
        else:
            val_b = self.loss_b(logits, targets)

        return (self.weight_a * val_a) + (self.weight_b * val_b)


# ============================================================================
# 4. BUILDER FACTORY
# ============================================================================


def build_loss_criterion(
    config: dict, class_weights: torch.Tensor = None  # type: ignore
) -> nn.Module:
    """Factory Function untuk membangun Loss Function berbasis CONFIG."""
    loss_type_raw = config.get("loss_type", "dice+ce").lower()
    is_generalized = config.get("is_generalized", False)

    w_a = config.get("compound_weight_a", 1.0)
    w_b = config.get("compound_weight_b", 1.0)

    use_weights = config.get("use_class_weights", True) and class_weights is not None
    weights = class_weights if use_weights else None

    tversky_alpha = config.get("tversky_alpha", 0.3)
    tversky_beta = config.get("tversky_beta", 0.7)
    focal_gamma = config.get("focal_gamma", 2.0)

    if use_weights:
        cw_list = (
            class_weights.view(-1).cpu().tolist()
            if hasattr(class_weights, "cpu")
            else list(class_weights)
        )
        cw_formatted = [round(float(w), 4) for w in cw_list]
        weight_str = f"Weighted=True | Weights: {cw_formatted}"
    else:
        weight_str = "Weighted=False"

    def _instantiate_single_loss(name: str) -> nn.Module:
        if name in ["dice", "soft_dice"]:
            return DiceLoss(is_generalized=is_generalized)
        elif name in ["tversky"]:
            return TverskyLoss(
                alpha=tversky_alpha,
                beta=tversky_beta,
                is_generalized=is_generalized,
            )
        elif name in ["ce", "cross_entropy", "wce"]:
            # Buat custom class wrapper untuk BCEWithLogitsLoss agar 100% aman dari casting Long/Float
            class CustomBCELoss(nn.Module):
                def __init__(self, pos_weight=None):
                    super().__init__()
                    self.bce = nn.BCEWithLogitsLoss(pos_weight=pos_weight)

                def forward(self, logits, targets):
                    logits, targets = _align_shapes(logits, targets)
                    return self.bce(logits.float(), targets.float())

            pos_w = (
                weights.view(1, -1, 1, 1).float().to(weights.device)
                if weights is not None
                else None
            )
            return CustomBCELoss(pos_weight=pos_w)
        elif name in ["focal", "focal_loss"]:
            return FocalLoss2D(gamma=focal_gamma, weight=weights)  # type: ignore
        elif name in ["gsl", "surface", "surface_loss"]:
            return GeneralizedSurfaceLoss()
        else:
            raise ValueError(f"🚨 Sub-loss '{name}' tidak dikenali!")

    components = [c.strip() for c in loss_type_raw.split("+")]

    if len(components) == 1:
        single_name = components[0]
        loss_obj = _instantiate_single_loss(single_name)
        gen_str = (
            f" | Generalized={is_generalized}"
            if single_name in ["dice", "tversky"]
            else ""
        )
        print(
            f"🎯 [Loss Criterion] Active: {single_name.upper()}"
            f" ({weight_str}{gen_str})"
        )
        return loss_obj

    elif len(components) == 2:
        comp_a, comp_b = components[0], components[1]
        loss_a_obj = _instantiate_single_loss(comp_a)
        loss_b_obj = _instantiate_single_loss(comp_b)

        compound_loss = CompoundLoss(
            loss_a=loss_a_obj,
            loss_b=loss_b_obj,
            weight_a=w_a,
            weight_b=w_b,
        )

        gen_str = (
            f" | Generalized={is_generalized}"
            if any(x in ["dice", "tversky"] for x in [comp_a, comp_b])
            else ""
        )
        hparam_str = f"w_a={w_a}, w_b={w_b}"
        if "tversky" in components:
            hparam_str += f", alpha={tversky_alpha}, beta={tversky_beta}"
        if "focal" in components:
            hparam_str += f", gamma={focal_gamma}"

        print(
            f"🎯 [Loss Criterion] Active Compound: {comp_a.upper()} +"
            f" {comp_b.upper()} ({hparam_str} | {weight_str}{gen_str})"
        )
        return compound_loss

    else:
        raise ValueError(
            f"🚨 Format loss_type '{loss_type_raw}' terlalu kompleks. Maksimal 2"
            " komponen dipisah '+'."
        )


# ============================================================================
# 5. SANITY CHECK RUNNER (VERIFIKASI 5 MATRIKS ABLASI)
# ============================================================================

if __name__ == "__main__":
    device = torch.device("cpu")
    num_classes = 2  # 2 Kanal Biner (Ischemia & Bleeding)
    batch_size, h, w = 2, 64, 64
    num_samples = 8  # Total sampel dummy untuk dataset bayangan

    # ==========================================================================
    # 1. BUAT TRAIN LOADER BAYANGAN (MOCK DATALOADER)
    # ==========================================================================
    dummy_images = torch.randn(num_samples, 1, h, w)
    # Target mask 2-kanal biner: [N, 2, H, W]
    dummy_masks = torch.randint(0, 2, (num_samples, num_classes, h, w)).float()

    # Wrap ke dalam TensorDataset dan DataLoader
    dummy_dataset = TensorDataset(dummy_images, dummy_masks)
    dummy_train_loader = DataLoader(dummy_dataset, batch_size=batch_size, shuffle=False)

    # ==========================================================================
    # 2. HITUNG DYNAMIC CLASS WEIGHTS DARI LOADER BAYANGAN
    # ==========================================================================
    weights, counts = compute_class_weights(
        train_loader=dummy_train_loader,
        num_classes=num_classes,
        device=device,
        use_class_weights=True,
        weight_mode="cui_cb",
    )

    dummy_logits = torch.randn(batch_size, num_classes, h, w, requires_grad=True)
    dummy_targets = torch.randint(0, 2, (batch_size, num_classes, h, w)).float()
    dummy_dtm = torch.randn(batch_size, num_classes, h, w)

    print("=== TESTING 5 ABLATION MATRICES (2-CHANNEL BINARY) ===\n")
    print(f"📊 Pixel Counts Matriks [Pos, Neg]:\n{counts.cpu().numpy()}")
    print(f"🎯 Calculated pos_weight per Channel: {weights.cpu().tolist()}\n")

    ablation_configs = [
        {
            "name": "L1: Dice + CE (Baseline)",
            "config": {"loss_type": "dice+ce", "use_class_weights": False},
        },
        {
            "name": "L2: Dice + WCE (Pixel-Weighted)",
            "config": {"loss_type": "dice+wce", "use_class_weights": True},
        },
        {
            "name": "L3: Tversky + CE (Asymmetric)",
            "config": {
                "loss_type": "tversky+ce",
                "use_class_weights": False,
                "tversky_alpha": 0.3,
                "tversky_beta": 0.7,
            },
        },
        {
            "name": "L4: Tversky + Focal (Hard-Pixel Champion)",
            "config": {
                "loss_type": "tversky+focal",
                "use_class_weights": True,
                "tversky_alpha": 0.3,
                "tversky_beta": 0.7,
                "focal_gamma": 2.0,
            },
        },
        {
            "name": "L5: Dice + GSL (Distance-Aware)",
            "config": {"loss_type": "dice+gsl", "use_class_weights": False},
        },
    ]

    for item in ablation_configs:
        print(f"--- {item['name']} ---")
        criterion = build_loss_criterion(item["config"], class_weights=weights)

        if "gsl" in item["config"]["loss_type"]:
            loss_val = criterion(dummy_logits, dummy_targets, dtm_targets=dummy_dtm)
        else:
            loss_val = criterion(dummy_logits, dummy_targets)

        loss_val.backward()
        print(f"Loss Output Value: {loss_val.item():.4f}\n")
