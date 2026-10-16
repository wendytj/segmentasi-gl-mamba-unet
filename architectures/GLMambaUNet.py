import os, sys

PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from architectures.patcher import apply_mamba_patch

apply_mamba_patch()

import torch
import torch.nn as nn
import torch.nn.functional as F
from mamba_ssm import Mamba


class GuidedBiM(nn.Module):
    """
    Guided Bidirectional Mamba Mechanism (Eq. 11, 12, 13)
    Diakselerasi menggunakan CUDA Kernel dari library official `mamba_ssm`.
    Global prompt g disuntikkan secara dinamis (x + g) ke dalam parameter SSM.
    """

    def __init__(self, d_model, d_state=16, d_conv=4, expand=2):
        super().__init__()
        # Forward & Backward Selective Scan Mamba
        self.fwd_mamba = Mamba(
            d_model=d_model, d_state=d_state, d_conv=d_conv, expand=expand
        )
        self.bwd_mamba = Mamba(
            d_model=d_model, d_state=d_state, d_conv=d_conv, expand=expand
        )
        self.out_proj = nn.Linear(d_model, d_model)

    def forward(self, x, g):
        """
        x: Local sequence feature (B, L, D)
        g: Global prompt context feature (B, L, D)
        """
        # 1. Injeksi Global Prompt Context (x_t + g_t) -> Eq. (8, 9, 10)
        prompt_input = x + g

        # 2. Forward Sequence Scanning
        y_fwd = self.fwd_mamba(prompt_input)

        # 3. Backward Sequence Scanning (Reversed Input)
        prompt_rev = torch.flip(prompt_input, dims=[1])
        y_bwd_rev = self.bwd_mamba(prompt_rev)
        y_bwd = torch.flip(y_bwd_rev, dims=[1])

        # 4. Bidirectional Sum Projection -> Eq. (13)
        y = y_fwd + y_bwd
        return self.out_proj(y)


# ==============================================================================
# 2. GLOBAL-TO-LOCAL MAMBA BLOCK (GLM Block)
# ==============================================================================
class GLMBlock(nn.Module):
    """
    Global-to-Local Mamba Block (Eq. 1 - 7)
    - Step 1: Global Context Extraction (GCM)
    - Step 2: Spatial Attention Modulation
    - Step 3: Local Guided Feature Extraction (LGM)
    """

    def __init__(self, in_channels, patch_size=4, d_state=16, d_conv=4, expand=2):
        super().__init__()
        self.in_channels = in_channels
        self.patch_size = patch_size

        # 1. Global Context Mamba (GCM)
        self.pool = nn.AdaptiveAvgPool2d((patch_size, patch_size))
        self.bim_global = GuidedBiM(
            d_model=in_channels, d_state=d_state, d_conv=d_conv, expand=expand
        )

        # 2. Spatial Attention Modulation
        self.conv_attn = nn.Conv2d(in_channels, 1, kernel_size=1)
        self.sigmoid = nn.Sigmoid()

        # 3. Local Guided Mamba (LGM)
        self.bim_local = GuidedBiM(
            d_model=in_channels, d_state=d_state, d_conv=d_conv, expand=expand
        )
        self.norm = nn.GroupNorm(1, in_channels)

    def forward(self, F_l):
        B, C, H, W = F_l.shape

        # ----------------------------------------------------------------------
        # Step 1: Global Context Extraction (GCM) -> Eq. (1, 2, 3)
        # ----------------------------------------------------------------------
        P_l = self.pool(F_l)  # (B, C, P, P) -> Eq. (1)
        P_l_flat = P_l.flatten(2).transpose(1, 2)  # (B, P*P, C)

        # Global BiM dengan self-prompting
        P_l_prime_flat = self.bim_global(P_l_flat, P_l_flat)  # Eq. (2)
        P_l_prime = P_l_prime_flat.transpose(1, 2).view(
            B, C, self.patch_size, self.patch_size
        )

        # Unpooling kembali ke resolusi asal (H, W)
        F_global = F.interpolate(
            P_l_prime, size=(H, W), mode="bilinear", align_corners=False
        )  # Eq. (3)

        # ----------------------------------------------------------------------
        # Step 2: Spatial Attention Modulation -> Eq. (4, 5)
        # ----------------------------------------------------------------------
        M_l = self.sigmoid(self.conv_attn(F_global))  # Spatial Attention Map Eq. (4)
        F_guided = F_l * M_l + F_l  # Modulated Feature Eq. (5)

        # ----------------------------------------------------------------------
        # Step 3: Local Guided Feature Extraction (LGM) -> Eq. (6, 7)
        # ----------------------------------------------------------------------
        F_guided_flat = F_guided.flatten(2).transpose(1, 2)  # (B, H*W, C)
        F_global_flat = F_global.flatten(2).transpose(1, 2)  # (B, H*W, C)

        # LGM dipandu oleh F_global sebagai continuous prompt
        F_l_double_prime = self.bim_local(F_guided_flat, F_global_flat)  # Eq. (7)

        # Reshape kembali ke 2D Spatial Feature Map
        out = F_l_double_prime.transpose(1, 2).view(B, C, H, W)
        return self.norm(out + F_l)


# ==============================================================================
# 3. HELPER ENCODER & DECODER BLOCKS
# ==============================================================================
class Stem(nn.Module):
    """Initial Feature Extraction via Depth-wise Separable Convolution."""

    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(
                in_channels, out_channels, kernel_size=3, padding=1, groups=in_channels
            ),
            nn.Conv2d(out_channels, out_channels, kernel_size=1),
            nn.BatchNorm2d(out_channels),
            nn.GELU(),
        )

    def forward(self, x):
        return self.conv(x)


class DownSample(nn.Module):
    """Downsampling Spatial Resolution (1/2x)."""

    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.down = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=2, stride=2),
            nn.BatchNorm2d(out_channels),
        )

    def forward(self, x):
        return self.down(x)


class UpSample(nn.Module):
    """Upsampling Spatial Resolution (2x)."""

    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.up = nn.Sequential(
            nn.Upsample(scale_factor=2, mode="bilinear", align_corners=False),
            nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1),
            nn.BatchNorm2d(out_channels),
            nn.GELU(),
        )

    def forward(self, x):
        return self.up(x)


# ==============================================================================
# 4. FULL GL-MAMBA UNET ARCHITECTURE
# ==============================================================================
class GLMambaUNet(nn.Module):
    """
    Arsitektur Lengkap GL-Mamba UNet (Wang et al., 2026)
    - Encoder: Stem + 3 GLM Stages + Downsampling
    - Bottleneck: GLM Block
    - Decoder: UpSampling + Skip Connection Fusion + Conv Refinement
    """

    def __init__(
        self,
        in_channels=1,
        num_classes=3,
        feature_dims=[32, 64, 128, 256],
        d_state=16,
        d_conv=4,
        expand=2,
    ):
        super().__init__()

        # 1. Stem Layer
        self.stem = Stem(in_channels, feature_dims[0])

        # 2. Encoder Path
        self.enc1 = GLMBlock(
            feature_dims[0], d_state=d_state, d_conv=d_conv, expand=expand
        )
        self.down1 = DownSample(feature_dims[0], feature_dims[1])

        self.enc2 = GLMBlock(
            feature_dims[1], d_state=d_state, d_conv=d_conv, expand=expand
        )
        self.down2 = DownSample(feature_dims[1], feature_dims[2])

        self.enc3 = GLMBlock(
            feature_dims[2], d_state=d_state, d_conv=d_conv, expand=expand
        )
        self.down3 = DownSample(feature_dims[2], feature_dims[3])

        # 3. Bottleneck
        self.bottleneck = GLMBlock(
            feature_dims[3], d_state=d_state, d_conv=d_conv, expand=expand
        )

        # 4. Decoder Path + Skip Connections
        self.up3 = UpSample(feature_dims[3], feature_dims[2])
        self.dec3 = nn.Sequential(
            nn.Conv2d(feature_dims[2] * 2, feature_dims[2], kernel_size=3, padding=1),
            nn.BatchNorm2d(feature_dims[2]),
            nn.GELU(),
        )

        self.up2 = UpSample(feature_dims[2], feature_dims[1])
        self.dec2 = nn.Sequential(
            nn.Conv2d(feature_dims[1] * 2, feature_dims[1], kernel_size=3, padding=1),
            nn.BatchNorm2d(feature_dims[1]),
            nn.GELU(),
        )

        self.up1 = UpSample(feature_dims[1], feature_dims[0])
        self.dec1 = nn.Sequential(
            nn.Conv2d(feature_dims[0] * 2, feature_dims[0], kernel_size=3, padding=1),
            nn.BatchNorm2d(feature_dims[0]),
            nn.GELU(),
        )

        # 5. Output Segmentation Head
        self.final_proj = nn.Conv2d(feature_dims[0], num_classes, kernel_size=1)

    def forward(self, x):
        # Stem
        x0 = self.stem(x)  # (B, 32, 224, 224)

        # Encoder
        e1 = self.enc1(x0)  # (B, 32, 224, 224)
        e2 = self.enc2(self.down1(e1))  # (B, 64, 112, 112)
        e3 = self.enc3(self.down2(e2))  # (B, 128, 56, 56)

        # Bottleneck
        b = self.bottleneck(self.down3(e3))  # (B, 256, 28, 28)

        # Decoder + Skip Connections
        d3 = self.up3(b)  # (B, 128, 56, 56)
        d3 = self.dec3(torch.cat([d3, e3], dim=1))

        d2 = self.up2(d3)  # (B, 64, 112, 112)
        d2 = self.dec2(torch.cat([d2, e2], dim=1))

        d1 = self.up1(d2)  # (B, 32, 224, 224)
        d1 = self.dec1(torch.cat([d1, e1], dim=1))

        # Final Segmentation Output
        out = self.final_proj(d1)  # (B, 3, 224, 224)
        return out


# ==============================================================================
# VERIFIKASI EXECUTION DI GPU
# ==============================================================================
if __name__ == "__main__":
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"🚀 Running test on device: {device}")

    # Dummy Input CT-Scan (Batch=2, Channel=1, H=224, W=224)
    dummy_input = torch.randn(2, 1, 224, 224, device=device)

    model = GLMambaUNet(
        in_channels=1,
        num_classes=3,
        feature_dims=[32, 64, 128, 256],
        d_state=16,
        d_conv=4,
        expand=2,
    ).to(device)

    # Forward Pass
    output = model(dummy_input)

    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)

    print("✅ Forward Pass GL-Mamba UNet Berhasil!")
    print(f" ├─ Input Shape       : {dummy_input.shape}")
    print(f" ├─ Output Shape      : {output.shape} (Expected: [2, 3, 224, 224])")
    print(f" ├─ Total Parameters  : {total_params:,}")
    print(f" └─ Trainable Params  : {trainable_params:,}")
