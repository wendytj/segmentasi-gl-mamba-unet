import os, sys

PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

import torch
import torch.nn as nn
import torch.nn.functional as F

from architectures.DWTMamba import DWTMamba
from architectures.GLMambaUNet import DownSample, GLMBlock, Stem, UpSample


class HybridDWTGLMambaUNet(nn.Module):

    def __init__(
        self,
        in_channels=1,
        num_classes=3,
        feature_dims=[32, 64, 128, 256],
        dwt_embed_dim=256,
        d_state=16,
        d_conv=4,
        expand=2,
    ):
        super().__init__()

        # =========================================================================
        # 1. ENCODER BRANCH A: DWT-MAMBA (PRETRAINED BACKBONE)
        # =========================================================================
        self.dwt_encoder = DWTMamba(
            in_channels=in_channels,
            num_classes=num_classes,
            embed_dim=dwt_embed_dim,
        )

        del self.dwt_encoder.latent_encoder
        del self.dwt_encoder.proj_fused
        del self.dwt_encoder.proj_latent
        del self.dwt_encoder.classifier
        del self.dwt_encoder.gap
        del self.dwt_encoder.drop

        # =========================================================================
        # 2. ENCODER BRANCH B: GL-MAMBA SPATIAL ENCODER
        # =========================================================================
        self.stem = Stem(in_channels, feature_dims[0])

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

        # =========================================================================
        # 3. FUSION ADAPTERS (SKIP CONNECTIONS INTEGRATION)
        # =========================================================================
        # Penyesuaian channel dari gabungan DWT-Mamba + GL-Mamba
        self.fuse_sk1 = nn.Sequential(
            nn.Conv2d(feature_dims[0] + dwt_embed_dim, feature_dims[0], 1),
            nn.BatchNorm2d(feature_dims[0]),
            nn.GELU(),
        )
        self.fuse_sk2 = nn.Sequential(
            nn.Conv2d(feature_dims[1] + dwt_embed_dim, feature_dims[1], 1),
            nn.BatchNorm2d(feature_dims[1]),
            nn.GELU(),
        )
        self.fuse_sk3 = nn.Sequential(
            nn.Conv2d(feature_dims[2] + dwt_embed_dim, feature_dims[2], 1),
            nn.BatchNorm2d(feature_dims[2]),
            nn.GELU(),
        )

        # =========================================================================
        # 4. BOTTLENECK & DECODER PATH
        # =========================================================================
        self.bottleneck = GLMBlock(
            feature_dims[3], d_state=d_state, d_conv=d_conv, expand=expand
        )

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

        self.final_proj = nn.Conv2d(feature_dims[0], num_classes, kernel_size=1)

    def load_dwt_pretrained(self, pretrained_path):
        if pretrained_path and os.path.exists(pretrained_path):
            print(f"✅ Memuat pretrained weights DWT-Mamba dari: {pretrained_path}")
            checkpoint = torch.load(pretrained_path, map_location="cpu")

            # Ambil state_dict dari checkpoint
            raw_state_dict = (
                checkpoint["model_state"]
                if isinstance(checkpoint, dict) and "model_state" in checkpoint
                else checkpoint
            )
            if isinstance(checkpoint, dict) and "state_dict" in checkpoint:
                raw_state_dict = checkpoint["state_dict"]

            # 🌟 Buat dictionary baru dan skip total semua key yang mengandung 'classifier'
            clean_state_dict = {}
            unused_keys = ["classifier", "latent_encoder", "proj_fused", "proj_latent"]
            for k, v in raw_state_dict.items():
                k_clean = k.replace("module.", "")
                if any(uk in k_clean for uk in unused_keys):
                    continue
                clean_state_dict[k_clean] = v

            # Load state dict dengan strict=False
            missing_keys, unexpected_keys = self.dwt_encoder.load_state_dict(
                clean_state_dict, strict=False
            )
            print(
                f"  - Berhasil memuat bobot backbone! Missing keys: {len(missing_keys)} Unexpected keys: {len(unexpected_keys)}"
            )
        else:
            print(f"⚠️ Pretrained path tidak ditemukan: {pretrained_path}")

    def forward(self, x):
        B, C, H, W = x.shape

        # 1. Ekstraksi Fitur Frekuensi via DWT-Mamba Branch
        subbands = self.dwt_encoder.dwt(x)
        branch_outputs = []
        for i in range(4):
            f = self.dwt_encoder.pe_branches[i](subbands[:, i])
            f = self.dwt_encoder.spase_branches[i](f)
            f = self.dwt_encoder.pm_branches[i](f)
            branch_outputs.append(f)

        feat_5d = torch.stack(branch_outputs, dim=1)
        dwt_fused_feat = self.dwt_encoder.mb_gsf(
            feat_5d
        )  # Feature map DWT (B, embed_dim, H_dwt, W_dwt)

        # 2. Ekstraksi Fitur Spasial via GL-Mamba Branch
        x0 = self.stem(x)
        e1 = self.enc1(x0)  # (B, 32, H, W)

        e2_in = self.down1(e1)
        e2 = self.enc2(e2_in)  # (B, 64, H/2, W/2)

        e3_in = self.down2(e2)
        e3 = self.enc3(e3_in)  # (B, 128, H/4, W/4)

        # 3. Fusion Skip Connections (DWT Feat Resized + GL-Mamba Enc Feat)
        dwt_sk1 = F.interpolate(
            dwt_fused_feat, size=e1.shape[2:], mode="bilinear", align_corners=False
        )
        sk1 = self.fuse_sk1(torch.cat([e1, dwt_sk1], dim=1))

        dwt_sk2 = F.interpolate(
            dwt_fused_feat, size=e2.shape[2:], mode="bilinear", align_corners=False
        )
        sk2 = self.fuse_sk2(torch.cat([e2, dwt_sk2], dim=1))

        dwt_sk3 = F.interpolate(
            dwt_fused_feat, size=e3.shape[2:], mode="bilinear", align_corners=False
        )
        sk3 = self.fuse_sk3(torch.cat([e3, dwt_sk3], dim=1))

        # 4. Bottleneck & Decoder
        b = self.bottleneck(self.down3(e3))

        d3 = self.up3(b)
        d3 = self.dec3(torch.cat([d3, sk3], dim=1))

        d2 = self.up2(d3)
        d2 = self.dec2(torch.cat([d2, sk2], dim=1))

        d1 = self.up1(d2)
        d1 = self.dec1(torch.cat([d1, sk1], dim=1))

        # Output logits segmentasi 2D
        out_mask = self.final_proj(d1)
        return out_mask


if __name__ == "__main__":
    # Inisialisasi Model
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model = HybridDWTGLMambaUNet(in_channels=1, num_classes=3).to(device)

    # ==============================================================================
    # 1. ISOLASI PARAMETER HANYA PADA MODUL AKTIF DWT-MAMBA
    # ==============================================================================
    dwt_active_modules = [
        model.dwt_encoder.pe_branches,
        model.dwt_encoder.spase_branches,
        model.dwt_encoder.pm_branches,
        model.dwt_encoder.mb_gsf,
    ]
    dwt_active_params = [
        p for m in dwt_active_modules for p in m.parameters() if p.requires_grad
    ]

    # Group B: GL-Mamba Spatial Encoder (From Scratch)
    glm_enc_modules = [
        model.stem,
        model.enc1,
        model.down1,
        model.enc2,
        model.down2,
        model.enc3,
        model.down3,
    ]
    glm_enc_params = [
        p for m in glm_enc_modules for p in m.parameters() if p.requires_grad
    ]

    # Group C: Fusion Adapters + Decoder Path + Segmentation Head (From Scratch)
    decoder_fusion_modules = [
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
    decoder_fusion_params = [
        p for m in decoder_fusion_modules for p in m.parameters() if p.requires_grad
    ]

    # ==============================================================================
    # 2. BREAKDOWN PARAMETER AKTIF TERPROSES
    # ==============================================================================
    count_p = lambda params: sum(p.numel() for p in params)

    p_dwt_active = count_p(dwt_active_params)
    p_glm_enc = count_p(glm_enc_params)
    p_dec_fuse = count_p(decoder_fusion_params)
    p_total = p_dwt_active + p_glm_enc + p_dec_fuse

    print("=" * 68)
    print("📊 BREAKDOWN TRAINABLE PARAMETERS (EXCLUDING UNUSED CLASSIFIER MODULES)")
    print("=" * 68)
    print(f" ├─ 1. DWT-Mamba Active Backbone (Pretrained) : {p_dwt_active:>10,} params")
    print(f" ├─ 2. GL-Mamba Encoder (From Scratch)       : {p_glm_enc:>10,} params")
    print(f" ├─ 3. Fusion & UNet Decoder (From Scratch)  : {p_dec_fuse:>10,} params")
    print(" ├" + "─" * 66)
    print(f" └─ 🔥 TOTAL ACTIVE TRAINABLE PARAMETERS     : {p_total:>10,} params")
    print("=" * 68)

    # ==============================================================================
    # 3. SETUP OPTIMIZER DENGAN DIFFERENTIAL LEARNING RATE
    # ==============================================================================
    base_lr = 1e-3  # LR untuk komponen baru (GL-Mamba Encoder & Decoder)
    dwt_lr = base_lr * 0.1  # LR 10% (1e-4) untuk Modul Aktif DWT-Mamba Pretrained

    optimizer = torch.optim.AdamW(
        [
            {
                "params": dwt_active_params,
                "lr": dwt_lr,
                "name": "dwt_active_backbone_pretrained",
            },
            {
                "params": glm_enc_params,
                "lr": base_lr,
                "name": "glm_encoder_scratch",
            },
            {
                "params": decoder_fusion_params,
                "lr": base_lr,
                "name": "decoder_fusion_scratch",
            },
        ],
        weight_decay=1e-2,
    )

    print("\n⚙️ OPTIMIZER PARAMETER GROUPS:")
    for group in optimizer.param_groups:
        num_p = sum(p.numel() for p in group["params"])
        print(f" ├─ Group: {group['name']:<32} | LR: {group['lr']} | Params: {num_p:,}")
