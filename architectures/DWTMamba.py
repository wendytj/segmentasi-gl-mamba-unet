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

class HaarDWT2D(nn.Module):
    """Mendekomposisi citra input menjadi 4 subband frekuensi ter-vektorisasi (FP16-Safe)."""
    def __init__(self, in_channels):
        super().__init__()
        self.in_channels = in_channels

        ll = torch.tensor([[0.5, 0.5], [0.5, 0.5]])
        lh = torch.tensor([[0.5, -0.5], [0.5, -0.5]])
        hl = torch.tensor([[0.5, 0.5], [-0.5, -0.5]])
        hh = torch.tensor([[0.5, -0.5], [-0.5, 0.5]])

        filters = torch.stack([ll, lh, hl, hh], dim=0).repeat(in_channels, 1, 1)
        filters = filters.unsqueeze(1)
        self.register_buffer('filters', filters)

    def forward(self, x):
        B, C, H, W = x.shape
        out = F.conv2d(x, self.filters, stride=2, groups=C) # type: ignore
        out = out.view(B, C, 4, H // 2, W // 2).permute(0, 2, 1, 3, 4)

        sb_min = out.amin(dim=(-2, -1), keepdim=True)
        sb_max = out.amax(dim=(-2, -1), keepdim=True)
        
        # 🌟 PERBAIKAN KRUSIAL FOR FP16: Gunakan clamp min 1e-4 agar denominator tidak membengkak
        denom = (sb_max - sb_min).clamp(min=1e-4)
        out_norm = (out - sb_min) / denom

        return out_norm  # Shape: [B, 4, C, H//2, W//2]

class SqueezeAndExcitation(nn.Module):
    def __init__(self, channels, reduction=16):
        super().__init__()
        self.fc = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Flatten(),
            nn.Linear(channels, max(1, channels // reduction)),
            nn.ReLU(inplace=True),
            nn.Linear(max(1, channels // reduction), channels),
            nn.Sigmoid()
        )

    def forward(self, x):
        b, c, _, _ = x.shape
        return self.fc(x).view(b, c, 1, 1)

def channel_shuffle(x, groups):
    batchsize, num_channels, height, width = x.size()
    channels_per_group = num_channels // groups
    x = x.view(batchsize, groups, channels_per_group, height, width)
    x = torch.transpose(x, 1, 2).contiguous()
    return x.view(batchsize, -1, height, width)

class SpaSE_SSM(nn.Module):
    """Spatial-Aware SE State Space Model Block menggunakan mamba-ssm."""
    def __init__(self, dim, d_state=16, d_conv=4, expand=2, se_reduction=16):
        super().__init__()
        self.dim = dim
        self.half_dim = dim // 2

        self.conv_branch = nn.Sequential(
            nn.Conv2d(self.half_dim, self.half_dim, 3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(self.half_dim, self.half_dim, 3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(self.half_dim, self.half_dim, 1)
        )

        self.norm = nn.LayerNorm(self.half_dim)
        
        self.row_mamba = Mamba(
            d_model=self.half_dim,
            d_state=d_state,
            d_conv=d_conv,
            expand=expand
        )
        self.col_mamba = Mamba(
            d_model=self.half_dim,
            d_state=d_state,
            d_conv=d_conv,
            expand=expand
        )
        
        self.se = SqueezeAndExcitation(self.half_dim, reduction=se_reduction)

    def forward(self, x):
        B, C, H, W = x.shape
        x_conv, x_ssm = torch.chunk(x, 2, dim=1)

        c_out = self.conv_branch(x_conv)

        x_norm = self.norm(x_ssm.permute(0, 2, 3, 1)) 

        x_row = x_norm.reshape(B * H, W, self.half_dim).contiguous()
        y_row = self.row_mamba(x_row)
        y_row = y_row.reshape(B, H, W, self.half_dim)

        x_col = y_row.permute(0, 2, 1, 3).reshape(B * W, H, self.half_dim).contiguous()
        y_col = self.col_mamba(x_col)

        y_col = y_col.reshape(B, W, H, self.half_dim).permute(0, 2, 1, 3).contiguous()
        y_ssm = y_col.permute(0, 3, 1, 2) 

        a_ssm = self.se(y_ssm)
        o_ssm = y_ssm * a_ssm

        y_cat = torch.cat([c_out, o_ssm], dim=1)
        y_shuff = channel_shuffle(y_cat, groups=2)
        return y_shuff + x

class MB_GSF(nn.Module):
    """Menggabungkan fitur 4 subband DWT secara ter-vektorisasi tanpa Python loop (FP16-Safe)."""
    def __init__(self, dim, reduction=4):
        super().__init__()
        self.dim = dim

        self.gate_fc = nn.Sequential(
            nn.Linear(dim, max(1, dim // reduction)),
            nn.GELU(),
            nn.Linear(max(1, dim // reduction), dim),
            nn.Sigmoid()
        )

        self.proj = nn.Conv2d(dim * 4, dim, kernel_size=1)
        self.ln = nn.LayerNorm(dim)

    def forward(self, x_5d):
        # x_5d: [B, 4, C, H, W]
        B, N, C, H, W = x_5d.shape

        # Global Average Pooling paralel [B, 4, C]
        v = x_5d.mean(dim=(-2, -1))
        
        # Gated Channel Recalibration paralel [B, 4, C, 1, 1]
        g = self.gate_fc(v).view(B, N, C, 1, 1)
        recalibrated = x_5d * g

        # Concatenation & Channel Shuffle
        f_cat = recalibrated.view(B, N * C, H, W)
        f_shuff = channel_shuffle(f_cat, groups=4)
        f_proj = self.proj(f_shuff)

        # Residual Sum & LayerNorm (Ditambah proteksi FP32 cast internal LayerNorm)
        f_sum = recalibrated.sum(dim=1)
        f_fused = f_proj + f_sum

        # LayerNorm dieksekusi dengan aman
        x_ln = f_fused.permute(0, 2, 3, 1)
        out_ln = self.ln(x_ln)
        return out_ln.permute(0, 3, 1, 2).contiguous()

class LatentEncoder(nn.Module):
    def __init__(self, in_channels, latent_dim=128):
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv2d(in_channels, 32, kernel_size=3, stride=2, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(32, 64, kernel_size=3, stride=2, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(64, 128, kernel_size=3, stride=2, padding=1),
            nn.ReLU(inplace=True),
            nn.AdaptiveAvgPool2d(1),
            nn.Flatten(),
            nn.Linear(128, latent_dim)
        )

    def forward(self, x):
        return self.net(x)

class DWTMamba(nn.Module):
    def __init__(
        self,
        in_channels=1,
        num_classes=3,
        embed_dim=256,
        depth=3,
        mamba_d_state=16,
        mamba_d_conv=4,
        mamba_expand=2,
        se_reduction=16,
        mb_gsf_reduction=4,
        latent_dim=128,
        proj_dim=256,
        drop_rate=0
    ):
        super().__init__()
        self.in_channels = in_channels
        self.dwt = HaarDWT2D(in_channels)

        self.pe_branches = nn.ModuleList([
            nn.Conv2d(in_channels, embed_dim, kernel_size=4, stride=4)
            for _ in range(4)
        ])

        self.spase_branches = nn.ModuleList([
            nn.Sequential(*[
                SpaSE_SSM(
                    dim=embed_dim,
                    d_state=mamba_d_state,
                    d_conv=mamba_d_conv,
                    expand=mamba_expand,
                    se_reduction=se_reduction
                ) for _ in range(depth)
            ])
            for _ in range(4)
        ])

        self.pm_branches = nn.ModuleList([
            nn.Sequential(
                nn.Conv2d(embed_dim, embed_dim, kernel_size=2, stride=2),
                nn.GroupNorm(1, embed_dim)
            ) for _ in range(4)
        ])

        self.mb_gsf = MB_GSF(dim=embed_dim, reduction=mb_gsf_reduction)
        self.latent_encoder = LatentEncoder(in_channels, latent_dim=latent_dim)

        self.gap = nn.AdaptiveAvgPool2d(1)
        self.proj_fused = nn.Linear(embed_dim, proj_dim)
        self.proj_latent = nn.Linear(latent_dim, proj_dim)
        self.drop = nn.Dropout(p=drop_rate)
        self.classifier = nn.Linear(proj_dim * 2, num_classes)

        self.apply(self._init_weights)

    def _init_weights(self, m):
        if isinstance(m, (nn.Conv2d, nn.Linear)):
            nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
            if m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, (nn.GroupNorm, nn.LayerNorm)):
            if m.weight is not None:
                nn.init.constant_(m.weight, 1.0)
            if m.bias is not None:
                nn.init.constant_(m.bias, 0)

    def forward(self, x):
        subbands = self.dwt(x) # Output DWT: [B, 4, C, H//2, W//2]

        branch_outputs = []
        for i in range(4):
            f = self.pe_branches[i](subbands[:, i])
            f = self.spase_branches[i](f)
            f = self.pm_branches[i](f)
            branch_outputs.append(f)

        feat_5d = torch.stack(branch_outputs, dim=1)

        f_fused = self.mb_gsf(feat_5d)
        f_global = self.gap(f_fused).view(f_fused.size(0), -1)
        u_fused = self.proj_fused(f_global)

        z_latent = self.latent_encoder(x)
        l_latent = self.proj_latent(z_latent)

        h_final = torch.cat([u_fused, l_latent], dim=1)
        h_final = self.drop(h_final)
        logits = self.classifier(h_final)
        return logits
    
    def load_pretrained_weights(self, checkpoint_path, device="cpu"):
        """
        Memuat weights pre-trained (misal OrganCMNIST) ke DWTMamba.
        Otomatis melakukan Inflation (1-ch ke 3-ch) atau Deflation (3-ch ke 1-ch)
        serta menangani ketidakcocokan jumlah kelas pada classifier.
        """
        if not os.path.exists(checkpoint_path):
            print(f"⚠️ Checkpoint tidak ditemukan di '{checkpoint_path}'. Training dimulai dari scratch.")
            return

        checkpoint = torch.load(checkpoint_path, map_location=device)
        
        # Ekstrak state_dict jika dibungkus dalam dictionary
        if isinstance(checkpoint, dict):
            if "state_dict" in checkpoint:
                state_dict = checkpoint["state_dict"]
            elif "model" in checkpoint:
                state_dict = checkpoint["model"]
            else:
                state_dict = checkpoint
        else:
            state_dict = checkpoint

        model_dict = self.state_dict()
        adapted_state_dict = {}

        for k, v in state_dict.items():
            # Hilangkan prefix 'module.' jika pretraining menggunakan DataParallel
            clean_key = k.replace("module.", "")

            if clean_key in model_dict:
                target_shape = model_dict[clean_key].shape

                # 1. Skip buffer HaarDWT2D (akan digenerate otomatis saat init)
                if "dwt.filters" in clean_key:
                    continue

                # 2. Skip classifier jika num_classes berbeda
                if "classifier" in clean_key and v.shape != target_shape:
                    print(f"⏩ Skip layer '{clean_key}' (Shape Mismatch: {v.shape} vs {target_shape})")
                    continue

                # 3. Handle Weight Inflation / Deflation jika in_channels berbeda
                if v.shape != target_shape:
                    if v.ndim == 4 and model_dict[clean_key].ndim == 4:
                        ckpt_in_c = v.shape[1]
                        curr_in_c = target_shape[1]

                        # Case A: Inflation (1 Channel -> 3 Channel)
                        if ckpt_in_c == 1 and curr_in_c == 3:
                            v = v.repeat(1, 3, 1, 1) / 3.0
                            print(f"🔄 Inflated weight '{clean_key}': {state_dict[k].shape} -> {v.shape}")

                        # Case B: Deflation (3 Channel -> 1 Channel)
                        elif ckpt_in_c == 3 and curr_in_c == 1:
                            v = v.mean(dim=1, keepdim=True)
                            print(f"🔄 Deflated weight '{clean_key}': {state_dict[k].shape} -> {v.shape}")
                        else:
                            print(f"⏩ Skip layer '{clean_key}' (Incompatible 4D Shape: {v.shape} vs {target_shape})")
                            continue
                    else:
                        # Skip layer 1D/2D/Linear lain jika ukurannya tidak cocok agar tidak memicu RuntimeError
                        print(f"⏩ Skip layer '{clean_key}' (Size Mismatch: {v.shape} vs {target_shape})")
                        continue

                adapted_state_dict[clean_key] = v

        missing_keys, unexpected_keys = self.load_state_dict(adapted_state_dict, strict=False)
        print(f"✅ Pretrained weights berhasil dimuat dari: {checkpoint_path}")
        print(f"   - Loaded keys    : {len(adapted_state_dict)}")
        print(f"   - Missing keys   : {len(missing_keys)}")
        print(f"   - Unexpected keys: {len(unexpected_keys)}")

if __name__ == "__main__":
    device = "cuda" if torch.cuda.is_available() else "cpu"
    dummy_input = torch.randn(2, 1, 224, 224, device=device)
    
    model = DWTMamba(
        in_channels=1,
        num_classes=3,
        embed_dim=256,
        depth=3,
        mamba_d_state=16,
        mamba_d_conv=4,
        mamba_expand=2,
        se_reduction=16,
        mb_gsf_reduction=4,
        latent_dim=128,
        proj_dim=256
    ).to(device)

    output = model(dummy_input)
    print("✅ Testing DWT-Mamba dengan Fleksibilitas Hyperparameter Selesai!")
    print(f"📦 Input Shape  : {dummy_input.shape}")
    print(f"🎯 Output Shape : {output.shape}")

    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)

    print(f"📊 Total Parameter Model : {total_params:,}")
    print(f"🔥 Parameter yang Dilatih : {trainable_params:,}")