import math
import torch
import torch.nn as nn
import torch.nn.functional as F

class ProDiaLLinear(nn.Module):
    """
    ProDiaL Layer - High-Performance & Ultra-Low Memory Overhead Version.
    Paper: "Parameter Efficient Mamba Tuning via Projector-targeted Diagonal-centric 
            Linear Transformation" (KAIST, 2025).
    """
    def __init__(self, original_linear: nn.Linear, r_b: int = 16, r_eps: int = 8):
        super().__init__()
        self.in_features = original_linear.in_features
        self.out_features = original_linear.out_features
        self.r_b = r_b
        self.r_eps = r_eps

        # 1. Freeze layer linear asli
        self.original_linear = original_linear
        for param in self.original_linear.parameters():
            param.requires_grad = False

        if self.in_features % r_b != 0:
            raise ValueError(f"❌ in_features ({self.in_features}) harus habis dibagi r_b ({r_b})!")
        self.block_size = self.in_features // r_b

        # 2. Block-Diagonal Parameter [r_b, block_size, block_size]
        self.D_blocks = nn.Parameter(
            torch.eye(self.block_size).unsqueeze(0).repeat(r_b, 1, 1)
        )

        # 3. Scaling Factor s [out_features]
        self.s = nn.Parameter(torch.ones(self.out_features))

        # 4. Off-Diagonal LoRA Parameters
        self.A_eps = nn.Parameter(torch.empty(r_eps, self.in_features))
        self.B_eps = nn.Parameter(torch.zeros(self.out_features, r_eps))

        nn.init.kaiming_uniform_(self.A_eps, a=math.sqrt(5))

    @property
    def weight(self) -> torch.Tensor:
        """
        Rekonstruksi W' = s * (W @ D_b) + B_eps @ A_eps TANPA torch.block_diag.
        Sangat cepat dan hemat VRAM!
        """
        W = self.original_linear.weight  # [out_features, in_features]
        
        # Reshape W ke bentuk blok [out_features, r_b, block_size]
        W_reshaped = W.view(self.out_features, self.r_b, self.block_size)
        
        # Perkalian blok W x D_blocks secara paralel tanpa membuat matriks nol penuh
        W_Db = torch.einsum('o r k, r k m -> o r m', W_reshaped, self.D_blocks).reshape(self.out_features, self.in_features)
        
        # Scaling + Off-diagonal LoRA
        scaled_W = self.s.unsqueeze(1) * W_Db
        eps = torch.matmul(self.B_eps, self.A_eps)
        
        return scaled_W + eps

    @property
    def bias(self):
        return self.original_linear.bias

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        orig_shape = x.shape
        
        # A. Block-Diagonal Multiplication via Reshape + Einsum
        x_reshaped = x.reshape(-1, self.r_b, self.block_size)
        x_trans = torch.einsum('b r k, r m k -> b r m', x_reshaped, self.D_blocks).reshape(orig_shape)

        # B. Base Linear Forward
        base_out = F.linear(x_trans, self.original_linear.weight, bias=None)

        # C. Channel Scaling
        scaled_out = base_out * self.s

        # D. Off-Diagonal LoRA Forward
        lora_out = F.linear(F.linear(x, self.A_eps), self.B_eps)

        # E. Final Output
        out = scaled_out + lora_out
        if self.original_linear.bias is not None:
            out = out + self.original_linear.bias

        return out
    
    def merge_weights(self) -> None:
        """
        Menggabungkan W' langsung ke layer linear asli untuk fase inferensi.
        """
        self.original_linear.weight.data = self.weight.data
        self.original_linear.weight.requires_grad = True


def apply_prodial_to_model(
    model: nn.Module,
    target_modules: list = None, # type: ignore
    r_b: int = 16,
    r_eps: int = 8
) -> nn.Module:
    """
    Injektor otomatis ProDiaL ke layer-layer linear target dalam model.
    """
    for param in model.parameters():
        param.requires_grad = False

    if target_modules is None:
        target_modules = ["in_proj", "out_proj", "proj_fused", "proj_latent", "classifier"]

    def _inject_recursive(module_to_check):
        for name, child in module_to_check.named_children():
            if isinstance(child, nn.Linear) and any(target in name for target in target_modules):
                try:
                    setattr(module_to_check, name, ProDiaLLinear(child, r_b=r_b, r_eps=r_eps))
                except ValueError as e:
                    print(f"  ⚠️ [ProDiaL Skipped] {name}: {e}")
            else:
                _inject_recursive(child)

    _inject_recursive(model)
    return model


def merge_prodial_weights(model: nn.Module) -> nn.Module:
    """
    Menggabungkan seluruh layer ProDiaL kembali ke modul linear standar.
    """
    for name, module in model.named_children():
        if isinstance(module, ProDiaLLinear):
            module.merge_weights()
            setattr(model, name, module.original_linear)
        else:
            merge_prodial_weights(module)
    return model