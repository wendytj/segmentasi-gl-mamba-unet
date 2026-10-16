import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.checkpoint import checkpoint

class CoLoRAConv2d(nn.Module):
    def __init__(self, original_conv: nn.Conv2d):
        super().__init__()
        self.in_channels = original_conv.in_channels
        self.out_channels = original_conv.out_channels
        self.kernel_size = original_conv.kernel_size
        self.stride = original_conv.stride
        self.padding = original_conv.padding
        self.dilation = original_conv.dilation
        self.groups = original_conv.groups
        self.has_bias = original_conv.bias is not None

        # 1. Bekukan bobot Conv2d asli
        self.original_conv = original_conv
        for param in self.original_conv.parameters():
            param.requires_grad = False

        # Target Device & Dtype dari layer asli
        target_device = original_conv.weight.device
        target_dtype = original_conv.weight.dtype

        # 2. Depthwise Convolution
        self.depthwise = nn.Conv2d(
            in_channels=self.in_channels,
            out_channels=self.in_channels,
            kernel_size=self.kernel_size, # type: ignore
            stride=self.stride, # type: ignore
            padding=self.padding, # type: ignore
            dilation=self.dilation, # type: ignore
            groups=self.in_channels,
            bias=False,
            device=target_device,
            dtype=target_dtype
        )

        # 3. Pointwise Convolution
        self.pointwise = nn.Conv2d(
            in_channels=self.in_channels,
            out_channels=self.out_channels,
            kernel_size=1,
            stride=1,
            padding=0,
            bias=self.has_bias,
            device=target_device,
            dtype=target_dtype
        )

        # 4. Inisialisasi Kp & Kd
        nn.init.xavier_uniform_(self.pointwise.weight)
        nn.init.zeros_(self.depthwise.weight)
        if self.has_bias and self.pointwise.bias is not None:
            nn.init.zeros_(self.pointwise.bias)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if self.training and x.requires_grad:
            def _custom_forward(x_in):
                base_out = self.original_conv(x_in)
                colora_out = self.pointwise(self.depthwise(x_in))
                return base_out + colora_out

            # use_reentrant=False direkomendasikan untuk PyTorch modern & AMP/autocast
            return checkpoint(_custom_forward, x, use_reentrant=False) # type: ignore
        else:
            base_out = self.original_conv(x)
            colora_out = self.pointwise(self.depthwise(x))
            return base_out + colora_out

    def merge_weights(self) -> None:
        W_orig = self.original_conv.weight.data
        W_p = self.pointwise.weight.data
        W_d = self.depthwise.weight.data

        delta_W = W_p * W_d.view(1, self.in_channels, self.kernel_size[0], self.kernel_size[1])

        self.original_conv.weight.data = W_orig + delta_W
        self.original_conv.weight.requires_grad = True

        if self.has_bias and self.pointwise.bias is not None:
            self.original_conv.bias.data += self.pointwise.bias.data # type: ignore
            self.original_conv.bias.requires_grad = True # type: ignore


def apply_colora_to_model(
    model: nn.Module, target_modules: list = None  # type: ignore
) -> nn.Module:
    """Injektor presisi CoLoRA ke seluruh layer Conv2d target dengan traversal nama lengkap."""
    if target_modules is None:
        target_modules = [
            "pe_branches",
            "pm_branches",
            "conv_branch",
            "mb_gsf.proj",
            "latent_encoder",
        ]

    modules_to_replace = []

    # Traversal menggunakan named_modules agar kompatibel dengan nn.Sequential / nn.ModuleList
    for full_name, module in model.named_modules():
        for child_name, child in module.named_children():
            if isinstance(child, nn.Conv2d):
                child_full_name = (
                    f"{full_name}.{child_name}" if full_name else child_name
                )
                if any(
                    target in child_full_name for target in target_modules
                ):
                    modules_to_replace.append(
                        (module, child_name, child, child_full_name)
                    )

    injected_count = 0
    for parent, child_name, child_conv, child_full_name in modules_to_replace:
        try:
            # FIX: Penanganan aman jika parent adalah nn.ModuleList
            if isinstance(parent, nn.ModuleList):
                parent[int(child_name)] = CoLoRAConv2d(child_conv)
            else:
                setattr(parent, child_name, CoLoRAConv2d(child_conv))
            injected_count += 1
        except Exception as e:
            print(f"  ⚠️ [CoLoRA Skipped] {child_full_name}: {e}")

    print(f"✅ Total {injected_count} layer Conv2d berhasil disuntik CoLoRA!")
    return model


def merge_colora_weights(model: nn.Module) -> nn.Module:
    """Melebur bobot CoLoRA ke original conv dan mengembalikan struktur asli model."""
    for name, module in model.named_children():
        if isinstance(module, CoLoRAConv2d):
            module.merge_weights()
            if isinstance(model, nn.ModuleList):
                model[int(name)] = module.original_conv
            else:
                setattr(model, name, module.original_conv)
        else:
            merge_colora_weights(module)
    return model