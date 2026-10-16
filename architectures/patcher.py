import os

os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"

import triton

def apply_mamba_patch():
    """Menerapkan monkey-patch pada Triton secara idempoten."""
    if not hasattr(triton, 'set_allocator'):
        triton.set_allocator = lambda *args, **kwargs: None # type: ignore

    if hasattr(triton, 'Config') and not getattr(triton.Config, '_is_patched', False):
        _old_init = triton.Config.__init__
        def _new_init(self, *args, maxnreg=None, **kwargs):
            _old_init(self, *args, **kwargs)
            self.maxnreg = maxnreg
        triton.Config.__init__ = _new_init
        triton.Config._is_patched = True # type: ignore
        print("[INFO] Monkey-patch Triton berhasil diterapkan!")

if __name__ == "__main__":
    apply_mamba_patch()
    
    import torch
    from mamba_ssm import Mamba 

    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Menggunakan perangkat: {device}")

    batch_size = 2
    seq_len = 128
    d_model = 768 

    mamba_block = Mamba(
        d_model=d_model,
        d_state=16,
        d_conv=4,
        expand=2,
    ).to(device)

    x = torch.randn(batch_size, seq_len, d_model, device=device)
    output = mamba_block(x)

    print("Shape Input  :", x.shape)
    print("Shape Output :", output.shape)
    print("Uji coba Mamba block berhasil dijalankan tanpa error!")