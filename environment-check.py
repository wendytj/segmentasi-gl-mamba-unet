import sys
import torch

print("--- INFORMASI ENVIRONMENT ---")
print(f"Versi Python         : {sys.version.split()[0]}")
print(f"Versi PyTorch        : {torch.__version__}")
print(f"Versi CUDA (PyTorch) : {torch.version.cuda}") # type: ignore[attr-defined]  
print(f"Ketersediaan GPU     : {torch.cuda.is_available()}")
print("CXX11 ABI:", torch.compiled_with_cxx11_abi())

if torch.cuda.is_available():
    print(f"Nama Perangkat GPU   : {torch.cuda.get_device_name(0)}")