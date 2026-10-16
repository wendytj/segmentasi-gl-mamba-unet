import os
import sys
import torch

PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

# Import DWTMamba dan PEFT helper
from architectures.DWTMamba import DWTMamba

try:
    from fine_tuning.peft_wrapper import apply_peft, merge_peft_and_unload
except ImportError:
    from peft_wrapper import apply_peft, merge_peft_and_unload

# 1. Path Checkpoint & Output Sesuai Konfigurasi
PEFT_CHECKPOINT_PATH = "logs-vastai/prodial-sqrt-conv-02/best_model_val_mcc.pth"
MERGED_OUTPUT_PATH = "fine_tuning/result/classification_merged_best.pth"

device = "cuda" if torch.cuda.is_available() else "cpu"

# 2. Inisialisasi Model Basah (Tanpa .to(device) terlebih dahulu)
model = DWTMamba(in_channels=1, num_classes=3)

# 3. Injeksi PEFT (ProDiaL r_eps=32, r_b=128 & CoLoRA)
model = apply_peft(
    model, method="prodial", r=32, r_b=128, use_colora_for_conv=True
)

# 4. Pindahkan SELURUH Model + Adaptor PEFT ke Device (Mencegah CUDA-CPU Device Mismatch)
model = model.to(device)

# 5. Load Weights PEFT
checkpoint = torch.load(PEFT_CHECKPOINT_PATH, map_location=device)

# Ekstrak state_dict secara presisi berdasarkan key di checkpoint
if isinstance(checkpoint, dict):
    if "model_state" in checkpoint:
        state_dict = checkpoint["model_state"]
    elif "state_dict" in checkpoint:
        state_dict = checkpoint["state_dict"]
    elif "model" in checkpoint:
        state_dict = checkpoint["model"]
    else:
        state_dict = checkpoint
else:
    state_dict = checkpoint

# Load weights yang sudah diekstraksi
model.load_state_dict(state_dict)
print("✅ Weights PEFT Klasifikasi Berhasil Dimuat!")

# 6. Pastikan kembali seluruh tensor berada di device yang sama sebelum Merge
model = model.to(device)

# 7. Merge & Unload (Penggabungan Permanen)
model_merged = merge_peft_and_unload(model)
print("⚡ Adaptor ProDiaL & CoLoRA Berhasil Di-merge ke Base Model!")

# 8. Simpan Checkpoint Clean Standalone
os.makedirs(os.path.dirname(MERGED_OUTPUT_PATH), exist_ok=True)
torch.save(model_merged.state_dict(), MERGED_OUTPUT_PATH)
print(f"🎉 Checkpoint Merged Standalone Tersimpan di: {MERGED_OUTPUT_PATH}")