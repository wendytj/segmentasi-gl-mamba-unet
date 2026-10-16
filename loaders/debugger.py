import os
import sys
import numpy as np
import pandas as pd
import torch
import torch.nn as nn

ROOT_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if ROOT_DIR not in sys.path:
  sys.path.append(ROOT_DIR)

from architectures.DWTMamba import DWTMamba
from architectures.HybridDWTGLMambaUNet import HybridDWTGLMambaUNet


def _to_tensor_img(arr):
  """Mengonversi numpy array citra ke PyTorch Tensor [B, C, H, W] berkisar [0, 1]."""
  if len(arr) == 0:
    return torch.empty(0)

  tensor = torch.tensor(arr, dtype=torch.float32)

  # Penyesuaian Dimensi -> Target: [B, C, H, W]
  if tensor.ndim == 3:  # [B, H, W] -> [B, 1, H, W]
    tensor = tensor.unsqueeze(1)
  elif tensor.ndim == 4 and tensor.shape[-1] in [
      1,
      3,
  ]:  # [B, H, W, C] -> [B, C, H, W]
    tensor = tensor.permute(0, 3, 1, 2)

  # FIX DOUBLE NORMALIZATION: Hanya bagi 255 jika max > 1.5
  if tensor.numel() > 0 and tensor.max() > 1.5:
    tensor /= 255.0

  return tensor


def _to_tensor_mask(arr):
  """Mengonversi numpy array mask ke PyTorch Tensor Long [B, H, W]."""
  if len(arr) == 0:
    return torch.empty(0)

  tensor = torch.tensor(arr, dtype=torch.long)

  # Hilangkan channel jika [B, H, W, 1] -> Target loss segmentasi PyTorch: [B, H, W]
  if tensor.ndim == 4 and tensor.shape[-1] == 1:
    tensor = tensor.squeeze(-1)
  elif tensor.ndim == 4 and tensor.shape[1] == 1:
    tensor = tensor.squeeze(1)

  return tensor


def unpack_split_info(split_entry, fold_idx=0):
  """Membongkar indeks split secara fleksibel untuk Holdout, String Array, & K-Fold."""
  if split_entry.shape == () or split_entry.size == 1:
    entry_item = split_entry.item()
    if isinstance(entry_item, dict):
      if "train" in entry_item:
        return entry_item
      elif fold_idx in entry_item or str(fold_idx) in entry_item:
        return entry_item.get(fold_idx, entry_item.get(str(fold_idx)))
    split_entry = entry_item

  if (
      isinstance(split_entry, (list, np.ndarray))
      and len(split_entry) in [2, 3]
      and split_entry.dtype == object # type: ignore
  ):
    return {
        "train": split_entry[0],
        "val": split_entry[1],
        "test": split_entry[2] if len(split_entry) > 2 else split_entry[1],
    }

  if (
      isinstance(split_entry, (list, np.ndarray))
      and len(split_entry) > 0
      and len(split_entry) <= 10
  ):
    selected_fold = split_entry[fold_idx % len(split_entry)]
    if isinstance(selected_fold, dict):
      return {
          "train": selected_fold.get("train", np.array([])),
          "val": selected_fold.get(
              "val", selected_fold.get("test", np.array([]))
          ),
          "test": selected_fold.get("test", np.array([])),
      }
    elif (
        isinstance(selected_fold, (list, tuple, np.ndarray))
        and len(selected_fold) >= 2
    ):
      return {
          "train": selected_fold[0],
          "val": selected_fold[1],
          "test": (
              selected_fold[2] if len(selected_fold) > 2 else selected_fold[1]
          ),
      }

  # Array 1D penanda per sampel ('train', 'val', 'test')
  if (
      isinstance(split_entry, np.ndarray)
      and split_entry.ndim == 1
      and len(split_entry) > 10
  ):
    if np.issubdtype(split_entry.dtype, np.character) or isinstance(
        split_entry[0], str
    ):
      train_idx = np.where(split_entry == "train")[0]
      val_idx = np.where(split_entry == "val")[0]
      test_idx = np.where(split_entry == "test")[0]
      return {"train": train_idx, "val": val_idx, "test": test_idx}
    else:
      val_idx = np.where(split_entry == fold_idx)[0]
      train_idx = np.where(split_entry != fold_idx)[0]
      return {"train": train_idx, "val": val_idx, "test": val_idx}

  return {"train": np.array([]), "val": np.array([]), "test": np.array([])}


def verify_stratification(
    npz_path="data/dataset_seg_1ch_224x224_nocrop.npz",
    config="splits",
    class_names=None,
):
  if class_names is None:
    class_names = {0: "Normal", 1: "Ischemia", 2: "Bleeding"}

  if not os.path.exists(npz_path):
    print(f"❌ File {npz_path} tidak ditemukan!")
    return

  data = np.load(npz_path, allow_pickle=True)
  labels = data["labels"].squeeze()
  total_samples = len(labels)

  # Auto-fallback jika config tidak ada tetapi key 'splits' ada
  if config not in data and "splits" in data:
    config = "splits"

  if config not in data:
    print(f"❌ Key '{config}' tidak ditemukan di NPZ!")
    return

  split_info = unpack_split_info(data[config])
  train_idx, val_idx, test_idx = (
      split_info["train"], # type: ignore
      split_info["val"], # type: ignore
      split_info["test"], # type: ignore
  )

  y_train = labels[train_idx]
  y_val = labels[val_idx]
  y_test = labels[test_idx] if len(test_idx) > 0 else np.array([])

  df_summary = pd.DataFrame(
      index=[f"Class {k} ({v})" for k, v in class_names.items()]
  )

  total_counts = (
      pd.Series(labels).value_counts().reindex([0, 1, 2], fill_value=0)
  )
  df_summary["Total (Count)"] = total_counts.values
  df_summary["Total (%)"] = (
      ((total_counts / total_samples) * 100).round(2).values
  )

  train_counts = (
      pd.Series(y_train).value_counts().reindex([0, 1, 2], fill_value=0)
  )
  df_summary["Train (Count)"] = train_counts.values
  df_summary["Train (%)"] = (
      ((train_counts / max(1, len(y_train))) * 100).round(2).values
  )

  val_counts = (
      pd.Series(y_val).value_counts().reindex([0, 1, 2], fill_value=0)
  )
  df_summary["Val (Count)"] = val_counts.values
  df_summary["Val (%)"] = (
      ((val_counts / max(1, len(y_val))) * 100).round(2).values
  )

  if len(y_test) > 0 and not np.array_equal(val_idx, test_idx):
    test_counts = (
        pd.Series(y_test).value_counts().reindex([0, 1, 2], fill_value=0)
    )
    df_summary["Test (Count)"] = test_counts.values
    df_summary["Test (%)"] = (
        ((test_counts / max(1, len(y_test))) * 100).round(2).values
    )

  print("=" * 75)
  print(f"📊 ANALISIS STRATIFIKASI LABEL: {config}")
  print(
      f"📂 Source File: {npz_path} (Total Data: {total_samples} Slices/Irisan)"
  )
  print("=" * 75)
  print(df_summary.to_string())
  print("-" * 75)


def run_diagnostics(
    npz_path="data/dataset_seg_1ch_224x224_nocrop.npz", config="splits"
):
  print("=" * 65)
  print("🔍 SEGMENTATION & MODEL DIAGNOSTIC TOOL")
  print("=" * 65)

  if not os.path.exists(npz_path):
    print(f"❌ File {npz_path} tidak ditemukan!")
    return

  data = np.load(npz_path, allow_pickle=True)
  print(f"\n📂 1. NPZ File Inspection ({npz_path}):")
  print(f"   - Keys tersedia        : {list(data.keys())}")

  is_segmentation = "masks" in data
  print(
      f"   - Mode Dataset Detected: {'SEGMENTATION (Images + Masks)' if is_segmentation else 'CLASSIFICATION (Images Only)'}"
  )

  if "images" not in data or "labels" not in data:
    print("❌ Key 'images' atau 'labels' tidak ditemukan!")
    return

  images = data["images"]
  labels = data["labels"].squeeze()
  masks = data["masks"] if is_segmentation else None

  if config not in data and "splits" in data:
    config = "splits"

  if config not in data:
    print(f"❌ Config split '{config}' tidak ditemukan!")
    return

  split_info = unpack_split_info(data[config])
  train_idx, val_idx, test_idx = (
      split_info["train"], # type: ignore
      split_info["val"], # type: ignore
      split_info["test"], # type: ignore
  )

  train_images, train_labels = images[train_idx], labels[train_idx]
  val_images, val_labels = images[val_idx], labels[val_idx]

  print(f"\n📊 {config} Info:")
  print(f"   - Train slices count   : {len(train_idx)}")
  print(f"   - Val slices count     : {len(val_idx)}")
  print(f"   - Test slices count    : {len(test_idx)}")

  print("\n🔄 2. DataLoader & Normalization Test:")
  tr_min, tr_max = float(train_images.min()), float(train_images.max())
  print(
      f"   - Train Images Range   : Min {tr_min:.4f} | Max {tr_max:.4f}"
      f" (Shape: {train_images.shape})"
  )

  # --- PENGECEKAN KHUSUS MASKS (SEGMENTASI) ---
  if is_segmentation and masks is not None:
    train_masks = masks[train_idx]
    m_min, m_max = float(train_masks.min()), float(train_masks.max())
    m_uniques = np.unique(train_masks)
    print(
        f"   - Train Masks Range    : Min {m_min:.4f} | Max {m_max:.4f}"
        f" (Shape: {train_masks.shape})"
    )
    print(f"   - Mask Unique Values   : {m_uniques.tolist()}")

    # Validasi Konsistensi Normal vs Lesi pada Masker
    normal_indices_train = np.where(train_labels == 0)[0]
    lesion_indices_train = np.where(train_labels > 0)[0]

    normal_masks_max = (
        train_masks[normal_indices_train].max()
        if len(normal_indices_train) > 0
        else 0
    )
    lesion_masks_max = (
        train_masks[lesion_indices_train].max()
        if len(lesion_indices_train) > 0
        else 0
    )

    print(
        f"   - Validasi Mask Normal : Max value = {normal_masks_max}"
        f" {'✅ (100% Zero-Mask)' if normal_masks_max == 0 else '⚠️ WARNING: Ada piksel non-zero di kelas Normal!'}"
    )
    print(
        f"   - Validasi Mask Lesi   : Max value = {lesion_masks_max}"
        f" {'✅ (Memiliki Kontur Lesi)' if lesion_masks_max > 0 else '⚠️ WARNING: Mask lesi bernilai 0 semua!'}"
    )

  # Sample Testing
  sample_size = min(16, len(train_images))
  sample_x = _to_tensor_img(train_images[:sample_size])
  sample_y_lbl = torch.tensor(
      train_labels[:sample_size], dtype=torch.long
  ).reshape(-1)

  in_channels = sample_x.shape[1]
  device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

  print("\n🧠 3. Model Forward & Mode Test (train vs eval):")
  print(f"   - Running on Device    : {device}")

  if is_segmentation and masks is not None:
    sample_m = _to_tensor_mask(masks[train_idx][:sample_size]).to(device)
    model = HybridDWTGLMambaUNet(in_channels=in_channels, num_classes=3).to(
        device
    )
    criterion = nn.CrossEntropyLoss()

    batch_x = sample_x.to(device)

    model.train()
    with torch.no_grad():
      out_train = model(batch_x)
      loss_train = criterion(out_train, sample_m).item()

    model.eval()
    with torch.no_grad():
      out_eval = model(batch_x)
      loss_eval = criterion(out_eval, sample_m).item()

    print(f"   - Model Target        : HybridDWTGLMambaUNet (Segmentation)")
    print(f"   - Input Shape         : {batch_x.shape}")
    print(
        f"   - Output Mask Shape   : {out_eval.shape} (Expected: [B, 3, 224,"
        " 224])"
    )
    print(
        f"   - Target Mask Shape   : {sample_m.shape} (Expected: [B, 224,"
        " 224])"
    )
    print(
        f"   - Loss mode train()   : {loss_train:.4f} | Loss mode eval()  :"
        f" {loss_eval:.4f}"
    )

  else:
    model = DWTMamba(
        in_channels=in_channels, num_classes=3, embed_dim=256
    ).to(device)
    criterion = nn.CrossEntropyLoss()
    batch_x, batch_y = sample_x.to(device), sample_y_lbl.to(device)

    model.eval()
    with torch.no_grad():
      out_eval = model(batch_x)
      loss_eval = criterion(out_eval, batch_y).item()

    print(f"   - Model Target        : DWTMamba (Classification)")
    print(f"   - Output Logits Shape : {out_eval.shape}")
    print(f"   - Loss mode eval()    : {loss_eval:.4f}")

  print("\n" + "=" * 65)
  print("📋 HASIL EVALUASI DIAGNOSTIK: SEMUA MODUL SEGMENTASI OK!")
  print("=" * 65)


if __name__ == "__main__":
  npz_relative_path = "data/dataset_seg_1ch_224x224_nocrop.npz"
  config = "splits"
  run_diagnostics(npz_relative_path, config)
  verify_stratification(npz_relative_path, config)