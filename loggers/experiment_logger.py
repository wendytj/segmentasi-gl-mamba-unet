import json
import os
import matplotlib

matplotlib.use("Agg")  # Aman untuk server headless
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.ndimage import distance_transform_edt
from train_modules.configs import CONFIG
import seaborn as sns
from sklearn.metrics import (
    accuracy_score,
    confusion_matrix,
    f1_score,
    matthews_corrcoef,
    precision_score,
    recall_score,
    roc_auc_score,
)


def compute_hd95_binary(pred_binary, gt_binary, voxel_spacing=(1.0, 1.0)):
    """Hitung 95th percentile Hausdorff Distance (HD95) biner menggunakan distance_transform_edt.

    Penanganan aman jika mask kosong.
    """
    pred_pts = np.argwhere(pred_binary)
    gt_pts = np.argwhere(gt_binary)

    if len(pred_pts) == 0 and len(gt_pts) == 0:
        return 0.0
    if len(pred_pts) == 0 or len(gt_pts) == 0:
        return np.nan

    dt_gt = distance_transform_edt(~gt_binary, sampling=voxel_spacing)
    dt_pred = distance_transform_edt(~pred_binary, sampling=voxel_spacing)

    dist_pred_to_gt = dt_gt[pred_binary]  # type: ignore
    dist_gt_to_pred = dt_pred[gt_binary]  # type: ignore

    if len(dist_pred_to_gt) == 0 or len(dist_gt_to_pred) == 0:
        return np.nan

    hd95 = max(np.percentile(dist_pred_to_gt, 95), np.percentile(dist_gt_to_pred, 95))
    return float(hd95)


class ExperimentLogger:

    def __init__(
        self,
        save_dir="./logs",
        class_names=None,
        hparams=None,
    ):
        self.save_dir = save_dir
        self.class_names = class_names or ["Background", "Ischemia", "Bleeding"]
        self.hparams = hparams or {}
        os.makedirs(save_dir, exist_ok=True)

        # History mencatat metrik segmentasi dan derivatif klasifikasi
        self.history = {
            "epoch": [],
            "lr": [],
            "train_loss": [],
            "val_loss": [],
            "train_dsc": [],
            "val_dsc": [],
            "val_iou": [],
            "val_cls_f1": [],
            "val_cls_mcc": [],
            "patience": [],
        }

        if self.hparams:
            self.save_hparams()

    def log_error(self, error_msg, context="TRAINING_ERROR"):
        """Mencatat error atau interupsi ke log riwayat / file teks."""
        error_log_path = os.path.join(self.save_dir, "error_log.txt")
        with open(error_log_path, "a", encoding="utf-8") as f:
            f.write(f"[{context}] {str(error_msg)}\n")
        print(f"📝 Error tercatat di: {error_log_path}")

    def save_hparams(self, filename="hparams.json"):
        path = os.path.join(self.save_dir, filename)
        with open(path, "w", encoding="utf-8") as f:
            json.dump(self.hparams, f, indent=4, default=str)
        print(f"⚙️ Hyperparameters disimpan di: {path}")

    def log_epoch(
        self,
        epoch,
        train_loss,
        val_loss,
        train_dsc,
        val_dsc,
        val_iou,
        val_cls_f1,
        val_cls_mcc,
        lr=0.0,
        patience=0,
    ):
        """Catat metrik pelatihan segmentasi per epoch (tanpa bobot kelas konstan)."""
        self.history["epoch"].append(int(epoch))
        self.history["lr"].append(float(lr))
        self.history["train_loss"].append(float(train_loss))
        self.history["val_loss"].append(float(val_loss))
        self.history["train_dsc"].append(float(train_dsc))
        self.history["val_dsc"].append(float(val_dsc))
        self.history["val_iou"].append(float(val_iou))
        self.history["val_cls_f1"].append(float(val_cls_f1))
        self.history["val_cls_mcc"].append(float(val_cls_mcc))
        self.history["patience"].append(int(patience))

    def export_csv(self, filename="training_history.csv"):
        if not self.history["epoch"]:
            print("⚠️ Belum ada data history untuk diekspor ke CSV.")
            return None
        df = pd.DataFrame(self.history)
        path = os.path.join(self.save_dir, filename)
        df.to_csv(path, index=False)
        return path

    def compute_metrics(
        self, y_true_masks, y_pred_masks, y_pred_probs=None, compute_hd95=False
    ):
        if hasattr(y_true_masks, "cpu"):
            y_true_masks = y_true_masks.cpu().numpy()
        if hasattr(y_pred_masks, "cpu"):
            y_pred_masks = y_pred_masks.cpu().numpy()
        if y_pred_probs is not None and hasattr(y_pred_probs, "cpu"):
            y_pred_probs = y_pred_probs.cpu().numpy()

        # Pastikan shape format kanal berada di belakang [N, H, W, 2] atau ditranspose jika [N, 2, H, W]
        if y_true_masks.ndim == 4 and y_true_masks.shape[1] == 2:
            y_true_masks = np.transpose(y_true_masks, (0, 2, 3, 1))
        if y_pred_masks.ndim == 4 and y_pred_masks.shape[1] == 2:
            y_pred_masks = np.transpose(y_pred_masks, (0, 2, 3, 1))

        total_slices = len(y_true_masks)

        # ==================== 1. METRIK SEGMENTASI PIKSEL (2 KANAL) ====================
        dsc_per_class = []
        iou_per_class = []
        prec_per_class = []
        rec_per_class = []
        hd95_per_class = []

        class_names_2ch = ["Ischemia", "Bleeding"]

        # 2 Kanal independen: 0 = Ischemia, 1 = Bleeding
        for c in range(2):
            gt_bin = y_true_masks[:, :, :, c].astype(bool)
            pred_bin = y_pred_masks[:, :, :, c].astype(bool)

            intersection = np.logical_and(pred_bin, gt_bin).sum()
            union = np.logical_or(pred_bin, gt_bin).sum()
            cardinality = pred_bin.sum() + gt_bin.sum()

            # Dice & IoU
            dsc = (2.0 * intersection) / (cardinality + 1e-6)
            iou = intersection / (union + 1e-6)

            # Precision & Recall Piksel
            precision = intersection / (pred_bin.sum() + 1e-6)
            recall = intersection / (gt_bin.sum() + 1e-6)

            dsc_per_class.append(float(dsc))
            iou_per_class.append(float(iou))
            prec_per_class.append(float(precision))
            rec_per_class.append(float(recall))

            # HD95 Kalkulasi (Iterasi Per-Slice 2D per Kanal)
            if compute_hd95:
                print(
                    f"   📏 Menghitung HD95 untuk Kanal {c} ({class_names_2ch[c]}) pada {total_slices} slice..."
                )
                hd95_slice_list = []

                # Gunakan tqdm jika tersedia, atau perulangan biasa dengan status
                for i in range(total_slices):
                    gt_2d = gt_bin[i]
                    pred_2d = pred_bin[i]

                    if not gt_2d.any() and not pred_2d.any():
                        continue

                    hd_val = compute_hd95_binary(
                        pred_2d, gt_2d, voxel_spacing=(1.0, 1.0)  # type: ignore
                    )
                    if not np.isnan(hd_val):
                        hd95_slice_list.append(hd_val)

                hd95_val = float(np.mean(hd95_slice_list)) if hd95_slice_list else None
                hd95_per_class.append(
                    round(hd95_val, 4) if hd95_val is not None else None
                )
                print(f"   ✓ HD95 {class_names_2ch[c]} Selesai: {hd95_per_class[-1]}")
            else:
                hd95_per_class.append(None)

        # Rata-rata metrik untuk kedua kanal lesi
        mean_dsc_lesion = float(np.mean(dsc_per_class))
        mean_iou_lesion = float(np.mean(iou_per_class))
        mean_prec_lesion = float(np.mean(prec_per_class))
        mean_rec_lesion = float(np.mean(rec_per_class))

        valid_hd95_lesions = [v for v in hd95_per_class if v is not None]
        mean_hd95_lesion = (
            float(np.mean(valid_hd95_lesions)) if valid_hd95_lesions else None
        )

        # ==================== 2. METRIK KLASIFIKASI DERIVATIF 3-KELAS (SLICE LEVEL) ====================
        min_lesion_pixels = CONFIG["min_lesion_pixels"]

        # Total piksel per kanal lesi per slice [B] dari format 2-kanal
        isch_pixels_pred = (
            y_pred_masks[:, :, :, 0].reshape(total_slices, -1).sum(axis=1)
        )
        bleed_pixels_pred = (
            y_pred_masks[:, :, :, 1].reshape(total_slices, -1).sum(axis=1)
        )

        isch_pixels_gt = y_true_masks[:, :, :, 0].reshape(total_slices, -1).sum(axis=1)
        bleed_pixels_gt = y_true_masks[:, :, :, 1].reshape(total_slices, -1).sum(axis=1)

        # Penentuan label 3-kelas per slice (0: Normal, 1: Ischemia, 2: Bleeding)
        cls_pred_slice = np.zeros(total_slices, dtype=int)
        cls_gt_slice = np.zeros(total_slices, dtype=int)

        for i in range(total_slices):
            # Decision Prediksi berdasarkan dominasi kanal & min_lesion_pixels
            p_isch = isch_pixels_pred[i] >= min_lesion_pixels
            p_bleed = bleed_pixels_pred[i] >= min_lesion_pixels

            if p_isch and p_bleed:
                cls_pred_slice[i] = (
                    1 if isch_pixels_pred[i] >= bleed_pixels_pred[i] else 2
                )
            elif p_isch:
                cls_pred_slice[i] = 1
            elif p_bleed:
                cls_pred_slice[i] = 2
            else:
                cls_pred_slice[i] = 0  # Normal

            # Decision Ground Truth
            g_isch = isch_pixels_gt[i] >= min_lesion_pixels
            g_bleed = bleed_pixels_gt[i] >= min_lesion_pixels

            if g_isch and g_bleed:
                cls_gt_slice[i] = 1 if isch_pixels_gt[i] >= bleed_pixels_gt[i] else 2
            elif g_isch:
                cls_gt_slice[i] = 1
            elif g_bleed:
                cls_gt_slice[i] = 2
            else:
                cls_gt_slice[i] = 0  # Normal

        # Confusion Matrix 3x3 (Normal, Ischemia, Bleeding)
        cls_cm = confusion_matrix(cls_gt_slice, cls_pred_slice, labels=[0, 1, 2])
        cls_acc = accuracy_score(cls_gt_slice, cls_pred_slice)
        cls_prec_macro = precision_score(
            cls_gt_slice, cls_pred_slice, average="macro", zero_division=0
        )
        cls_rec_macro = recall_score(
            cls_gt_slice, cls_pred_slice, average="macro", zero_division=0
        )
        cls_f1_macro = f1_score(
            cls_gt_slice, cls_pred_slice, average="macro", zero_division=0
        )
        cls_mcc = matthews_corrcoef(cls_gt_slice, cls_pred_slice)

        # Slice-Level Probabilities per Class (Macro-Averaged One-vs-Rest Binary AUC)
        cls_auc = 0.0
        try:
            if y_pred_probs is not None:
                if y_pred_probs.ndim == 4 and y_pred_probs.shape[1] == 2:
                    bg_prob = np.clip(
                        1.0 - (y_pred_probs[:, 0, :, :] + y_pred_probs[:, 1, :, :]),
                        0,
                        1,
                    )
                    probs_3ch = np.stack(
                        [bg_prob, y_pred_probs[:, 0, :, :], y_pred_probs[:, 1, :, :]],
                        axis=1,
                    )
                else:
                    probs_3ch = y_pred_probs

                # Rata-rata spasial per kanal -> shape [N, 3]
                slice_probs = probs_3ch.reshape(probs_3ch.shape[0], 3, -1).mean(axis=2)

                unique_classes = np.unique(cls_gt_slice)
                if len(unique_classes) > 1:
                    auc_scores = []
                    for c_idx in range(3):
                        binary_gt = (cls_gt_slice == c_idx).astype(int)
                        if len(np.unique(binary_gt)) > 1:
                            score_c = roc_auc_score(binary_gt, slice_probs[:, c_idx])
                            auc_scores.append(score_c)

                    if auc_scores:
                        cls_auc = float(np.mean(auc_scores))
        except Exception as e:
            print(f"⚠️ AUC calculation skipped: {e}")
            cls_auc = 0.0

        # Susun Kamus Hasil per Kelas (Ischemia & Bleeding)
        per_class_res = {}
        for i, name in enumerate(class_names_2ch):
            per_class_res[name] = {
                "dsc": round(dsc_per_class[i], 4),
                "iou": round(iou_per_class[i], 4),
                "precision": round(prec_per_class[i], 4),
                "recall": round(rec_per_class[i], 4),
                "hd95": hd95_per_class[i],
            }

        metrics = {
            "confusion_matrix": cls_cm.tolist(),
            "global_metrics": {
                # Segmentasi Piksel
                "mean_lesion_dsc": round(mean_dsc_lesion, 4),
                "mean_lesion_iou": round(mean_iou_lesion, 4),
                "mean_lesion_precision": round(mean_prec_lesion, 4),
                "mean_lesion_recall": round(mean_rec_lesion, 4),
                "mean_lesion_hd95": (
                    round(mean_hd95_lesion, 4) if mean_hd95_lesion is not None else None
                ),
                # Slice-Level Klasifikasi Biner
                "slice_cls_accuracy": round(float(cls_acc), 4),
                "slice_cls_precision": round(float(cls_prec_macro), 4),
                "slice_cls_recall": round(float(cls_rec_macro), 4),
                "slice_cls_f1": round(float(cls_f1_macro), 4),
                "slice_cls_mcc": round(float(cls_mcc), 4),
                "slice_cls_auc": round(float(cls_auc), 4),
            },
            "per_class_metrics": per_class_res,
        }
        return metrics, cls_cm

    def load_history_from_csv(self, filename="training_history.csv"):
        """Memuat kembali riwayat pelatihan dari CSV jika training di-resume."""
        path = os.path.join(self.save_dir, filename)
        if os.path.exists(path):
            try:
                df = pd.read_csv(path)
                self.history = df.to_dict(orient="list")
                print(
                    f"🔄 Riwayat log sebelumnya berhasil dimuat ({len(self.history['epoch'])} epoch ditemukan)."
                )
            except Exception as e:
                print(f"⚠️ Gagal membaca history CSV sebelumnya: {e}")

    def save_best_results_json(
        self,
        class_counts=None,
        initial_class_weights=None,
        best_dsc_epoch=-1,
        best_val_dsc=-1.0,
        best_loss_epoch=-1,
        best_val_loss=float("inf"),
        dsc_eval_data=None,  # Tuple: (train_eval, val_eval, test_eval)
        loss_eval_data=None,  # Tuple: (train_eval, val_eval, test_eval)
        filename="best_model_metrics.json",
    ):
        """Simpan snapshot performa model terbaik (By DSC & By Loss) ke JSON."""

        def format_eval_res(eval_tuple):
            if eval_tuple is None or eval_tuple[0] is None:
                return None
            # Sertakan HD95 pada evaluasi best model
            metrics, cm = self.compute_metrics(*eval_tuple, compute_hd95=True)
            return {
                "confusion_matrix": cm.tolist() if hasattr(cm, "tolist") else cm,
                "global_metrics": metrics["global_metrics"],
                "per_class_metrics": metrics["per_class_metrics"],
            }

        tr_dsc_res, va_dsc_res, te_dsc_res = None, None, None
        if dsc_eval_data is not None:
            tr_dsc_res = format_eval_res(dsc_eval_data[0])
            va_dsc_res = format_eval_res(dsc_eval_data[1])
            te_dsc_res = format_eval_res(dsc_eval_data[2])

        if loss_eval_data is not None:
            tr_loss_res = format_eval_res(loss_eval_data[0])
            va_loss_res = format_eval_res(loss_eval_data[1])
            te_loss_res = format_eval_res(loss_eval_data[2])
        else:
            tr_loss_res, va_loss_res, te_loss_res = (
                tr_dsc_res,
                va_dsc_res,
                te_dsc_res,
            )

        cw_list = []
        if initial_class_weights is not None:
            cw_list = (
                initial_class_weights.cpu().tolist()
                if hasattr(initial_class_weights, "cpu")
                else list(initial_class_weights)
            )

        cc_list = []
        if class_counts is not None:
            cc_list = (
                class_counts.tolist()
                if hasattr(class_counts, "tolist")
                else list(class_counts)
            )

        summary_json = {
            "dataset_info": {
                "num_classes": len(self.class_names),
                "class_labels": self.class_names,
                "train_class_counts": cc_list,
                "initial_class_weights": cw_list,
            },
            "hyperparameters": self.hparams,
            "model_by_val_dsc": {
                "criterion": "best_val_dsc",
                "best_epoch": int(best_dsc_epoch),
                "best_metric_value": round(float(best_val_dsc), 4),
                "train_results": tr_dsc_res,
                "val_results": va_dsc_res,
                "test_results": te_dsc_res,
            },
            "model_by_val_loss": {
                "criterion": "best_val_loss",
                "best_epoch": int(best_loss_epoch),
                "best_metric_value": round(float(best_val_loss), 4),
                "train_results": tr_loss_res,
                "val_results": va_loss_res,
                "test_results": te_loss_res,
            },
        }

        json_path = os.path.join(self.save_dir, filename)
        with open(json_path, "w", encoding="utf-8") as f:
            json.dump(summary_json, f, indent=4, default=str)

        print(f"📄 Snapshot JSON Best Model disimpan di: {json_path}")
        return summary_json

    def plot_confusion_matrix(self, cm, filename="slice_confusion_matrix.png"):
        """Plot Confusion Matrix 3x3 Slice-Level (Normal, Ischemia, Bleeding)."""
        plt.figure(figsize=(6, 5))
        sns.heatmap(
            cm,
            annot=True,
            fmt="d",
            cmap="Blues",
            xticklabels=["Normal", "Ischemia", "Bleeding"],
            yticklabels=["Normal", "Ischemia", "Bleeding"],
        )
        plt.title(
            "Slice-Level Lesion Classification Confusion Matrix", fontweight="bold"
        )
        plt.xlabel("Predicted")
        plt.ylabel("True Ground Truth")
        plt.tight_layout()
        plt.savefig(os.path.join(self.save_dir, filename), dpi=300)
        plt.close()

    def plot_learning_curves(
        self,
        test_res=None,
        filename="training_dashboard.png",
        save_individual=True,
    ):
        """Render Dashboard Grafik Pelatihan Segmentasi & Ekspor File Terpisah (Total 7 PNG)."""
        if not self.history["epoch"]:
            print("⚠️ Belum ada riwayat epoch untuk digambar grafiknya.")
            return
        sns.set_theme(style="whitegrid")

        epochs = self.history["epoch"]
        best_idx = (
            int(np.argmax(self.history["val_dsc"])) if self.history["val_dsc"] else 0
        )
        best_epoch = epochs[best_idx] if epochs else 1
        best_dsc_val = (
            self.history["val_dsc"][best_idx] if self.history["val_dsc"] else 0.0
        )
        patience_limit = self.hparams.get("early_stop_patience", 10)

        summary_text = (
            f"   === SEGMENTATION TRAINING SUMMARY ===\n\n"
            f" • Total Epochs Executed : {len(epochs)} / {self.hparams.get('max_epochs', '-')}\n"
            f" • Peak Model Saved Epoch: {best_epoch}\n"
            f" • Best Val Mean DSC    : {best_dsc_val:.4f}\n"
            f" • Best Val Mean IoU    : {self.history['val_iou'][best_idx]:.4f}\n"
            f" • Slice Val Cls F1     : {self.history['val_cls_f1'][best_idx]:.4f}\n"
            f" • Slice Val Cls MCC    : {self.history['val_cls_mcc'][best_idx]:.4f}\n\n"
        )

        if test_res is not None:
            g = test_res.get("global_metrics", test_res)
            summary_text += (
                f"   === FINAL TEST BENCHMARK ===\n\n"
                f" • Test Mean Lesion DSC : {g['mean_lesion_dsc']:.4f}\n"
                f" • Test Mean Lesion IoU : {g['mean_lesion_iou']:.4f}\n"
                f" • Test Mean HD95 (mm)  : {g['mean_lesion_hd95'] if g['mean_lesion_hd95'] else 'N/A'}\n"
                f" • Slice Test Cls Acc   : {g['slice_cls_accuracy']:.4f}\n"
                f" • Slice Test Cls MCC   : {g['slice_cls_mcc']:.4f}\n"
                f" • Slice Test Cls AUC   : {g['slice_cls_auc']:.4f}"
            )

        # -------------------------------------------------------------
        # 1. RENDER DASHBOARD GABUNGAN (File 1/7)
        # -------------------------------------------------------------
        fig, axes = plt.subplots(2, 3, figsize=(18, 10))

        # Panel 1: Loss
        axes[0, 0].plot(
            epochs,
            self.history["train_loss"],
            label="Train Loss",
            color="crimson",
            linewidth=2,
        )
        axes[0, 0].plot(
            epochs,
            self.history["val_loss"],
            label="Val Loss",
            color="navy",
            linestyle="--",
            linewidth=2,
        )
        axes[0, 0].axvline(
            x=best_epoch,
            color="gold",
            linestyle=":",
            linewidth=2,
            label=f"Best Ep ({best_epoch})",
        )
        axes[0, 0].set_title("Loss Dynamics", fontweight="bold")
        axes[0, 0].set_xlabel("Epoch")
        axes[0, 0].set_ylabel("Loss")
        axes[0, 0].legend()

        # Panel 2: DSC
        axes[0, 1].plot(
            epochs,
            self.history["train_dsc"],
            label="Train DSC",
            color="forestgreen",
            linewidth=2,
        )
        axes[0, 1].plot(
            epochs,
            self.history["val_dsc"],
            label="Val DSC",
            color="darkorange",
            linestyle="--",
            linewidth=2,
        )
        axes[0, 1].axvline(
            x=best_epoch,
            color="gold",
            linestyle=":",
            linewidth=2,
            label=f"Best Ep ({best_epoch})",
        )
        axes[0, 1].set_title("Mean Dice Coefficient (DSC)", fontweight="bold")
        axes[0, 1].set_xlabel("Epoch")
        axes[0, 1].set_ylabel("DSC Score")
        axes[0, 1].legend()

        # Panel 3: Val IoU & Slice MCC
        axes[0, 2].plot(
            epochs,
            self.history["val_iou"],
            label="Val Mean IoU",
            color="teal",
            linewidth=2,
        )
        axes[0, 2].plot(
            epochs,
            self.history["val_cls_mcc"],
            label="Slice Val MCC",
            color="purple",
            linewidth=2,
        )
        axes[0, 2].axvline(x=best_epoch, color="gold", linestyle=":", linewidth=2)
        axes[0, 2].set_title("Validation Quality (IoU vs Slice MCC)", fontweight="bold")
        axes[0, 2].set_xlabel("Epoch")
        axes[0, 2].set_ylabel("Score")
        axes[0, 2].legend()

        # Panel 4: LR
        axes[1, 0].plot(
            epochs,
            self.history["lr"],
            label="Effective LR",
            color="darkmagenta",
            linewidth=2,
        )
        axes[1, 0].set_title("Learning Rate Schedule", fontweight="bold")
        axes[1, 0].set_xlabel("Epoch")
        axes[1, 0].set_ylabel("Learning Rate")
        axes[1, 0].ticklabel_format(style="scientific", scilimits=(0, 0), axis="y")
        axes[1, 0].legend()

        # Panel 5: Patience
        axes[1, 1].plot(
            epochs,
            self.history["patience"],
            label="Patience Count",
            color="orangered",
            linewidth=2,
            marker="o",
            markersize=4,
        )
        axes[1, 1].axhline(
            y=patience_limit,
            color="red",
            linestyle="--",
            linewidth=2,
            label=f"Patience Limit ({patience_limit})",
        )
        axes[1, 1].set_title("Early Stopping Patience Tracker", fontweight="bold")
        axes[1, 1].set_xlabel("Epoch")
        axes[1, 1].set_ylabel("Stagnant Epochs")
        axes[1, 1].legend()

        # Panel 6: Summary Text
        axes[1, 2].axis("off")
        axes[1, 2].text(
            0.05,
            0.95,
            summary_text,
            transform=axes[1, 2].transAxes,
            fontsize=10.5,
            verticalalignment="top",
            fontfamily="monospace",
            bbox=dict(
                boxstyle="round,pad=0.8",
                facecolor="whitesmoke",
                alpha=0.9,
                edgecolor="lightgray",
            ),
        )

        plt.tight_layout()
        dashboard_path = os.path.join(self.save_dir, filename)
        plt.savefig(dashboard_path, dpi=300)
        plt.close()

        print(f"📊 [1/7] Dashboard Gabungan 6-in-1 disimpan di: {dashboard_path}")

        # -------------------------------------------------------------
        # 2. EKSPOR GRAFIK TERPISAH (File 2-7/7)
        # -------------------------------------------------------------
        if save_individual:
            plots_dir = os.path.join(self.save_dir, "plots")
            os.makedirs(plots_dir, exist_ok=True)

            # File 2: Loss Curve
            plt.figure(figsize=(7, 5))
            plt.plot(
                epochs,
                self.history["train_loss"],
                label="Train Loss",
                color="crimson",
                linewidth=2,
            )
            plt.plot(
                epochs,
                self.history["val_loss"],
                label="Val Loss",
                color="navy",
                linestyle="--",
                linewidth=2,
            )
            plt.axvline(
                x=best_epoch,
                color="gold",
                linestyle=":",
                linewidth=2,
                label=f"Best Ep ({best_epoch})",
            )
            plt.title("Loss Dynamics per Epoch", fontweight="bold")
            plt.xlabel("Epoch")
            plt.ylabel("Loss")
            plt.legend()
            plt.tight_layout()
            plt.savefig(os.path.join(plots_dir, "loss_curve.png"), dpi=300)
            plt.close()

            # File 3: DSC Curve
            plt.figure(figsize=(7, 5))
            plt.plot(
                epochs,
                self.history["train_dsc"],
                label="Train DSC",
                color="forestgreen",
                linewidth=2,
            )
            plt.plot(
                epochs,
                self.history["val_dsc"],
                label="Val DSC",
                color="darkorange",
                linestyle="--",
                linewidth=2,
            )
            plt.axvline(
                x=best_epoch,
                color="gold",
                linestyle=":",
                linewidth=2,
                label=f"Best Ep ({best_epoch})",
            )
            plt.title("Mean Dice Coefficient (DSC) Progression", fontweight="bold")
            plt.xlabel("Epoch")
            plt.ylabel("DSC Score")
            plt.legend()
            plt.tight_layout()
            plt.savefig(os.path.join(plots_dir, "dsc_curve.png"), dpi=300)
            plt.close()

            # File 4: Val Metrics (IoU vs Slice MCC)
            plt.figure(figsize=(7, 5))
            plt.plot(
                epochs,
                self.history["val_iou"],
                label="Val Mean IoU",
                color="teal",
                linewidth=2,
            )
            plt.plot(
                epochs,
                self.history["val_cls_mcc"],
                label="Slice Val MCC",
                color="purple",
                linewidth=2,
            )
            plt.axvline(x=best_epoch, color="gold", linestyle=":", linewidth=2)
            plt.title("Validation Quality (IoU vs Slice MCC)", fontweight="bold")
            plt.xlabel("Epoch")
            plt.ylabel("Score")
            plt.legend()
            plt.tight_layout()
            plt.savefig(os.path.join(plots_dir, "val_metrics_iou_mcc.png"), dpi=300)
            plt.close()

            # File 5: Learning Rate Schedule
            plt.figure(figsize=(7, 5))
            plt.plot(
                epochs,
                self.history["lr"],
                label="Effective LR",
                color="darkmagenta",
                linewidth=2,
            )
            plt.title("Learning Rate Schedule per Epoch", fontweight="bold")
            plt.xlabel("Epoch")
            plt.ylabel("Learning Rate")
            plt.ticklabel_format(style="scientific", scilimits=(0, 0), axis="y")
            plt.legend()
            plt.tight_layout()
            plt.savefig(os.path.join(plots_dir, "lr_schedule.png"), dpi=300)
            plt.close()

            # File 6: Patience Tracker
            plt.figure(figsize=(7, 5))
            plt.plot(
                epochs,
                self.history["patience"],
                label="Patience Count",
                color="orangered",
                linewidth=2,
                marker="o",
                markersize=4,
            )
            plt.axhline(
                y=patience_limit,
                color="red",
                linestyle="--",
                linewidth=2,
                label=f"Patience Limit ({patience_limit})",
            )
            plt.title("Early Stopping Patience Tracker", fontweight="bold")
            plt.xlabel("Epoch")
            plt.ylabel("Stagnant Epochs")
            plt.legend()
            plt.tight_layout()
            plt.savefig(os.path.join(plots_dir, "patience_tracker.png"), dpi=300)
            plt.close()

            # File 7: Performance & Test Benchmark Summary Card
            plt.figure(figsize=(7, 5))
            plt.axis("off")
            plt.text(
                0.05,
                0.95,
                summary_text,
                transform=plt.gca().transAxes,
                fontsize=10.5,
                verticalalignment="top",
                fontfamily="monospace",
                bbox=dict(
                    boxstyle="round,pad=0.8",
                    facecolor="whitesmoke",
                    alpha=0.9,
                    edgecolor="lightgray",
                ),
            )
            plt.title("Performance & Benchmark Summary", fontweight="bold")
            plt.tight_layout()
            plt.savefig(os.path.join(plots_dir, "summary_card.png"), dpi=300)
            plt.close()

            print(
                f"📁 [2-7/7] 6 Grafik terpisah berhasil disimpan di folder: {plots_dir}"
            )
