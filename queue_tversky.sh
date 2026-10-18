#!/bin/bash
set -e

echo "=========================================================="
echo "🚀 Memulai Antrean Eksperimen Ablasi Loss Tversky"
echo "$(date)"
echo "=========================================================="

echo ""
echo "▶️ [1/4] Menjalankan: Tversky + Unweighted BCE..."

python train.py --experiment_code tversky_ce_unweighted --loss_type tversky+ce --no-use_class_weights

echo "✅ [1/4] Selesai: $(date)"
sleep 10

echo ""
echo "▶️ [2/4] Menjalankan: Tversky + Weighted BCE..."

python train.py --experiment_code tversky_ce --loss_type tversky+ce --use_class_weights --weight_mode median

echo "✅ [2/4] Selesai: $(date)"
sleep 10

echo ""
echo "▶️ [3/4] Menjalankan: Tversky + Focal..."

python train.py --experiment_code tversky_focal --loss_type tversky+focal --no-use_class_weights

echo "✅ [3/4] Selesai: $(date)"
sleep 10

echo ""
echo "▶️ [4/4] Menjalankan: Tversky + GSL..."

python train.py --experiment_code tversky_gsl --loss_type tversky+gsl --no-use_class_weights --npz_path data/dataset_seg_2ch_224x224_nocrop_with_dtm_ratio_0_1.npz

echo "✅ [4/4] Selesai: $(date)"

echo ""
echo "=========================================================="
echo "🎉 Semua antrean eksperimen selesai: $(date)"
echo "=========================================================="