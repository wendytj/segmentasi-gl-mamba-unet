#!/bin/bash
set -e

echo "=========================================================="
echo "🚀 Memulai [MOCK] Antrean Eksperimen Ablasi Loss Dice"
echo "$(date)"
echo "=========================================================="

echo ""
echo "▶️ [1/3] Menjalankan: Dice + Unweighted BCE..."

python train.py --exp_name mock_dice_ce_unweighted --loss_type dice+ce --no-use_class_weights --max_epochs 1 --warmup_epochs 0

echo "✅ [1/3] Selesai: $(date)"
sleep 10

echo ""
echo "▶️ [2/3] Menjalankan: Dice + Focal..."

python train.py --exp_name mock_dice_focal --loss_type dice+focal --no-use_class_weights --max_epochs 1 --warmup_epochs 0

echo "✅ [2/3] Selesai: $(date)"
sleep 10

echo ""
echo "▶️ [3/3] Menjalankan: Dice + GSL..."

python train.py --exp_name mock_dice_gsl --loss_type dice+gsl --no-use_class_weights --max_epochs 1 --warmup_epochs 0 --npz_path data/dataset_seg_2ch_224x224_nocrop_with_dtm_ratio_0_1.npz

echo "✅ [3/3] Selesai: $(date)"

echo ""
echo "=========================================================="
echo "🎉 Semua antrean eksperimen selesai: $(date)"
echo "=========================================================="