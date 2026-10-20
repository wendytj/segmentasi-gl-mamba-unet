#!/bin/bash
set -e

echo "=========================================================="
echo "🚀 Memulai Antrean Eksperimen Ablasi LR Conv Ratio 0.1 & 1.0"
echo "$(date)"
echo "=========================================================="

echo ""
echo "▶️ [1/2] Menjalankan: LR Conv Ratio 0.1"

python train.py --experiment_code conv_01 --lr_conv_ratio 0.1

echo "✅ [1/2] Selesai: $(date)"
sleep 10

echo ""
echo "▶️ [2/2] Menjalankan: LR Conv Ratio 0.1"

python train.py --experiment_code conv_10 --lr_conv_ratio 1.0

echo "✅ [2/2] Selesai: $(date)"
sleep 10

echo ""
echo "=========================================================="
echo "🎉 Semua antrean eksperimen selesai: $(date)"
echo "=========================================================="