#!/bin/bash
set -e

echo "=========================================================="
echo "🚀 Memulai Antrean Eksperimen Ablasi Weight Decay"
echo "$(date)"
echo "=========================================================="

echo ""
echo "▶️ [1/3] Menjalankan: Decay 0"

python train.py --experiment_code decay-0 --weight_decay 0

echo "✅ [1/3] Selesai: $(date)"
sleep 10

echo ""
echo "▶️ [2/3] Menjalankan: Decay 1e-4"

python train.py --experiment_code decay-1e-4 --weight_decay 1e-4

echo "✅ [2/3] Selesai: $(date)"
sleep 10

echo ""
echo "▶️ [3/3] Menjalankan: Decay 1e-2"

python train.py --experiment_code decay-1e-2 --weight_decay 1e-2

echo "✅ [3/3] Selesai: $(date)"
sleep 10

echo ""
echo "=========================================================="
echo "🎉 Semua antrean eksperimen selesai: $(date)"
echo "=========================================================="