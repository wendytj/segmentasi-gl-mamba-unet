#!/bin/bash
set -e

echo "=========================================================="
echo "🚀 Memulai [MOCK] Antrean Eksperimen Ablasi BS 8 - 32"
echo "$(date)"
echo "=========================================================="

echo ""
echo "▶️ [1/3] Menjalankan: Eff BS 8"

python train.py --experiment_code eff_bs_8 --batch_size 8 --accumulation_steps 1

echo "✅ [1/3] Selesai: $(date)"
sleep 10

echo ""
echo "▶️ [2/3] Menjalankan: Eff BS 16"

python train.py --experiment_code eff_bs_16 --batch_size 16 --accumulation_steps 1

echo "✅ [2/3] Selesai: $(date)"
sleep 10

echo ""
echo "▶️ [2/3] Menjalankan: Eff BS 32"

python train.py --experiment_code eff_bs_32 --batch_size 16 --accumulation_steps 2

echo "✅ [2/3] Selesai: $(date)"

echo ""
echo "=========================================================="
echo "🎉 Semua antrean eksperimen selesai: $(date)"
echo "=========================================================="