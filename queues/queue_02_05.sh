#!/bin/bash
set -e

echo "=========================================================="
echo "🚀 Memulai Antrean Eksperimen Ablasi Data Scaling 0.2 - 0.5"
echo "$(date)"
echo "=========================================================="

gdown 1t8UgXRpXQI3rWTgqLspexcb9NZR3db75 -O data/dataset_seg_2ch_224x224_nocrop_ratio_0_2.npz
gdown 1RjgJUFsN-MD44_wNqOmmbRyHWwdjcBCJ -O data/dataset_seg_2ch_224x224_nocrop_ratio_0_5.npz

echo ""
echo "▶️ [1/2] Menjalankan: Rasio 0.2"

python train.py --experiment_code scaling_02 --npz_path data/dataset_seg_2ch_224x224_nocrop_ratio_0_2.npz

echo "✅ [1/2] Selesai: $(date)"
sleep 10

echo ""
echo "▶️ [2/2] Menjalankan: Rasio 0.5"

python train.py --experiment_code scaling_05 --npz_path data/dataset_seg_2ch_224x224_nocrop_ratio_0_5.npz

echo "✅ [2/2] Selesai: $(date)"

echo ""
echo "=========================================================="
echo "🎉 Semua antrean eksperimen selesai: $(date)"
echo "=========================================================="