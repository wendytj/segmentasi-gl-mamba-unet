#!/bin/bash
set -e

echo "=========================================================="
echo "🚀 Memulai Antrean Eksperimen Ablasi Data Scaling 1.0 - 1.5"
echo "$(date)"
echo "=========================================================="

# https://drive.google.com/file/d/1xhiEWlZbf9e_mmBJxSKzxJO8WE2wlCur/view?usp=drive_link
# https://drive.google.com/file/d/1oe58FWxkz-8mbMtRJ_ouD0z1H2QxbNv1/view?usp=drive_link


gdown 1xhiEWlZbf9e_mmBJxSKzxJO8WE2wlCur -O data/dataset_seg_2ch_224x224_nocrop_ratio_1_0.npz
gdown 1oe58FWxkz-8mbMtRJ_ouD0z1H2QxbNv1 -O data/dataset_seg_2ch_224x224_nocrop_ratio_1_5.npz

echo ""
echo "▶️ [1/2] Menjalankan: Rasio 1.0"

python train.py --experiment_code scaling_10 --npz_path data/dataset_seg_2ch_224x224_nocrop_ratio_1_0.npz

echo "✅ [1/2] Selesai: $(date)"
sleep 10

echo ""
echo "▶️ [2/2] Menjalankan: Rasio 1.5"

python train.py --experiment_code scaling_15 --npz_path data/dataset_seg_2ch_224x224_nocrop_ratio_1_5.npz

echo "✅ [2/2] Selesai: $(date)"
sleep 10

echo ""
echo "=========================================================="
echo "🎉 Semua antrean eksperimen selesai: $(date)"
echo "=========================================================="