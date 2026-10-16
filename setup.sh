#!/bin/bash

# Hentikan skrip jika ada satu perintah yang gagal
set -e

echo "🚀 [1/4] Installing system CLI tools..."
apt-get update && apt-get install -y tmux tree libz3-dev

echo "📦 [2/4] Installing Mamba dependencies & pre-compiled wheels..."

pip install --no-deps "triton>=3.5.0" "tilelang==0.1.8" "quack-kernels>=0.3.4" "apache-tvm-ffi<=0.1.9"
pip install --no-deps https://github.com/Dao-AILab/causal-conv1d/releases/download/v1.7.0/causal_conv1d-1.7.0+cu12torch2.6cxx11abiTRUE-cp311-cp311-linux_x86_64.whl
pip install --no-deps https://github.com/state-spaces/mamba/releases/download/v2.3.2.post1/mamba_ssm-2.3.2.post1+cu12torch2.6cxx11abiTRUE-cp311-cp311-linux_x86_64.whl

echo "📚 [3/4] Installing Python dependencies & gdown..."
pip install einops huggingface_hub transformers pandas scikit-learn matplotlib seaborn gdown pandas peft

echo "📂 [4/4] Preparing Dataset & Pretrained Weights..."
mkdir -p data
mkdir -p weights

if [ ! -f "data/dataset_seg_2ch_224x224_nocrop_ratio_0_1.npz" ]; then
    echo "Downloading dataset Turkey 1-Channel 224x224 Segmentation Without Crop Ratio 0.1!"
    gdown 10c1E30NsSXV6WeJZ7Zs5M8MRo7hC14HH -O data/dataset_seg_2ch_224x224_nocrop_ratio_0_1.npz
else
    echo "✅ Dataset Turkey 1-Channel 224x224 Segmentation Without Crop Ratio 0.1 sudah ada, melewati proses unduh."
fi

# 2. Download Pretrained Weights
if [ ! -f "weights/classification_merged_best.pth" ]; then
    echo "Downloading Classification Merged DWT Weights..."
    gdown 1L0iEtgaLjb_huJ6HSNwsURM6WQsxn3Vw -O weights/classification_merged_best.pth
else
    echo "✅ Classification Merged DWT Weights sudah ada, melewati proses unduh."
fi

echo "================================================="
echo "🎉 Setup Selesai! Environment dan Data Siap."
echo "================================================="

