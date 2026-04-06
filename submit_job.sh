#!/bin/bash
#SBATCH --account=lp_h_ecoom_uhasselt
#SBATCH --cluster=genius
#SBATCH --partition=gpu_v100
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=9
#SBATCH --gpus-per-node=1
#SBATCH --mem=32G
#SBATCH --time=6:00:00
#SBATCH --job-name=ANNDL_train
#SBATCH --output=anndl_train_%j.log

echo "=== Job started at $(date) ==="
echo "Node: $(hostname)"
echo "GPU:  $(nvidia-smi --query-gpu=name,memory.total --format=csv,noheader)"

# ── Load modules ──────────────────────────────────────────────────────────────
module purge
module load Python/3.11.5-GCCcore-13.2.0
module load PyTorch/2.1.2-foss-2023b-CUDA-12.1.1
module load scikit-learn/1.4.0-gfbf-2023b
module load Pillow/10.2.0-GCCcore-13.2.0
module load matplotlib/3.8.2-gfbf-2023b

# ── Set Keras to use PyTorch backend ──────────────────────────────────────────
export KERAS_BACKEND=torch
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

# ── Install Keras 3 if not available ──────────────────────────────────────────
pip install --user keras --quiet 2>/dev/null

# ── Set paths ─────────────────────────────────────────────────────────────────
PROJECT_DIR="$HOME/ANNDL-PROJECT"
OUTPUT_DIR="$VSC_SCRATCH/anndl_output"
mkdir -p "$OUTPUT_DIR"

cd "$PROJECT_DIR"

echo "=== Starting training at $(date) ==="
python train_vsc.py \
    --data-dir "$VSC_DATA/ANNDL/data/VOCtrainval_11-May-2012_2" \
    --output-dir "$OUTPUT_DIR"

echo "=== Job finished at $(date) ==="
echo "Output saved to: $OUTPUT_DIR"
