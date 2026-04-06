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
module load IPython/8.17.2-GCCcore-13.2.0
module load jupyter-server/2.14.2-GCCcore-13.2.0

# ── Set Keras to use PyTorch backend ──────────────────────────────────────────
export KERAS_BACKEND=torch
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

# ── Set data path (the notebook reads this) ───────────────────────────────────
export DATA_DIR="$VSC_DATA/ANNDL/data/VOCtrainval_11-May-2012_2"

# ── Install Keras 3 (user-level) ─────────────────────────────────────────────
pip install --user keras nbconvert --quiet 2>/dev/null

# ── Navigate to project ──────────────────────────────────────────────────────
cd "$HOME/ANNDL-PROJECT"

echo "=== Starting notebook execution at $(date) ==="
echo "Data directory: $DATA_DIR"

# ── Execute the notebook ─────────────────────────────────────────────────────
# This runs every cell top-to-bottom and saves the output into a new notebook
jupyter nbconvert \
    --to notebook \
    --execute \
    --ExecutePreprocessor.timeout=7200 \
    --ExecutePreprocessor.kernel_name=python3 \
    --output ANNDL2526_Project_EXECUTED.ipynb \
    ANNDL2526_Project_Template_vsc.ipynb

echo ""
echo "=== Job finished at $(date) ==="
echo "Executed notebook saved as: ANNDL2526_Project_EXECUTED.ipynb"
echo "Download it with:  scp vsc37509@login.hpc.kuleuven.be:~/ANNDL-PROJECT/ANNDL2526_Project_EXECUTED.ipynb ."
