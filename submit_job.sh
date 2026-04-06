#!/bin/bash
#SBATCH --account=lp_h_ecoom_uhasselt
#SBATCH --cluster=genius
#SBATCH --partition=gpu_v100
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --gpus-per-node=1
#SBATCH --mem=32G
#SBATCH --time=6:00:00
#SBATCH --job-name=ANNDL_train
#SBATCH --output=anndl_train_%j.log

echo "=== Job started at $(date) ==="
echo "Node: $(hostname)"
echo "GPU:  $(nvidia-smi --query-gpu=name,memory.total --format=csv,noheader)"

# ── Use miniconda environment ─────────────────────────────────────────────────
source ~/miniconda3/bin/activate
conda activate anndl

# ── Set Keras to use PyTorch backend ──────────────────────────────────────────
export KERAS_BACKEND=torch
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

# ── Set data path (the notebook reads this) ───────────────────────────────────
export DATA_DIR="$VSC_DATA/ANNDL/data/VOCtrainval_11-May-2012_2"

# ── Navigate to project ──────────────────────────────────────────────────────
cd "$HOME/ANNDL-PROJECT"

echo "Python: $(python --version)"
echo "PyTorch: $(python -c 'import torch; print(torch.__version__)')"
echo "CUDA available: $(python -c 'import torch; print(torch.cuda.is_available())')"
echo "Data directory: $DATA_DIR"
echo "=== Starting notebook execution at $(date) ==="

# ── Execute the notebook ─────────────────────────────────────────────────────
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
