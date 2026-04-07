#!/bin/bash
#SBATCH --account=lp_h_ecoom_uhasselt
#SBATCH --cluster=wice
#SBATCH --partition=gpu_a100
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=16
#SBATCH --gpus-per-node=1
#SBATCH --mem=32G
#SBATCH --time=24:00:00
#SBATCH --job-name=ANNDL_train
#SBATCH --output=%x_%j.log

# ── All output goes to SCRATCH to avoid Home Quota issues ─────────────────────
SCRATCH_OUT="$VSC_SCRATCH/anndl_output"
mkdir -p "$SCRATCH_OUT"

# Redirect all stdout/stderr to scratch log
LOGFILE="$SCRATCH_OUT/anndl_train_${SLURM_JOB_ID}.log"
exec > >(tee "$LOGFILE") 2>&1

echo "========================================================================"
echo "SLURM_JOB_ID: $SLURM_JOB_ID"
echo "SLURM_NODELIST: $SLURM_NODELIST"
echo "Date: $(date)"
echo "Walltime: $SBATCH_TIMELIMIT"
echo "Log file: $LOGFILE"
echo "========================================================================"

echo "=== Job started at $(date) ==="
echo "Node: $(hostname)"
echo "GPU:  $(nvidia-smi --query-gpu=name,memory.total --format=csv,noheader)"

# ── Use miniconda environment ─────────────────────────────────────────────────
source ~/miniconda3/bin/activate
conda activate anndl

# ── Set Keras to use PyTorch backend ──────────────────────────────────────────
export KERAS_BACKEND=torch
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
export KERAS_HOME="$VSC_SCRATCH/.keras"      # pretrained weights go to Scratch, not Home

# ── Set data path (the notebook reads this) ───────────────────────────────────
export DATA_DIR="/vsc-hard-mounts/leuven-data/375/vsc37509/ANNDL/data/VOCtrainval_11-May-2012_2"

# ── Navigate to project ──────────────────────────────────────────────────────
cd "$HOME/ANNDL-PROJECT"

echo "Python: $(python --version)"
echo "PyTorch: $(python -c 'import torch; print(torch.__version__)')"
echo "CUDA available: $(python -c 'import torch; print(torch.cuda.is_available())')"
echo "Data directory: $DATA_DIR"
echo "Output directory: $SCRATCH_OUT"
echo "=== Starting notebook execution at $(date) ==="

# ── Execute notebook: output goes to SCRATCH ──────────────────────────────────
papermill \
    ANNDL2526_Project_Template_vsc.ipynb \
    "$SCRATCH_OUT/ANNDL2526_Project_EXECUTED.ipynb" \
    --log-output \
    --progress-bar \
    --request-save-on-cell-execute \
    --execution-timeout 7200

echo ""
echo "=== Job finished at $(date) ==="
echo "Executed notebook saved to: $SCRATCH_OUT/ANNDL2526_Project_EXECUTED.ipynb"
echo ""
echo "To download, run locally:"
echo "  scp vsc37509@login.hpc.kuleuven.be:$SCRATCH_OUT/ANNDL2526_Project_EXECUTED.ipynb ."
