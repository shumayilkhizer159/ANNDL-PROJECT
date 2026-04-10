#!/bin/bash
#SBATCH --account=lp_h_ecoom_uhasselt
#SBATCH --cluster=wice
#SBATCH --partition=gpu_a100
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=16
#SBATCH --gpus-per-node=1
#SBATCH --mem=32G
#SBATCH --time=12:00:00
#SBATCH --job-name=ANNDL_shir
#SBATCH --output=%x_%j.log

# ── Shireen's Output Directory ──────────────────────────────────────────────
SCRATCH_OUT="$VSC_SCRATCH/anndl_output_shir"
mkdir -p "$SCRATCH_OUT"

# Log stdout/stderr
LOGFILE="$SCRATCH_OUT/shir_train_${SLURM_JOB_ID}.log"
exec > >(tee "$LOGFILE") 2>&1

echo "=== Shireen's Job: MobileNetV2 Approach ==="
echo "SLURM_JOB_ID: $SLURM_JOB_ID"
echo "Date: $(date)"

# ── Environment ──────────────────────────────────────────────────────────────
source ~/miniconda3/bin/activate
conda activate anndl
export KERAS_BACKEND=torch
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
export KERAS_HOME="$VSC_SCRATCH/.keras_shir" 

# ── Data Cache ───────────────────────────────────────────────────────────────
echo "=== Copying dataset to local node SSD ==="
rsync -a "/vsc-hard-mounts/leuven-data/375/vsc37509/ANNDL/data/VOCtrainval_11-May-2012_2" "$TMPDIR/"
export DATA_DIR="$TMPDIR/VOCtrainval_11-May-2012_2"

# ── Execution ────────────────────────────────────────────────────────────────
cd "$HOME/ANNDL-PROJECT"

echo "=== Starting notebook execution at $(date) ==="
papermill \
    ANNDL2526_Project_Template_vsc_shir.ipynb \
    "$SCRATCH_OUT/ANNDL2526_Project_EXECUTED_shir.ipynb" \
    --log-output \
    --progress-bar \
    --request-save-on-cell-execute \
    --execution-timeout 36000

echo "=== Job finished at $(date) ==="
echo "Executed notebook: $SCRATCH_OUT/ANNDL2526_Project_EXECUTED_shir.ipynb"
