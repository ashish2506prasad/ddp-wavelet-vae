#!/bin/bash

#SBATCH --job-name=training-wavelet-vae
#SBATCH --error=slurm_logs/yoloe.%J.err
#SBATCH --output=yoloe-training-%j.log
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gres=gpu:1
#SBATCH --partition=
#SBATCH --qos=dgx
#SBATCH --cpus-per-task=8
#SBATCH --mem=24G
#SBATCH --time=50:00:00     # Reduced to 5 hours for testing

export OMP_NUM_THREADS=$SLURM_CPUS_PER_TASK

echo "========================================================"
echo "SLURM JOB: $SLURM_JOB_ID"
echo "Running on node: $(hostname)"
echo "Allocated GPU: $CUDA_VISIBLE_DEVICES"
echo "CPUs allocated: $SLURM_CPUS_PER_TASK"
echo "========================================================"

echo ">>> Starting YOLOE training script..."

# Path to your fine-tune script
SCRIPT_PATH="/home/agipml/21d070064/research/models/yoloe/yoloe.py"
WEIGHTS_PATH="/home/agipml/21d070064/research/models/pretrained_weights/yoloe-11l-seg.pt" 

python $SCRIPT_PATH --weights $WEIGHTS_PATH  > yoloe.out 2>&1

echo ">>> YOLOE training script finished."