#!/bin/bash

#SBATCH --account=stf218
#SBATCH --partition=batch-gpu
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=64
#SBATCH --gpus-per-node=8
#SBATCH --time=24:00:00
#SBATCH --job-name=train
#SBATCH --output=train_%A_%a.out
#SBATCH --array=0

# activate venv
source /lustre/polis/stf218/scratch/emin/defiantvenv/bin/activate

# set misc env vars
export TRITON_CACHE_DIR="/lustre/polis/stf218/scratch/emin/triton"
export PYTORCH_KERNEL_CACHE_PATH="/lustre/polis/stf218/scratch/emin/pytorch_kernel_cache"
export LOGLEVEL=INFO
export GPUS_PER_NODE=8
export HF_HOME="/lustre/polis/stf218/scratch/emin/huggingface"
export HF_DATASETS_CACHE="/lustre/polis/stf218/scratch/emin/huggingface"
export TRITON_CACHE_DIR="/lustre/polis/stf218/scratch/emin/triton"
export HF_HUB_OFFLINE=1

# set network
export MASTER_ADDR=$(scontrol show hostnames $SLURM_JOB_NODELIST | head -n 1)
export MASTER_PORT=3442

srun torchrun --nnodes $SLURM_NNODES --nproc_per_node $GPUS_PER_NODE --max_restarts 1 --node_rank $SLURM_NODEID --rdzv_id 101 --rdzv_backend c10d --rdzv_endpoint "$MASTER_ADDR:$MASTER_PORT" ./train.py

echo "Done"