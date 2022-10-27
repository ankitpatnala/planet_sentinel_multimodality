#!/usr/bin/env bash

#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --account=deepacf
#SBATCH --cpus-per-task=32
#SBATCH --output=lstm_outputs/run-out.%j
#SBATCH --error=lstm_outputs/run-err.%j
#SBATCH --time=01:00:00
#SBATCH --gres=gpu:1
#SBATCH --partition=develgpus

ml stages/2022
ml PyTorch
module unload typing-extensions/3.10.0.0
module unload Pillow-SIMD/9.0.1

source ./multimodality_cluster_env/bin/activate

export PYTHONPATH="${PYTHONPATH}:/p/project/deepacf/kiste/patnala1/planet_sentinel_multimodality"

module list

#export CUDA_VISIBLE_DEVICES="0,1,2,3"

export MASTER_PORT=12340
master_addr=$(scontrol show hostnames "$SLURM_JOB_NODELIST" | head -n 1)
export MASTER_ADDR=$master_addr

srun python /p/project/deepacf/kiste/patnala1/planet_sentinel_multimodality/models/lstm.py
