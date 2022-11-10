#!/usr/bin/env bash

#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --account=deepacf
#SBATCH --cpus-per-task=32
#SBATCH --output=run-out.%j
#SBATCH --error=run-err.%j
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

srun python /p/project/deepacf/kiste/patnala1/planet_sentinel_multimodality/main_self_supervised.py \
		--baseline_model_type transformer \
		--pretrain_type mlp \
		--temperature 0.7 \
		--baseline_hyper_param_file ../hyp_tune_transformer.pkl \
		--num_layers 2 4 6 8 \
		--hidden_dim 256 512 1024 \
		--lr 1e-5 1e-3  \
		--dropout 0.0 0.6 \
		--hyperparameter_tuning
		

		


