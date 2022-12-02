#!/usr/bin/env bash

#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --account=deepacf
#SBATCH --cpus-per-task=32
#SBATCH --output=run-out.%j
#SBATCH --error=run-err.%j
#SBATCH --time=24:00:00
#SBATCH --gres=gpu:1
#SBATCH --partition=gpus

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

python /p/project/deepacf/kiste/patnala1/planet_sentinel_multimodality/main_downstream.py \
		--method transformer \
		--d_model 32 64 128  \
		--n_layers 2 3 4 5 6 \
		--n_head 2 4 8 \
		--lr 1e-5 1e-3    \
		--dropout 0.0 0.6 \
		--baseline_hyper_param_file ../hyp_tune_transformer2.pkl \
		--trial_number $1 \
		--pretrain_type mlp \
		--is_normalize \
		--self_supervised_loss barlow_twins \
		--temperature 1.0 \
		--scarf 20 \
                --self_supervised_ckpt mlp/barlow_twins/1.0/20/epoch=99-step=703200.ckpt

python /p/project/deepacf/kiste/patnala1/planet_sentinel_multimodality/main_downstream.py \
		--method lstm \
		--hidden_dims 32 64 128 256\
		--num_layers 2 3 4 5 6\
		--lr 1e-5 1e-3    \
		--dropout 0.0 0.6 \
		--baseline_hyper_param_file ../hyp_tune_lstm2.pkl \
		--trial_number $1 \
		--pretrain_type mlp \
		--is_normalize \
		--self_supervised_loss barlow_twins \
		--temperature 1.0 \
		--scarf 20 \
                --self_supervised_ckpt mlp/barlow_twins/1.0/20/epoch=99-step=703200.ckpt

python /p/project/deepacf/kiste/patnala1/planet_sentinel_multimodality/main_downstream.py \
		--method inception \
		--hidden_dims 128 256 512\
		--num_layers 2 4 6 8\
		--lr 1e-5 1e-3    \
		--kernel_size 40 80 120 160 \
		--baseline_hyper_param_file ../hyp_tune_inception2.pkl \
		--trial_number $1 \
		--pretrain_type mlp \
		--is_normalize \
		--self_supervised_loss barlow_twins \
		--temperature 1.0 \
		--scarf 20 \
                --self_supervised_ckpt mlp/barlow_twins/1.0/20/epoch=99-step=703200.ckpt

#temporal_transformer_1e-5/simclr/0.7/epoch=99-step=58600.ckpt
#temporal_transformer_1e-5/barlow_twins/1.0/epoch=99-step=58600.ckpt
#temporal_transformer_seasonal_1e-5/simclr/0.07/epoch=99-step=58600.ckpt

#mlp/barlow_twins/1.0/0/epoch=_1e-599-step=703200.ckpt
#temporal_transformer/barlow_twins/1.0/epoch=99-step=58600.ckpt
#temporal_transformer/simclr/0.07/epoch=99-step=58600.ckpt
#mlp/simclr/0.07/60/epoch=99-step=703200.ckpt
#resmlp/simclr/0.07/60/epoch=99-step=703200.ckpt
#resmlp/barlow_twins/1.0/60/epoch=99-step=703200.ckpt
#resmlp/simclr/0.07/0/epoch=99-step=703200.ckpt


#resmlp/barlow_twins/1.0/20/epoch=99-step=703200.ckpt
mlp/barlow_twins/1.0/20/epoch=99-step=703200.ckpt
