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



export MASTER_PORT=12340
master_addr=$(scontrol show hostnames "$SLURM_JOB_NODELIST" | head -n 1)
export MASTER_ADDR=$master_addr

export CUDA_VISIBLE_DEVICES="0"

#srun python /p/project/deepacf/kiste/patnala1/planet_sentinel_multimodality/main_temporal_self_supervised.py \
#		--baseline_model_type transformer \
#		--pretrain_type temporal_transformer \
#		--temperature 0.07 \
#		--hyperparameter_tuning \
#		--d_model 32 64 128 \
#		--n_head 4 8 \
#		--num_layer 2 4 6 8 \
#		--mlp_dim 64 128 \
#		--lr 1e-5 1e-3 \
#		--baseline_hyper_param_file ../hyp_tune_transformer.pkl \
#		--projector_layer 0 \ 
#		--is_mixup


python /p/project/deepacf/kiste/patnala1/planet_sentinel_multimodality/main_temporal_self_supervised.py \
  	--baseline_model_type transformer \
  	--pretrain_type temporal_transformer \
	--loss $1 \
  	--temperature $2 \
  	--d_model 64 \
  	--n_head 4 \
  	--num_layer 4  \
  	--mlp_dim 128 \
  	--lr 1e-3  1e-3 \
	--is_seasonal \
  	--baseline_hyper_param_file ../hyp_tune_transformer.pkl \
  	--projector_layer 2  



#-------------------------------
#1e-5
# simclr 1.0
# simclr 1.0 seasonal
# barlow_twins 1.0 
# barlow_twins 1.0 seasonal 
# simclr 0.7
# simclr 0.7 seasonal
# barlow_twins 0.7
# barlow_twins 0.7 seasonal


# simclr 0.07
# simclr 0.07 seasonal
# barlow_twins 0.07
# barlow_twins 0.07 seasonal



#------------------------------

#----------------------------
#1e-3
#barlow_twins 1.0

#simclr 0.04 
#simclr 0.07
#simclr 0.7
#simclr 1.0

#barlow_twins 1.0 seasonal

#simclr 0.04 seasonal
#simclr 0.07 seasonal
#simclr 0.7 seasonal
#simclr 1.0 seasonal
