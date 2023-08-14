#!/usr/bin/env bash

#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --account=deepacf
#SBATCH --cpus-per-task=32
#SBATCH --output=run-out.%j
#SBATCH --error=run-err.%j
#SBATCH --time=24:00:00
#SBATCH --gres=gpu:1
#SBATCH --partition=batch


source /p/scratch/deepacf/kiste/patnala1/planet_sentinel_multimodality/environments/modules.sh
#source /p/scratch/deepacf/kiste/patnala1/planet_sentinel_multimodality/environments/multimodality_env2/bin/activate
source multimodality_env2/bin/activate

export PYTHONPATH="${PYTHONPATH}:/p/project/deepacf/kiste/patnala1/planet_sentinel_multimodality"

module list

pip list

#export CUDA_VISIBLE_DEVICES="0,1,2,3"

export MASTER_PORT=12340
master_addr=$(scontrol show hostnames "$SLURM_JOB_NODELIST" | head -n 1)
export MASTER_ADDR=$master_addr


for scarf in 20 60 

do

srun python /p/project/deepacf/kiste/patnala1/planet_sentinel_multimodality/main_sentinel2_temporal_self_supervised.py \
  	--baseline_model_type transformer \
  	--pretrain_type temporal_transformer \
	--loss simclr \
  	--temperature 0.07 \
  	--d_model 64 \
  	--n_head 4 \
  	--num_layer 4  \
  	--mlp_dim 128 \
  	--lr 1e-3  1e-3 \
	--scarf $scarf \
        --projector_layer 2 	


done 
