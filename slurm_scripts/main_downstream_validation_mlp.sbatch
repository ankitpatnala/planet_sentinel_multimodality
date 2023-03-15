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
source /p/scratch/deepacf/kiste/patnala1/planet_sentinel_multimodality/environments/multimodality_env/bin/activate

export PYTHONPATH="${PYTHONPATH}:/p/project/deepacf/kiste/patnala1/planet_sentinel_multimodality"

module list

#export CUDA_VISIBLE_DEVICES="0,1,2,3"

export MASTER_PORT=12340
master_addr=$(scontrol show hostnames "$SLURM_JOB_NODELIST" | head -n 1)
export MASTER_ADDR=$master_addr

#for scarf in 20 60 
#do
#for((number=0;number<10;number++))
#do
#python /p/project/deepacf/kiste/patnala1/planet_sentinel_multimodality/main_downstream.py \
#		--method lstm \
#		--hidden_dims 32 64 128 256\
#		--num_layers 2 3 4 5 6\
#		--lr 1e-5 1e-3    \
#		--dropout 0.0 0.6 \
#		--baseline_hyper_param_file ../hyp_tune_lstm2.pkl \
#		--trial_number $number \
#		--is_normalize \
#		--pretrain_type sentinel2_mlp \
#		--self_supervised_loss simclr \
#		--temperature 0.07 \
#		--scarf $scarf \
#		--dataset train \
#		--self_supervised_ckpt sentinel_mlp/simclr/0.07/$scarf/epoch=99-step=703200.ckpt
#done
#done


#for scarf in 20 60 
#do
#for((number=0;number<10;number++))
#do
#python /p/project/deepacf/kiste/patnala1/planet_sentinel_multimodality/main_downstream.py \
#		--method inception \
#		--hidden_dims 128 256 512\
#		--num_layers 2 4 6 8\
#		--lr 1e-5 1e-3    \
#		--kernel_size 40 80 120 160 \
#		--baseline_hyper_param_file ../hyp_tune_inception2.pkl \
#		--trial_number $number \
#		--is_normalize \
#		--pretrain_type sentinel2_mlp \
#		--self_supervised_loss simclr \
#		--temperature 0.07 \
#		--scarf $scarf \
#		--dataset train \
#		--self_supervised_ckpt sentinel_mlp/simclr/0.07/$scarf/epoch=99-step=703200.ckpt
#
#done
#done
#

for scarf in 60
do
for number in  6 7 8 9
do
python /p/project/deepacf/kiste/patnala1/planet_sentinel_multimodality/main_downstream.py \
		--method transformer \
		--d_model 32 64 128  \
		--n_layers 2 3 4 5 6 \
		--n_head 2 4 8 \
		--lr 1e-5 1e-3    \
		--dropout 0.0 0.6 \
		--baseline_hyper_param_file ../hyp_tune_transformer2.pkl \
		--trial_number $number \
		--is_normalize \
		--pretrain_type mlp\
		--self_supervised_loss simclr \
		--temperature 0.07 \
		--scarf $scarf \
		--dataset train \
		--self_supervised_ckpt /p/scratch/deepacf/kiste/patnala1/planet_sentinel_multimodality/checkpoint_files/mlp/simclr/0.07/$scarf/epoch=99-step=703200.ckpt 
done
done
