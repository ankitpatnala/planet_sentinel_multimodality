#!/bin/bash


for loss in simclr barlow_twins
do
 for pretrain_type in resmlp mlp
 do
  for temperature in 0.07 0.7 1.0
  do
   for scarf in 20 0
   do
    sbatch run_self_supervised.sbatch $pretrain_type $temperature $scarf $loss
   done
  done
 done
done
