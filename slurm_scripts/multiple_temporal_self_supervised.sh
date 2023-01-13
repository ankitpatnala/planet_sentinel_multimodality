#!/bin/bash

for temperature in 0.04 0.07 0.7 1.0
 do
  sbatch run_temporal_self_supervised.sbatch simclr $temperature
 done

sbatch run_temporal_self_supervised.sbatch barlow_twins 1.0
