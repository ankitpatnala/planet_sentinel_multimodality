#!/bin/bash

for((number=0;number<$1;number++))
do
 sbatch run_main_downstream.sbatch $number
done
