#!/bin/bash

#declare -a elems=(
#        "mlp barlow_twins 1.0 60  mlp/barlow_twins/1.0/60/epoch=99-step=703200.ckpt"
#        "mlp simclr 0.07 0 mlp/simclr/0.07/0/epoch=99-step=703200.ckpt"
#        "mlp simclr 0.07 20 mlp/simclr/0.07/20/epoch=99-step=703200.ckpt"
#        "mlp simclr 0.7 0 mlp/simclr/0.7/0/epoch=99-step=703200.ckpt"
#        "mlp simclr 0.7 20 mlp/simclr/0.7/20/epoch=99-step=703200.ckpt"
#        "mlp simclr 0.7 60 mlp/simclr/0.7/60/epoch=99-step=703200.ckpt"
#        "mlp simclr 1.0 0 mlp/simclr/1.0/0/epoch=99-step=703200.ckpt"
#        "mlp simclr 1.0 20 mlp/simclr/1.0/20/epoch=99-step=703200.ckpt"
#        "mlp simclr 1.0 60 mlp/simclr/1.0/60/epoch=99-step=703200.ckpt"
#        "resmlp barlow_twins 1.0 0 resmlp/barlow_twins/1.0/0/epoch=99-step=703200.ckpt"
#        "resmlp barlow_twins 1.0 60 resmlp/barlow_twins/1.0/60/epoch=99-step=703200.ckpt"
#        "resmlp simclr 0.07 20 resmlp/simclr/0.07/20/epoch=99-step=703200.ckpt"
#        "resmlp simclr 0.07 60 resmlp/simclr/0.07/60/epoch=99-step=703200.ckpt"
#        "resmlp simclr 0.7 0 resmlp/simclr/0.7/0/epoch=99-step=703200.ckpt"
#        "resmlp simclr 0.7 20 resmlp/simclr/0.7/20/epoch=99-step=703200.ckpt"
#        "resmlp simclr 0.7 60 resmlp/simclr/0.7/60/epoch=99-step=703200.ckpt"
#        "resmlp simclr 1.0 0 resmlp/simclr/1.0/0/epoch=99-step=703200.ckpt"
#        "resmlp simclr 1.0 20 resmlp/simclr/1.0/20/epoch=99-step=703200.ckpt"
#        "resmlp simclr 1.0 60 resmlp/simclr/1.0/60/epoch=99-step=703200.ckpt"
#        "temporal_transformer simclr 0.04 20 temporal_transformer/simclr/0.04/epoch=99-step=58600.ckpt"
#        "temporal_transformer simclr 0.07 20 temporal_transformer/simclr/0.07/epoch=99-step=58600.ckpt"
#        "temporal_transformer simclr 0.7 20 temporal_transformer/simclr/0.7/epoch=99-step=58600.ckpt"
#        "temporal_transformer simclr 1.0 20 temporal_transformer/simclr/1.0/epoch=99-step=58600.ckpt"
#	)
#
#
#declare -a elems=(
#        'temporal_transformer barlow_twins 1.0 20 temporal_transformer_seasonal/barlow_twins/1.0/epoch=99-step=58600.ckpt'  
#	'temporal_transformer simclr 0.07 20 temporal_transformer_seasonal/simclr/0.07/epoch=99-step=58600.ckpt'
#	'temporal_transformer simclr 0.04 20 temporal_transformer_seasonal/simclr/0.04/epoch=99-step=58600.ckpt'  
#   	'temporal_transformer simclr 0.7 20 temporal_transformer_seasonal/simclr/0.7/epoch=99-step=58600.ckpt'
#	'temporal_transformer simclr 1.0 20 temporal_transformer_seasonal/simclr/1.0/epoch=99-step=58600.ckpt'
#	)

#for((number=0;number<25;number++))
#do
#for elem in "${elems[@]}"; 
#do
#	read -a strarr <<< "$elem"
#	#sbatch run_main_downstream2.sbatch  $number ${strarr[0]} ${strarr[1]} ${strarr[2]} ${strarr[3]} ${strarr[4]}
#	echo "$number ${strarr[0]} ${strarr[1]} ${strarr[2]} ${strarr[3]} ${strarr[4]}" >> scripts2.txt
#done
#done

#declare -a elems=(
#	"22 mlp simclr 1.0 0 mlp/simclr/1.0/0/epoch=99-step=703200.ckpt"
#	"22 mlp simclr 1.0 20 mlp/simclr/1.0/20/epoch=99-step=703200.ckpt"
#	"22 mlp simclr 1.0 60 mlp/simclr/1.0/60/epoch=99-step=703200.ckpt" 
#	"22 resmlp barlow_twins 1.0 0 resmlp/barlow_twins/1.0/0/epoch=99-step=703200.ckpt"
#	"22 resmlp barlow_twins 1.0 60 resmlp/barlow_twins/1.0/60/epoch=99-step=703200.ckpt"
#	"22 resmlp simclr 0.07 20 resmlp/simclr/0.07/20/epoch=99-step=703200.ckpt"
#	"22 resmlp simclr 0.07 60 resmlp/simclr/0.07/60/epoch=99-step=703200.ckpt"
#	"22 resmlp simclr 0.7 0 resmlp/simclr/0.7/0/epoch=99-step=703200.ckpt"
#	"22 resmlp simclr 0.7 20 resmlp/simclr/0.7/20/epoch=99-step=703200.ckpt"
#	"22 resmlp simclr 0.7 60 resmlp/simclr/0.7/60/epoch=99-step=703200.ckpt"
#	"22 resmlp simclr 1.0 0 resmlp/simclr/1.0/0/epoch=99-step=703200.ckpt"
#	"22 resmlp simclr 1.0 20 resmlp/simclr/1.0/20/epoch=99-step=703200.ckpt"
#	"22 resmlp simclr 1.0 60 resmlp/simclr/1.0/60/epoch=99-step=703200.ckpt"
#	"22 temporal_transformer simclr 0.04 20 temporal_transformer/simclr/0.04/epoch=99-step=58600.ckpt"
#	"22 temporal_transformer simclr 0.07 20 temporal_transformer/simclr/0.07/epoch=99-step=58600.ckpt"
#	"22 temporal_transformer simclr 0.7 20 temporal_transformer/simclr/0.7/epoch=99-step=58600.ckpt"
#	"22 temporal_transformer simclr 1.0 20 temporal_transformer/simclr/1.0/epoch=99-step=58600.ckpt"
#	"23 mlp barlow_twins 1.0 60 mlp/barlow_twins/1.0/60/epoch=99-step=703200.ckpt"
#	"23 mlp simclr 0.07 0 mlp/simclr/0.07/0/epoch=99-step=703200.ckpt"
#	"23 mlp simclr 0.07 20 mlp/simclr/0.07/20/epoch=99-step=703200.ckpt"
#	"23 mlp simclr 0.7 0 mlp/simclr/0.7/0/epoch=99-step=703200.ckpt"
#	"23 mlp simclr 0.7 20 mlp/simclr/0.7/20/epoch=99-step=703200.ckpt"
#	"23 mlp simclr 0.7 60 mlp/simclr/0.7/60/epoch=99-step=703200.ckpt"
#	"23 mlp simclr 1.0 0 mlp/simclr/1.0/0/epoch=99-step=703200.ckpt"
#	"23 mlp simclr 1.0 20 mlp/simclr/1.0/20/epoch=99-step=703200.ckpt"
#	"23 mlp simclr 1.0 60 mlp/simclr/1.0/60/epoch=99-step=703200.ckpt"
#	"23 resmlp barlow_twins 1.0 0 resmlp/barlow_twins/1.0/0/epoch=99-step=703200.ckpt"
#	"23 resmlp barlow_twins 1.0 60 resmlp/barlow_twins/1.0/60/epoch=99-step=703200.ckpt"
#	"23 resmlp simclr 0.07 20 resmlp/simclr/0.07/20/epoch=99-step=703200.ckpt"
#	"23 resmlp simclr 0.07 60 resmlp/simclr/0.07/60/epoch=99-step=703200.ckpt"
#	"23 resmlp simclr 0.7 0 resmlp/simclr/0.7/0/epoch=99-step=703200.ckpt"
#	"23 resmlp simclr 0.7 20 resmlp/simclr/0.7/20/epoch=99-step=703200.ckpt"
#	"23 resmlp simclr 0.7 60 resmlp/simclr/0.7/60/epoch=99-step=703200.ckpt"
#	"23 resmlp simclr 1.0 0 resmlp/simclr/1.0/0/epoch=99-step=703200.ckpt"
#	"23 resmlp simclr 1.0 20 resmlp/simclr/1.0/20/epoch=99-step=703200.ckpt"
#	"23 resmlp simclr 1.0 60 resmlp/simclr/1.0/60/epoch=99-step=703200.ckpt"
#	"23 temporal_transformer simclr 0.04 20 temporal_transformer/simclr/0.04/epoch=99-step=58600.ckpt"
#	"23 temporal_transformer simclr 0.07 20 temporal_transformer/simclr/0.07/epoch=99-step=58600.ckpt"
#	"23 temporal_transformer simclr 0.7 20 temporal_transformer/simclr/0.7/epoch=99-step=58600.ckpt"
#	"23 temporal_transformer simclr 1.0 20 temporal_transformer/simclr/1.0/epoch=99-step=58600.ckpt"
#	"24 mlp barlow_twins 1.0 60 mlp/barlow_twins/1.0/60/epoch=99-step=703200.ckpt"
#	"24 mlp simclr 0.07 0 mlp/simclr/0.07/0/epoch=99-step=703200.ckpt"
#	"24 mlp simclr 0.07 20 mlp/simclr/0.07/20/epoch=99-step=703200.ckpt"
#	"24 mlp simclr 0.7 0 mlp/simclr/0.7/0/epoch=99-step=703200.ckpt"
#	"24 mlp simclr 0.7 20 mlp/simclr/0.7/20/epoch=99-step=703200.ckpt"
#	"24 mlp simclr 0.7 60 mlp/simclr/0.7/60/epoch=99-step=703200.ckpt"
#	"24 mlp simclr 1.0 0 mlp/simclr/1.0/0/epoch=99-step=703200.ckpt"
#	"24 mlp simclr 1.0 20 mlp/simclr/1.0/20/epoch=99-step=703200.ckpt"
#	"24 mlp simclr 1.0 60 mlp/simclr/1.0/60/epoch=99-step=703200.ckpt"
#	"24 resmlp barlow_twins 1.0 0 resmlp/barlow_twins/1.0/0/epoch=99-step=703200.ckpt"
#	"24 resmlp barlow_twins 1.0 60 resmlp/barlow_twins/1.0/60/epoch=99-step=703200.ckpt"
#	"24 resmlp simclr 0.07 20 resmlp/simclr/0.07/20/epoch=99-step=703200.ckpt"
#	"24 resmlp simclr 0.07 60 resmlp/simclr/0.07/60/epoch=99-step=703200.ckpt"
#	"24 resmlp simclr 0.7 0 resmlp/simclr/0.7/0/epoch=99-step=703200.ckpt"
#	"24 resmlp simclr 0.7 20 resmlp/simclr/0.7/20/epoch=99-step=703200.ckpt"
#	"24 resmlp simclr 0.7 60 resmlp/simclr/0.7/60/epoch=99-step=703200.ckpt"
#	"24 resmlp simclr 1.0 0 resmlp/simclr/1.0/0/epoch=99-step=703200.ckpt"
#	"24 resmlp simclr 1.0 20 resmlp/simclr/1.0/20/epoch=99-step=703200.ckpt"
#	"24 resmlp simclr 1.0 60 resmlp/simclr/1.0/60/epoch=99-step=703200.ckpt"
#	"24 temporal_transformer simclr 0.04 20 temporal_transformer/simclr/0.04/epoch=99-step=58600.ckpt"
#	"24 temporal_transformer simclr 0.07 20 temporal_transformer/simclr/0.07/epoch=99-step=58600.ckpt"
#	"24 temporal_transformer simclr 0.7 20 temporal_transformer/simclr/0.7/epoch=99-step=58600.ckpt"
#	"24 temporal_transformer simclr 1.0 20 temporal_transformer/simclr/1.0/epoch=99-step=58600.ckpt"
#	)


declare -a elems=(
"16 temporal_transformer simclr 0.04 20 temporal_transformer_seasonal/simclr/0.04/epoch=99-step=58600.ckpt"
"16 temporal_transformer simclr 0.7 20 temporal_transformer_seasonal/simclr/0.7/epoch=99-step=58600.ckpt"
"16 temporal_transformer simclr 1.0 20 temporal_transformer_seasonal/simclr/1.0/epoch=99-step=58600.ckpt"
"17 temporal_transformer barlow_twins 1.0 20 temporal_transformer_seasonal/barlow_twins/1.0/epoch=99-step=58600.ckpt"
"17 temporal_transformer simclr 0.07 20 temporal_transformer_seasonal/simclr/0.07/epoch=99-step=58600.ckpt"
"17 temporal_transformer simclr 0.04 20 temporal_transformer_seasonal/simclr/0.04/epoch=99-step=58600.ckpt"
"17 temporal_transformer simclr 0.7 20 temporal_transformer_seasonal/simclr/0.7/epoch=99-step=58600.ckpt"
"17 temporal_transformer simclr 1.0 20 temporal_transformer_seasonal/simclr/1.0/epoch=99-step=58600.ckpt"
"18 temporal_transformer barlow_twins 1.0 20 temporal_transformer_seasonal/barlow_twins/1.0/epoch=99-step=58600.ckpt"
"18 temporal_transformer simclr 0.07 20 temporal_transformer_seasonal/simclr/0.07/epoch=99-step=58600.ckpt"
"18 temporal_transformer simclr 0.04 20 temporal_transformer_seasonal/simclr/0.04/epoch=99-step=58600.ckpt"
"18 temporal_transformer simclr 0.7 20 temporal_transformer_seasonal/simclr/0.7/epoch=99-step=58600.ckpt"
"18 temporal_transformer simclr 1.0 20 temporal_transformer_seasonal/simclr/1.0/epoch=99-step=58600.ckpt"
"19 temporal_transformer barlow_twins 1.0 20 temporal_transformer_seasonal/barlow_twins/1.0/epoch=99-step=58600.ckpt"
"19 temporal_transformer simclr 0.07 20 temporal_transformer_seasonal/simclr/0.07/epoch=99-step=58600.ckpt"
"19 temporal_transformer simclr 0.04 20 temporal_transformer_seasonal/simclr/0.04/epoch=99-step=58600.ckpt"
"19 temporal_transformer simclr 0.7 20 temporal_transformer_seasonal/simclr/0.7/epoch=99-step=58600.ckpt"
"19 temporal_transformer simclr 1.0 20 temporal_transformer_seasonal/simclr/1.0/epoch=99-step=58600.ckpt"
"20 temporal_transformer barlow_twins 1.0 20 temporal_transformer_seasonal/barlow_twins/1.0/epoch=99-step=58600.ckpt"
"20 temporal_transformer simclr 0.07 20 temporal_transformer_seasonal/simclr/0.07/epoch=99-step=58600.ckpt"
"20 temporal_transformer simclr 0.04 20 temporal_transformer_seasonal/simclr/0.04/epoch=99-step=58600.ckpt"
"20 temporal_transformer simclr 0.7 20 temporal_transformer_seasonal/simclr/0.7/epoch=99-step=58600.ckpt"
"20 temporal_transformer simclr 1.0 20 temporal_transformer_seasonal/simclr/1.0/epoch=99-step=58600.ckpt"
"21 temporal_transformer barlow_twins 1.0 20 temporal_transformer_seasonal/barlow_twins/1.0/epoch=99-step=58600.ckpt"
"21 temporal_transformer simclr 0.07 20 temporal_transformer_seasonal/simclr/0.07/epoch=99-step=58600.ckpt"
"21 temporal_transformer simclr 0.04 20 temporal_transformer_seasonal/simclr/0.04/epoch=99-step=58600.ckpt"
"21 temporal_transformer simclr 0.7 20 temporal_transformer_seasonal/simclr/0.7/epoch=99-step=58600.ckpt"
"21 temporal_transformer simclr 1.0 20 temporal_transformer_seasonal/simclr/1.0/epoch=99-step=58600.ckpt"
"22 temporal_transformer barlow_twins 1.0 20 temporal_transformer_seasonal/barlow_twins/1.0/epoch=99-step=58600.ckpt"
"22 temporal_transformer simclr 0.07 20 temporal_transformer_seasonal/simclr/0.07/epoch=99-step=58600.ckpt"
"22 temporal_transformer simclr 0.04 20 temporal_transformer_seasonal/simclr/0.04/epoch=99-step=58600.ckpt"
"22 temporal_transformer simclr 0.7 20 temporal_transformer_seasonal/simclr/0.7/epoch=99-step=58600.ckpt"
"22 temporal_transformer simclr 1.0 20 temporal_transformer_seasonal/simclr/1.0/epoch=99-step=58600.ckpt"
"23 temporal_transformer barlow_twins 1.0 20 temporal_transformer_seasonal/barlow_twins/1.0/epoch=99-step=58600.ckpt"
"23 temporal_transformer simclr 0.07 20 temporal_transformer_seasonal/simclr/0.07/epoch=99-step=58600.ckpt"
"23 temporal_transformer simclr 0.04 20 temporal_transformer_seasonal/simclr/0.04/epoch=99-step=58600.ckpt"
"23 temporal_transformer simclr 0.7 20 temporal_transformer_seasonal/simclr/0.7/epoch=99-step=58600.ckpt"
"23 temporal_transformer simclr 1.0 20 temporal_transformer_seasonal/simclr/1.0/epoch=99-step=58600.ckpt"
"24 temporal_transformer barlow_twins 1.0 20 temporal_transformer_seasonal/barlow_twins/1.0/epoch=99-step=58600.ckpt"
"24 temporal_transformer simclr 0.07 20 temporal_transformer_seasonal/simclr/0.07/epoch=99-step=58600.ckpt"
"24 temporal_transformer simclr 0.04 20 temporal_transformer_seasonal/simclr/0.04/epoch=99-step=58600.ckpt"
"24 temporal_transformer simclr 0.7 20 temporal_transformer_seasonal/simclr/0.7/epoch=99-step=58600.ckpt"
"24 temporal_transformer simclr 1.0 20 temporal_transformer_seasonal/simclr/1.0/epoch=99-step=58600.ckpt"
)

for elem in "${elems[@]}";
do
read -a strarr <<< "$elem"
	sbatch run_main_downstream2.sbatch ${strarr[0]} ${strarr[1]} ${strarr[2]} ${strarr[3]} ${strarr[4]}  ${strarr[5]}
done	
