#!/usr/bin/env bash

gpu_id=1
now=$(date +"%Y%m%d_%H%M%S")
fofe_alpha=0.9
fofe_len=64
#name=a_${fofe_alpha}-l_${fofe_len}
name=baseliner8

mkdir -p ./exps/${name}

CUDA_VISIBLE_DEVICES=${gpu_id} \
python -u train_fofe.py --model_dir ./exps/${name} \
                --tune_partial 1000 \
	       	--batch_size 8 \
		--fofe_alpha ${fofe_alpha} \
                --fofe_max_length ${fofe_len} \
                --sample_num 32 \
		--neg_ratio 0.875 \
		--max_len 16 \
		--optimizer adamax \
                2>&1|tee ./exps/${name}/train-${name}-$now.log
