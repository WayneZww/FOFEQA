#!/usr/bin/env bash
export LD_LIBRARY_PATH=/usr/local/lib:/mnt/lustre/share/nccl2/lib:/mnt/lustre/share/cuda-9.0/lib64:$LD_LIBRARY_PATH

gpu_id=2
now=$(date +"%Y%m%d_%H%M%S")
fofe_alpha=0.9
fofe_len=64
#name=a_${fofe_alpha}-l_${fofe_len}
name=baseliner0.125

mkdir -p ./exps/${name}

CUDA_VISIBLE_DEVICES=${gpu_id} \
python -u train_fofe.py --model_dir ./exps/${name} \
                --tune_partial 1000 \
                --batch_size 16 \
                --fofe_alpha ${fofe_alpha} \
                --fofe_max_length ${fofe_len} \
                --sample_num 1024 \
		--neg_ratio 0.125 \
		--max_len 16 \
		--optimizer adamax \
                2>&1|tee ./exps/${name}/train-${name}-$now.log
