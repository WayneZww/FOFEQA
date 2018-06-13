#!/usr/bin/env bash
export LD_LIBRARY_PATH=/usr/local/lib:/mnt/lustre/share/nccl2/lib:/mnt/lustre/share/cuda-9.0/lib64:$LD_LIBRARY_PATH

nGPU=1
now=$(date +"%Y%m%d_%H%M%S")
data_root=/mnt/lustre/share/daixing/data/
name=length_5-sgd_b32

mkdir -p ./exps/${name}


python -u train_fofe.py --model_dir ./exps/${name} \
                --fix_embeddings \
                --tune_partial 0 \
                --batch_size 32 \
                --fofe_alpha 0.7 \
                --fofe_max_length 5 \
                --pos False --ner False \
                --optimizer sgd \
                2>&1|tee ./exps/${name}/train-${name}-$now.log &
