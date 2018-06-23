#!/usr/bin/env bash
export LD_LIBRARY_PATH=/usr/local/lib:/mnt/lustre/share/nccl2/lib:/mnt/lustre/share/cuda-9.0/lib64:$LD_LIBRARY_PATH

gpu_id=0
now=$(date +"%Y%m%d_%H%M%S")
fofe_alpha=0
fofe_len=3
name=l_${fofe_alpha}-a_${fofe_len}

mkdir -p ./exps/${name}

CUDA_VISIBLE_DEVICES=${gpu_id} \
python -u train_fofe.py --model_dir ./exps/${name} \
                --fix_embeddings \
                --tune_partial 0 \
                --batch_size 2 \
                --encoder fofe_biatt_selfatt_aspp \
                --planes 256 \
                --fofe_alpha ${fofe_alpha} \
                --fofe_max_length ${fofe_len} \
                --pos False --ner False \
                --optimizer adamax \
                2>&1|tee ./exps/${name}/train-${name}-$now.log
