#!/usr/bin/env bash
name=test

python -u train_fofe.py --model_dir ./exps/${name} \
                --fix_embeddings \
                --tune_partial 0 \
                --batch_size 2 \
                --fofe_alpha 0.7 \
                --fofe_max_length 9 \
                --pos False --ner False \

