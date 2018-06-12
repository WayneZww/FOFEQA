#!/usr/bin/env bash
name=test

python -u train_fofe.py --model_dir ./exps/${name} \
                --fofe_alpha 0.8 \
                --fofe_max_length 9 \
                --pos False --ner False \

