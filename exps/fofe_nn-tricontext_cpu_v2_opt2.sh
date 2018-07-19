#!/usr/bin/env bash
#work_dir="/local/scratch/watchara/Project_FOFE_QA/FOFEQA"
work_dir="$HOME/CSE/FOFE_QA/FOFEQA"
now=$(date +"%Y%b%d_%Hh%Mm%Ss")
data_root="$work_dir/data/"
ver_n_opt="v2_opt2"

batch_size=1
sample_num=8
neg_ratio=0.875
max_cand_len=16
fofe_alpha=0.8
name=${ver_n_opt}_${now}__a${fofe_alpha}_mcl${max_cand_len}_sn${sample_num}
models_n_logs_dir="$work_dir/models_n_logs/${name}"

mkdir -p ${models_n_logs_dir}

python -u train_fofe.py --model_dir ${models_n_logs_dir} \
                --tune_partial 1000 \
                --batch_size ${batch_size} \
                --sample_num ${sample_num} \
                --neg_ratio ${neg_ratio} \
                --max_len ${max_cand_len} \
                --fofe_alpha ${fofe_alpha} \
                --pos False \
                --ner False \
                --contexts_incl_cand True \
                --contexts_excl_cand True \
                --optimizer sgd