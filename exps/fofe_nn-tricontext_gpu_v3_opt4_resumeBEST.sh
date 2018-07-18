#!/usr/bin/env bash
work_dir="/local/scratch/watchara/Project_FOFE_QA/FOFEQA_SED"
now=$(date +"%Y%b%d_%Hh%Mm%Ss")
data_root="$work_dir/data/"
ver_n_opt="v3_opt4_resumeBEST"

gpu_id=0
epoch_num=100
batch_size=4
sample_num=0
neg_ratio=0
max_cand_len=16
fofe_alpha=0.7
#name=${ver_n_opt}_${now}__a${fofe_alpha}_mcl${max_cand_len}_sn${sample_num}_ctx5
name="v3_opt4_2018Jul17_18h48m46s__a0.7_mcl16_sn0_ctx5"
resume="best_model.pt"

models_n_logs_dir="$work_dir/models_n_logs/${name}"
mkdir -p ${models_n_logs_dir}

CUDA_VISIBLE_DEVICES=${gpu_id} \
python -u train_fofe.py --model_dir ${models_n_logs_dir} \
                --resume ${resume} \
                --test_only \
                --tune_partial 1000 \
                --epochs ${epoch_num} \
                --batch_size ${batch_size} \
                --sample_num ${sample_num} \
                --neg_ratio ${neg_ratio} \
                --max_len ${max_cand_len} \
                --fofe_alpha ${fofe_alpha} \
                --pos False \
                --ner False \
                --contexts_incl_cand True \
                --contexts_excl_cand True \
                --optimizer adamax
