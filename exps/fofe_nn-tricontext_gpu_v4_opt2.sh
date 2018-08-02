#!/usr/bin/env bash
work_dir="/local/scratch/watchara/Project_FOFE_QA/FOFEQA_SED"
now=$(date +"%Y%b%d_%Hh%Mm%Ss")
data_dir="/local/scratch/FOFEQA/data/SQuAD"
ver_n_opt="v4_opt2"

gpu_id=3
epoch_num=100
batch_size=4
sample_num=0
neg_ratio=0
hidden_size=512
learning_rate=0.1
max_cand_len=16
fofe_alpha="0.5,0.9"
ctx_incl_cand=True
ctx_excl_cand=True
n_ctx_types=1
if [ ${ctx_incl_cand} ]; then
    ((n_ctx_types+=2))
fi
if [ ${ctx_excl_cand} ]; then
    ((n_ctx_types+=2))
fi

name=${ver_n_opt}_${now}__a${fofe_alpha}_mcl${max_cand_len}_sn${sample_num}_ctx${n_ctx_types}

models_n_logs_dir="$work_dir/models_n_logs/${name}"
mkdir -p ${models_n_logs_dir}

CUDA_VISIBLE_DEVICES=${gpu_id} \
python -u train_fofe.py --model_dir ${models_n_logs_dir} \
                --data_file ${data_dir}/data-test.msgpack \
                --meta_file ${data_dir}/meta-test.msgpack \
                --test_train \
                --tune_partial 0 \
                --epochs ${epoch_num} \
                --batch_size ${batch_size} \
                --sample_num ${sample_num} \
                --neg_ratio ${neg_ratio} \
                --hidden_size ${hidden_size} \
                --max_len ${max_cand_len} \
                --fofe_alpha ${fofe_alpha} \
                --pos True \
                --ner True \
                --contexts_incl_cand ${ctx_incl_cand} \
                --contexts_excl_cand ${ctx_excl_cand} \
                --optimizer sgd \
                --learning_rate ${learning_rate}

