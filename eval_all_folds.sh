#!/bin/bash
CONFIG="config/sleepedfx.py"
GPU=0

for fold in 0 1 2 3 4 5 6 7 8 9; do
    echo "===== Evaluating Fold ${fold} ====="
    python predict.py \
        --config_file ${CONFIG} \
        --model_dir ./output/bilstm_attn_tune_fold${fold} \
        --fold_idx ${fold} \
        --output_dir ./output/bilstm_attn_tune_fold${fold}/predict \
        --log_file ./output/bilstm_attn_tune_fold${fold}/predict.log \
        --use-best \
        --gpu ${GPU}
done