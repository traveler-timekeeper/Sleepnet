#!/bin/bash
CONFIG="config/sleepedfx.py"
GPU=0

for fold in 0 1 2 3 4 5 6 7 8 9; do
    echo "===== Evaluating Fold ${fold} ====="
    python eval_single_fold.py \
        --config_file ${CONFIG} \
        --model_dir ./output/bilstm_attn_tune_fold${fold} \
        --fold_idx ${fold} \
        --output_dir ./output/bilstm_attn_tune_fold${fold}/eval \
        --gpu ${GPU}
done