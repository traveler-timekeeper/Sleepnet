#!/bin/bash
# 10折交叉验证循环（跳过已完成的 fold 4）

CONFIG_FILE="./config/sleepedfx.py"
BASE_OUTPUT="./output/bilstm_attn_tune_fold"

for fold in 0 1 2 3 5 6 7 8 9; do
    echo "===== Starting Fold ${fold} ====="
    python train.py \
        --config_file ${CONFIG_FILE} \
        --fold_idx ${fold} \
        --output_dir ${BASE_OUTPUT}${fold} \
        --restart
    echo "===== Fold ${fold} Finished ====="
done