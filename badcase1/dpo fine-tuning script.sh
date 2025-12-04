#!/bin/bash

##########################################################

# 模型相关参数
export MODEL_NAME="Qwen2-72B-Instruct/v2.0"
export MODEL_BASE_PATH="/data/sft/merge_lora/b16s1/saves"
export SCENE_CODE="b16s1"
export WANDB_PROJECT="${SCENE_CODE}_dpo"
export WANDB_LOG_MODEL="${MODEL_NAME}"
export CUTLASS_PATH="/data/llm/cutlass-3.5.0"
export LORA_TARGET="all"
# export RESUME_FROM_CHECKPOINT=true


# 数据集相关参数
export DATASET_DIR="../../data"
export MAX_SAMPLES=10000000000000
export VAL_SIZE=0.05
export CUTOFF_LEN=4096
# export RESIZE_VOCAB="true"
unset RESIZE_VOCAB

# 训练相关参数
export ADAM_BETA2=0.95

export DATASET="generalized_dpo_preference_dataset"

export PER_DEVICE_TRAIN_BATCH_SIZE=1
export PER_DEVICE_EVAL_BATCH_SIZE=1
export GRADIENT_ACCUMULATION_STEPS=1

export VERSION="v7.0"
export LEARNING_RATE=1e-4    # 学习率
export NUM_TRAIN_EPOCHS=1    # 训练轮数
export WARMUP_RATIO=0.05     # 预热比例
export WEIGHT_DECAY=0.05     # 权重衰减
export LORA_RANK=8           # LoRA秩
export LORA_ALPHA=16         # LoRA缩放因子
export LORA_DROPOUT=0.05     # LoRA dropout比例
unset RESUME_FROM_CHECKPOINT



export OUTPUT_DIR="/data/dpo/lora/${SCENE_CODE}/saves/${MODEL_NAME}/${VERSION}"
export DS_CONFIG=${DS_CONFIG:-"../../deepspeed/qwen_ds_config_zero3.json"}


mkdir -p ${OUTPUT_DIR}

echo RESUME_FROM_CHECKPOINT: ${RESUME_FROM_CHECKPOINT}

start_time=$(date +%s)
deepspeed -i localhost:0,1,2,3,4,5,6,7 ../../../src/llamafactory/launcher.py \
    --deepspeed ${DS_CONFIG} \
    --stage dpo \
    --pref_beta 0.1 \
    --pref_loss sigmoid \
    --do_train \
    --model_name_or_path ${MODEL_BASE_PATH}/${MODEL_NAME} \
    --dataset ${DATASET} \
    --dataset_dir ${DATASET_DIR} \
    --template qwen \
    --finetuning_type lora \
    --lora_target ${LORA_TARGET:=all} \
    --output_dir ${OUTPUT_DIR} \
    --overwrite_cache \
    --overwrite_output_dir \
    --cutoff_len ${CUTOFF_LEN}  \
    --preprocessing_num_workers 180 \
    --per_device_train_batch_size ${PER_DEVICE_TRAIN_BATCH_SIZE} \
    --per_device_eval_batch_size ${PER_DEVICE_EVAL_BATCH_SIZE} \
    --gradient_accumulation_steps ${GRADIENT_ACCUMULATION_STEPS} \
    --lr_scheduler_type cosine \
    --logging_steps 2 \
    --save_steps 500 \
    $([ "${RESUME_FROM_CHECKPOINT}" != "" ] && echo "--resume_from_checkpoint ${RESUME_FROM_CHECKPOINT}") \
    $([ "${RESIZE_VOCAB}" != "" ] && echo "--resize_vocab --additional_target embed_tokens,lm_head") \
    --eval_steps 10 \
    --eval_strategy steps \
    --learning_rate ${LEARNING_RATE} \
    --num_train_epochs ${NUM_TRAIN_EPOCHS} \
    --max_samples ${MAX_SAMPLES} \
    --val_size ${VAL_SIZE} \
    --ddp_timeout 180000000 \
    --plot_loss \
    --report_to wandb \
    --run_name "${MODEL_NAME//\//_}_${VERSION}_${start_time}" \
    --logging_dir /data/wandb/logs \
    --lora_rank ${LORA_RANK} \
    --lora_alpha ${LORA_ALPHA} \
    --lora_dropout ${LORA_DROPOUT} \
    --ddp_find_unused_parameters false \
    --warmup_ratio ${WARMUP_RATIO} \
    --weight_decay ${WEIGHT_DECAY} \
    --adam_beta2 ${ADAM_BETA2} \
    --use_fast_tokenizer \
    --bf16
end_time=$(date +%s)
# 计算脚本耗时(秒)
elapsed_seconds=$((end_time - start_time))
# 计算小时数
elapsed_hours=$((elapsed_seconds / 3600))
# 计算剩余的分钟数
remaining_minutes=$((elapsed_seconds % 3600 / 60))
# 计算剩余的秒数
remaining_seconds=$((elapsed_seconds % 60))
# 输出统计信息
echo "开始时间: $(date -d @$start_time)"
echo "结束时间: $(date -d @$end_time)"
echo "总耗时: $elapsed_hours 小时 $remaining_minutes 分 $remaining_seconds 秒"

cat >${OUTPUT_DIR}/time_consuming_statistics.txt<<EOF
"开始时间: $(date -d @$start_time)"
"结束时间: $(date -d @$end_time)"
"总耗时: $elapsed_hours 小时 $remaining_minutes 分 $remaining_seconds 秒"
EOF
