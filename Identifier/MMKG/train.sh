
# CL stage 1:
DATASET_NAME="FB15K_DB15K_ratio0.2_Name_easy"
BASE_MODEL="/home/VEA/Ranker/BAAI/bge-reranker-v2-m3"
OUTPUT_MODEL="FB15K_DB15K_ratio0.2_Name_CL_1"
BATCH_SIZE=2 
EPOCHS=1
LEARNING_RATE="3e-5"
MAX_LEN=2048

CUDA_VISIBLE_DEVICES=0 torchrun --nproc_per_node 1 --master_port 29500 \
    -m FlagEmbedding.finetune.reranker.encoder_only.base.__main__ \
    --output_dir "/data/Ranker/MMKG/${OUTPUT_MODEL}" \
    --model_name_or_path ${BASE_MODEL} \
    --train_data "../data/MMKG/Rerank_data/${DATASET_NAME}.json" \
    --learning_rate ${LEARNING_RATE} \
    --num_train_epochs ${EPOCHS} \
    --per_device_train_batch_size ${BATCH_SIZE} \
    --gradient_accumulation_steps 2 \
    --dataloader_drop_last True \
    --train_group_size 16 \
    --max_len ${MAX_LEN} \
    --warmup_ratio 0.1 \
    --weight_decay 0.01 \
    --logging_steps 10 \
    --save_strategy "no" \
    --save_total_limit 0 
    # --fp16

# CL stage 2:
DATASET_NAME="FB15K_DB15K_ratio0.2_Name_easy_complex"
BASE_MODEL="/data/Ranker/MMKG/FB15K_DB15K_ratio0.2_Name_CL_1"
OUTPUT_MODEL="FB15K_DB15K_ratio0.2_Name_CL_2"
BATCH_SIZE=2 
EPOCHS=1
LEARNING_RATE="8e-6"
MAX_LEN=2048


CUDA_VISIBLE_DEVICES=0 torchrun --nproc_per_node 1 --master_port 29500 \
    -m FlagEmbedding.finetune.reranker.encoder_only.base.__main__ \
    --output_dir "/data/Ranker/MMKG/${OUTPUT_MODEL}" \
    --model_name_or_path ${BASE_MODEL} \
    --train_data "../data/MMKG/Rerank_data/${DATASET_NAME}.json" \
    --learning_rate ${LEARNING_RATE} \
    --num_train_epochs ${EPOCHS} \
    --per_device_train_batch_size ${BATCH_SIZE} \
    --gradient_accumulation_steps 2 \
    --dataloader_drop_last True \
    --train_group_size 16 \
    --max_len ${MAX_LEN} \
    --warmup_ratio 0.1 \
    --weight_decay 0.01 \
    --logging_steps 10 \
    --save_strategy "no" \
    --save_total_limit 0 
    # --fp16