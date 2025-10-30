
# # # CL stage 1 fr-en
DATASET_NAME="fr_en_easy_kv_on"
BASE_MODEL="Reranker/BAAI/bge-reranker-v2-m3"
OUTPUT_MODEL="fr_en_1014_CL_on_1"
BATCH_SIZE=2
EPOCHS=1
LEARNING_RATE="2e-5"
MAX_LEN=1024

CUDA_VISIBLE_DEVICES=0 torchrun --nproc_per_node 1 --master_port 29500 \
    -m FlagEmbedding.finetune.reranker.encoder_only.base.__main__ \
    --output_dir "Reranker/${OUTPUT_MODEL}" \
    --model_name_or_path ${BASE_MODEL} \
    --train_data "reranker_data/${DATASET_NAME}.json" \
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


# # # CL stage 2: 2025-10-14 17:43:06
DATASET_NAME="fr_en_easy_complex_kv_on"
BASE_MODEL="Reranker/fr_en_1014_CL_on_1"
OUTPUT_MODEL="fr_en_1014_CL_on_2"
BATCH_SIZE=2
EPOCHS=1
LEARNING_RATE="5e-6"
MAX_LEN=1024

CUDA_VISIBLE_DEVICES=0 torchrun --nproc_per_node 1 --master_port 29500 \
    -m FlagEmbedding.finetune.reranker.encoder_only.base.__main__ \
    --output_dir "Reranker/${OUTPUT_MODEL}" \
    --model_name_or_path ${BASE_MODEL} \
    --train_data "reranker_data/${DATASET_NAME}.json" \
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
