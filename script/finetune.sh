export PROXY_TASK_NAME=sts-b
export CUDA_VISIBLE_DEVICES=0

python ../code/finetune.py \
    --model_type xlmroberta \
    --model_name_or_path xlm-roberta-large \
    --task_name $PROXY_TASK_NAME \
    --do_train \
    --do_eval \
    --evaluate_during_training  \
    --logging_steps=5000 \
    --save_steps=20000   \
    --examples_cache_dir ../cache \
    --max_seq_length 256 \
    --per_gpu_eval_batch_size=64   \
    --per_gpu_train_batch_size=16   \
    --learning_rate 2e-6 \
    --warmup_steps 1000 \
    --num_train_epochs 2.0 \
    --gradient_accumulation_steps=1 \
    --training_examples_subsample 0.1  \
    --negative_random_sampling 8 \
    --fuzzy_ratio=2 \
    --fuzzy_max_score=60 \
    --positive_oversampling=1   \
    --two_way_neighbour_sampling    \
    --output_dir ../outputs/$PROXY_TASK_NAME   \
    --valid_src_data_dir  /VAILD_SRC_DATA_PATH/valid.ps  \
    --valid_trg_data_dir  /VAILD_TRG_DATA_PATH/valid.en  \
    --train_src_data_dirs /TRAIN_SRC_DATA_PATH/train.ps \
    --train_trg_data_dirs /TRAIN_TRG_DATA_PATH/train.en \