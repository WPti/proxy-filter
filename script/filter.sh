export CUDA_VISIBLE_DEVICES=0

python ../code/filter.py \
    --max_seq_length 256 \
    --batch_size 256 \
    --model_checkpoint_path ../outputs/model-name-that-you-trained-before \
    --src_data /SRC_DATA_PATH/sents.ps  \
    --trg_data /TRG_DATA_PATH/sents.en  \
    --output_dir ../filter.output/  \