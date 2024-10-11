#BASE_MODEL="meta-llama/Llama-2-7b-hf"
BASE_MODEL="meta-llama/Meta-Llama-3-8B"

CUDA_VISIBLE_DEVICES=$4 python main_commonsense_tuning.py \
    --base_model $BASE_MODEL \
    --data_path 'data/commonsense/commonsense_170k.json' \
    --output_dir $3 \
    --lora_dropout 0.0 \
    --batch_size 64  --micro_batch_size 1 --num_epochs 3 \
    --learning_rate 2e-3 --cutoff_len 256 --val_set_size 120 \
    --eval_step 40 --save_step 40  --adapter_name loraxs \
    --target_modules '["q_proj", "k_proj", "v_proj", "o_proj", "up_proj", "down_proj", "gate_proj"]' \
    --lora_r $1 --lora_alpha $2 --use_gradient_checkpointing
