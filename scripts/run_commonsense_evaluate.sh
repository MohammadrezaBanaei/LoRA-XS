MERGED_PATH="output_model"

python -m utils.merge_adapter_to_base_model --base_model $1 --adapter $2 --output_path "$MERGED_PATH"

CUDA_VISIBLE_DEVICES=$3 python instruction_tuning_eval/commonsense_eval.py \
    --model LLaMA3-8B \
    --adapter LoRAXS \
    --dataset boolq \
    --base_model "$MERGED_PATH" \
    --batch_size 1 \

CUDA_VISIBLE_DEVICES=$3 python instruction_tuning_eval/commonsense_eval.py \
    --model LLaMA3-8B \
    --adapter LoRAXS \
    --dataset piqa \
    --base_model "$MERGED_PATH" \
    --batch_size 1
#
CUDA_VISIBLE_DEVICES=$3 python instruction_tuning_eval/commonsense_eval.py \
    --model LLaMA3-8B \
    --adapter LoRAXS \
    --dataset social_i_qa \
    --base_model "$MERGED_PATH" \
    --batch_size 1

CUDA_VISIBLE_DEVICES=$3 python instruction_tuning_eval/commonsense_eval.py \
    --model LLaMA3-8B \
    --adapter LoRAXS \
    --dataset hellaswag \
    --base_model "$MERGED_PATH" \
    --batch_size 1

CUDA_VISIBLE_DEVICES=$3 python instruction_tuning_eval/commonsense_eval.py \
    --model LLaMA3-8B \
    --adapter LoRAXS \
    --dataset winogrande \
    --base_model "$MERGED_PATH" \
    --batch_size 1

CUDA_VISIBLE_DEVICES=$3 python instruction_tuning_eval/commonsense_eval.py \
    --model LLaMA3-8B \
    --adapter LoRAXS \
    --dataset ARC-Challenge \
    --base_model "$MERGED_PATH" \
    --batch_size 1

CUDA_VISIBLE_DEVICES=$3 python instruction_tuning_eval/commonsense_eval.py \
    --model LLaMA3-8B \
    --adapter LoRAXS \
    --dataset ARC-Easy \
    --base_model "$MERGED_PATH" \
    --batch_size 1

CUDA_VISIBLE_DEVICES=$3 python instruction_tuning_eval/commonsense_eval.py \
    --model LLaMA3-8B \
    --adapter LoRAXS \
    --dataset openbookqa \
    --base_model "$MERGED_PATH" \
    --batch_size 1

rm -r $MERGED_PATH
