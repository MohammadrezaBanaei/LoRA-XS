import os
import argparse


def glue_main(args):
    epoch = 50
    task = args.target_task  # should be one of MRPC, RTE and STSB tasks
    model_name = "roberta-large"

    mnli_models_path = "model_checkpoints/RoBERTa-large/MNLI"

    for seed in [0, 1, 2, 3, 4]:
        for rank in [4, 8, 12, 16, 20, 25]:
            mnli_trained_model = os.path.join(mnli_models_path, f"rank_{rank}")
            results_dir = f'results_{task}_{rank}'
            for classifier_LR in [6e-4, 1e-3]:
                for learning_rate in [6e-4, 1e-3]:
                    run_str = f'''CUDA_VISIBLE_DEVICES="0" \
                           WANDB_DISABLED="true" \
                           python main_glue_from_pretrained.py \
                             --model_name_or_path {model_name} \
                             --lora_rank {rank} \
                             --task_name {task} \
                             --do_train \
                             --do_eval \
                             --seed {seed}\
                             --max_seq_length 128 \
                             --per_device_train_batch_size 32 \
                             --learning_rate {learning_rate} \
                             --cls_lr {classifier_LR}\
                             --num_train_epochs {epoch} \
                             --save_steps -1 \
                             --evaluation_strategy epoch  \
                             --logging_steps 20 \
                             --mnli_model_path "{mnli_trained_model}"\
                             --overwrite_output_dir \
                             --output_dir {results_dir}'''
                    os.system(run_str)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--target_task', required=True)
    args = parser.parse_args()

    glue_main(args)
