import argparse
import os


def glue_main(args):
    epoch = 10
    task = args.target_task  # should be one of COLA, SST2 and QNLI tasks
    model_name = "roberta-large"

    for rank in [4, 8, 12, 16, 20, 25]:
        results_dir = f'results_{task}_{rank}'
        for lr in [1e-4, 5e-4, 1e-3]:
            for cls_lr in [5e-4, 1e-3, 5e-3]:
                for seed in [0, 1, 2, 3, 4]:
                    run_str = f'''CUDA_VISIBLE_DEVICES="0" \
                       WANDB_DISABLED="true" \
                       python main_glue.py \
                         --model_name_or_path {model_name} \
                         --lora_rank {rank} \
                         --task_name {task} \
                         --do_train \
                         --do_eval \
                         --seed {seed}\
                         --max_seq_length 128 \
                         --per_device_train_batch_size 32 \
                         --learning_rate {lr} \
                         --cls_learning_rate {cls_lr} \
                         --num_train_epochs {epoch} \
                         --save_steps -1 \
                         --evaluation_strategy epoch  \
                         --logging_steps 1 \
                         --overwrite_output_dir \
                         --output_dir {results_dir}'''
                    os.system(run_str)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--target_task', required=True)
    args = parser.parse_args()

    glue_main(args)
