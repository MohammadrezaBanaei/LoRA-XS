import argparse
import os

RES_DIR = "results"

TASK_CONFIG = {
    4: {
        'cola': {'epochs': 50, 'lr': [5e-4, 5e-3], 'cls_lr': [5e-3, 1e-3, 5e-4]},
        'sst2': {'epochs': 20, 'lr': [5e-4, 5e-3], 'cls_lr': [5e-3, 1e-3, 5e-4]},
        'mrpc': {'epochs': 50, 'lr': [5e-4, 5e-3], 'cls_lr': [5e-3, 1e-3, 5e-4]},
        'qnli': {'epochs': 10, 'lr': [5e-4, 5e-3], 'cls_lr': [5e-3, 1e-3, 5e-4]},
    }
}

def glue_main(args):
    task = args.target_task.lower()  # Ensure task is in lowercase to match the config keys
    init_loraxs_config_path = args.init_loraxs_config_path
    output_dir = args.output_dir
    model_name = "roberta-large"

    for rank in [4]:
        epoch = TASK_CONFIG[rank][task]['epochs']
        lrs = TASK_CONFIG[rank][task]['lr']
        cls_lrs = TASK_CONFIG[rank][task]['cls_lr']
        if task not in TASK_CONFIG[rank]:
            raise ValueError(f"Task {task} not recognized. Available tasks: {', '.join(TASK_CONFIG[rank].keys())}")
        for cls_lr in cls_lrs:
            for lr in lrs:
                for seed in [0, 1, 2, 3, 4]:
                    run_str = f'''CUDA_VISIBLE_DEVICES="0" \
                       WANDB_DISABLED="true" \
                       python main_glue.py \
                         --model_name_or_path {model_name} \
                         --lora_rank {rank} \
                         --task_name {task} \
                         --do_train \
                         --do_eval \
                         --seed {seed} \
                         --max_seq_length 128 \
                         --per_device_train_batch_size 32 \
                         --learning_rate {lr} \
                         --cls_learning_rate {cls_lr} \
                         --num_train_epochs {epoch} \
                         --save_steps -1 \
                         --evaluation_strategy epoch  \
                         --logging_steps 1 \
                         --overwrite_output_dir \
                         --init_loraxs_config_path {init_loraxs_config_path} \
                         --output_dir {output_dir}'''
                    os.system(run_str)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--target_task', required=True, help="Name of the GLUE task (e.g., cola, sst2, mrpc, qnli)")
    parser.add_argument('--output_dir', required=True, help="Output directory")
    parser.add_argument('--init_loraxs_config_path', default="config/loraxs_init_config.yaml", help="Path to the LoRAXS config file")
    args = parser.parse_args()

    print(args)

    glue_main(args)
