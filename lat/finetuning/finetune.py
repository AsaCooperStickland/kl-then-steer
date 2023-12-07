from llmtuner.model import get_train_args
import argparse
import datetime
import os
import sys
from llmtuner.extras.callbacks import LogCallback
from workflow import run_sft
# os.environ['CUDA_LAUNCH_BLOCKING'] = "1"


def main():
    script_dir = os.path.dirname(os.path.abspath(sys.argv[0]))
    os.chdir(script_dir)

    parser = argparse.ArgumentParser()
    parser.add_argument('--wandb_dir', default='wandb')
    parser.add_argument('--output_dir', default='results/tmp')
    parser.add_argument('--flash_attn', action='store_true')
    # parser.add_argument('--run_name', default=tmp_dir)
    cmd_args = parser.parse_args()
    
    os.environ['WANDB_PROJECT'] = 'lat'
    os.environ['WANDB_DIR'] = cmd_args.wandb_dir

    custom_args = {
        "steering_data_path": "../datasets",
        'steering_dataset': 'refusal',
    }

    input_args = {
        "stage": "sft",
        "model_name_or_path": "meta-llama/Llama-2-7b-chat-hf",
        "do_train": True,
        'dataset_dir': 'finetuning_data',
        "dataset": "training_0",
        # "dataset": "alpaca_gpt4_en",
        "template": "default",
        "finetuning_type": "full",
        # "lora_target": "q_proj,v_proj",
        "output_dir": cmd_args.output_dir,
        # "output_dir": os.path.join('results', datetime.datetime.now().strftime('%Y-%m-%d_%H:%M:%S') + '_' + cmd_args.run_name),
        "overwrite_cache": True,
        "per_device_train_batch_size": 4,
        "gradient_accumulation_steps": 4,
        "lr_scheduler_type": "cosine",
        "logging_steps": 10,
        "save_steps": 10,
        "learning_rate": 5e-5,
        "num_train_epochs": 1.0,
        "plot_loss": True,
        "bfloat16": True,
        "overwrite_output_dir": True,
        "seed": 15,
        "flash_attn": cmd_args.flash_attn,
        # "do_eval": True,  # Enable evaluation
        # "evaluation_strategy": "steps",
        # "eval_steps": 8,
        # 'val_size': 8,
    }

    model_args, data_args, training_args, finetuning_args, generating_args = get_train_args(input_args)
    callbacks = [LogCallback()]
    run_sft(model_args, data_args, training_args, finetuning_args, generating_args, callbacks, custom_args)

if __name__ == "__main__":
    main()
