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
    parser.add_argument('--finetuning_type', default='full', choices=['full', 'lora'])
    parser.add_argument('--steering_data_path', default="/scratch/alc9734/latent-adversarial-training/datasets")
    parser.add_argument('--dataset_dir', default='/scratch/alc9734/latent-adversarial-training/lat/finetuning/finetuning_data')
    parser.add_argument('--dataset', default='training_0')
    parser.add_argument('--steering_dataset', default='refusal')
    # parser.add_argument('--run_name', default=tmp_dir)
    cmd_args = parser.parse_args()
    
    os.environ['WANDB_PROJECT'] = 'lat'
    # set wandb off for now
    # os.environ["WANDB_DISABLED"] = "true"
    os.environ['WANDB_DIR'] = cmd_args.wandb_dir

    custom_args = {
        "steering_data_path": cmd_args.steering_data_path,
        'steering_dataset': cmd_args.steering_dataset,
    }

    input_args = {
        "stage": "sft",
        "model_name_or_path": "meta-llama/Llama-2-7b-chat-hf",
        "do_train": True,
        "template": "llama2chatsimple",
        'dataset_dir': cmd_args.dataset_dir,
        "dataset": cmd_args.dataset,
        # "dataset": "alpaca_gpt4_en",
        # "template": "default",
        "finetuning_type": cmd_args.finetuning_type,
        "lora_target": "q_proj,v_proj",
        "output_dir": cmd_args.output_dir,
        # "output_dir": os.path.join('results', datetime.datetime.now().strftime('%Y-%m-%d_%H:%M:%S') + '_' + cmd_args.run_name),
        "overwrite_cache": True,
        "per_device_train_batch_size": 2,
        "gradient_accumulation_steps": 4,
        "lr_scheduler_type": "cosine",
        "logging_steps": 10,
        "save_steps": 1000,
        "learning_rate": 5e-5,
        "num_train_epochs": 1.0,
        "plot_loss": True,
        "bf16": True,
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
    custom_args['finetuning_type'] = finetuning_args.finetuning_type
    custom_args['model_name_or_path'] = input_args['model_name_or_path']
    run_sft(model_args, data_args, training_args, finetuning_args, generating_args, callbacks, custom_args)

if __name__ == "__main__":
    main()
