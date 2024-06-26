from llmtuner.hparams import get_train_args
import argparse
from datetime import datetime
import os
import random
import numpy as np
import torch
import sys
from llmtuner.extras.callbacks import LogCallback
from workflow import run_sft
from ppo_workflow import run_ppo
from dpo_workflow import run_dpo

# os.environ['CUDA_LAUNCH_BLOCKING'] = "1"


def set_random_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def main():
    script_dir = os.path.dirname(os.path.abspath(sys.argv[0]))
    os.chdir(script_dir)

    parser = argparse.ArgumentParser()
    parser.add_argument('--wandb_dir', default='wandb')
    parser.add_argument('--output_dir', default='results/tmp')
    parser.add_argument('--flash_attn', action='store_true')
    parser.add_argument('--stage', default='sft', choices=['sft', 'ppo', 'dpo'])
    parser.add_argument('--finetuning_type', default='lora', choices=['full', 'lora'])
    parser.add_argument('--adapter_name_or_path', type=str, default=None)
    parser.add_argument('--merge_adapter', action='store_true')
    parser.add_argument('--steering_data_path', default="/scratch/alc9734/latent-adversarial-training/datasets")
    parser.add_argument('--dataset_dir', default='/scratch/alc9734/latent-adversarial-training/lat/finetuning/finetuning_data')
    parser.add_argument('--base_directory', default='/scratch/alc9734/latent-adversarial-training/')
    parser.add_argument('--dataset', default='training_0')
    parser.add_argument('--steering_dataset', default='refusal')
    parser.add_argument('--num_train_epochs', type=int, default=1)
    parser.add_argument('--samples_dir', default='/scratch/jp6263/latent-adversarial-training/samples')
    parser.add_argument('--samples_freq', default=8000, type=int)  # measured in training steps
    parser.add_argument('--batch_size', default=2, type=int)  # measured in training steps
    parser.add_argument('--learning_rate', default=None, type=float)
    parser.add_argument('--run_name', default=datetime.now().strftime("%Y-%m-%d_%H:%M"))
    parser.add_argument('--num_return_sequences', type=int, default=2)
    parser.add_argument('--buffer_size', type=int, default=0)
    parser.add_argument('--rep_token', default=-1)
    parser.add_argument('--direction_method', default='pca', choices=['random', 'pca', 'cluster_mean'])
    parser.add_argument('--steering_unnormalized', action='store_true')
    parser.add_argument('--loss_function', default='vanilla', choices=['vanilla', 'kl'])
    parser.add_argument('--steering_coeff', type=float, default=None)
    parser.add_argument('--steering_coeff_range', type=str, default='positive', choices=['positive', 'both'])
    parser.add_argument('--token_pos', type=str, default=None)
    parser.add_argument('--steering_probability', type=float, default=0.5)
    parser.add_argument('--do_steer', action='store_true')
    parser.add_argument('--batch_lora', action='store_true')
    parser.add_argument('--train_bias', action='store_true')
    parser.add_argument('--optimize_steering', action='store_true')
    parser.add_argument('--template', default='llama2chatsimple')
    parser.add_argument('--seed', type=int, default=19)
    parser.add_argument('--neftune_noise_alpha', type=float, default=0.0)
    parser.add_argument('--local_rank', type=int, default=0)
    parser.add_argument('--deepspeed', type=str, default=None)
    cmd_args = parser.parse_args()

    set_random_seed(cmd_args.seed)
    if cmd_args.neftune_noise_alpha == 0.0:
        cmd_args.neftune_noise_alpha = None
    
    os.environ['WANDB_PROJECT'] = 'lat'
    # os.environ["WANDB_DISABLED"] = "true"  # set wandb off for faster debug
    os.environ['WANDB_DIR'] = cmd_args.wandb_dir

    custom_args = {
        "steering_data_path": cmd_args.steering_data_path,
        "base_directory": cmd_args.base_directory,
        'steering_dataset': cmd_args.steering_dataset,
        'do_steer': cmd_args.do_steer,
        'optimize_steering': cmd_args.optimize_steering,
        'samples_dir': cmd_args.samples_dir,
        'samples_freq': cmd_args.samples_freq,
        'buffer_size': cmd_args.buffer_size,
        'run_name': cmd_args.run_name,
        'rep_token': cmd_args.rep_token,
        'direction_method': cmd_args.direction_method,
        'steering_unnormalized': cmd_args.steering_unnormalized,
        'loss_function': cmd_args.loss_function,
        'steering_probability': cmd_args.steering_probability,
        'steering_coeff_range': cmd_args.steering_coeff_range,
        'subsample_steering_data': False,
        'token_pos': cmd_args.token_pos,
        'normalize': False,
        "batch_lora": cmd_args.batch_lora,
        "bias": "all" if cmd_args.train_bias else "none",
        "num_return_sequences": cmd_args.num_return_sequences,  # for samples generation
    }
    
    if cmd_args.stage == "sft":
        gradient_accumulation_steps = 2 if cmd_args.finetuning_type == "lora" else 4
    else:
        gradient_accumulation_steps = 2 if cmd_args.finetuning_type == "lora" else 4
    if cmd_args.learning_rate is not None:
        learning_rate = cmd_args.learning_rate
    else:
        learning_rate = 5e-4 if cmd_args.stage == "sft" else 1e-5
    input_args = {
        "stage": cmd_args.stage,
        "model_name_or_path": "/vast/work/public/ml-datasets/llama-2/Llama-2-7b-chat-hf",
        "adapter_name_or_path": cmd_args.adapter_name_or_path,
        
        "do_train": True,
        "template": cmd_args.template,
        'dataset_dir': cmd_args.dataset_dir,
        # "dataset": "alpaca_gpt4_en",
        "dataset": cmd_args.dataset,
        "finetuning_type": cmd_args.finetuning_type,
        "lora_target": "all",
        # "use_rslora": True,
        "lora_rank": 128 if not cmd_args.batch_lora else 1,
        "output_dir": cmd_args.output_dir,
        # "output_dir": os.path.join('results', datetime.now().strftime('%Y-%m-%d_%H:%M:%S') + '_' + cmd_args.run_name),
        "overwrite_cache": True,
        "per_device_train_batch_size": cmd_args.batch_size,
        # "gradient_accumulation_steps": 4,
        "gradient_accumulation_steps": gradient_accumulation_steps,
        "lr_scheduler_type": "linear" if cmd_args.adapter_name_or_path is not None else "cosine",
        "warmup_steps": 100 if cmd_args.adapter_name_or_path is not None else 0,
        "logging_steps": 1,
        "save_steps": 4000,
        "learning_rate": learning_rate,
        "reward_model": "starling",
        'reward_model_type': 'starling',
        # "reward_model": "alikhan0100u/Llama-2-7b-oasst-preference-reward-model-adapter",
        # 'reward_model_type': 'lora',
        # "ref_model_quantization_bit": 4,
        "num_train_epochs": cmd_args.num_train_epochs,
        "plot_loss": True,
        # "bf16": True,
        "overwrite_output_dir": True,
        "seed": cmd_args.seed,
        "neftune_noise_alpha": cmd_args.neftune_noise_alpha,
        "flash_attn": cmd_args.flash_attn,
        "val_size": 0.2,
        "do_sample": True,
        "max_new_tokens": 80,
        "temperature": 1.0,
        "top_p": 1 if cmd_args.stage == "sft" else 0.9,
        "top_k": 50 if cmd_args.stage == "sft" else 0,
        "length_penalty": 1.0,
        "deepspeed": cmd_args.deepspeed,
        "local_rank": cmd_args.local_rank,
    }

    model_args, data_args, training_args, finetuning_args, generating_args = get_train_args(input_args)
    callbacks = [LogCallback()]
    custom_args['finetuning_type'] = finetuning_args.finetuning_type
    custom_args['model_name_or_path'] = input_args['model_name_or_path']
    if cmd_args.steering_coeff is not None:
        custom_args['steering_coeff'] = cmd_args.steering_coeff
    else:
        if custom_args['model_name_or_path'] == 'meta-llama/Llama-2-7b-chat-hf':
            custom_args['steering_coeff'] = 1.5
        elif custom_args['model_name_or_path'] == 'meta-llama/Llama-2-13b-chat-hf':
            custom_args['steering_coeff'] = 3.0
        else:
            custom_args['steering_coeff'] = 1.5
    if finetuning_args.stage == "sft":
        run_sft(model_args, data_args, training_args, finetuning_args, generating_args, callbacks, custom_args)
    elif finetuning_args.stage == "dpo":
        run_dpo(model_args, data_args, training_args, finetuning_args, generating_args, callbacks, custom_args)
    else:
        run_ppo(model_args, data_args, training_args, finetuning_args, generating_args, callbacks, custom_args)

if __name__ == "__main__":
    main()
