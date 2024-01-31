import os
import argparse
import sys
import json
import jsonlines
import torch
from datetime import datetime
from dotenv import load_dotenv
from tqdm import tqdm
from typing import TYPE_CHECKING, Optional, List
from copy import deepcopy

from transformers import Seq2SeqTrainingArguments
from llmtuner.model import load_model_and_tokenizer, get_train_args
from llmtuner.extras.callbacks import LogCallback
from lat.utils import system_prompt, data_path, jailbreaks_path
from lat.finetuning.trainer import SteeringTrainer
from lat.finetuning.steering import Steering

if TYPE_CHECKING:
    from transformers import TrainerCallback
    from llmtuner.hparams import ModelArguments, DataArguments, FinetuningArguments, GeneratingArguments


load_dotenv()
token = os.getenv("HF_TOKEN")


def prompt_format(instruction):
    B_INST, E_INST = "[INST]", "[/INST]"
    B_SYS, E_SYS = "<<SYS>>\n", "\n<</SYS>>\n\n"
    dialog_content = B_SYS + system_prompt + E_SYS + instruction.strip()
    dialog_content = f"{B_INST} {dialog_content.strip()} {E_INST}"
    return dialog_content


def generate_with_vector(trainer, tokenizer, questions, directory, custom_args, question_type="", temperature=0.0):
    # Define the layer range and block name for steering
    layer_ids = list(range(-11, -30, -1))
    block_name = "decoder_block"

    # Define parameters for generation
    results_file = f"{custom_args['results_path']}/{custom_args['steering_dataset']}_{question_type}results.json"
    if custom_args["overwrite_results"]:
        existing_results = {}
    else:
        if os.path.exists(results_file):
            with open(results_file, "r") as jfile:
                existing_results = json.load(jfile)
                existing_results = {item["multiplier"]: item["answers"] for item in existing_results}
        else:
            existing_results = {}
    max_new_tokens = 1200
    batch_size = 8
    all_results = []

    if tokenizer.pad_token_id is None:
        tokenizer.pad_token = tokenizer.eos_token

    # Loop through each multiplier
    for multiplier in [-3.0, -1.5, 0.0, 1.5, 3.0]:
        answers = []
        trainer.steering.wrapped_model.reset()
        activations = trainer.steering.get_shift(coeff=multiplier, layer_id=layer_ids, mode="test", num_pairs=200)
        for key in activations:
            activations[key] = activations[key].to(torch.bfloat16)
        trainer.steering.wrapped_model.set_controller(layer_ids, activations, block_name)
        trainer.steering.wrapped_model.to(torch.bfloat16)
        if multiplier in existing_results:
            answers = existing_results[multiplier]
            # filter question is for existing answers
            existing_questions = set([a["question"] for a in answers])
            new_questions = [q for q in questions if q["question"] not in existing_questions]
        else:
            new_questions = deepcopy(questions)

        # Batch processing of questions
        for i in tqdm(range(0, len(new_questions), batch_size)):
            batched_questions = [q["question"]
                                 for q in new_questions[i: min(i + batch_size, len(new_questions))]]
            batched_categories = [q["category"]
                                  for q in new_questions[i: min(i + batch_size, len(new_questions))]]

            # Generate texts

            inputs = tokenizer(batched_questions, return_tensors="pt", padding=True).to("cuda")
            # Generate
            generate_ids = trainer.model.generate(
                inputs.input_ids, max_length=max_new_tokens, temperature=temperature)

            generated_texts = tokenizer.batch_decode(
                generate_ids, skip_special_tokens=True, clean_up_tokenization_spaces=False)

            # Process generated texts
            for question, category, text in zip(batched_questions, batched_categories, generated_texts):
                text = text.split("[/INST]")[-1].strip()
                print(f"Question: {question}")
                print(f"Category: {category}")
                print(f"Answer: {text}")
                print(
                    f"Settings: multiplier {multiplier}, directory {directory}, question_type {question_type}")
                answers.append({"question": question, "answer": text,
                               "category": category, "multiplier": multiplier})

        all_results.append({"multiplier": multiplier, "answers": answers})

    # Save results
    with open(results_file, "w") as jfile:
        json.dump(all_results, jfile)


def run_generation(
    model_args: "ModelArguments",
    training_args: "Seq2SeqTrainingArguments",
    finetuning_args: "FinetuningArguments",
    callbacks: Optional[List["TrainerCallback"]] = None,
    custom_args=None,
):
    model, tokenizer = load_model_and_tokenizer(
        model_args, finetuning_args, training_args.do_train, stage="sft")
    

    tokenizer.padding_side = "left"  # use left-padding in generation
        
    questions = []
    
    if custom_args['test_setting'] == "manual_jailbreaks":
        file_path = f"{custom_args['base_directory']}/datasets/refusal/filtered_questions.jsonl"
    else:
        file_path = f"{custom_args['base_directory']}/datasets/refusal/augmented_questions.jsonl"
    
    # Open the JSONL file and extract questions.
    with jsonlines.open(file_path) as reader:
        for item in reader:
            if 'question' in item:
                item["question"] = prompt_format(item["question"])
                questions.append(item)
    
    steering = Steering(custom_args['steering_dataset'], model, tokenizer, custom_args['steering_data_path'], custom_args)
    trainer = SteeringTrainer(
        model=model,
        steering=steering,
        args=training_args,
        custom_args=custom_args,
        tokenizer=tokenizer,
        callbacks=callbacks,
    )

    if custom_args['test_setting'] == "manual_jailbreaks":
        with open(f"{custom_args['base_directory']}/{jailbreaks_path}", "r") as f:
            jailbreaks = json.load(f)
        for jailbreak in jailbreaks:
            jailbreak_name = jailbreak["name"]
            print(f"Generating for {jailbreak_name}")
            jailbreak_prompt = jailbreak["prompt"]
            jailbreak_questions = []
            for question in questions:
                jailbreak_questions.append(
                    {"question": jailbreak_prompt + question["question"], "category": question["category"]})
            generate_with_vector(trainer, tokenizer, jailbreak_questions,
                                    jailbreak_name, custom_args, question_type=f"{jailbreak_name}_")
    elif custom_args["test_setting"] == "vanilla":
        generate_with_vector(trainer, tokenizer, questions,
                             "vanilla_steering", custom_args)
    else:
        raise ValueError(f"Invalid test setting: {custom_args['test_setting']}, must be one of 'manual_jailbreaks' or 'vanilla'")


def main():
    script_dir = os.path.dirname(os.path.abspath(sys.argv[0]))
    os.chdir(script_dir)

    parser = argparse.ArgumentParser()
    parser.add_argument('--wandb_dir', default='wandb')
    parser.add_argument('--output_dir', default='results/tmp')
    parser.add_argument('--flash_attn', action='store_true')
    parser.add_argument('--overwrite_results', action='store_true')
    parser.add_argument('--finetuning_type', default='full',
                        choices=['full', 'lora'])
    parser.add_argument('--model_name_or_path',
                        default="/scratch/alc9734/latent-adversarial-training/results/run_1")
    parser.add_argument('--steering_data_path',
                        default="/scratch/alc9734/latent-adversarial-training/datasets")
    parser.add_argument('--base_directory', default='/scratch/alc9734/latent-adversarial-training/')
    parser.add_argument(
        '--dataset_dir', default='/scratch/alc9734/latent-adversarial-training/lat/finetuning/finetuning_data')
    parser.add_argument('--dataset', default='training_0')
    parser.add_argument('--steering_dataset', default='refusal_test')
    parser.add_argument('--test_setting', default='vanilla', choices=['vanilla', 'manual_jailbreaks'])
    parser.add_argument('--samples_dir', default='samples')
    parser.add_argument('--samples_freq', default=1000, type=int)  # measured in training steps
    parser.add_argument('--run_name', default=datetime.now().strftime("%Y-%m-%d_%H:%M"))
    parser.add_argument('--num_return_sequences', type=int, default=2)
    parser.add_argument('--steering_coeff', type=float, default=None)
    parser.add_argument('--buffer_size', type=int, default=0)
    # parser.add_argument('--run_name', default=tmp_dir)
    cmd_args = parser.parse_args()
    
 
    os.environ['WANDB_PROJECT'] = 'lat'
    # set wandb off for now
    # os.environ["WANDB_DISABLED"] = "true"
    os.environ['WANDB_DIR'] = cmd_args.wandb_dir
    model_sizes = ["7", "13"]
    name_to_path = {}
    for size in model_sizes:
        name_to_path[f'/vast/work/public/ml-datasets/llama-2/Llama-2-{size}b-chat-hf'] = f'{cmd_args.output_dir}/llama-2-{size}b-chat'
        name_to_path[f'meta-llama/Llama-2-{size}b-chat-hf'] = f'{cmd_args.output_dir}/llama-2-{size}b-chat'
                    
    custom_args = {
        "base_directory": cmd_args.base_directory,
        "steering_data_path": cmd_args.steering_data_path,
        'steering_dataset': cmd_args.steering_dataset,
        'test_setting': cmd_args.test_setting,
        'samples_dir': cmd_args.samples_dir,
        'buffer_size': cmd_args.buffer_size,
        'samples_freq': cmd_args.samples_freq,
        'run_name': cmd_args.run_name,
        'mix_with_clean_data': False,
        'subsample_steering_data': False,
        "num_return_sequences": cmd_args.num_return_sequences,  # for samples generation
        "overwrite_results": cmd_args.overwrite_results,
    }

    input_args = {
        "stage": "sft",
        "model_name_or_path": cmd_args.model_name_or_path,
        "do_train": False,
        "template": "llama2chatsimple",
        'dataset_dir': cmd_args.dataset_dir,
        "dataset": cmd_args.dataset,
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
        "hf_hub_token": token,
        # "do_eval": True,  # Enable evaluation
        # "evaluation_strategy": "steps",
        # "eval_steps": 8,
        # 'val_size': 8,
    }

    model_args, data_args, training_args, finetuning_args, generating_args = get_train_args(
        input_args)
    callbacks = [LogCallback()]
    custom_args['finetuning_type'] = finetuning_args.finetuning_type
    custom_args['model_name_or_path'] = input_args['model_name_or_path']
    output_folder_name = "vanilla_steering"
    directory_or_model_name_or_path = name_to_path[custom_args['model_name_or_path']] if custom_args['model_name_or_path'] in name_to_path else custom_args['model_name_or_path']
    custom_args['results_path'] = f"{directory_or_model_name_or_path}/{output_folder_name}"
    os.makedirs(f"{directory_or_model_name_or_path}/{output_folder_name}", exist_ok=True)
    run_generation(model_args, training_args, finetuning_args,
         callbacks, custom_args)


if __name__ == "__main__":
    main()
