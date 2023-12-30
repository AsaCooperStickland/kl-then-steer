import os
import argparse
import datetime
import sys
import json
import jsonlines
from dotenv import load_dotenv
from tqdm import tqdm
from typing import TYPE_CHECKING, Optional, List

from transformers import DataCollatorForSeq2Seq, Seq2SeqTrainingArguments
from llmtuner.model import load_model_and_tokenizer, get_train_args
from llmtuner.extras.callbacks import LogCallback
from repe import rep_control_reading_vec
from lat.utils import system_prompt, data_path, jailbreaks_path
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


def generate_with_vector(model, tokenizer, questions, directory, question_type="", temperature=0.0):
    # Define the layer range and block name for steering
    layer_ids = list(range(-11, -30, -1))
    block_name = "decoder_block"

    # Initialize Steering and WrappedReadingVecModel
    model.steering = Steering(model.custom_args['steering_dataset'], model.model.model,
                              model.tokenizer, model.custom_args['steering_data_path'])
    model.wrapped_model = rep_control_reading_vec.WrappedReadingVecModel(
        model.model.model, model.tokenizer)
    model.wrapped_model.unwrap()
    model.wrapped_model.wrap_block(layer_ids, block_name=block_name)

    # Define parameters for generation
    max_new_tokens = 400
    batch_size = 8
    all_results = []

    # Loop through each multiplier
    for multiplier in [-3.0, -1.5, 1.5, 3.0]:
        answers = []

        # Batch processing of questions
        for i in tqdm(range(0, len(questions), batch_size)):
            batched_questions = [q["question"]
                                 for q in questions[i: min(i + batch_size, len(questions))]]
            batched_categories = [q["category"]
                                  for q in questions[i: min(i + batch_size, len(questions))]]

            # Generate texts

            inputs = tokenizer(batched_questions, return_tensors="pt")
            # Generate
            generate_ids = model.generate(
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
    with open(f"{directory}/{question_type}results.json", "w") as jfile:
        json.dump(all_results, jfile)


def run_generation(
    model_args: "ModelArguments",
    data_args: "DataArguments",
    training_args: "Seq2SeqTrainingArguments",
    finetuning_args: "FinetuningArguments",
    generating_args: "GeneratingArguments",
    callbacks: Optional[List["TrainerCallback"]] = None,
    custom_args=None,
):
    model, tokenizer = load_model_and_tokenizer(
        model_args, finetuning_args, training_args.do_train, stage="sft")

    tokenizer.padding_side = "left"  # use left-padding in generation
        
    questions = []
    
    file_path = 'datasets/refusal/filtered_questions.jsonl'
    
    # Open the JSONL file and extract questions.
    with jsonlines.open(file_path) as reader:
        for item in reader:
            if 'question' in item:
                item["question"] = prompt_format(item["question"])
                questions.append(item)


    generate_with_vector(model, tokenizer, questions,
                         "vanilla_steering", saved_vector=True)

    # Override the decoding parameters of Seq2SeqTrainer
    # training_args_dict = training_args.to_dict()
    # training_args_dict.update(dict(
    #     generation_max_length=training_args.generation_max_length or data_args.cutoff_len,
    # generation_num_beams=data_args.eval_num_beams or training_args.generation_num_beams
    # ))
    # training_args = Seq2SeqTrainingArguments(**training_args_dict)


def main():
    script_dir = os.path.dirname(os.path.abspath(sys.argv[0]))
    os.chdir(script_dir)
    output_folder_name = "vanilla_steering"
    os.makedirs(output_folder_name, exist_ok=True)

    parser = argparse.ArgumentParser()
    parser.add_argument('--wandb_dir', default='wandb')
    parser.add_argument('--output_dir', default='results/tmp')
    parser.add_argument('--flash_attn', action='store_true')
    parser.add_argument('--finetuning_type', default='full',
                        choices=['full', 'lora'])
    parser.add_argument('--model_name_or_path',
                        default="/scratch/alc9734/latent-adversarial-training/results/run_1")
    parser.add_argument('--steering_data_path',
                        default="/scratch/alc9734/latent-adversarial-training/datasets")
    parser.add_argument(
        '--dataset_dir', default='/scratch/alc9734/latent-adversarial-training/lat/finetuning/finetuning_data')
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
        "model_name_or_path": cmd_args.model_name_or_path,
        "do_train": False,
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

    model_args, data_args, training_args, finetuning_args, generating_args = get_train_args(
        input_args)
    callbacks = [LogCallback()]
    custom_args['finetuning_type'] = finetuning_args.finetuning_type
    custom_args['model_name_or_path'] = input_args['model_name_or_path']
    run_generation(model_args, data_args, training_args, finetuning_args,
         generating_args, callbacks, custom_args)


if __name__ == "__main__":
    main()
