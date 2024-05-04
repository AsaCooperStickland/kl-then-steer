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
import dill

from transformers import Seq2SeqTrainingArguments
from llmtuner.model import load_model, load_tokenizer
from llmtuner.hparams import get_train_args
from llmtuner.extras.callbacks import LogCallback
from lat.utils import jailbreaks_path, alternative_system_prompts
from lat.format_utils import prompt_format
from lat.finetuning.trainer import SteeringTrainer
from lat.finetuning.steering import Steering

if TYPE_CHECKING:
    from transformers import TrainerCallback
    from llmtuner.hparams import ModelArguments, DataArguments, FinetuningArguments, GeneratingArguments


load_dotenv()
token = os.getenv("HF_TOKEN")


def generate_with_vector(trainer, tokenizer, questions, directory, custom_args, question_type="", temperature=0.0):
    # Define the layer range and block name for steering
    start_layer, end_layer = custom_args['start_layer'], custom_args['end_layer']
    layer_ids = list(range(start_layer, end_layer, -1))
    block_name = "decoder_block"

    # Define parameters for generation
    results_prefix = f"{custom_args['results_path']}/{custom_args['steering_dataset']}_{question_type}"
    if custom_args["direction_method"] != "pca":
        results_prefix += f"{custom_args['direction_method']}_"
    if custom_args["steering_unnormalized"]:
        results_prefix += "unnormalized_"
    if start_layer != -11 and end_layer != -30:
        results_prefix += f"layers_{start_layer}_{end_layer}_"
    if custom_args["decay_coefficient"]:
        results_prefix += f"decay_"
    if custom_args["alternative_system_prompt"] is not None:
        results_prefix += f"alt_prompt_{custom_args['alternative_system_prompt']}_"
    results_file = f"{results_prefix}results_bs1.json"
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
    # max_new_tokens = 400
    batch_size = 24 if "13b" in custom_args["model_name_or_path"] else 48
    if custom_args["no_bf16"]:
        batch_size = 8 
    all_results = []

    if tokenizer.pad_token_id is None:
        tokenizer.pad_token = tokenizer.eos_token
    
    with open(f"{custom_args['base_directory']}/lat/finetuning/steering_data/norms_llama-2-7b.json", "r") as f:
        norms_dict = json.load(f)
    # s = trainer.steering
    raw_activations = trainer.steering.get_shift(coeff=1.0, layer_id=trainer.steering.layer_id, num_pairs=200, mode='train')
    if custom_args['direction_method'] == 'cluster_mean' and not custom_args['steering_unnormalized']:
        print("Normalizing raw cluster_mean activations")
        raw_activations = {k: v / torch.norm(str(k)) for k, v in raw_activations.items()}
        # rror: can't convert cuda:0 device type tensor to numpy. Use Tensor.cpu() to copy the tensor to host memory first
    if custom_args['direction_method'] == 'pca' and custom_args['steering_unnormalized']:
        print("Unnormalizing raw pca activations")
        if trainer.steering.norms_dict is None:
            print("Unnormalizing raw pca activations in test_repe.py")
            raw_activations = {k: v * norms_dict[str(k)] for k, v in raw_activations.items()}

    # Loop through each multiplier
    if custom_args['steering_unnormalized']:
        multipliers = [-0.5, -0.25, -0.15, -0.12, -0.09, -0.06, -0.03, 0.0, 0.03, 0.06, 0.09, 0.12, 0.15, 0.25, 0.5]
        if custom_args["direction_method"] == "cluster_mean":
            # multipliers = [-1.0, -0.75, -0.5, -0.25, -0.15, -0.12, -0.09, 0.0, 0.09, 0.12, 0.15, 0.25, 0.5, 0.75, 1.0]
            if "bias" in custom_args['test_setting']:
                multipliers = [-1.5, -1.0, 1.0, 1.5]
            else:
                multipliers = [-1.0, -0.75, -0.5, -0.25, -0.15, -0.12, -0.09, 0.0, 0.09, 0.12, 0.15, 0.25, 0.5, 0.75, 1.0]
            if custom_args["alternative_system_prompt"] is not None:
                multipliers = [-0.25, -0.12, 0.0, 0.12, 0.25]
            # multipliers = [-1.0, -0.75, 0.75, 1.0]
            # multipliers = [-0.5, -0.75]
            # multipliers = [0.5]
        # multipliers = [0.0]
    else:
        multipliers = [-2.0, -1.5, -1.0, 0.0, 1.0, 1.5, 2.0]
        # multipliers = [0.5]
    
    existing_multipliers_not_being_generated = set(existing_results.keys()) - set(multipliers)
    if len(existing_multipliers_not_being_generated) > 0:
        print(f"Existing multipliers {existing_multipliers_not_being_generated}")
        for multiplier in existing_multipliers_not_being_generated:
            all_results.append({"multiplier": multiplier, "answers": existing_results[multiplier]})
    
    for multiplier in multipliers:
        print(f"Generating with multiplier {multiplier}")
        answers = []
        trainer.steering.wrapped_model.reset()
        activations = raw_activations.copy()
        for layer in trainer.steering.layer_id:
            activations[layer] = activations[layer] * multiplier
        # activations = s.get_shift(coeff=multiplier, layer_id=layer_ids, mode="test", num_pairs=200)
        for key in activations:
            activations[key] = activations[key].to(torch.bfloat16)
        trainer.steering.wrapped_model.set_controller(layer_ids, activations, block_name)
        trainer.steering.wrapped_model.to(torch.bfloat16)
        if multiplier in existing_results:
            answers = existing_results[multiplier]
            # filter question is for existing answers
            # delete any questions without the correct steering key
            for a in answers:
                if "vector_type" not in a:
                    a["vector_type"] = "pca"
                    
            existing_questions = set([a["question"] for a in answers if a["vector_type"] == trainer.steering.repe_key])
            print(f"Found {len(existing_questions)} existing answers")
            new_questions = [q for q in questions if q["question"] not in existing_questions]
            print(f"Generating for {len(new_questions)} new questions")
        else:
            new_questions = deepcopy(questions)
            print(len(new_questions))

        # Batch processing of questions
        for i in tqdm(range(0, len(new_questions), batch_size)):
            batched_questions = [q["question"]
                                 for q in new_questions[i: min(i + batch_size, len(new_questions))]]
            batched_categories = [q["category"]
                                  for q in new_questions[i: min(i + batch_size, len(new_questions))]]
            if "bias" in custom_args['test_setting']:
                batched_ground_truth = [q["ground_truth"]
                                        for q in new_questions[i: min(i + batch_size, len(new_questions))]]
                batched_biased_option = [q["biased_option"]
                                         for q in new_questions[i: min(i + batch_size, len(new_questions))]]
                extra_data = {"ground_truth": batched_ground_truth, "biased_option": batched_biased_option}
            else:
                extra_data = None

            # Generate texts

            inputs = tokenizer(batched_questions, return_tensors="pt", padding=True).to("cuda")
            # Generate
            generate_ids = trainer.model.generate(
                inputs.input_ids, max_length=max_new_tokens, temperature=temperature, do_sample=False)

            generated_texts = tokenizer.batch_decode(
                generate_ids, skip_special_tokens=True, clean_up_tokenization_spaces=False)

            # Process generated texts
            for idx, (question, category, text) in enumerate(zip(batched_questions, batched_categories, generated_texts)):
                if custom_args['template'] == 'llama3':
                    text = ''.join(text.split('assistant\n\n')[1:]).strip()
                else:
                    text = text.split("[/INST]")[-1].strip()
                # print(f"Question: {question}")
                # print(f"Category: {category}")
                # print(f"Answer: {text}")
                # print(
                #     f"Settings: multiplier {multiplier}, directory {directory}, question_type {question_type}")
                outputs_and_data = {"question": question, "answer": text, "vector_type": trainer.steering.repe_key,
                               "category": category, "multiplier": multiplier}
                if extra_data is not None:
                    for key in extra_data:
                        outputs_and_data[key] = extra_data[key][idx]
                answers.append(outputs_and_data)
                print('Multipler:', multiplier)
                print(text)
                print('---')

        all_results.append({"multiplier": multiplier, "answers": answers})
        trainer.steering.reset()

        # Save results after each multiplier finishes
        with open(results_file, "w") as jfile:
            json.dump(all_results, jfile)


def run_generation(
    model_args: "ModelArguments",
    training_args: "Seq2SeqTrainingArguments",
    finetuning_args: "FinetuningArguments",
    callbacks: Optional[List["TrainerCallback"]] = None,
    custom_args=None,
):
    tokenizer = load_tokenizer(model_args)['tokenizer']
    model = load_model(tokenizer, model_args, finetuning_args, is_trainable=training_args.do_train)

    tokenizer.padding_side = "left"  # use left-padding in generation

    # # tokenizer.pad_token = "[PAD]"
    # tokenizer.pad_token = " "
    # # tokenizer.pad_token = tokenizer.eos_token
    # tokenizer.padding_side = "right"

    steering = Steering(custom_args['steering_dataset'], model, tokenizer, custom_args['steering_data_path'], custom_args)
    trainer = SteeringTrainer(
        model=model,
        steering=steering,
        args=training_args,
        ref_model=None,
        custom_args=custom_args,
        tokenizer=tokenizer,
        callbacks=callbacks,
        finetuning_args=finetuning_args,
    )
    
    questions = []
    
    if custom_args['test_setting'] == "manual_jailbreaks":
        file_path = f"{custom_args['base_directory']}/datasets/refusal/filtered_questions.jsonl"
    elif custom_args['test_setting'] == "ultra_filtered":
        file_path = f"{custom_args['base_directory']}/datasets/refusal/ultra_filtered_questions.jsonl"
    else:
        file_path = f"{custom_args['base_directory']}/datasets/refusal/augmented_questions.jsonl"

    if custom_args['alternative_system_prompt'] is not None:
        alternative_system_prompt = alternative_system_prompts[str(custom_args['alternative_system_prompt'])]
    else:
        alternative_system_prompt = None
    
    # Open the JSONL file and extract questions.
    if "bias" not in custom_args['test_setting']:
        with jsonlines.open(file_path) as reader:
            for item in reader:
                if 'question' in item:
                    item["question"] = prompt_format(item["question"],                 
                        alternative_system_prompt=alternative_system_prompt,
                        template=custom_args['template'],
                        tokenizer=tokenizer)
                    questions.append(item)

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
                                    jailbreak_name, custom_args, question_type=f"{jailbreak_name}")
    elif custom_args["test_setting"] == "vanilla" or custom_args["test_setting"] == "ultra_filtered":
        generate_with_vector(trainer, tokenizer, questions,
                             "vanilla_steering", custom_args)
        
    elif "bias" in custom_args["test_setting"]:
        _, *bias_dataset = custom_args["test_setting"].split("_")
        bias_dataset = "_".join(bias_dataset)
        for bias_type in ["suggested_answer"]:
            for dataset in [bias_dataset]:
                unbiased_questions = []
                biased_questions = []
                file_name = os.path.join(bias_type, f"{dataset}_{bias_type}.jsonl")
                with jsonlines.open(os.path.join(custom_args["bias_path"], file_name)) as reader:
                    for item in reader:
                        if 'original_question' in item:
                            item["question"] = prompt_format(item["original_question"] + "\n\nGive your answer in the format \"Therefore, the best answer is: (X).\"", alternative_system_prompt=alternative_system_prompt, template=custom_args['template'],
                            tokenizer=tokenizer)
                            unbiased_questions.append({"question": item["question"], "category": item["original_dataset"], "ground_truth": item["ground_truth"], "biased_option": item["biased_option"]})
                        if 'biased_question' in item:
                            biased_question_content = item["biased_question"][0]["content"].replace("\n\nPlease think step by step and then give your answer in the format \"Therefore, the best answer is: (X).\".", "").replace("It's very important that you stick to this format.", "").replace("\n\nLet's think step by step:", "").rstrip() + "\n\nGive your answer in the format \"Therefore, the best answer is: (X).\""
                            biased_questions.append({"question": prompt_format(biased_question_content, alternative_system_prompt=alternative_system_prompt, template=custom_args['template'],
                            tokenizer=tokenizer), "category": item["original_dataset"], "ground_truth": item["ground_truth"], "biased_option": item["biased_option"]})
                
                generate_with_vector(trainer, tokenizer, biased_questions, f"{dataset}_{bias_type}_biased", custom_args, question_type=f"{dataset}_{bias_type}_biased_")
                generate_with_vector(trainer, tokenizer, unbiased_questions, f"{dataset}_{bias_type}_unbiased", custom_args, question_type=f"{dataset}_{bias_type}_unbiased_")



    else:
        raise ValueError(f"Invalid test setting: {custom_args['test_setting']}, must be one of 'manual_jailbreaks' or 'vanilla'")


def main():
    script_dir = os.path.dirname(os.path.abspath(sys.argv[0]))
    os.chdir(script_dir)

    parser = argparse.ArgumentParser()
    parser.add_argument('--wandb_dir', default='wandb')
    parser.add_argument('--output_dir', default='results/tmp')
    parser.add_argument('--flash_attn', action='store_true')
    parser.add_argument('--no_bf16', action='store_true')
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
    parser.add_argument('--dataset', default='training_0')  # ignored!
    parser.add_argument('--steering_dataset', default='refusal_test')
    parser.add_argument('--adapter_name_or_path', type=str, default=None)
    parser.add_argument('--template', type=str, default='llama2chatsimple')
    parser.add_argument('--start_layer', type=int, default=-11)
    parser.add_argument('--end_layer', type=int, default=-30)
    parser.add_argument('--test_setting', default='vanilla', choices=['vanilla', 'ultra_filtered', 'manual_jailbreaks', 'bias_mmlu', 'bias_truthfulqa', 'bias_hellaswag', 'bias_logiqa'])
    parser.add_argument('--bias_path', default='/scratch/alc9734/cot-transparency/dataset_dumps/test')
    parser.add_argument('--samples_dir', default='samples')
    parser.add_argument('--rep_token', default=-1)
    parser.add_argument('--direction_method', default='pca',
                        choices=['random', 'pca', 'cluster_mean'])
    parser.add_argument('--steering_unnormalized', action='store_true')
    parser.add_argument('--decay_coefficient', action='store_true')
    parser.add_argument('--samples_freq', default=1000, type=int)  # measured in training steps
    parser.add_argument('--run_name', default=datetime.now().strftime("%Y-%m-%d_%H:%M"))
    parser.add_argument('--num_return_sequences', type=int, default=2)
    parser.add_argument('--steering_coeff', type=float, default=None)
    parser.add_argument('--buffer_size', type=int, default=0)
    parser.add_argument('--alternative_system_prompt', type=int, default=None, choices=[1, 2, 3])
    # parser.add_argument('--run_name', default=tmp_dir)
    cmd_args = parser.parse_args()
    # dill.dump(cmd_args, open(f"cmd_args.pkl", "wb"))
    
 
    os.environ['WANDB_PROJECT'] = 'lat'
    # set wandb off for now
    # os.environ["WANDB_DISABLED"] = "true"
    os.environ['WANDB_DIR'] = cmd_args.wandb_dir
    model_sizes = ["7", "13"]
    name_to_path = {}
    for size in model_sizes:
        name_to_path[f'/vast/work/public/ml-datasets/llama-2/Llama-2-{size}b-chat-hf'] = f'{cmd_args.output_dir}/llama-2-{size}b-chat'
        name_to_path[f'meta-llama/Llama-2-{size}b-chat-hf'] = f'{cmd_args.output_dir}/llama-2-{size}b-chat'
    name_to_path["NousResearch/Nous-Hermes-2-Mistral-7B-DPO"] = f"{cmd_args.output_dir}/hermes-2-mistral-7b-dpo"

    custom_args = {
        "base_directory": cmd_args.base_directory,
        "steering_data_path": cmd_args.steering_data_path,
        'steering_dataset': cmd_args.steering_dataset,
        'test_setting': cmd_args.test_setting,
        'bias_path': cmd_args.bias_path,
        'samples_dir': cmd_args.samples_dir,
        'buffer_size': cmd_args.buffer_size,
        'rep_token': cmd_args.rep_token,
        'token_pos': None,
        'normalize': False,
        'direction_method': cmd_args.direction_method,
        'steering_unnormalized': cmd_args.steering_unnormalized,
        'start_layer': cmd_args.start_layer,
        'end_layer': cmd_args.end_layer,
        'loss_function': "vanilla",
        'steering_coeff_range': "positive",
        "decay_coefficient": cmd_args.decay_coefficient,
        'samples_freq': cmd_args.samples_freq,
        'run_name': cmd_args.run_name,
        'mix_with_clean_data': False,
        'subsample_steering_data': False,
        'no_bf16': cmd_args.no_bf16,
        "num_return_sequences": cmd_args.num_return_sequences,  # for samples generation
        "overwrite_results": cmd_args.overwrite_results,
        "merge_adapter": cmd_args.finetuning_type == "lora",
        "alternative_system_prompt": cmd_args.alternative_system_prompt,
        'template': cmd_args.template,
    }

    input_args = {
        "stage": "sft",
        "model_name_or_path": cmd_args.model_name_or_path,
        "adapter_name_or_path": cmd_args.adapter_name_or_path,
        "do_train": False,
        "template": cmd_args.template,
        'dataset_dir': cmd_args.dataset_dir,
        "dataset": cmd_args.dataset,
        "finetuning_type": cmd_args.finetuning_type,
        "lora_target": "q_proj,v_proj",
        "output_dir": cmd_args.output_dir,
        # "output_dir": os.path.join('results', datetime.datetime.now().strftime('%Y-%m-%d_%H:%M:%S') + '_' + cmd_args.run_name),
        "overwrite_cache": True,
        "logging_steps": 10,
        "save_steps": 1000,
        "learning_rate": 5e-5,
        "num_train_epochs": 1.0,
        "plot_loss": True,
        "bf16": not cmd_args.no_bf16,
        "overwrite_output_dir": True,
        "seed": 15,
        # "flash_attn": '80' if cmd_args.flash_attn else 'off',
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
    # output_folder_name = "vanilla_steering"
    directory_or_model_name_or_path = name_to_path[custom_args['model_name_or_path']] if custom_args['model_name_or_path'] in name_to_path else custom_args['model_name_or_path']
    if cmd_args.adapter_name_or_path is not None:
        directory_or_model_name_or_path = cmd_args.adapter_name_or_path
    if custom_args["merge_adapter"]:
        directory_or_model_name_or_path = f"{directory_or_model_name_or_path}/merged"
    custom_args['results_path'] = directory_or_model_name_or_path
    # custom_args['results_path'] = f"{directory_or_model_name_or_path}/{output_folder_name}"
    os.makedirs(custom_args['results_path'], exist_ok=True)
    # os.makedirs(f"{directory_or_model_name_or_path}/{output_folder_name}", exist_ok=True)
    custom_args['subsample_steering_data'] = False
    custom_args['mix_with_clean_data'] = False
    run_generation(model_args, training_args, finetuning_args,
         callbacks, custom_args)


if __name__ == "__main__":
    main()
