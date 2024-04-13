import json
import os
import random
import jsonlines
import argparse
from collections import defaultdict

from lat.format_utils import check_source_for_refusal, ProbeQuestionAugmenter

random.seed(42)


def load_from_json(input_paths):
    data = []

    for input_path in input_paths:
        with open(input_path, 'r') as file:
            data_subset = json.load(file)
        data.extend(data_subset)
    return data


def transform_data(data, output_path):
    transformed_data = []

    for entry in data:
        new_entry = {
            "instruction": entry["question"],
            "input": "",
            "output": entry["answer"],
            "category": entry["category"],
            "source": entry["source"]
        }
        transformed_data.append(new_entry)

    with open(output_path, 'w') as file:
        json.dump(transformed_data, file, indent=4)


def main(args):
    if args.mode == "sft":
        input_file_paths = [
            f"datasets/refusal/training/llama2_chat_7b/generated_training_{i}/vanilla_results_temp.json" for i in range(7)]
        input_persuasion_file_paths = [
            f"datasets/refusal/training/llama2_chat_7b/generated_training_persuasion_{i}/vanilla_results_temp.json" for i in range(7)]
        print(input_file_paths)
        if args.persuasion_fraction > 0:
            data = load_from_json(input_file_paths)
            persuasion_data_with_answers = load_from_json(input_persuasion_file_paths)
            persuasion_json_output_file = "/scratch/alc9734/llm-jailbreaks/data/persuasion/jailbreak_dataset_rephrased_5.jsonl"
            persuasion_mapping = defaultdict(list)
            with jsonlines.open(persuasion_json_output_file) as reader:
                for item in reader:
                    persuasion_mapping[item["bad_request"]].append(item["prompt"])
    
            persuasion_data = []
            for d in persuasion_data_with_answers:
                # search for if the bad request is an persuasion mapping and pop off a prompt
                if d["question"] in persuasion_mapping:
                    d["question"] = persuasion_mapping[d["question"]].pop()
                    persuasion_data.append(d)
            refusal_data = [d for d in data if check_source_for_refusal(d["source"])]
            num_refusal = len(refusal_data)
            non_refusal_data = [d for d in data if not check_source_for_refusal(d["source"])]
            # replace args.persuasion_fraction of refusal data with persuasion data
            num_persuasion = min(int(num_refusal * args.persuasion_fraction), len(persuasion_data))
            new_persuasion_fraction = num_persuasion / num_refusal
            # record persuasion function to 2 decimal places
            print(f"Persuasion fraction: {new_persuasion_fraction:.2f}")
            output_file_path = f"lat/finetuning/finetuning_data/training_persuasion{new_persuasion_fraction:.2f}.json"
            persuasion_data = persuasion_data[:num_persuasion]
            refusal_data = refusal_data[num_persuasion:]
            print(f"Dataset proportions: {len(non_refusal_data)} non-refusal, {len(refusal_data)} refusal, {len(persuasion_data)} persuasion")
            data = non_refusal_data + refusal_data + persuasion_data
            transform_data(data, output_file_path)
        else:
            output_file_path = "lat/finetuning/finetuning_data/training_0.json"
            data = load_from_json(input_file_paths)
            transform_data(data, output_file_path)
    elif args.mode == "ppo":
            input_file_paths = [
            f"datasets/refusal/training/llama2_chat_7b/generated_training_{i}/vanilla_results_temp.json" for i in range(7)]
            input_persuasion_file_paths = [
                f"datasets/refusal/training/llama2_chat_7b/generated_training_persuasion_{i}/vanilla_results_temp.json" for i in range(7)]
            input_ant_file_paths = ["datasets/refusal/refusal_data.json"]
            print(input_file_paths)
            if args.persuasion_fraction > 0:
                data = load_from_json(input_file_paths)
                # load anthropic data
                ant_data = load_from_json(input_ant_file_paths)
                for entry in ant_data:
                    entry["source"] = "anthropic"
                    entry["answer"] = ""
                    entry["category"] = "anthropic"
                persuasion_data_with_answers = load_from_json(input_persuasion_file_paths)
                persuasion_json_output_file = "/scratch/alc9734/llm-jailbreaks/data/persuasion/jailbreak_dataset_rephrased_5.jsonl"
                persuasion_mapping = defaultdict(list)
                with jsonlines.open(persuasion_json_output_file) as reader:
                    for item in reader:
                        persuasion_mapping[item["bad_request"]].append(item["prompt"])
        
                persuasion_data = []
                for d in persuasion_data_with_answers:
                    # search for if the bad request is an persuasion mapping and pop off a prompt
                    if d["question"] in persuasion_mapping:
                        d["question"] = persuasion_mapping[d["question"]].pop()
                        persuasion_data.append(d)
                refusal_data = [d for d in data if check_source_for_refusal(d["source"])]
                refusal_data.extend(ant_data)
                num_refusal = len(refusal_data)
                print(f"Num refusal: {num_refusal}")
                # replace args.persuasion_fraction of refusal data with persuasion data
                num_persuasion = min(int(num_refusal * args.persuasion_fraction), len(persuasion_data))
                new_persuasion_fraction = num_persuasion / num_refusal
                # record persuasion function to 2 decimal places
                print(f"Persuasion fraction: {new_persuasion_fraction:.2f}")
                persuasion_data = persuasion_data[:num_persuasion]
                refusal_data = refusal_data[num_persuasion:]
                for refusal_proportion in [0.125, 0.25, 0.5]:
                    output_file_path = f"lat/finetuning/finetuning_data/training_persuasion{new_persuasion_fraction:.2f}_refusal{refusal_proportion:.2f}.json"
                    non_refusal_data = [d for d in data if not check_source_for_refusal(d["source"])]
                    print(f"Dataset proportions before filtering: {len(non_refusal_data)} non-refusal, {len(refusal_data)} refusal, {len(persuasion_data)} persuasion")
                    # remove normal data so we have refusal_proportion of refusal data
                    total_data_size = int(num_refusal / refusal_proportion)
                    non_refusal_data = non_refusal_data[:int(total_data_size * (1 - refusal_proportion))]
                    print(f"Dataset proportions: {len(non_refusal_data)} non-refusal, {len(refusal_data)} refusal, {len(persuasion_data)} persuasion")                    
                    data = non_refusal_data + refusal_data + persuasion_data
                    # do the same thing as fractions
                    print(f"Dataset proportions: {len(non_refusal_data) / len(data)} non-refusal, {len(refusal_data) / len(data)} refusal, {len(persuasion_data) / len(data)} persuasion")
                    transform_data(data, output_file_path)
                output_file_path = f"lat/finetuning/finetuning_data/training_persuasion{new_persuasion_fraction:.2f}.json"
                
                print(f"Dataset proportions: {len(non_refusal_data)} non-refusal, {len(refusal_data)} refusal, {len(persuasion_data)} persuasion")
                data = non_refusal_data + refusal_data + persuasion_data
                print(f"Dataset proportions: {len(non_refusal_data) / len(data)} non-refusal, {len(refusal_data) / len(data)} refusal, {len(persuasion_data) / len(data)} persuasion")
                transform_data(data, output_file_path)
            else:
                output_file_path = "lat/finetuning/finetuning_data/training_0.json"
                data = load_from_json(input_file_paths)
                transform_data(data, output_file_path)
    elif args.mode == "probing":
            input_file_paths = [
            f"datasets/refusal/training/llama2_chat_7b/generated_training_7/vanilla_results_temp.json"]
            input_persuasion_file_paths = [
                f"datasets/refusal/training/llama2_chat_7b/generated_training_persuasion_7/vanilla_results_temp.json"]
            input_ant_file_paths = ["datasets/refusal/refusal_data.json"]
            print(input_file_paths)
            if args.persuasion_fraction > 0:
                data = load_from_json(input_file_paths)
                # load anthropic data
                ant_data = load_from_json(input_ant_file_paths)
                print(ant_data)
                print(len(ant_data))
                for entry in ant_data:
                    entry["source"] = "anthropic"
                    entry["answer"] = ""
                    entry["category"] = "anthropic"
                persuasion_data_with_answers = load_from_json(input_persuasion_file_paths)
                persuasion_json_output_file = "/scratch/alc9734/llm-jailbreaks/data/persuasion/jailbreak_dataset_rephrased_5.jsonl"
                persuasion_mapping = defaultdict(list)
                with jsonlines.open(persuasion_json_output_file) as reader:
                    for item in reader:
                        persuasion_mapping[item["bad_request"]].append(item["prompt"])
        
                persuasion_data = []
                for d in persuasion_data_with_answers:
                    # search for if the bad request is an persuasion mapping and pop off a prompt
                    if d["question"] in persuasion_mapping:
                        d["question"] = persuasion_mapping[d["question"]].pop()
                        persuasion_data.append(d)
                refusal_data = [d for d in data if check_source_for_refusal(d["source"])]
                num_refusal = len(refusal_data)
                print(f"Num refusal before augmenting with anthropic: {num_refusal}")
                refusal_data.extend(ant_data)
                # shuffle, then limit refusal data size to 2500
                random.shuffle(refusal_data)
                refusal_data = refusal_data[:2500]
                num_refusal = len(refusal_data)
                print(f"Num refusal: {num_refusal}")
                # replace args.persuasion_fraction of refusal data with persuasion data
                num_persuasion = min(int(num_refusal * args.persuasion_fraction), len(persuasion_data))
                new_persuasion_fraction = num_persuasion / num_refusal
                # record persuasion function to 2 decimal places
                print(f"Persuasion fraction: {new_persuasion_fraction:.2f}")
                persuasion_data = persuasion_data[:num_persuasion]
                refusal_data = refusal_data[num_persuasion:]
                for refusal_proportion in [0.125, 0.25, 0.5]:
                    output_file_path = f"probing/training/training_persuasion{new_persuasion_fraction:.2f}_refusal{refusal_proportion:.2f}.jsonl"
                    non_refusal_data = [d for d in data if not check_source_for_refusal(d["source"])]
                    print(f"Dataset proportions before filtering: {len(non_refusal_data)} non-refusal, {len(refusal_data)} refusal, {len(persuasion_data)} persuasion")
                    # remove normal data so we have refusal_proportion of refusal data
                    total_data_size = int(num_refusal / refusal_proportion)
                    non_refusal_data = non_refusal_data[:int(total_data_size * (1 - refusal_proportion))]
                    print(f"Dataset proportions: {len(non_refusal_data)} non-refusal, {len(refusal_data)} refusal, {len(persuasion_data)} persuasion")                    
                    data = non_refusal_data + refusal_data + persuasion_data
                    # do the same thing as fractions
                    print(f"Dataset proportions: {len(non_refusal_data) / len(data)} non-refusal, {len(refusal_data) / len(data)} refusal, {len(persuasion_data) / len(data)} persuasion")
                    for item in data:
                        item["category"] = item["source"]
                    if not os.path.exists("datasets/probing/training"):
                        os.makedirs("datasets/probing/training")
                    with jsonlines.open(os.path.join("datasets", output_file_path), 'w') as writer:
                        writer.write_all(data)
                    
                    output_file_path = output_file_path.replace(".jsonl", "")
                    augmenter = ProbeQuestionAugmenter(dataset_path="datasets", 
                                    jailbreaks_path="/scratch/alc9734/latent-adversarial-training/datasets/refusal/jailbreaks.json",
                                    jinja_directory="/scratch/alc9734/llm-jailbreaks/prompts/wei-jailbreaks/", jinja_subset="train", questions_file=output_file_path)
                    augmenter.augment_questions()
            else:
                output_file_path = "lat/finetuning/finetuning_data/training_0.json"
                data = load_from_json(input_file_paths)
                transform_data(data, output_file_path)
    else:
        raise ValueError("Mode not recognized")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--persuasion_fraction", type=float, default=0.0)
    parser.add_argument("--mode", type=str, default="sft", choices=["sft", "ppo", "probing"])
    args = parser.parse_args()
    main(args)
