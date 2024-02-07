import json
import jsonlines
import argparse
from collections import defaultdict


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


def check_source_for_refusal(text):
    if ("gpt" in text or "claude" in text or "mistral" in text):
        return True
    return False


def main(args):
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


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--persuasion_fraction", type=float, default=0.0)
    args = parser.parse_args()
    main(args)
