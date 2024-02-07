import os
import argparse
import random
import json
import jsonlines
from dotenv import load_dotenv
from datasets import load_dataset

from lat.legacy.modeling import Llama7BChatHelper
from lat.utils import system_prompt
from lat.legacy.generation_utils import generate_with_vector

random.seed(64)
load_dotenv()
token = os.getenv("HF_TOKEN")


def ultrachat_processing():
    questions = []
    ultrachat_counter = 0
    dataset = load_dataset("HuggingFaceH4/ultrachat_200k")
    for example in dataset["train_sft"]:
        questions.append({"question": example["prompt"], "category": "train", "source": "ultrachat_200k"})
        ultrachat_counter += 1
        
    
    # Find all filepaths with "gpt" or "claude" in them ending in jsonl.
    base_file_paths = ['datasets/refusal/', 'datasets/refusal/mistralai']
    file_paths = []
    for base_file_path in base_file_paths:
        for root, dirs, files in os.walk(base_file_path):
            for file in files:
                if file.endswith(".jsonl") and ("gpt" in file or "claude" in file or "mistral" in file):
                    file_paths.append(os.path.join(root, file))
                
    
    refusal_counter = 0
    for file_path in file_paths:
        # Open the JSONL file and extract questions.
        with jsonlines.open(file_path) as reader:
            for item in reader:
                if 'question' in item:
                    item["source"] = file_path
                    questions.append(item)
                    refusal_counter += 1
    
    print(f"ultrachat_counter: {ultrachat_counter}")
    print(f"refusal_counter: {refusal_counter}")
    print(f"total: {ultrachat_counter + refusal_counter}")
    # shuffle the dataset
    random.shuffle(questions)
    return questions


def persuasion_processing(rephrased_requests=False):
    persuasion_json_output_file = "/scratch/alc9734/llm-jailbreaks/data/persuasion/jailbreak_dataset_rephrased_5.jsonl"
    persuasion_questions = []
    with jsonlines.open(persuasion_json_output_file) as reader:
        for item in reader:
            if rephrased_requests:
                potential_prompts = [("original", item["bad_request"])]
                for model in ["gpt-4", "gpt-3.5-turbo-16k-0613", "mistralai/Mixtral-8x7B-Instruct-v0.1"]:
                    model_success_key = "success" if model == "gpt-4" else f"{model}_success"
                    model_rephrase_key = "rephrased_request" if model == "gpt-4" else f"{model}_rephrased_request"
                    if item[model_success_key]:
                        potential_prompts.append((model, item[model_rephrase_key]))
                chosen_request = random.choice(potential_prompts)
                item["question"] = chosen_request[1]
                item["source"] = f"persuasion_{chosen_request[0]}"
                persuasion_questions.append(item)
            else:
                item["question"] = item["bad_request"]
                item["source"] = "persuasion"
                persuasion_questions.append(item)
    # save chosen persuasion questions
    if rephrased_requests:
        with jsonlines.open(persuasion_json_output_file, mode='w') as writer:
            for question in persuasion_questions:
                writer.write(question)
    random.shuffle(persuasion_questions)
    return persuasion_questions
    
        

if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Generate answers to questions.")
    parser.add_argument('--question_set_id', type=int, default=0,
                        help='Question set id to generate')
    parser.add_argument("--dataset", type=str, default="ultrachat_200k",
                        choices=["ultrachat", "persuasion"],
                        help="Dataset to use for generation")
    args = parser.parse_args()
    model = Llama7BChatHelper(token, system_prompt, generation=True)
    if args.dataset == "ultrachat":
        questions = ultrachat_processing()
        output_folder_name = f"llama2_chat_7b/generated_training_{args.question_set_id}"
    elif args.dataset == "persuasion":
        questions = persuasion_processing()
        output_folder_name = f"llama2_chat_7b/generated_training_persuasion_{args.question_set_id}"
    # split questions into 8 parts
    os.makedirs(output_folder_name, exist_ok=True)
    questions_split = []
    for i in range(8):
        batch_size = int(len(questions) / 8)
        questions_split.append(questions[i * batch_size: min(len(questions), (i + 1) * batch_size)])
    # generate questions
    generate_with_vector(model, questions_split[args.question_set_id], output_folder_name, saved_vector=False, question_type="vanilla_", temperature=1.0)