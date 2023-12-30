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
system_prompt = "You are a helpful, honest and concise assistant."
QUESTIONS = []
ultrachat_counter = 0
dataset = load_dataset("HuggingFaceH4/ultrachat_200k")
for example in dataset["train_sft"]:
    QUESTIONS.append({"question": example["prompt"], "category": "train", "source": "ultrachat_200k"})
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
                QUESTIONS.append(item)
                refusal_counter += 1

print(f"ultrachat_counter: {ultrachat_counter}")
print(f"refusal_counter: {refusal_counter}")
print(f"total: {ultrachat_counter + refusal_counter}")
# shuffle the dataset
random.shuffle(QUESTIONS)
    

if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Generate answers to questions.")
    parser.add_argument('--question_set_id', type=int, default=0,
                        help='Question set id to generate')
    args = parser.parse_args()
    model = Llama7BChatHelper(token, system_prompt, generation=True)
    output_folder_name = f"generated_training_{args.question_set_id}"
    os.makedirs(output_folder_name, exist_ok=True)
    # split QUESTIONS into 8 parts
    QUESTIONS_split = []
    for i in range(8):
        batch_size = int(len(QUESTIONS) / 8)
        QUESTIONS_split.append(QUESTIONS[i * batch_size: min(len(QUESTIONS), (i + 1) * batch_size)])
    # generate questions
    generate_with_vector(model, QUESTIONS_split[args.question_set_id], output_folder_name, saved_vector=False, question_type="vanilla_", temperature=1.0)