import os
import json
import jsonlines
from dotenv import load_dotenv
from datasets import load_dataset

from lat.modeling import Llama7BChatHelper
from lat.utils import system_prompt
from lat.generation_utils import generate_with_vector

load_dotenv()
token = os.getenv("HF_TOKEN")
system_prompt = "You are a helpful, honest and concise assistant."
QUESTIONS = []
dataset = load_dataset("HuggingFaceH4/ultrachat_200k")
for example in dataset["train_sft"]:
    QUESTIONS.append({"question": example["prompt"], "category": "train", "source": "ultrachat_200k"})

# Find all filepaths with "gpt" or "claude" in them ending in jsonl.
base_file_path = 'datasets/refusal/'
file_paths = []
for root, dirs, files in os.walk(base_file_path):
    for file in files:
        if file.endswith(".jsonl") and ("gpt" in file or "claude" in file):
            file_paths.append(os.path.join(root, file))

for file_path in file_paths:
    # Open the JSONL file and extract questions.
    with jsonlines.open(file_path) as reader:
        for item in reader:
            if 'question' in item:
                item["source"] = file_path
                QUESTIONS.append(item)
    

if __name__ == "__main__":
    model = Llama7BChatHelper(token, system_prompt, generation=True)
    output_folder_name = "generated_training"
    os.makedirs(output_folder_name, exist_ok=True)
    generate_with_vector(model, QUESTIONS, output_folder_name, saved_vector=True)