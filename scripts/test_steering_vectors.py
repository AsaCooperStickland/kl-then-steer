import os
import json
import jsonlines
from dotenv import load_dotenv

from lat.modeling import Llama7BChatHelper
from lat.data import ComparisonDataset
from lat.utils import system_prompt, data_path, jailbreaks_path
from lat.generation_utils import generate_with_vector

load_dotenv()
token = os.getenv("HF_TOKEN")
QUESTIONS = []

file_path = 'datasets/refusal/filtered_questions.jsonl'

# Open the JSONL file and extract questions.
with jsonlines.open(file_path) as reader:
    for item in reader:
        if 'question' in item:
            QUESTIONS.append(item)


if __name__ == "__main__":
    model = Llama7BChatHelper(token, system_prompt, generation=True)
    output_folder_name = "vanilla_steering"
    os.makedirs(output_folder_name, exist_ok=True)
    # generate_with_vector(model, QUESTIONS, output_folder_name, saved_vector=True)
    # generate_with_vector(model, QUESTIONS, output_folder_name, saved_vector=False, question_type="vanilla_")
    with open(data_path, "r") as f:
        data = json.load(f)
    dataset = ComparisonDataset(data, system_prompt)
    # for i in range(10):
    #     example = dataset[i]
    #     output_folder_name = f"example{i}_steering"
    #     os.makedirs(output_folder_name, exist_ok=True)
    #     generate_with_vector(model, QUESTIONS, output_folder_name, saved_vector=False, example=example)

    with open(jailbreaks_path, "r") as f:
        jailbreaks = json.load(f)
    for jailbreak in jailbreaks:
        jailbreak_name = jailbreak["name"]
        print(f"Generating for {jailbreak_name}")
        jailbreak_prompt = jailbreak["prompt"]
        jailbreak_questions = []
        for question in QUESTIONS:
            jailbreak_questions.append(
                {"question": jailbreak_prompt + question["question"], "category": question["category"]})

        output_folder_name = "vanilla_steering"
        generate_with_vector(model, jailbreak_questions, output_folder_name,
                             saved_vector=True, question_type=f"{jailbreak_name}_")
        # comparison with doing nothing
        generate_with_vector(model, jailbreak_questions, output_folder_name,
                             saved_vector=False, question_type=f"vanilla_{jailbreak_name}_")
