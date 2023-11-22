import json
import os
import torch
import jsonlines
from tqdm import tqdm
from dotenv import load_dotenv

from lat.modeling import Llama7BChatHelper
from lat.generate_steering_vectors import generate_single_steering_vector
from lat.data import ComparisonDataset
from lat.utils import system_prompt, data_path, jailbreaks_path

load_dotenv()
token = os.getenv("HF_TOKEN")
system_prompt = "You are a helpful, honest and concise assistant."
QUESTIONS = []

file_path = 'datasets/refusal/filtered_questions.jsonl'

# Open the JSONL file and extract questions.
with jsonlines.open(file_path) as reader:
    for item in reader:
        if 'question' in item:
            QUESTIONS.append(item)


def get_vec(directory, layer, vec_type=""):
    return torch.load(f"{directory}/vec{vec_type}_layer_{layer}.pt")


def generate_with_vector(model, questions, directory, saved_vector, example=None, question_type=""):
    if "vanilla" in question_type:
        layers = ["n/a"]
        multipliers = ["n/a"]
    else:
        layers = [10, 12, 14, 16]
        # [x / 10 for x in range(-32, 32, 4)]
        multipliers = [-3.2, -1.6, -0.8, -0.4]
    max_new_tokens = 200
    if not saved_vector and "vanilla" not in question_type:
        assert example is not None
        activations = generate_single_steering_vector(model, example)
    else:
        activations = None
    model.set_save_internal_decodings(False)
    all_results = []

    batch_size = 16
    vec_type2name = {"": "activations", "_value": "value",
                     "_query": "query", "n/a": "n/a"}
    if "vanilla" in question_type:
        vec_types = ["n/a"]
    else:
        vec_types = ["", "_value", "_query"]

    for layer in layers:
        layer_results = []
        # for multiplier in tqdm(multipliers):
        for multiplier in multipliers:
            answers = []
            for vec_type in vec_types:
                vec_name = vec_type2name[vec_type]
                model.reset_all()
                if saved_vector:
                    vec = get_vec(directory, layer, vec_type)
                elif "vanilla" in question_type:
                    vec = None
                else:
                    vec = activations[layer]
                if vec is not None:
                    if vec_type == "_value":
                        model.set_add_value_activations(
                            layer, multiplier * vec.cuda())
                    elif vec_type == "_query":
                        model.set_add_query_activations(
                            layer, multiplier * vec.cuda())
                    else:
                        model.set_add_activations(
                            layer, multiplier * vec.cuda())

                # Batch questions
                for i in range(0, len(questions), batch_size):
                    # batch, and handle the case where we have less than batch_size questions
                    batched_questions = [
                        q["question"] for q in questions[i: min(i + batch_size, len(questions))]]
                    batched_categories = [
                        q["category"] for q in questions[i: min(i + batch_size, len(questions))]]
                    generated_texts = model.generate_text(
                        batched_questions, max_new_tokens=max_new_tokens)

                    for question, category, text in zip(batched_questions, batched_categories, generated_texts):
                        text = text.split("[/INST]")[-1].strip()
                        print(f"Question: {question}")
                        print(f"Category: {category}")
                        print(f"Answer: {text}")
                        print(
                            f"Settings: layer {layer}, multiplier {multiplier}, directory {directory}, question_type {question_type}")
                        answers.append(
                            {"question": question, "answer": text, "category": category, "vec_name": vec_name})

            layer_results.append(
                {"multiplier": multiplier, "answers": answers})
        all_results.append({"layer": layer, "results": layer_results})

    with open(f"{directory}/{question_type}results.json", "w") as jfile:
        json.dump(all_results, jfile)


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
