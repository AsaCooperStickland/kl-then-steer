import os
import json
import jsonlines
from dotenv import load_dotenv
from tqdm import tqdm

from lat.modeling import Llama7BChatHelper
from lat.utils import system_prompt, data_path, jailbreaks_path
from lat.finetuning.steering import Steering
from repe import rep_control_reading_vec

load_dotenv()
token = os.getenv("HF_TOKEN")
QUESTIONS = []

file_path = 'datasets/refusal/filtered_questions.jsonl'

# Open the JSONL file and extract questions.
with jsonlines.open(file_path) as reader:
    for item in reader:
        if 'question' in item:
            QUESTIONS.append(item)


def generate_with_vector(model, questions, directory, question_type="", temperature=0.0):
    # Define the layer range and block name for steering
    layer_ids = list(range(-11, -30, -1))
    block_name = "decoder_block"

    # Initialize Steering and WrappedReadingVecModel
    model.steering = Steering(model.custom_args['steering_dataset'], model.model.model, model.tokenizer, model.custom_args['steering_data_path'])
    model.wrapped_model = rep_control_reading_vec.WrappedReadingVecModel(model.model.model, model.tokenizer)
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
            batched_questions = [q["question"] for q in questions[i: min(i + batch_size, len(questions))]]
            batched_categories = [q["category"] for q in questions[i: min(i + batch_size, len(questions))]]

            # Generate texts
            generated_texts = model.generate_text(batched_questions, max_new_tokens=max_new_tokens, temperature=temperature)

            # Process generated texts
            for question, category, text in zip(batched_questions, batched_categories, generated_texts):
                text = text.split("[/INST]")[-1].strip()
                print(f"Question: {question}")
                print(f"Category: {category}")
                print(f"Answer: {text}")
                print(f"Settings: multiplier {multiplier}, directory {directory}, question_type {question_type}")
                answers.append({"question": question, "answer": text, "category": category, "multiplier": multiplier})

        all_results.append({"multiplier": multiplier, "answers": answers})

    # Save results
    with open(f"{directory}/{question_type}results.json", "w") as jfile:
        json.dump(all_results, jfile)


if __name__ == "__main__":
    model = Llama7BChatHelper(token, system_prompt, generation=True)
    output_folder_name = "vanilla_steering"
    os.makedirs(output_folder_name, exist_ok=True)
    generate_with_vector(model, QUESTIONS, output_folder_name)
    # generate_with_vector(model, QUESTIONS, output_folder_name, saved_vector=False, question_type="vanilla_")
