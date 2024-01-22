import json
import os
import re
from openai import BadRequestError

from lat.api_utils import extract_score, call_model_with_retries_batched, batch_prompts


def remove_repeated_letters(input_string):
    """
    Remove a sequence of repeated letters with spaces in between longer than 7.
    
    Args:
    input_string (str): The input string to process.

    Returns:
    bool: True if a sequence was found and removed, False otherwise.
    str: The processed string.
    """
    # Regex pattern to find repeated letters with spaces (more than 7 times)
    pattern = r'(?:([a-zA-Z])\s+)(?=\1\s+){7,}'
    
    # Search for the pattern in the string
    if re.search(pattern, input_string):
        # Remove the found pattern
        output_string = re.sub(pattern, '', input_string)
        return True, output_string
    else:
        return False, input_string
    

def categorize_results(results, classifier_prompt, model_name, call_type, directory, question_type, evaluate_vanilla=False):
    """Categorizes the results using the classifier and stores them as JSON."""
    categorized_data = {}

    all_prompts = []
    for multiplier_data in results:
        multiplier = multiplier_data["multiplier"]
        if evaluate_vanilla:
            if multiplier != 0.0:
                continue
        for ans in multiplier_data["answers"]:
            input_prompt = {
                "question": ans["question"], "answer": ans["answer"]}
            classification_input = f"\nInput: {input_prompt}"
            prompt = classifier_prompt + classification_input
            prompt = f"{prompt}\nLet's think step by step:"
            all_prompts.append(prompt)

    # Send prompts in batches and get responses
    cleaned_prompts = []
    for prompt_idx, prompt in enumerate(all_prompts):
        _, prompt = remove_repeated_letters(prompt)
        cleaned_prompts.append(prompt)

    print(f"Evaluating {len(cleaned_prompts)} prompts.")
            
    batch_size = 1
    batched_prompts_gen = batch_prompts(cleaned_prompts, batch_size)
    all_responses = call_model_with_retries_batched(
        batched_prompts_gen, model_name, call_type)

    # Process and categorize responses
    response_idx = 0
    for multiplier_data in results:
        multiplier = multiplier_data["multiplier"]
        if evaluate_vanilla:
            if multiplier != 0.0:
                continue
        multiplier_key = f"multiplier_{multiplier}"
        categorized_data[multiplier_key] = []

        for ans in multiplier_data["answers"]:
            if "category" not in ans:
                ans["category"] = "n/a"

            # Constructing the prompt for classification
            response = all_responses[response_idx]
            response_idx += 1
            classification_output = response.content
            if response.stop_reason == "bad_request_error":
                classification_output = "{'score': 1}"
            # print(f"{classification_output}\n\nthis is the classification_output")
            score = extract_score(classification_output)
            if score is None:
                score = -1
            # print(f"{score}\n\nthis is the score")

            # Storing the result with its classification
            entry = {
                "question": ans["question"],
                "answer": ans["answer"],
                "category": ans["category"],
                "vec_type": "pca",
                "model_output": classification_output,
                "score": score,
            }
            categorized_data[multiplier_key].append(entry)

    # Store the data
    with open(f"{directory}/{question_type}_categorized_results.json", "w") as file:
        json.dump(categorized_data, file)

    return categorized_data


def get_average_scores_json(categorized_data):
    """Gets the average score per layer per multiplier."""
    average_scores = {}
    for multiplier_key, multiplier_data in categorized_data.items():
        scores = [entry for entry in multiplier_data if entry["score"] != -1]
        average_scores[multiplier_key] = {}
        for vec_name in ["pca"]:
            average_scores[multiplier_key][vec_name] = {}
            # print(scores[0])
            # print(scores[0]["vec_type"])
            # print(vec_name == scores[0]["vec_type"])
            vec_scores = [entry["score"]
                          for entry in scores if entry["vec_type"] == vec_name]
            # print(vec_scores)
            if len(vec_scores) > 0:
                average_scores[multiplier_key][vec_name]["average"] = sum(
                    vec_scores) / len(vec_scores)
            category_list = set(
                [entry["category"] for entry in scores if entry["vec_type"] == vec_name])
            for category in category_list:
                category_scores = [
                    entry["score"] for entry in scores if entry["vec_type"] == vec_name and entry["category"] == category]
                if len(category_scores) > 0:
                    average_scores[multiplier_key][vec_name][category] = sum(
                            category_scores) / len(category_scores)
    return average_scores


def get_csv_from_json_scores(average_scores, question_type="", steering_dataset="refusal_test", print_header=True):
    """Gets a CSV from the average scores. Separate out different categories"""
    category_list = ["average", "nationality_bias", "race_bias", "gender_bias",
                     "political_bias", "illegal_activity", "capability", "human_only", "power_seeking"]
    if print_header:
        # csv = ", , average, , , nationality_bias, , ,race_bias,,,gender_bias,,,political_bias,,,illegal_activity,,,capability,,,human_only,,,power_seeking,,,\n"
        csv = ", , , "
        for category in category_list:
            csv += f"{category},"
        csv += "\n"
        csv += "Question type, Steering dataset, Multiplier,pca\n"
    else:
        csv = ""
    for multiplier_key, multiplier_data in average_scores.items():
        multiplier_key = multiplier_key.replace("multiplier_", "")
        csv += f"{question_type}, {steering_dataset},{multiplier_key},"
        for category in category_list:
            for vec_name in ["pca"]:
                # if "n/a" in multiplier_data:
                #     if category in multiplier_data['n/a']:
                #         csv += f"{multiplier_data['n/a'][category]},,,"
                #         break
                csv += f"{multiplier_data[vec_name][category]},"
        csv += "\n"
    return csv


def main():
    overwrite = True
    path = "/scratch/alc9734/latent-adversarial-training/results/"
    models = ["run_no_steer_full/checkpoint-9000", "run_2/checkpoint-9000", "llama-2-7b-chat"]
    paths = [path + model for model in models]
    for path in paths:
        directory = f"{path}/vanilla_steering"
        # "4chan_", "vanilla_4chan_", "aligned_", "vanilla_aligned_", "confidant_", "vanilla_confidant_", "machiavelli_", "vanilla_machiavelli_"]
        jail_breaks = ["4chan", "aligned", "confidant", "machiavelli"]
        # question_types = ["refusal_test", "emotions_happiness", "emotions_anger"]
        question_types = ["refusal"]
        all_question_types = [f"{question_type}_{jail_break}" for jail_break in jail_breaks for question_type in question_types]
        # question_types = ["vanilla_"]
        for i, results_type in enumerate(all_question_types):
            results = json.load(
                open(f"{directory}/{results_type}_results.json", "r"))
            with open("lat/evaluation_prompt.txt", "r") as f:
                classifier_prompt = f.read()
            model_name = "gpt-3.5-turbo-16k-0613"
            call_type = "sample"  # or "logprobs"
            categorized_results_file = f"{directory}/{results_type}_categorized_results.json"
            if overwrite or not os.path.exists(categorized_results_file):
                categorized_results = categorize_results(
                    results, classifier_prompt, model_name, call_type, directory, results_type, evaluate_vanilla=False)
            else:
                with open(categorized_results_file, "r") as f:
                    categorized_results = json.load(f)
            average_scores = get_average_scores_json(categorized_results)
            jail_break_type = results_type.split("_")[-1]
            question_type = "_".join(results_type.split("_")[:-1])
            # print(average_scores, "average scores")
            csv_scores = get_csv_from_json_scores(
                average_scores, question_type=question_type, steering_dataset=jail_break_type, print_header=i == 0)
            print(f"Results for {results_type} from model {path}:")
            print(csv_scores)


if __name__ == "__main__":
    main()
