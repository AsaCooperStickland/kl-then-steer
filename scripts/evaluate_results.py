import json
import os
import re
import jsonlines
from collections import defaultdict

from lat.api_utils import extract_score, call_model_with_retries_batched, batch_prompts
from lat.format_utils import QuestionAugmenter, prompt_format
from lat.utils import jailbreaks_path


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
    

def categorize_results(results, classifier_prompt, model_name, call_type, 
                       directory, question_type, evaluate_vanilla=False,
                       use_existing=True, test_set_questions=None):
    """Categorizes the results using the classifier and stores them as JSON."""
    categorized_results_file = f"{directory}/{question_type}_categorized_results.json"
    if use_existing and os.path.exists(categorized_results_file):
        with open(categorized_results_file, "r") as f:
            categorized_data = json.load(f)
    else:
        categorized_data = {}

    all_prompts = defaultdict(list)
    for multiplier_data in results:
        multiplier = multiplier_data["multiplier"]
        if evaluate_vanilla:
            if multiplier != 0.0:
                continue
        multiplier_key = f"multiplier_{multiplier}"
        existing_results = categorized_data.get(
            multiplier_key, [])
        existing_questions = [entry["question"] for entry in existing_results]
        print(f"Existing questions for multiplier {multiplier}: {len(existing_questions)}")
        for ans in multiplier_data["answers"]:
            if ans["question"] in existing_questions:
                continue

            input_prompt = {
                "question": ans["question"], "answer": ans["answer"]}
            classification_input = f"\nInput: {input_prompt}"
            prompt = classifier_prompt + classification_input
            prompt = f"{prompt}\nLet's think step by step:"
            prompt = remove_repeated_letters(prompt)[1]
            entry = {
                "question": ans["question"],
                "answer": ans["answer"],
                "category": ans["category"],
                "vec_type": "pca", "prompt": prompt}
            all_prompts[multiplier_key].append(entry)

    # Send prompts in batches and get responses

    print(f"Evaluating {sum(len(v) for v in all_prompts.values())} prompts.")
            
    batch_size = 1
    multiplier_keys_sorted = sorted(all_prompts.keys())
    all_prompts_only = []
    for multiplier_key in multiplier_keys_sorted:
        all_prompts_only.extend([entry["prompt"] for entry in all_prompts[multiplier_key]])
    batched_prompts_gen = batch_prompts(all_prompts_only, batch_size)
    all_responses = call_model_with_retries_batched(
        batched_prompts_gen, model_name, call_type)
    print(f"Received {len(all_responses)} responses.")

    # Process and categorize responses
    response_idx = 0
    for multiplier_key in multiplier_keys_sorted:
        if multiplier_key not in categorized_data:
            categorized_data[multiplier_key] = []

        for entry in all_prompts[multiplier_key]:
            if "category" not in entry:
                entry["category"] = "n/a"

            # Constructing the prompt for classification
            response = all_responses[response_idx]
            response_idx += 1
            classification_output = response.content
            if response.stop_reason == "bad_request_error":
                classification_output = "{'score': 1}"
            score = extract_score(classification_output)
            # if score == 4:
            #     print(ans["answer"])
            #     print(classification_output)
            if score is None:
                score = -1
            # print(f"{score}\n\nthis is the score")

            # Storing the result with its classification
            entry["score"] = score
            categorized_data[multiplier_key].append(entry)

    # Store the data
    with open(categorized_results_file, "w") as file:
        json.dump(categorized_data, file)
    # make sure potential questions is a superset of test_set_questions
    if test_set_questions:
        categorized_data = filter_categorized_data(categorized_data, test_set_questions)

    return categorized_data


def filter_categorized_data(categorized_data, test_set_questions, verbose=False):
    potential_questions = set([entry["question"] for entry in categorized_data["multiplier_0.0"]])
    # assert test_set_questions.issubset(potential_questions)
    if not test_set_questions.issubset(potential_questions):
        if verbose:
            print(f"Test set questions: {len(test_set_questions)}")
            print(f"Potential questions: {len(potential_questions)}")
            print(f"Example questions: {list(test_set_questions)[:5]}")
            print(f"Example potential questions: {list(potential_questions)[:5]}")
            print("Warning! Test set questions are not a subset of potential questions.")
    if verbose:
        print(f"Potential questions: {len(potential_questions)}")
    # filter out any questions not in the test set
    categorized_data["multiplier_0.0"] = [entry for entry in categorized_data["multiplier_0.0"] 
                                          if entry["question"] in test_set_questions]
    if verbose:
        print(f"Filtered potential questions: {len(categorized_data['multiplier_0.0'])}")
    return categorized_data


def get_average_scores_json(categorized_data, grouped_categories_dict):
    """Gets the average score per layer per multiplier."""
    average_scores = {}
    for multiplier_key, multiplier_data in categorized_data.items():
        if multiplier_key != "multiplier_0.0":
            continue
        scores = [entry for entry in multiplier_data if entry["score"] != -1]
        average_scores[multiplier_key] = {}
        for vec_name in ["pca"]:
            average_scores[multiplier_key][vec_name] = {}
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
            for category_type in grouped_categories_dict:
                category_scores = []
                for category in grouped_categories_dict[category_type]:
                    if category in average_scores[multiplier_key][vec_name]:
                        category_scores.append(
                            average_scores[multiplier_key][vec_name][category])
                if len(category_scores) > 0:
                    average_scores[multiplier_key][vec_name][category_type] = sum(
                        category_scores) / len(category_scores)
    return average_scores


def get_csv_from_json_scores(average_scores, question_type="", steering_dataset="refusal_test", 
                             grouped_categories_dict=None, print_header=True, model_name=None):
    """Gets a CSV from the average scores. Separate out different categories"""
    if model_name is None:
        model_name = "n/a"
    if grouped_categories_dict:
        category_list = ["average"] 
        other_categories = sorted(list(grouped_categories_dict.keys()))
        for category_type in other_categories:
            for multiplier_key, multiplier_data in average_scores.items():
                for vec_name in ["pca"]:
                    if category_type in multiplier_data[vec_name]:
                        category_list.append(category_type)
    else:
        category_list = ["average", "nationality_bias", "race_bias", "gender_bias",
                         "political_bias", "illegal_activity", "capability", "human_only", "power_seeking"]
    if print_header:
        csv = ", , , "
        for category in category_list:
            csv += f"{category},"
        csv += "\n"
        csv += "Model, Steering dataset, Multiplier,pca\n"
    else:
        csv = ""
    for multiplier_key, multiplier_data in average_scores.items():
        multiplier_key = multiplier_key.replace("multiplier_", "")
        csv += f"{model_name}, {steering_dataset},{multiplier_key},"
        for category in category_list:
            for vec_name in ["pca"]:
                # if "n/a" in multiplier_data:
                #     if category in multiplier_data['n/a']:
                #         csv += f"{multiplier_data['n/a'][category]},,,"
                #         break
                # use 2 decimal places
                csv += f"{multiplier_data[vec_name].get(category, 0):.2f},"
        csv += "\n"
    return csv.strip()


def main():
    verbose = False
    overwrite = False
    path = "/scratch/alc9734/latent-adversarial-training/results"
    models = ["run_no_steer_full/checkpoint-1000", "run_persuasion_0.5/checkpoint-1000", "run_large_scale_concept_1/checkpoint-1000"] #, "llama-2-7b-chat"]
    models += ["run_2/checkpoint-4000", "run_no_steer_full/checkpoint-4000", "run_persuasion_0.5/checkpoint-4000", "run_large_scale_concept_1/checkpoint-4000"]
    models += ["run_2/checkpoint-16000", "run_no_steer_full/checkpoint-16000", "run_persuasion_0.5/checkpoint-16000", "run_large_scale_concept_1/checkpoint-16000"]
    models += ["llama-2-7b-chat", "llama-2-13b-chat"]
    models += ["run2_persuasion_0.5"]
    augmenter = QuestionAugmenter(dataset_path="datasets", 
                                  jailbreaks_path=jailbreaks_path,
                                  jinja_directory="/scratch/alc9734/llm-jailbreaks/prompts/wei-jailbreaks/")
    grouped_categories_dict = augmenter.get_all_category_keys()

    test_set_questions_file = "datasets/refusal/augmented_questions.jsonl"
    test_set_questions = set()
    with jsonlines.open(test_set_questions_file) as reader:
        for item in reader:
            if 'question' in item:
                formatted_question = prompt_format(item["question"])
                test_set_questions.add(formatted_question)

    for path_idx, model in enumerate(models):
        model_path = f"{path}/{model}"
        directory = f"{model_path}/vanilla_steering"
        # "4chan_", "vanilla_4chan_", "aligned_", "vanilla_aligned_", "confidant_", "vanilla_confidant_", "machiavelli_", "vanilla_machiavelli_"]
        jail_breaks = [""] # , "_4chan", "_aligned", "_confidant", "_machiavelli"]
        # jail_breaks = ["", "_4chan", "_aligned", "_confidant"]
        # question_types = ["refusal_test", "emotions_happiness", "emotions_anger"]
        question_types = ["refusal"]
        all_question_types = [f"{question_type}{jail_break}" for jail_break in jail_breaks for question_type in question_types]
        # question_types = ["vanilla_"]
        for i, results_type in enumerate(all_question_types):
            results_location = f"{directory}/{results_type}_results.json"
            if os.path.exists(results_location):
                results = json.load(
                    open(f"{directory}/{results_type}_results.json", "r"))
            else:
                if verbose:
                    print(f"Results for {results_type} from model {results_location} not found.")
                continue
            with open("lat/evaluation_prompt.txt", "r") as f:
                classifier_prompt = f.read()
            model_name = "gpt-4-0125-preview" #"gpt-3.5-turbo-16k-0613"
            call_type = "sample"  # or "logprobs"
            categorized_results_file = f"{directory}/{results_type}_categorized_results.json"
            if overwrite or not os.path.exists(categorized_results_file):
                categorized_results = categorize_results(
                    results, classifier_prompt, model_name, call_type, directory, results_type, evaluate_vanilla=True,
                    test_set_questions=test_set_questions)
            else:
                with open(categorized_results_file, "r") as f:
                    categorized_results = json.load(f)
                categorized_results = filter_categorized_data(categorized_results, test_set_questions)
            
            average_scores = get_average_scores_json(categorized_results, grouped_categories_dict)
            jail_break_type = results_type.split("_")[-1]
            question_type = "_".join(results_type.split("_")[:-1])
            csv_scores = get_csv_from_json_scores(
                average_scores, question_type=question_type, steering_dataset=jail_break_type, 
                grouped_categories_dict=grouped_categories_dict, print_header=path_idx==0, model_name=model)
            # print(f"Results for {results_type} from model {path}:")
            print(csv_scores)


if __name__ == "__main__":
    main()
