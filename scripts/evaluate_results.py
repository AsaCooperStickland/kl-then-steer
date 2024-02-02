import json
import re
from openai import BadRequestError
import argparse

from lat.api_utils import get_content, extract_score, call_model_with_retries_batched, batch_prompts


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
    

def categorize_results(results, classifier_prompt, model_name, call_type, directory, save_filename_prefix):
    """Categorizes the results using the classifier and stores them as JSON."""
    categorized_data = {}

    all_prompts = []
    for multiplier_data in results:
        for ans in multiplier_data["answers"]:
            input_prompt = {
                "question": ans["question"], "answer": ans["answer"]}
            classification_input = f"\nInput: {input_prompt}"
            prompt = classifier_prompt + classification_input
            # prompt = f"{prompt}\nLet's think step by step:"
            all_prompts.append(prompt)

    # Send prompts in batches and get responses
    cleaned_prompts = []
    for prompt_idx, prompt in enumerate(all_prompts):
        include, prompt = remove_repeated_letters(prompt)
        cleaned_prompts.append(prompt)
            
    batch_size = 128
    batched_prompts_gen = batch_prompts(cleaned_prompts, batch_size)
    all_responses = call_model_with_retries_batched(
        batched_prompts_gen, model_name, call_type)

    # Process and categorize responses
    response_idx = 0
    for multiplier_data in results:
        multiplier = multiplier_data["multiplier"]
        multiplier_key = f"multiplier_{multiplier}"
        categorized_data[multiplier_key] = []

        for ans in multiplier_data["answers"]:
            if "category" not in ans:
                ans["category"] = "n/a"

            # Constructing the prompt for classification
            response = all_responses[response_idx]
            response_idx += 1
            classification_output = get_content(response, model_name)
            if isinstance(classification_output, BadRequestError):
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
        with open(f"{directory}/{save_filename_prefix}_eval_results.json", "w") as file:
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


def get_csv_from_json_scores(average_scores, print_header=True):
    """Gets a CSV from the average scores. Separate out different categories"""
    category_list = ["average", "nationality_bias", "race_bias", "gender_bias",
                     "political_bias", "illegal_activity", "capability", "human_only", "power_seeking"]
    if print_header:
        # csv = ", , average, , , nationality_bias, , ,race_bias,,,gender_bias,,,political_bias,,,illegal_activity,,,capability,,,human_only,,,power_seeking,,,\n"
        csv = ", "
        for category in category_list:
            csv += f"{category},"
        csv += "\n"
        csv += "Multiplier,pca\n"
    else:
        csv = ""
    for multiplier_key, multiplier_data in average_scores.items():
        multiplier_key = multiplier_key.replace("multiplier_", "")
        csv += f"{multiplier_key},"
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
    # Create parser object
    parser = argparse.ArgumentParser(description="Get the 'exp' parameter from command line.")

    # Add 'exp' argument
    # You can modify `type` based on the expected type (int, float, etc.)
    # `required=True` makes the argument mandatory
    parser.add_argument('--exp', type=str, required=True, help="An example parameter.")

    # Parse arguments
    args = parser.parse_args()

    evaluate = True

    # path = "/scratch/alc9734/latent-adversarial-training/results/"
    # path = "/scratch/al6759/assets_lat/tmp/"
    path = "/scratch/al6759/lat/results/"

    # models = ["run_2/checkpoint-9000", "llama-2-7b-chat"]
    models = ["llama-2-7b-chat",]
    
    paths = [path + model for model in models]
    for path in paths:
        # directory = f"{path}/vanilla_steering"
        directory = path
        # "4chan_", "vanilla_4chan_", "aligned_", "vanilla_aligned_", "confidant_", "vanilla_confidant_", "machiavelli_", "vanilla_machiavelli_"]
        # question_types = ["refusal_test", "emotions_happiness", "emotions_anger"]
        # question_types = ["vanilla_"]  # I got these with refusal
        # for i, question_type in enumerate(question_types):
        # exp = 'refusal_data_A_B_cropped'
        filename = args.exp + '_gen_results'
        results = json.load(
            # open(f"{directory}/{question_type}results.json", "r"))
            open(f"{directory}/{filename}.json", "r"))
        with open("lat/evaluation_prompt.txt", "r") as f:
            classifier_prompt = f.read()
        # model_name = "gpt-3.5-turbo-0125"
        model_name = "gpt-3.5-turbo-1106"  # higher rate limit
        call_type = "sample"  # or "logprobs"
        if evaluate:
            categorized_results = categorize_results(
                results, classifier_prompt, model_name, call_type, directory, filename)
        else:
            with open(f"{directory}/{args.exp}_eval_results.json", "r") as f:
                categorized_results = json.load(f)
        average_scores = get_average_scores_json(categorized_results)
        # print(average_scores, "average scores")
        csv_scores = get_csv_from_json_scores(
            average_scores, print_header=True)
        print(f"Results for {filename} from model {path}:")
        print(csv_scores)


if __name__ == "__main__":
    main()
