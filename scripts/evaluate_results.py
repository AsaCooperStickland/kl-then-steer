import json
from lat.api_utils import get_content, extract_score, call_model_with_retries_batched, batch_prompts


def categorize_results(results, classifier_prompt, model_name, call_type, directory, question_type):
    """Categorizes the results using the classifier and stores them as JSON."""
    categorized_data = {}

    all_prompts = []
    for layer_data in results:
        for multiplier_data in layer_data["results"]:
            for ans in multiplier_data["answers"]:
                input_prompt = {
                    "question": ans["question"], "answer": ans["answer"]}
                classification_input = f"\nInput: {input_prompt}"
                prompt = classifier_prompt + classification_input
                prompt = f"{prompt}\nLet's think step by step:"
                all_prompts.append(prompt)

    # Send prompts in batches and get responses
    batch_size = 1
    batched_prompts_gen = batch_prompts(all_prompts, batch_size)
    all_responses = call_model_with_retries_batched(
        batched_prompts_gen, model_name, call_type)

    # Process and categorize responses
    response_idx = 0
    for layer_data in results:
        layer = layer_data["layer"]
        layer_key = f"layer_{layer}"
        if layer_key not in categorized_data:
            categorized_data[layer_key] = {}

        for multiplier_data in layer_data["results"]:
            multiplier = multiplier_data["multiplier"]
            multiplier_key = f"multiplier_{multiplier}"
            categorized_data[layer_key][multiplier_key] = []

            for ans in multiplier_data["answers"]:
                if "category" not in ans:
                    ans["category"] = "n/a"

                # Constructing the prompt for classification
                response = all_responses[response_idx]
                response_idx += 1
                classification_output = get_content(response, model_name)
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
                    "vec_type": ans["vec_name"],
                    "model_output": classification_output,
                    "score": score,
                }
                categorized_data[layer_key][multiplier_key].append(entry)

    # Store the data
    with open(f"{directory}/{question_type}categorized_results.json", "w") as file:
        json.dump(categorized_data, file)

    return categorized_data


def get_average_scores_json(categorized_data):
    """Gets the average score per layer per multiplier."""
    average_scores = {}
    for layer_key, layer_data in categorized_data.items():
        average_scores[layer_key] = {}
        for multiplier_key, multiplier_data in layer_data.items():
            scores = [entry for entry in multiplier_data if entry["score"] != -1]
            average_scores[layer_key][multiplier_key] = {}
            for vec_name in ["activations", "query", "value", "n/a"]:
                average_scores[layer_key][multiplier_key][vec_name] = {}
                # print(scores[0])
                # print(scores[0]["vec_type"])
                # print(vec_name == scores[0]["vec_type"])
                vec_scores = [entry["score"]
                              for entry in scores if entry["vec_type"] == vec_name]
                # print(vec_scores)
                if len(vec_scores) > 0:
                    average_scores[layer_key][multiplier_key][vec_name]["average"] = sum(
                        vec_scores) / len(vec_scores)
                category_list = set(
                    [entry["category"] for entry in scores if entry["vec_type"] == vec_name])
                for category in category_list:
                    category_scores = [
                        entry["score"] for entry in scores if entry["vec_type"] == vec_name and entry["category"] == category]
                    if len(category_scores) > 0:
                        average_scores[layer_key][multiplier_key][vec_name][category] = sum(
                            category_scores) / len(category_scores)
    return average_scores


def get_csv_from_json_scores(average_scores, print_header=True):
    """Gets a CSV from the average scores. Separate out different categories"""
    category_list = ["average", "nationality_bias", "race_bias", "gender_bias",
                     "political_bias", "illegal_activity", "capability", "human_only", "power_seeking"]
    if print_header:
        # csv = ", , average, , , nationality_bias, , ,race_bias,,,gender_bias,,,political_bias,,,illegal_activity,,,capability,,,human_only,,,power_seeking,,,\n"
        csv = ", ,"
        for category in category_list:
            csv += f"{category},,,"
        csv += "\n"
        csv += "Layer,Multiplier,Activations,Query,Value,Activations,Query,Value\n"
    else:
        csv = ""
    for layer_key, layer_data in average_scores.items():
        for multiplier_key, multiplier_data in layer_data.items():
            csv += f"{layer_key},{multiplier_key},"
            for category in category_list:
                for vec_name in ["activations", "query", "value"]:
                    if "n/a" in multiplier_data:
                        if category in multiplier_data['n/a']:
                            csv += f"{multiplier_data['n/a'][category]},,,"
                            break
                    csv += f"{multiplier_data[vec_name][category]},"
            csv += "\n"
    return csv


def main():
    evaluate = True
    path = "/scratch/alc9734/latent-adversarial-training/results/"
    models = ["run_2/checkpoint-9000", "llama-2-7b-chat"]
    paths = [path + model for model in models]
    for path in paths:
        directory = f"{path}/vanilla_steering"
        # "4chan_", "vanilla_4chan_", "aligned_", "vanilla_aligned_", "confidant_", "vanilla_confidant_", "machiavelli_", "vanilla_machiavelli_"]
        question_types = ["refusal_test", "emotions_happiness", "emotions_anger"]
        # question_types = ["vanilla_"]
        for i, question_type in enumerate(question_types):
            results = json.load(
                open(f"{directory}/{question_type}results.json", "r"))
            with open("lat/evaluation_prompt.txt", "r") as f:
                classifier_prompt = f.read()
            model_name = "gpt-3.5-turbo-16k-0613"
            call_type = "sample"  # or "logprobs"
            if evaluate:
                categorized_results = categorize_results(
                    results, classifier_prompt, model_name, call_type, directory, question_type)
            else:
                with open(f"{directory}/{question_type}categorized_results.json", "r") as f:
                    categorized_results = json.load(f)
            average_scores = get_average_scores_json(categorized_results)
            # print(average_scores, "average scores")
            csv_scores = get_csv_from_json_scores(
                average_scores, print_header=i == 0)
            print(f"Results for {question_type} from model {path}:")
            print(csv_scores)


if __name__ == "__main__":
    main()
