import json


def transform_data(input_paths, output_path):
    data = []

    for input_path in input_paths:
        with open(input_path, 'r') as file:
            data_subset = json.load(file)
        data.extend(data_subset)

    transformed_data = []

    for entry in data:
        new_entry = {
            "instruction": entry["question"],
            "input": "",
            "output": entry["answer"],
            "category": entry["category"],
            "source": entry["source"]
        }
        transformed_data.append(new_entry)

    with open(output_path, 'w') as file:
        json.dump(transformed_data, file, indent=4)


if __name__ == "__main__":
    # Call the function
    input_file_paths = [
        f"datasets/refusal/training/llama2_chat_7b/generated_training_{i}/vanilla_results_temp.json" for i in range(7)]
    print(input_file_paths)
    output_file_path = "lat/finetuning/finetuning_data/training_0.json"
    transform_data(input_file_paths, output_file_path)
