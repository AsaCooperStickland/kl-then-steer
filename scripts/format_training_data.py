
import json

# Replace 'input_file_path' with the path to your original JSON file
input_file_path = "datasets/refusal/training/llama2_chat_7b/generated_training_0/vanilla_results_temp.json"
output_file_path = "lat/finetuning/finetuning_data/training_0.json"

def transform_data(input_path, output_path):
    with open(input_path, 'r') as file:
        data = json.load(file)

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

# Call the function
transform_data(input_file_path, output_file_path)