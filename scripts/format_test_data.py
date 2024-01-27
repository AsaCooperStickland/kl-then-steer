import jsonlines
import json
import random
import os
import glob

from lat.utils import jailbreaks_path


def read_rephrased_questions(model, category, extra_diverse=False, test_set=False):
    extra = "_test" if test_set else ""
    extra += "_extra_diverse" if extra_diverse else ""
    path = f"datasets/refusal/{model}_{category}_rephrased_questions{extra}.jsonl"
    if not os.path.exists(path):
        return []
    questions = []
    with jsonlines.open(path, mode='r') as reader:
        for item in reader:
            questions.append(item)
    return questions

    
def load_jinja_files(directory):
    # Find all .jinja files in the specified directory
    file_paths = glob.glob(f"{directory}/*.jinja")
    
    file_contents = []

    for file_path in file_paths:
        # Read the contents of the file
        with open(file_path, 'r') as file:
            content = file.read()
            # append to the filename and content
            file_contents.append((os.path.basename(file_path), content))
            # Replace the placeholder with the actual prompt
            # content_with_prompt = content.replace("{prompt}", prompt)
    return file_contents


def jinja_format(jinja_files, questions, category, model, extra_diverse=False):
    # randomly assign each question to a jinja file, ensuring each one has an equal amount
    # of questions
    jinja_questions = []
    random.shuffle(questions)
    question_interval = len(questions) // len(jinja_files)
    for i, jinja_info in enumerate(jinja_files):
        start_index = i * question_interval
        end_index = start_index + question_interval
        jinja_name, jinja_prompt = jinja_info
        category_key = f"{jinja_name}_{category}_{model}" if not extra_diverse else f"{jinja_name}_{category}_{model}_extra_diverse"
        jinja_questions.extend([{"question": jinja_prompt.replace("{prompt}", item["question"]),
                                 "category": category_key} for item in questions[start_index:end_index]])
    return jinja_questions


def jailbreak_format(jailbreaks, questions, category, model, extra_diverse=False):
    # randomly assign each question to a jailbreak, ensuring each one has an equal amount
    # of questions
    jailbreak_questions = []
    random.shuffle(questions)
    question_interval = len(questions) // len(jailbreaks)
    for i, jailbreak in enumerate(jailbreaks):
        start_index = i * question_interval
        end_index = start_index + question_interval
        jailbreak_name = jailbreak["name"]
        print(f"Generating for {jailbreak_name}")
        jailbreak_prompt = jailbreak["prompt"]
        category_key = f"{jailbreak_name}_{category}_{model}" if not extra_diverse else f"{jailbreak_name}_{category}_{model}_extra_diverse"
        jailbreak_questions.extend([{"question": jailbreak_prompt + item["question"],
                                 "category": category_key} for item in questions[start_index:end_index]])
    return jailbreak_questions
    

def main():
    initial_questions_file_path = "datasets/refusal/filtered_questions.jsonl"
    initial_questions = []
    with jsonlines.open(initial_questions_file_path, mode='r') as reader:
        for item in reader:
            initial_questions.append(item)
    print(f"initial_questions: {len(initial_questions)}")
    
    # Load the jinja files
    directory = "/scratch/alc9734/llm-jailbreaks/prompts/wei-jailbreaks/"
    jinja_files = load_jinja_files(directory)
    print(f"jinja_files: {len(jinja_files)}")
    random.shuffle(jinja_files)
    jinja_files = jinja_files[:5]
    print(jinja_files)
    with open(f"{jailbreaks_path}", "r") as f:
        jailbreaks = json.load(f)
    augmented_categories = ["illegal_activity", "race_bias", "nationality_bias"]
    for category in augmented_categories:
        questions = [item for item in initial_questions if item["category"] == category]
        augmented_questions = jinja_format(jinja_files, questions, category, "initial")
        initial_questions.extend(augmented_questions)
        jailbreak_questions = jailbreak_format(jailbreaks, questions, category, "initial")
        initial_questions.extend(jailbreak_questions)

    for category in augmented_categories:
        for model in ["gpt-4", "gpt-3.5-turbo-16k-0613", "mistralai/Mixtral-8x7B-Instruct-v0.1"]:
            for extra_diverse in [True, False]:
                questions = read_rephrased_questions(model, category, extra_diverse=extra_diverse, test_set=True)
                print(f"questions and category {category} for model {model} using extra_diverse {extra_diverse}: {len(questions)}")
                new_category_key = f"{category}_{model}" if not extra_diverse else f"{category}_{model}_extra_diverse"
                questions = [{"question": item["question"], "category": new_category_key} for item in questions]
                initial_questions.extend(questions)
                jinja_questions = jinja_format(jinja_files, questions, category, model, extra_diverse=extra_diverse)
                initial_questions.extend(jinja_questions)
                jailbreak_questions = jailbreak_format(jailbreaks, questions, category, model, extra_diverse=extra_diverse)
                initial_questions.extend(jailbreak_questions)
    print(f"initial_questions after augmentation: {len(initial_questions)}")
    augmented_questions_file_path = "datasets/refusal/augmented_questions.jsonl"
    with jsonlines.open(augmented_questions_file_path, mode='w') as writer:
        writer.write_all(initial_questions)



if __name__ == "__main__":
    main()
    