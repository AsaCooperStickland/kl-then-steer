
import random
import json
import ast
import os
from collections import defaultdict
import jsonlines
from lat.evaluate_results import call_model_with_retries_batched, batch_prompts, get_content


def load_documents(file_path):
    documents = []
    with jsonlines.open(file_path) as reader:
        for item in reader:
            if 'question' in item:
                documents.append(item)
    return documents


def get_categories(documents):
    return set(doc["category"] for doc in documents)


def load_generated_questions(categories, model):
    generated_questions = defaultdict(list)
    for c in categories:
        path = f"datasets/refusal/{model}_{c}_generated_questions.jsonl"
        if os.path.exists(path):
            with jsonlines.open(path) as reader:
                for item in reader:
                    generated_questions[item["category"]].append(item)
    return generated_questions


def parse_json_string(string_json, category):
    try:
        string_json = string_json[string_json.find(
            "{"):string_json.rfind("}")+1]
        string_json = string_json.replace("': '", '\': "').replace(
            "'}", '"}').replace("', '", '", \'')
        example = ast.literal_eval(string_json)
        if example["category"] == category:
            return example["question"]
    except SyntaxError:
        return None


def generate_prompts(category_questions, category):
    sample = random.sample(category_questions, 5) if len(
        category_questions) > 5 else category_questions
    sample_str = "\n".join([str(q) for q in sample])
    return f"Generate 20 more questions like the following, with the category {category}. Make sure to stick to the JSON format, and make the questions diverse and interesting. Don't stick to the format of the example questions, for example always beginning with the same phrase, make sure you begin with diverse phrases and scenarios:\n{sample_str}"


def generate_questions_for_category(category, model, documents, safe_models, existing_questions):
    question_set = set(q["question"] for q in existing_questions[category])
    category_questions = [q for q in documents if q["category"] == category]
    generated_questions = defaultdict(list)
    if category not in ["illegal_activity", "race_bias"] or model not in safe_models:
        num_tries = 0
        while len(generated_questions[category]) < 100 and num_tries < 25:
            prompt = generate_prompts(category_questions, category)
            responses = call_model_with_retries_batched(batch_prompts(
                [prompt], batch_size=1), model, "sample", temperature=0.5)
            num_tries += 1
            for resp in responses:
                for string_json in get_content(resp, model).splitlines():
                    question = parse_json_string(string_json, category)
                    if question and question not in question_set:
                        generated_questions[category].append(
                            {"category": category, "question": question})
                        question_set.add(question)
    return generated_questions


def write_generated_questions(generated_questions, model):
    for c, questions in generated_questions.items():
        if questions:
            path = f"datasets/refusal/{model}_{c}_generated_questions.jsonl"
            with jsonlines.open(path, mode='w') as writer:
                for question in questions:
                    writer.write(question)


def main():
    random.seed(64)
    file_path = 'datasets/refusal/filtered_questions.jsonl'
    safe_models = ["claude-2", "gpt-4-1106-preview"]
    model = "gpt-3.5-turbo-16k-0613"
    model = "gpt-4-1106-preview"

    documents = load_documents(file_path)
    categories = get_categories(documents)
    existing_questions = load_generated_questions(categories, model)

    for c in categories:
        generated_questions = generate_questions_for_category(
            c, model, documents, safe_models, existing_questions)
        write_generated_questions(generated_questions, model)


if __name__ == "__main__":
    main()
