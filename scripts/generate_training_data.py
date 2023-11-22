import random
import ast
import jsonlines
from lat.evaluate_results import call_model_with_retries_batched, batch_prompts, get_content

# Load documents


file_path = 'datasets/refusal/filtered_questions.jsonl'
documents = []
# Open the JSONL file and extract questions.
with jsonlines.open(file_path) as reader:
    for item in reader:
        if 'question' in item:
            documents.append(item)

categories = set()
for doc in documents:
    categories.add(doc["category"])

expanded_questions = []
for c in categories:
    # Get existing questions for this category
    category_questions = [q for q in documents if q["category"] == c]

    if len(category_questions) > 5:
        # Sample 5 random questions
        sample = random.sample(category_questions, 5)
    else:
        # Use all questions
        sample = category_questions

    sample_str = "\n".join([str(q) for q in sample])
    prompts = [f"Generate 10 more questions like the following:\n{sample_str}"]
    print(prompts)
    # raise ValueError

    # Call model to generate 10 more questions per prompt
    model = "gpt-4-1106-preview"
    # responses = call_model_with_retries_batched(batch_prompts(prompts, batch_size=1), "claude-2", "sample")
    # responses = call_model_with_retries_batched(batch_prompts(prompts, batch_size=1), "gpt-3.5-turbo-16k-0613", "sample")
    responses = call_model_with_retries_batched(
        batch_prompts(prompts, batch_size=1), model, "sample")

    # Extract generated questions
    for prompt, resp in zip(prompts, responses):
        for string_json in get_content(resp, model).splitlines():
            print(string_json)
            if string_json:
                # parse string representation of dictionary into dictionary
                try:
                    if "{" not in string_json and "}" not in string_json:
                        continue
                    # remove anything not enclosed in curly braces
                    string_json = string_json[string_json.find(
                        "{"):string_json.rfind("}")+1]
                    example = ast.literal_eval(string_json)
                    if example["category"] != c:
                        continue
                    expanded_questions.append(
                        {"category": c, "question": example["question"]})
                except SyntaxError:
                    print("Error parsing question: ", string_json)
                    continue

print(len(expanded_questions))
print(expanded_questions)
