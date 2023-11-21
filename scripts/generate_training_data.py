import random
import json
import jsonlines
from lat.evaluate_results import call_model_with_retries_batched

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
    responses = call_model_with_retries_batched(prompts, "claude-2", "sample")
    # responses = call_model_with_retries_batched(prompts, "gpt-3.5-turbo-16k-0613", "sample")
    
    # Extract generated questions
    for prompt, resp in zip(prompts, responses):  
        for q in resp.splitlines():
            if q:
                # parse into json
                q = json.loads(q)
                expanded_questions.append({"category": c, "question": q["question"]})
                
print(len(expanded_questions))
print(expanded_questions)
