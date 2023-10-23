import json
import math
import time
import re
import asyncio
import os
from collections import namedtuple
from typing import Dict, List, Tuple, Union
from dotenv import load_dotenv

from anthropic import Anthropic
import openai

MAX_NUM_RETRIES = 5
CHAT_MODELS = ['gpt-3.5-turbo-16k-0613', 'gpt-4']
OPENAI_MODELS = ['text-ada-001', 'text-babbage-001', 'text-curie-001', 'text-davinci-002', 'text-davinci-003'] + CHAT_MODELS
ANTHROPIC_MODELS = ['claude-2']
Example = namedtuple('Example', ['question', 'choice1', 'choice2', 'choice3', 'choice4', 'correct_index'])

load_dotenv()
openai.api_key = os.getenv("OPENAI_API_KEY")
anthropic = Anthropic(api_key=os.getenv("ANTHROPIC_API_KEY"))


def extract_score(classification_str):
    """Extracts the numerical score from the classification string."""
    match = re.search(r'"score": (\d+)', classification_str)
    if match:
        return int(match.group(1))
    match = re.search(r"'score': (\d+)", classification_str)
    if match:
        return int(match.group(1))
    return None  # or return 0 or any default value


def call_model_with_retries(prompt: str,
                            model_name: str = None,
                            call_type: str = 'logprobs',
                            temperature: float = 0.0,
                            stop: str = None) -> Union[str, Dict[str, List[Union[str, float]]]]:
    if model_name is None:
        raise ValueError("Model name must be specified.")
    num_retries = 0
    while True:
        try:
            response = select_and_call_model(prompt, model_name, call_type, temperature, stop)
        except Exception as e:
            if num_retries == MAX_NUM_RETRIES:
                raise e
            print(f"Error calling model {model_name}: {e}, sleeping for {math.pow(3, num_retries)} seconds and retrying...")
            time.sleep(math.pow(3, num_retries))
            num_retries += 1
            continue
        break
    return response


def select_and_call_model(prompt: str,
                          model_name: str,
                          call_type: str,
                          temperature: float = 0.0,
                          stop: str = None) -> Union[str, Dict[str, List[Union[str, float]]]]:
    """Selects the appropriate model and calls it with the given prompt."""
    if model_name in OPENAI_MODELS:
        if call_type == 'sample' and model_name in CHAT_MODELS:
            # for now, we don't need to use this
            messages = [
                {"role": "system", "content": "You are a very intelligent assistant, who follows instructions directly."},
                {"role": "user", "content": prompt}
            ]
            response = openai.ChatCompletion.create(
                model=model_name,
                temperature=temperature,
                messages=messages,
                max_tokens=1000
            )
            print(response['choices'][0]['message'])
        elif call_type == 'sample':
            response = openai.Completion.create(model=model_name, prompt=prompt, max_tokens=600, temperature=0.0)
        elif call_type == 'logprobs':
            response = openai.Completion.create(model=model_name, prompt=prompt, max_tokens=0, echo=True, logprobs=5)
    elif model_name in ANTHROPIC_MODELS:
        if call_type == 'logprobs':
            # we don't actually care about this being async - asyncio.run() will block until the call is complete
            response = asyncio.run(anthropic.top_k_log_probs(prompt=prompt, model_name=model_name, bearer_auth=False))
        elif call_type == 'sample':
            prompt = "\n\nHuman: " + prompt + "\n\nAssistant:"
            response = anthropic.completions.create(model=model_name, max_tokens_to_sample=1000, prompt=prompt)
    else:
        raise ValueError(f"Model {model_name} not supported.")
    return response


def categorize_results(results, classifier_prompt, model_name, call_type, directory, question_type):
    """Categorizes the results using the classifier and stores them as JSON."""
    categorized_data = {}
    
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
                input_prompt = {"question": ans["question"], "answer": ans["answer"]}
                if "category" not in ans:
                    ans["category"] = "n/a"
                
                # Constructing the prompt for classification
                classification_input = f"\nInput: {input_prompt}"
                prompt = classifier_prompt + classification_input
                prompt = f"{prompt}\nLet's think step by step:"
                print(f"{prompt}\n\nthis is the prompt")
                response = call_model_with_retries(
                    prompt=prompt,
                    model_name=model_name,
                    call_type=call_type
                )
                
                # Assuming the model returns the result in 'text' or 'message' field
                classification_output = response["choices"][0]["message"]["content"]
                print(f"{classification_output}\n\nthis is the classification_output")
                score = extract_score(classification_output)
                if score is None:
                    score = -1
                print(f"{score}\n\nthis is the score")
                
                # Storing the result with its classification
                entry = {
                    "question": ans["question"],
                    "answer": ans["answer"],
                    "category": ans["category"],
                    "model_output": classification_output,
                    "score": score,
                }
                categorized_data[layer_key][multiplier_key].append(entry)

    # Store the data
    with open(f"{directory}_{question_type}categorized_results.json", "w") as file:
        json.dump(categorized_data, file)

    return categorized_data

directory = "vanilla_steering"
question_types = ["", "vanilla_", "opposite_", "vanilla_opposite_"]
for question_type in question_types:
    results = json.load(open(f"{directory}/{question_type}results.json", "r"))
    with open("lat/evaluation_prompt.txt", "r") as f:
        classifier_prompt = f.read()
    model_name = "gpt-4"
    call_type = "sample" # or "logprobs"
    categorized_results = categorize_results(results, classifier_prompt, model_name, call_type, directory, question_type)
    