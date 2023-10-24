import json
from collections import defaultdict
import pickle
import concurrent.futures
import math
import time
import re
import asyncio
import os
from collections import namedtuple
from typing import Dict, List, Tuple, Union
from tqdm import tqdm
from dotenv import load_dotenv
from tenacity import retry, wait_random_exponential, stop_after_attempt

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


# load pickled cache if it exists
if os.path.exists('cache.pkl'):
    with open('cache.pkl', 'rb') as f:
        CACHE = pickle.load(f)
else:
    # Global cache for model responses
    CACHE = defaultdict(dict)

def batch_prompts(prompts, batch_size=5):
    """Batches prompts."""
    for i in range(0, len(prompts), batch_size):
        max_index = min(i + batch_size, len(prompts))
        yield prompts[i: max_index]

def call_model_with_retries_batched(batched_prompts, model_name, call_type):
    """Calls the model with retries for batched prompts and caches the results."""
    responses = []
    for prompts in tqdm(batched_prompts):
        # Check cache first
        cached_responses = [CACHE[(model_name, prompt)] for prompt in prompts if (model_name, prompt) in CACHE]
        uncached_prompts = [prompt for prompt in prompts if (model_name, prompt) not in CACHE]

        if uncached_prompts:
            model_responses = call_model_with_retries(
                prompt=uncached_prompts,
                model_name=model_name,
                call_type=call_type
            )
            for prompt, response in zip(uncached_prompts, model_responses):
                CACHE[(model_name, prompt)] = response
                responses.append(response)
            # save cache
            with open('cache.pkl', 'wb') as f:
                pickle.dump(CACHE, f)
        
        responses.extend(cached_responses)

    return responses


def call_model_with_retries(prompt: List[str],
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


def select_and_call_model(prompts: List[str],
                          model_name: str,
                          call_type: str,
                          temperature: float = 0.0,
                          stop: str = None) -> Union[str, Dict[str, List[Union[str, float]]]]:
    """Selects the appropriate model and calls it with the given prompt."""
    if model_name in OPENAI_MODELS:
        if call_type == 'sample' and model_name in CHAT_MODELS:
            # for now, we don't need to use this
            system_message = "You are a very intelligent assistant, who follows instructions directly."
            response = chat_batch_generate(prompts, len(prompts), model_name, system_message)
        elif call_type == 'sample':
            response = openai.Completion.create(model=model_name, prompt=prompts, max_tokens=600, temperature=0.0)
        elif call_type == 'logprobs':
            response = openai.Completion.create(model=model_name, prompt=prompts, max_tokens=0, echo=True, logprobs=5)
    elif model_name in ANTHROPIC_MODELS:
        if call_type == 'logprobs':
            # we don't actually care about this being async - asyncio.run() will block until the call is complete
            response = asyncio.run(anthropic.top_k_log_probs(prompt=prompt, model_name=model_name, bearer_auth=False))
        elif call_type == 'sample':
            messages = []
            for prompt in prompts:
                prompt = "\n\nHuman: " + prompt + "\n\nAssistant:"
                messages.append(prompt)
            response = anthropic.completions.create(model=model_name, max_tokens_to_sample=1000, prompt=messages)
    else:
        raise ValueError(f"Model {model_name} not supported.")
    return response


def chat_batch_generate(
    messages: list,
    n_threads: int,
    model: str = "gpt-3.5-turbo",
    system_message: str = "You are a helpful assistant.",
):

    @retry(
        wait=wait_random_exponential(min=3, max=60),
        stop=stop_after_attempt(2),
    )
    def retry_with_exp_backoff(func, *args, **kwargs):
        return func(*args, **kwargs)

    def api_call(message):
        response = retry_with_exp_backoff(
            openai.ChatCompletion.create,  # type: ignore
            model=model,
            messages=[
                {"role": "system", "content": system_message},
                {"role": "user", "content": message},
            ],
        )
        return response

    answers = []
    with concurrent.futures.ThreadPoolExecutor(max_workers=n_threads) as executor:
        for response in executor.map(api_call, messages):
            answers.append(response)

    return answers


def categorize_results(results, classifier_prompt, model_name, call_type, directory, question_type):
    """Categorizes the results using the classifier and stores them as JSON."""
    categorized_data = {}
    
    all_prompts = []
    for layer_data in results:
        for multiplier_data in layer_data["results"]:
            for ans in multiplier_data["answers"]:
                input_prompt = {"question": ans["question"], "answer": ans["answer"]}
                classification_input = f"\nInput: {input_prompt}"
                prompt = classifier_prompt + classification_input
                prompt = f"{prompt}\nLet's think step by step:"
                all_prompts.append(prompt)
    
    # Send prompts in batches and get responses
    batch_size = 1
    batched_prompts_gen = batch_prompts(all_prompts, batch_size)
    all_responses = call_model_with_retries_batched(batched_prompts_gen, model_name, call_type)

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
                classification_output = response["choices"][0]["message"]["content"]
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
                    "model_output": classification_output,
                    "score": score,
                }
                categorized_data[layer_key][multiplier_key].append(entry)

    # Store the data
    with open(f"{directory}_{question_type}categorized_results.json", "w") as file:
        json.dump(categorized_data, file)

    return categorized_data


def get_average_score_per_layer_per_multiplier(categorized_data):
    """Gets the average score per layer per multiplier."""
    average_scores = {}
    for layer_key, layer_data in categorized_data.items():
        average_scores[layer_key] = {}
        for multiplier_key, multiplier_data in layer_data.items():
            scores = [entry["score"] for entry in multiplier_data if entry["score"] != -1]
            average_scores[layer_key][multiplier_key] = sum(scores) / len(scores)
    return average_scores
    


directory = "vanilla_steering"
question_types = ["", "vanilla_", "opposite_", "vanilla_opposite_"]
# question_types = ["vanilla_"]
for question_type in question_types:
    results = json.load(open(f"{directory}/{question_type}results.json", "r"))
    with open("lat/evaluation_prompt.txt", "r") as f:
        classifier_prompt = f.read()
    model_name = "gpt-3.5-turbo-16k-0613"
    call_type = "sample" # or "logprobs"
    categorized_results = categorize_results(results, classifier_prompt, model_name, call_type, directory, question_type)
    average_scores = get_average_score_per_layer_per_multiplier(categorized_results)
    print(average_scores)
    
# save cache
with open('cache.pkl', 'wb') as f:
    pickle.dump(CACHE, f)
