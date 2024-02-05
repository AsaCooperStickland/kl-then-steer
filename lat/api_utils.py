from collections import defaultdict
import concurrent.futures
import math
import json
import time
import re
import asyncio
import os
from collections import namedtuple
from typing import Dict, List, Tuple, Union
from dataclasses import dataclass
from tqdm import tqdm
from dotenv import load_dotenv

from anthropic import Anthropic
from openai import OpenAI, BadRequestError


MAX_NUM_RETRIES = 5
CHAT_MODELS = ['gpt-3.5-turbo-16k-0613', 'gpt-4', 'gpt-4-1106-preview', 'gpt-4-0125-preview',
               'gpt-3.5-turbo', 'gpt-3.5-turbo-0125', 'gpt-3.5-turbo-1106']
OPENAI_MODELS = ['text-ada-001', 'text-babbage-001', 'text-curie-001',
                 'text-davinci-002', 'text-davinci-003'] + CHAT_MODELS
ANTHROPIC_MODELS = ['claude-2']
ANYSCALE_MODELS = ['HuggingFaceH4/zephyr-7b-beta', 'mistralai/Mistral-7B-Instruct-v0.1', 'mistralai/Mixtral-8x7B-Instruct-v0.1']
Example = namedtuple('Example', [
                     'question', 'choice1', 'choice2', 'choice3', 'choice4', 'correct_index'])

load_dotenv()
openai_key = os.getenv("OPENAI_API_KEY")
anyscale_token = os.getenv("ANYSCALE_TOKEN")

anthropic = Anthropic(api_key=os.getenv("ANTHROPIC_API_KEY"))
client = OpenAI(api_key=openai_key,
                organization='org-4L2GWAH28buzKOIhEAb3L5aq')
anyscale_client = OpenAI(
           base_url = "https://api.endpoints.anyscale.com/v1",
           api_key=anyscale_token)


@dataclass
class ModelResponse:
    model_name: str
    content: str
    stop_reason: str


def get_generic_response(response, model_name):
    if isinstance(response, BadRequestError):
        model_response = ModelResponse(
            model_name=model_name,
            content="",
            stop_reason="bad_request_error",
        )
        return model_response
    if model_name in CHAT_MODELS or model_name in ANYSCALE_MODELS:
        content = response.choices[0].message.content
        stop_reason = response.choices[0].finish_reason
        model_response = ModelResponse(
            model_name=model_name,
            content=content,
            stop_reason=stop_reason,
        )
    elif model_name in ANTHROPIC_MODELS:
        content = response.completion
        stop_reason = response.stop_reason
        model_response = ModelResponse(
            model_name=model_name,
            content=content,
            stop_reason=stop_reason,
        )
    else:
        raise ValueError(f"Model {model_name} not supported.")
    return model_response


def extract_score(classification_str):
    """Extracts the numerical score from the classification string."""
    match = re.search(r'"score": (\d+)', classification_str)
    if match:
        return int(match.group(1))
    match = re.search(r"'score': (\d+)", classification_str)
    if match:
        return int(match.group(1))
    return None  # or return 0 or any default value


do_cache = True
if do_cache:
    # load cache if it exists
    if os.path.exists('cache.json'):
        with open('cache.json', 'r') as f:
            CACHE = json.load(f)
    else:
        # Global cache for model responses
        CACHE = defaultdict(dict)
else:
    CACHE = defaultdict(dict)


def batch_prompts(prompts, batch_size=5):
    """Batches prompts."""
    for i in range(0, len(prompts), batch_size):
        max_index = min(i + batch_size, len(prompts))
        yield prompts[i: max_index]


def call_model_with_retries_batched(batched_prompts, model_name, call_type, temperature=0.0):
    """Calls the model with retries for batched prompts and caches the results."""
    responses = []

    def format_key(model_name, prompt):
        return f"MODEL_NAME: {model_name} PROMPT: {prompt}"
    
    def model_response_to_json(model_response):
        return {
            "model_name": model_response.model_name,
            "content": model_response.content,
            "stop_reason": model_response.stop_reason,
        }
    
    def json_to_model_response(json_response):
        return ModelResponse(
            model_name=json_response["model_name"],
            content=json_response["content"],
            stop_reason=json_response["stop_reason"],
        )
    
    for prompts in tqdm(batched_prompts):
        # Check cache first
        cached_responses = [json_to_model_response(CACHE[format_key(model_name, prompt)])
                            for prompt in prompts if format_key(model_name, prompt) in CACHE]
        cached_prompts = [prompt for prompt in prompts if format_key(model_name, prompt) in CACHE]
        uncached_prompts = [prompt for prompt in prompts if format_key(model_name, prompt) not in CACHE]

        if uncached_prompts:
            model_responses = call_model_with_retries(
                prompt=uncached_prompts,
                model_name=model_name,
                call_type=call_type,
                temperature=temperature,
            )
            
            for prompt, response in zip(uncached_prompts, model_responses):
                response = get_generic_response(response, model_name)
                CACHE[format_key(model_name, prompt)] = model_response_to_json(response)
                responses.append(response)
            # save cache
            if do_cache:
                with open('cache.json', 'w') as f:
                    json.dump(CACHE, f)
                
        for cached_prompt, cached_response in zip(cached_prompts, cached_responses):
            if not isinstance(cached_response, ModelResponse):
                cached_response = get_generic_response(
                    cached_response, model_name)
                CACHE[format_key(model_name, cached_prompt)] = cached_response
            responses.append(cached_response)
        # responses.extend(cached_responses)

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
            response = select_and_call_model(
                prompt, model_name, call_type, temperature, stop)
        except Exception as e:
            # return BadRequestError if we encounter it
            if isinstance(e, BadRequestError) and len(prompt) == 1:
                return [e]
            if num_retries == MAX_NUM_RETRIES:
                raise e
            print(
                f"Error calling model {model_name}: {e}, sleeping for {math.pow(3, num_retries)} seconds and retrying...")
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
    if model_name in ANYSCALE_MODELS:
        client_to_use = anyscale_client
        print("using any scale")
    else:
        client_to_use = client
    if model_name in OPENAI_MODELS or model_name in ANYSCALE_MODELS:
        if call_type == 'sample' and (model_name in CHAT_MODELS or model_name in ANYSCALE_MODELS):
            # for now, we don't need to use this
            system_message = "You are a very intelligent assistant, who follows instructions directly."
            response = chat_batch_generate(
                prompts, len(prompts), client_to_use, model_name, system_message, temperature=temperature)
        elif call_type == 'sample':
            response = client_to_use.completions.create(
                model=model_name, prompt=prompts, max_tokens=600, temperature=temperature)
        elif call_type == 'logprobs':
            response = client_to_use.completions.create(
                model=model_name, prompt=prompts, max_tokens=0, echo=True, logprobs=5)
    elif model_name in ANTHROPIC_MODELS:
        if call_type == 'logprobs':
            # we don't actually care about this being async - asyncio.run() will block until the call is complete
            response = asyncio.run(anthropic.top_k_log_probs(
                prompt=prompt, model_name=model_name, bearer_auth=False))
        elif call_type == 'sample':
            messages = []
            for prompt in prompts:
                prompt = "\n\nHuman: " + prompt + "\n\nAssistant:"
                messages.append(prompt)
            response = []
            for message in messages:
                print(message)
                response.append(anthropic.completions.create(
                    model=model_name, max_tokens_to_sample=1000, prompt=message, temperature=temperature))
                print("finished")
    else:
        raise ValueError(f"Model {model_name} not supported.")
    return response


def chat_batch_generate(
    messages: list,
    n_threads: int,
    api_client: OpenAI,
    model: str = "gpt-3.5-turbo",
    system_message: str = "You are a helpful assistant.",
    temperature: float = 0.0,
):

    # @retry(
    #     wait=wait_random_exponential(min=3, max=60),
    #     stop=stop_after_attempt(2),
    # )
    def retry_with_exp_backoff(func, *args, **kwargs):
        return func(*args, **kwargs)

    def api_call(message):
        response = retry_with_exp_backoff(
            api_client.chat.completions.create,  # type: ignore
            model=model,
            messages=[
                {"role": "system", "content": system_message},
                {"role": "user", "content": message},
            ],
            temperature=temperature,
        )
        return response

    answers = []
    if len(messages) == 1:
        return [api_call(messages[0])]
    with concurrent.futures.ThreadPoolExecutor(max_workers=n_threads) as executor:
        for response in executor.map(api_call, messages):
            answers.append(response)

    return answers
