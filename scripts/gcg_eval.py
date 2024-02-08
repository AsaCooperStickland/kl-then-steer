import random
from typing import Optional, Callable, Tuple
from baselines import autoDAN, jailbreak_utils
import json
from transformers import AutoModelForCausalLM, AutoTokenizer
import os
import argparse
import torch
from lat import format_utils 
from lat.utils import formatted_system_prompt

def eval_gcg_difficulty(test_model, #an HF causal LM
                unaligned_model, #an HF causal LM
                test_tokenizer, #an HF tokenizer
                unaligned_tokenizer,  #an HF tokenizer
                test_queries_json_path: str,
                output_json_path: str,
                jailbreak_length: int = 20,
                num_target_tokens: int = 12,
                num_continuation_tokens: int = 10, # number of additional tokens beyond the target on which to evaluate (generated from the unaligned model)
                target_only: bool = False,
                gcg_epoch_budget: int = 500,
                formatted_system_prompt: str = formatted_system_prompt,
                connecting_string: str = ' ' + '[/INST]' + '\n', #' ' or newline after?
                gcg_kwargs: Optional[dict] = None, # If None, use default kwargs
                jailbreak_target_generator: Optional[Callable] = jailbreak_utils.generate_target_string,
) -> Tuple[float, list, list]:
    questions = format_utils.json_to_questions(test_queries_json_path)
    print(f"Loaded {len(questions)} questions from {test_queries_json_path}")
    if not gcg_kwargs:
        llama_tokenizer_correction = 'llama' in str(type(test_model))
        gcg = autoDAN.PromptOptimizer(test_model, test_tokenizer, n_epochs=gcg_epoch_budget, n_proposals=400, verbose=False,
                              revert_on_loss_increase=False, llama_tokenizer_correction=llama_tokenizer_correction,) #default n_proposals is 500 in the paper, 400 used here for memory constraints on A100
    else:
        gcg = autoDAN.PromptOptimizer(test_model, test_tokenizer, **gcg_kwargs)
    
    full_unaligned_responses = [jailbreak_target_generator(question, unaligned_model, unaligned_tokenizer, target_tokens=num_target_tokens+num_continuation_tokens) for question in questions]
    targets = [test_tokenizer.decode(test_tokenizer.encode(response, add_special_tokens=False)[:num_target_tokens]) for response in full_unaligned_responses]

    print(f"Targets generated e.g. {targets[0]} with full unaligned response e.g. {full_unaligned_responses[0]}")
    losses, sample_outputs, jailbreaks = [], [], []
    for i, question in enumerate(questions):
        target = targets[i]
        print(f"Optimizing for question {i+1}/{len(questions)}")
        initial_jailbreak = autoDAN.get_jailbreak_init(test_tokenizer, jailbreak_length)
        initial_input = initial_jailbreak
        preamble = formatted_system_prompt + question
        _, jailbreak, target_loss, _ = gcg.optimize(preamble, initial_input, connecting_string, target, early_stop=0.01) #Note this is optimizing as post-query jailbreak a la GCG paper, pre-query jailbreak is also possible
        if target_only:
            loss = target_loss.item()
        else:
            loss = autoDAN.forward_loss(test_model, (preamble+jailbreak+connecting_string, full_unaligned_responses[i]), test_tokenizer,)[1].item()
        losses.append(loss)
        jailbreaks.append(jailbreak)
        sample_outputs.append(jailbreak_utils.sample_response(test_model, test_tokenizer, preamble+jailbreak+connecting_string, num_target_tokens+num_continuation_tokens))

    with open(output_json_path, 'a') as file:
        for i, question in enumerate(questions):
            data = {
                "question": question,
                "target": targets[i],
                "loss": losses[i],
                "jailbreak": jailbreaks[i],
                "sample_output": sample_outputs[i],
            }
            file.write(json.dumps(data) + "\n")
    print(f"Results written to {output_json_path}")
    mean_loss = sum(losses)/len(losses)
    print(f"Average loss: {mean_loss}")
    return mean_loss, losses, sample_outputs

def parse_args():
    parser = argparse.ArgumentParser(description="Evaluate GCG difficulty for a set of questions against a given model.")
    
    parser.add_argument("-m", "--test_model", type=str, default='/vast/work/public/ml-datasets/llama-2/Llama-2-7b-chat-hf/', help="Path to the test model. Default is '/vast/work/public/ml-datasets/llama-2/Llama-2-7b-chat-hf/'.")
    parser.add_argument("-u", "--unaligned_model", type=str, default='teknium/OpenHermes-2.5-Mistral-7B', help="Path to the unaligned model. Default is 'teknium/OpenHermes-2.5-Mistral-7B'.")
    parser.add_argument("-t", "--test_tokenizer", type=str, default='/vast/work/public/ml-datasets/llama-2/Llama-2-7b-chat-hf/', help="Path to the test tokenizer. Default is '/vast/work/public/ml-datasets/llama-2/Llama-2-7b-chat-hf/'.")
    parser.add_argument("-n", "--unaligned_tokenizer", type=str, default='teknium/OpenHermes-2.5-Mistral-7B', help="Path to the unaligned tokenizer. Default is 'teknium/OpenHermes-2.5-Mistral-7B'.")
    parser.add_argument("-c", "--cache_dir", type=str, default=os.getenv('SCRATCH', './'), help="Cache directory for saving and loading models. Default is $SCRATCH or './' if $SCRATCH is not set.")
    parser.add_argument("-j", "--jailbreak_length", type=int, default=20, help="Length of the jailbreak initialization. Default is 20.",)
    parser.add_argument("-tt", "--num_target_tokens", type=int, default=12, help="Number of target tokens to generate. Default is 12.",)
    parser.add_argument("-ct", "--num_continuation_tokens", type=int, default=10, help="Number of continuation tokens. Default is 10.")
    parser.add_argument("-to", "--target_only", action='store_true', help="Evaluate loss on the target only. Default is False.",)
    parser.add_argument("-g", "--gcg_epoch_budget", type=int, default=500, help="GCG epoch budget. Default is 500.",)
    parser.add_argument("-sp", "--system_prompt", type=str, default=formatted_system_prompt, help="Formatted system prompt. Default is defined by SYSTEM_PROMPT variable.",)
    parser.add_argument("-cs", "--connecting_string", type=str, default=' [/INST]\n', help="Connecting string used between parts of the input. Default is ' [/INST]\\n'.",)
    parser.add_argument("-k", "--gcg_kwargs", type=json.loads, default=None, help="Additional kwargs for GCG in JSON format. Default is None.")
    parser.add_argument("-q", "--test_queries_json_path", type=str, required=True, help="Path to the JSON file containing test queries.")
    parser.add_argument("-o", "--output_json_path", type=str, required=True, help="Path to output the results in JSON format.")
    return parser.parse_args()

def main():
    args = parse_args()

    test_model = AutoModelForCausalLM.from_pretrained(
        args.test_model,
        cache_dir=args.cache_dir,
        torch_dtype=torch.float16,
    ).to('cuda')
    unaligned_model = AutoModelForCausalLM.from_pretrained(
        args.unaligned_model,
        cache_dir=args.cache_dir,
        torch_dtype=torch.float16,
    ).to('cuda')
    
    test_tokenizer = AutoTokenizer.from_pretrained(
        args.test_tokenizer,
        cache_dir=args.cache_dir,
    )
    unaligned_tokenizer = AutoTokenizer.from_pretrained(
        args.unaligned_tokenizer,
        cache_dir=args.cache_dir,
    )

    eval_gcg_difficulty(
        test_model=test_model,
        unaligned_model=unaligned_model,
        test_tokenizer=test_tokenizer,
        unaligned_tokenizer=unaligned_tokenizer,
        test_queries_json_path=args.test_queries_json_path,
        output_json_path=args.output_json_path,
        jailbreak_length=args.jailbreak_length,
        num_target_tokens=args.num_target_tokens,
        num_continuation_tokens=args.num_continuation_tokens,
        target_only=args.target_only,
        gcg_epoch_budget=args.gcg_epoch_budget,
        formatted_system_prompt=args.system_prompt,
        connecting_string=args.connecting_string,
        gcg_kwargs=args.gcg_kwargs
    )
    return

if __name__ == "__main__":
    main()