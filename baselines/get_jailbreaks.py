import argparse
from lat import utils
from baselines import autoDAN
from transformers import AutoModelForCausalLM, AutoTokenizer
import torch
import json

def main(args):
    llama = AutoModelForCausalLM.from_pretrained(args.model_path, cache_dir=args.cache_dir, torch_dtype=torch.float16).to('cuda')
    llama_tokenizer = AutoTokenizer.from_pretrained(args.model_path)

    # Get questions
    questions = []
    with open(args.question_file, 'r') as file:
        for line in file:
            data = json.loads(line)
            if any(category in data['category'] for category in args.categories):
                questions.append(data)

    formatted_system_prompt = f"<s>[INST] <<SYS>>\n{utils.system_prompt}\n<</SYS>>\n\n"
    jailbreaks = []

    for data in questions:
        question = data['question']
        connecting_string = question.strip() + ' ' + '[/INST]' + ' '
        for i in range(args.BON):
            initial_input = llama_tokenizer.decode(torch.randint(0, len(llama_tokenizer), (1, args.num_tokens)).to('cuda')[0])
            gcg = autoDAN.PromptOptimizer(llama, llama_tokenizer, n_epochs=args.n_epochs, n_proposals=args.n_proposals, verbose=False)
            out = gcg.optimize(formatted_system_prompt, initial_input, connecting_string, args.target_string)
            print(out)
            if out[2] < args.early_stop: 
                new_entry = data.copy()
                new_entry['jailbreak'] = out[1]
                jailbreaks.append(new_entry)
                break

    with open(args.output_file, 'w') as outfile:
        for entry in jailbreaks:
            json.dump(entry, outfile)
            outfile.write('\n')

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Script for finding jailbreaks for refusal questions")
    parser.add_argument('--model_path', type=str, default='/vast/work/public/ml-datasets/llama-2/Llama-2-7b-chat-hf/', help='Path to the model')
    parser.add_argument('--cache_dir', type=str, default='/scratch/jp6263/slackV2/hf/models/', help='Cache directory for the model')
    parser.add_argument('--question_file', type=str, default='./datasets/refusal/questions.jsonl', help='Path to the question file')
    parser.add_argument('--output_file', type=str, default='./datasets/refusal/jailbreak_questions.jsonl', help='Path for the output file')
    parser.add_argument('--num_tokens', type=int, default=10, help='Number of tokens for jailbreak')
    parser.add_argument('--n_proposals', type=int, default=64, help='Number of proposals for GCG (batch size)')
    parser.add_argument('--n_epochs', type=int, default=250, help='Number of epochs for GCG')
    parser.add_argument('--BON', type=int, default=2, help='Number of runs of GCG')
    parser.add_argument('--categories', nargs='+', default=['bias', 'illegal'], help='List of categories to filter questions')
    parser.add_argument('--target_string', type=str, default='Sure, here', help='Target string for optimization')
    parser.add_argument('--early_stop', type=float, default=0.05, help='Early stopping threshol for GCG')

    args = parser.parse_args()
    main(args)
