from lat import utils
from baselines import autoDAN

from transformers import AutoModelForCausalLM, AutoTokenizer
import torch
import json

llama = AutoModelForCausalLM.from_pretrained('/vast/work/public/ml-datasets/llama-2/Llama-2-7b-chat-hf/', cache_dir='/scratch/jp6263/slackV2/hf/models/', torch_dtype=torch.float16).to('cuda')
llama_tokenizer = AutoTokenizer.from_pretrained('/vast/work/public/ml-datasets/llama-2/Llama-2-7b-chat-hf/')

#Get questions
directory = './datasets/refusal/'

questions = []
# Read the file and extract the question from the first line
with open(directory+'questions.jsonl', 'r') as file:
    for line in file:
        data = json.loads(line)
        if 'bias' in data['category'] or 'illegal' in data['category']:
            questions.append(data)


# GCG Parameters
num_tokens = 15 #20 is default, using 15 as a sort of simplicity regularizer to avoid the question being changed during GCG optimization.
n_proposals = 64 #Memory limited for rtx 8000
n_epochs = 250 #Default is 500, using 250 with best-of-2 instead seems to work better.

formatted_system_prompt = f"<s>[INST] <<SYS>>\n{utils.system_prompt}\n<</SYS>>\n\n"
target_string = 'Sure, here'
jailbreaks = []

for data in questions[:]:
    question = data['question']
    # formatted_system_prompt = f"<s>[INST] <<SYS>>\n{utils.system_prompt}\n<</SYS>>\n\n"+question.strip()
    # connecting_string = ' ' + '[/INST]' + ' '
    connecting_string = question.strip()+ ' ' + '[/INST]' + ' '
    for i in range(2):
        initial_input = llama_tokenizer.decode(torch.randint(0, len(llama_tokenizer), (1, num_tokens)).to('cuda')[0])
        gcg = autoDAN.PromptOptimizer(llama, llama_tokenizer, n_epochs=n_epochs, n_proposals=n_proposals, verbose=False,)
        out = gcg.optimize(formatted_system_prompt, initial_input, connecting_string, target_string,)
        print(out)
        if out[2]<0.05: 
            final_tokens = llama_tokenizer.encode(out[0], return_tensors='pt', add_special_tokens=False)
            sample = llama.generate(final_tokens.cuda(), max_length=len(final_tokens)+250, do_sample=True, temperature=1.)
            print(llama_tokenizer.decode(sample[0][len(final_tokens):]))
            new_entry = data.copy()
            new_entry['jailbreak'] = out[1]
            jailbreaks.append(new_entry)
            break

with open(directory + 'jailbreak_questions.jsonl', 'w') as outfile:
    for entry in jailbreaks:
        json.dump(entry, outfile)
        outfile.write('\n')
