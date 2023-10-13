import torch
from torch.utils.data import Dataset
from transformers import AutoTokenizer
import os
from dotenv import load_dotenv


load_dotenv()
token = os.getenv("HF_TOKEN")


def prompt_to_tokens(tokenizer, system_prompt, instruction, model_output, prompt_beginning):
    B_INST, E_INST = "[INST]", "[/INST]"
    B_SYS, E_SYS = "<<SYS>>\n", "\n<</SYS>>\n\n"
    dialog_content = B_SYS + system_prompt + E_SYS + prompt_beginning + instruction.strip()
    dialog_tokens = tokenizer.encode(
        f"{B_INST} {dialog_content.strip()} {E_INST} {model_output.strip()}"
    )
    return torch.tensor(dialog_tokens).unsqueeze(0)


class ComparisonDataset(Dataset):
    def __init__(self, data, system_prompt, prompt_beginning=""):
        self.data = data
        self.system_prompt = system_prompt
        self.prompt_beginning = prompt_beginning
        self.tokenizer = AutoTokenizer.from_pretrained(
            "meta-llama/Llama-2-7b-chat-hf", use_auth_token=token
        )
        self.tokenizer.pad_token = self.tokenizer.eos_token

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        item = self.data[idx]
        question = item["question"]
        pos_answer = item["answer_matching_behavior"]
        neg_answer = item["answer_not_matching_behavior"]
        pos_tokens = prompt_to_tokens(
            self.tokenizer,
            self.system_prompt,
            question,
            pos_answer,
            self.prompt_beginning,
        )
        neg_tokens = prompt_to_tokens(
            self.tokenizer,
            self.system_prompt,
            question,
            neg_answer,
            self.prompt_beginning,
        )
        return pos_tokens, neg_tokens
