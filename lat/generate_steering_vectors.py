import torch
import json
import os
from tqdm import tqdm
from dotenv import load_dotenv
from lat.modeling import Llama7BChatHelper
from lat.data import ComparisonDataset


load_dotenv()
token = os.getenv("HF_TOKEN")
system_prompt = "You are a helpful, honest and concise assistant."
data_path = "refusal_data_A_B.json"


def generate_and_save_steering_vectors(
    model, dataset, start_layer=0, end_layer=32, token_idx=-2
):
    layers = list(range(start_layer, end_layer + 1))
    positive_activations = dict([(layer, []) for layer in layers])
    negative_activations = dict([(layer, []) for layer in layers])
    model.set_save_internal_decodings(False)
    model.reset_all()
    for p_tokens, n_tokens in tqdm(dataset, desc="Processing prompts"):
        p_tokens = p_tokens.to(model.device)
        n_tokens = n_tokens.to(model.device)
        model.reset_all()
        model.get_logits(p_tokens)
        for layer in layers:
            p_activations = model.get_last_activations(layer)
            p_activations = p_activations[0, token_idx, :].detach().cpu()
            positive_activations[layer].append(p_activations)
        model.reset_all()
        model.get_logits(n_tokens)
        for layer in layers:
            n_activations = model.get_last_activations(layer)
            n_activations = n_activations[0, token_idx, :].detach().cpu()
            negative_activations[layer].append(n_activations)
    for layer in layers:
        positive = torch.stack(positive_activations[layer])
        negative = torch.stack(negative_activations[layer])
        vec = (positive - negative).mean(dim=0)
        torch.save(vec, f"vec_layer_{layer}.pt")
        torch.save(positive, f"positive_layer_{layer}.pt")
        torch.save(
            negative,
            f"negative_layer_{layer}.pt",
        )


if __name__ == "__main__":
    model = Llama7BChatHelper(token, system_prompt)
    data = []
    with open(data_path, "r") as f:
        data = json.load(f)
    dataset = ComparisonDataset(data, system_prompt)
    generate_and_save_steering_vectors(model, dataset)