import torch
import json
import os
from tqdm import tqdm
from dotenv import load_dotenv
from lat.modeling import Llama7BChatHelper
from lat.data.data import ComparisonDataset
from lat.utils import system_prompt, data_path


load_dotenv()
token = os.getenv("HF_TOKEN")


def generate_and_save_steering_vectors(
    model, dataset, output_folder, start_layer=0, end_layer=31, token_idx=-2
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
            p_activations_value = model.get_last_value_activations(layer)
            p_activations_query = model.get_last_query_activations(layer)
            p_activations = p_activations[0, token_idx, :].detach().cpu()
            p_activations_value = p_activations_value[0, token_idx, :].detach(
            ).cpu()
            p_activations_query = p_activations_query[0, token_idx, :].detach(
            ).cpu()
            positive_activations[layer].append(
                (p_activations, p_activations_value, p_activations_query))
        model.reset_all()
        model.get_logits(n_tokens)
        for layer in layers:
            n_activations = model.get_last_activations(layer)
            n_activations_value = model.get_last_value_activations(layer)
            n_activations_query = model.get_last_query_activations(layer)
            n_activations = n_activations[0, token_idx, :].detach().cpu()
            n_activations_value = n_activations_value[0, token_idx, :].detach(
            ).cpu()
            n_activations_query = n_activations_query[0, token_idx, :].detach(
            ).cpu()
            negative_activations[layer].append(
                (n_activations, n_activations_value, n_activations_query))
    for layer in layers:
        positive_activations_layer = [x[0]
                                      for x in positive_activations[layer]]
        positive = torch.stack(positive_activations_layer)
        negative_activations_layer = [x[0]
                                      for x in negative_activations[layer]]
        negative = torch.stack(negative_activations_layer)
        vec = (positive - negative).mean(dim=0)
        positive_activations_value_layer = [x[1]
                                            for x in positive_activations[layer]]
        positive_value = torch.stack(positive_activations_value_layer)
        negative_activations_value_layer = [x[1]
                                            for x in negative_activations[layer]]
        negative_value = torch.stack(negative_activations_value_layer)
        vec_value = (positive_value - negative_value).mean(dim=0)
        positive_activations_query_layer = [x[2]
                                            for x in positive_activations[layer]]
        positive_query = torch.stack(positive_activations_query_layer)
        negative_activations_query_layer = [x[2]
                                            for x in negative_activations[layer]]
        negative_query = torch.stack(negative_activations_query_layer)
        vec_query = (positive_query - negative_query).mean(dim=0)
        for activation_type, vectors in zip(["", "_value", "_query"],
                                            [(vec, positive, negative),
                                             (vec_value, positive_value,
                                              negative_value),
                                             (vec_query, positive_query, negative_query)]):
            vec_, positive_, negative_ = vectors
            torch.save(
                vec_, f"{output_folder}/vec{activation_type}_layer_{layer}.pt")
            torch.save(
                positive_, f"{output_folder}/positive{activation_type}_layer_{layer}.pt")
            torch.save(
                negative_,
                f"{output_folder}/negative{activation_type}_layer_{layer}.pt",
            )


def generate_single_steering_vector(
    model, data, start_layer=0, end_layer=31, token_idx=-2
):
    layers = list(range(start_layer, end_layer + 1))
    activations = {}
    model.set_save_internal_decodings(False)
    model.reset_all()
    p_tokens, n_tokens = data
    p_tokens = p_tokens.to(model.device)
    n_tokens = n_tokens.to(model.device)
    model.reset_all()
    model.get_logits(p_tokens)
    for layer in layers:
        p_activations = model.get_last_activations(layer)
        p_activations = p_activations[0, token_idx, :].detach().cpu()
        activations[layer] = p_activations
    model.reset_all()
    model.get_logits(n_tokens)
    for layer in layers:
        n_activations = model.get_last_activations(layer)
        n_activations = n_activations[0, token_idx, :].detach().cpu()
        activations[layer] = activations[layer] - n_activations
    return activations


if __name__ == "__main__":
    model = Llama7BChatHelper(token, system_prompt)
    with open(data_path, "r") as f:
        data = json.load(f)
    dataset = ComparisonDataset(data, system_prompt)
    output_folder_name = "vanilla_steering"
    os.makedirs(output_folder_name, exist_ok=True)
    generate_and_save_steering_vectors(
        model, dataset, output_folder=output_folder_name)
