import json
import os
from tqdm import tqdm

from sklearn.preprocessing import LabelEncoder

import torch
import numpy as np
import pickle

from transformers import AutoModelForCausalLM, AutoTokenizer, AutoConfig
from transformers import LlamaForCausalLM, LlamaTokenizer
from transformer_lens import HookedTransformer
from sklearn.model_selection import train_test_split
import random

from configs.models import model_config, model_lookup
from probe import *

def get_model_tokenizer(model):
    AutoConfig.token = model_config.hf_token
    hf_name = model_lookup[model]['hf_name']
    tl_name = model_lookup[model]['tl_name']

    if str.find(model_config.model, 'qwen') > -1:
        tokenizer = AutoTokenizer.from_pretrained(hf_name, trust_remote_code=True)
        hf_model = AutoModelForCausalLM.from_pretrained(hf_name,
                                                        device_map=model_config.device,
                                                        fp16=True,
                                                        use_logn_attn=False,
                                                        use_dynamic_ntk = False,
                                                        scale_attn_weights = False,
                                                        trust_remote_code = True
                                                        ).eval()
        tl_model = HookedTransformer.from_pretrained(
                                                    tl_name,
                                                    device=model_config.device,
                                                    fp16=True,
                                                    dtype=torch.float16,
                                                    fold_ln=False,
                                                    center_writing_weights=False, 
                                                    center_unembed=False,
                                                    )
    elif str.find(model_config.model, 'llama')>-1:
        tokenizer = LlamaTokenizer.from_pretrained(hf_name, 
                                                    token = model_config.hf_token, 
                                                    device = model_config.device)
        hf_model = AutoModelForCausalLM.from_pretrained(hf_name, 
                                                        torch_dtype=torch.float16, 
                                                        token = model_config.hf_token)
        tl_model = HookedTransformer.from_pretrained(tl_name, 
                                            hf_model=hf_model, 
                                            device=model_config.device, 
                                            dtype=torch.float16,
                                            fold_ln=False, 
                                            center_writing_weights=False, 
                                            center_unembed=False, 
                                            tokenizer=tokenizer)
    else:
        tokenizer = AutoTokenizer.from_pretrained(hf_name)
        hf_model = AutoModelForCausalLM.from_pretrained(hf_name)
        tl_model = HookedTransformer.from_pretrained(
                                                    tl_name, 
                                                    hf_model=hf_model, 
                                                    device=model_config.device, 
                                                    fold_ln=False, 
                                                    center_writing_weights=False, 
                                                    center_unembed=False, 
                                                    tokenizer=tokenizer)
    return tokenizer, tl_model
    


def load_activations(activations_file):
    X = np.load(activations_file)
    X = X.astype(np.float32)
    return X

def get_activations(model, tokenizer, dataset, layer, activations_file= None):
   
    print(layer)
    X = []
    y = []
    for example in tqdm(dataset):
        # Cache the activations of the model over the example
        tokens = tokenizer(example["question"], return_tensors="pt")['input_ids']
        if tokens.shape[1]>0:
            if tokens.shape[1]>1024:
                tokens = tokens[:,:1024]
            with torch.no_grad():
                output, activations = model.run_with_cache(tokens)
            X.append(activations[f"blocks.{layer}.hook_resid_post"][:, -1].detach().cpu().numpy())
            #y.append(example["label"])
    # print(X)
    if len(X) < 1:
        return None
    if np.array(X).shape[0] == 1:
        X = np.array(X).reshape(-1, np.array(X).shape[-1])
    else:
        X = np.concatenate(X, axis=0)
    nan_idx = np.isnan(X).any(axis=1)
    X = X[~nan_idx]
    X = X.astype(np.float32)

    if activations_file:
        np.save(activations_file, X)
    

    return X 

def load_probe(probe_file_path):
    probe = LinearClsProbe(device = model_config.device, max_iter=1000, verbose=True)
    probe.load(probe_file_path)
    
    return probe


def train_probe(X_train, y_train, device, weight_decay, probe_file_path = None):
    probe = LinearClsProbe(device = model_config.device, max_iter=1000, verbose=True)
    probe.fit(X_train, y_train.astype(np.int64), float(weight_decay))
    probe.save(probe_file_path)

    return probe