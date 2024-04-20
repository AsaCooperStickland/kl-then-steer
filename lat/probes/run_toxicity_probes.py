from transformer_lens import HookedTransformer
import json
from collections import defaultdict
import os
from tqdm import tqdm

from sklearn.preprocessing import LabelEncoder

import torch
import numpy as np
import pickle
import pandas as pd
import configparser
import argparse
import fire

from transformers import AutoModelForCausalLM, AutoTokenizer, AutoConfig
from transformers import LlamaForCausalLM, LlamaTokenizer

from transformers.generation import GenerationConfig

from functools import partial

from probe import *
from utils.config_utils import *
from utils.data_utils import *
from utils.model_utils import *
from configs import DataConfig, model_config, model_lookup
from lat.format_utils import ProbeQuestionAugmenter, QuestionAugmenter
from lat.utils import jailbreaks_path

from sklearn.model_selection import train_test_split
import random

from pathlib import Path

random.seed(model_config.seed)

def main(**kwargs):
    update_config((DataConfig, model_config), **kwargs)

    print(model_config)
    print(model_config.weight_decay, type(model_config.weight_decay))
    data_config = DataConfig()
    print(data_config)

    # probe_type = f'{data_config.toxic_data[0]}_{data_config.toxic_data[-1]}_v_{data_config.normal_data[0]}_{data_config.normal_data[-1]}'
    probe_type = "vanilla"

    probe_dir = os.path.join(data_config.probe_dir, probe_type) 
    results_dir = os.path.join(data_config.results_dir, probe_type) 
    predictions_dir = os.path.join(data_config.predictions_dir, probe_type) 
    

    if not os.path.exists(os.path.join(probe_dir, model_config.model)):
        os.makedirs(os.path.join(probe_dir, model_config.model))
        print("Creating", os.path.join(probe_dir, model_config.model))
    
    if not os.path.exists(os.path.join(results_dir, model_config.model)):
        os.makedirs(os.path.join(results_dir, model_config.model))
        print("Creating", os.path.join(results_dir, model_config.model))
    
    if not os.path.exists(os.path.join(predictions_dir, model_config.model)):
        os.makedirs(os.path.join(predictions_dir, model_config.model))
        print("Creating", os.path.join(predictions_dir, model_config.model))
    

    tokenizer, model = get_model_tokenizer(model_config.model)

    layers = model_lookup[model_config.model]['layers']
    X = {}
    y = {}
    if data_config.toxic_data is None or data_config.normal_data is None:
        augmenter = ProbeQuestionAugmenter(dataset_path="datasets", 
                                    jailbreaks_path="/scratch/alc9734/latent-adversarial-training/datasets/refusal/jailbreaks_extra.json",
                                    jinja_directory="/scratch/alc9734/llm-jailbreaks/prompts/wei-jailbreaks/", jinja_subset="train", questions_file=f"probing/training/{data_config.data_type}")
        category_keys = augmenter.get_all_category_keys()
        toxic_data = set(category_keys["all_toxic_categories"])
        normal_data = set(category_keys["all_normal_categories"])
        test_augmenter = QuestionAugmenter(dataset_path="datasets", 
                                  jailbreaks_path=jailbreaks_path,
                                  jinja_directory="/scratch/alc9734/llm-jailbreaks/prompts/wei-jailbreaks/",
                    output_dataset_path="datasets/probing/testing")
        test_category_keys = test_augmenter.get_all_category_keys()
        for k, v in test_category_keys.items():
            toxic_data.update(v)
        output_file_path = "question"
        mt_augmenter = ProbeQuestionAugmenter(dataset_path="FastChat/fastchat/llm_judge/data/mt_bench/", 
                    jailbreaks_path=jailbreaks_path,
                    jinja_directory="/scratch/alc9734/llm-jailbreaks/prompts/wei-jailbreaks/", jinja_subset="test", questions_file=output_file_path,
                    output_dataset_path="datasets/probing/mt_bench", assume_normal=True)
        mt_category_keys = mt_augmenter.get_all_category_keys()
        for k, v in mt_category_keys.items():
            normal_data.update(v)
        toxic_data = list(toxic_data)
        normal_data = list(normal_data)

        if data_config.toxic_data is None:
            data_config.toxic_data = toxic_data
        if data_config.normal_data is None:
            data_config.normal_data = normal_data
    #Get all of the past data
    for domain in data_config.domains:
        X[domain], y[domain] = {}, {}
        if domain == "testing":
            full_dataset = defaultdict(list)
            with open(os.path.join(data_config.data_dir, domain, f'augmented_questions.jsonl'), 'r') as f:
                for line in f:
                    line = json.loads(line)
                    print(line)
                    full_dataset[line['source']].append(line)
        elif domain == "mt_bench":
            full_dataset = json.load(open(os.path.join(data_config.data_dir, domain, f'question_questions.json'), 'r'))
        else:
            full_dataset = json.load(open(os.path.join(data_config.data_dir, domain, f'{data_config.data_type}_questions.json'), 'r'))
        # print(full_dataset.keys())
        # raise ValueError
        for source in data_config.toxic_data + data_config.normal_data:
            # dataset = json.load(open(os.path.join(data_config.data_dir, source, f'{domain}_{data_config.data_type}_questions.json'), 'r'))
            print(f"Getting activations for {source}")
            if source not in full_dataset:
                print(f"Warning: Skipping {source}")
                continue
            dataset = full_dataset[source]
            X[domain][source], y[domain][source] = {}, {}
            
            if not os.path.exists(os.path.join(data_config.activations_dir, source, model_config.model)):
                os.makedirs(os.path.join(data_config.activations_dir, source, model_config.model))
            for layer in layers:
                activations_file = os.path.join(data_config.activations_dir, source, model_config.model, f'{domain}_layer{layer}_activations.npy')
                if Path(activations_file).exists() and domain != "training":
                    print(f"Loading activations from {activations_file}")
                    X[domain][source][layer] = load_activations(activations_file)
                else:
                    X[domain][source][layer] = get_activations(model, tokenizer, dataset, layer, activations_file)

                #if source in data_config.toxic_data:
                #    y[source][domain][layer] = np.zeros(X[source][domain][layer].shape[0])
                #else:
                #    y[source][domain][layer] = np.ones(X[source][domain][layer].shape[0])
    
    #Train single domain only probes          
    if model_config.single_topic_probe:
        print("TRAINING SINGLE domain PROBES")
        single_topic_results = pd.DataFrame(columns = ['train_domain', 'layer', 'test_domain', 'test_score', 'train_size', 'test_size'])   
        for train_domain in data_config.train_domains:
            for l in layers:
                X_train, X_test, y_train, y_test = get_single_domain_data(X, data_config, train_domain, layer, model_config.seed)

                #Train probe
                probe_path = os.path.join(probe_dir, model_config.model, f'{train_domain}_layer{l}_probe_l2_{model_config.weight_decay}.pt')
                
                if os.path.exists(probe_path):
                    print(f"Loading probe from {probe_path}")
                    trained_probe = load_probe(probe_path) 
                else:
                    print(f"Training probe for {model} layer {l} with l2 {model_config.weight_decay}")
                    trained_probe = train_probe(X_train, y_train, model_config.device, model_config.weight_decay, probe_path)
                
                score = trained_probe.score(X_test, y_test.astype(np.int64))
                #predictions = trained_probe.predict(X_test, y_test.astype(np.int64))
                
                add = {'train_domain': train_domain,
                        'layer':l,
                        'test_domain': train_domain,
                        'test_score':score,
                        'train_size': X_train.shape[0],
                        'test_size': X_test.shape[0]}

                print(f"TEST ACCURACY {train_domain} LAYER {l}: {score}")
                single_topic_results = single_topic_results._append(add, ignore_index = True)
                for domain in data_config.test_domains:
                    _, X_test, _, y_test = get_single_domain_data(X, data_config, domain, layer, model_config.seed, test_size=1.0)
                    score = trained_probe.score(X_test, y_test.astype(np.int64))
                    add = {'train_domain': train_domain,
                            'layer':l,
                            'test_domain':domain,
                            'test_score':score,
                            'train_size': X_train.shape[0],
                            'test_size': X_test.shape[0]}
                    print(f"TEST ACCURACY {domain} LAYER {l}: {score}")
                    single_topic_results = single_topic_results._append(add, ignore_index = True)
                    if domain == "testing":
                        for source_name, sources in test_category_keys.items():
                            _, X_test, _, y_test = get_single_domain_data(X, data_config, domain, layer, model_config.seed, test_size=1.0, custom_toxic_sources=sources)
                            if X_test is None:
                                print(f"Skipping {source_name}")
                                continue
                            score = trained_probe.score(X_test, y_test.astype(np.int64))
                            add = {'train_domain': train_domain,
                                    'layer':l,
                                    'test_domain':f"{domain}_{source_name}",
                                    'test_score':score,
                                    'train_size': X_train.shape[0],
                                    'test_size': X_test.shape[0]}
                            print(f"TEST ACCURACY {source_name} LAYER {l}: {score}")
                            single_topic_results = single_topic_results._append(add, ignore_index = True)

        single_topic_results.to_csv(os.path.join(results_dir, model_config.model, f'single_topic_l2_{model_config.weight_decay}_results.csv'), index = False)
    
    #Train hold one domain out
    if model_config.hold_one_out_probe:
        print("TRAINING HOLD ONE OUT domain PROBES")
        hold_one_out_results = pd.DataFrame(columns = ['train_topic', 'layer', 'test_topic', 'test_score', 'train_size', 'test_size'])
        
        for domain in data_config.domains:
            print("domain", domain)
            for l in layers:
                print("LAYER", l)
                X_train, X_test, y_train, y_test = get_hold_one_out_data(X, data_config, domain, layer, model_config.seed)
                
                #Train probe
                probe_path = os.path.join(probe_dir, model_config.model, f'hold_out_{domain}_layer{l}_probe_l2_{model_config.weight_decay}.pt')
                
                if os.path.exists(probe_path):
                    print(f"Loading probe from {probe_path}")
                    trained_probe = load_probe(probe_path) 
                else:
                    print(f"Training probe for {model} layer {l} with l2 {model_config.weight_decay}")
                    trained_probe = train_probe(X_train, y_train, model_config.device, model_config.weight_decay, probe_path)
                
                score = trained_probe.score(X_test, y_test.astype(np.int64))

                add = {'train_topic': 'mixed',
                        'layer':l,
                        'test_topic':domain,
                        'test_score':score,
                        'train_size': X_train.shape[0],
                        'test_size': X_test.shape[0]}

                print(f"TEST ACCURACY {domain} LAYER {l}: {score}")
                hold_one_out_results = hold_one_out_results._append(add, ignore_index = True)

                if model_config.get_predictions:
                    predictions = trained_probe.predict(X_test)
                    y_test = y_test.reshape(-1,1)
                    predictions = np.concatenate([predictions, y_test], axis = 1)

                    np.save(os.path.join(predictions_dir, model_config.model, f'{domain}_layer{l}_l2_{model_config.weight_decay}_preds.npz'), predictions)

        hold_one_out_results.to_csv(os.path.join(results_dir, model_config.model, f'hold_one_out_l2_{model_config.weight_decay}_results.csv'), index = False)

    
    #Train mixed probe 
    if model_config.mixed_probe:
        print("TRAINING MIXED PROBES")
        mixed_results = pd.DataFrame(columns = ['train_topic', 'layer', 'test_topic', 'test_score', 'train_size', 'test_size'])

        for l in layers:
            X_train, X_test, y_train, y_test = get_mixed_data(X, data_config, layer, model_config.seed)
            
            #Train probe
            probe_path = os.path.join(probe_dir, model_config.model, f'mixed_layer{l}_probe_l2_{model_config.weight_decay}.pt')
            
            if os.path.exists(probe_path):
                print(f"Loading probe from {probe_path}")
                trained_probe = load_probe(probe_path) 
            else:
                print(f"Training probe for {model} layer {l} with l2 {model_config.weight_decay}")
                trained_probe = train_probe(X_train, y_train, model_config.device, model_config.weight_decay, probe_path)
            
            score = trained_probe.score(X_test, y_test.astype(np.int64))

            add = {'train_topic': 'all',
                    'layer':l,
                    'test_topic': 'all',
                    'test_score':score,
                    'train_size': X_train.shape[0],
                    'test_size': X_test.shape[0]}

            print(f"TEST ACCURACY {domain} LAYER {l}: {score}")
            mixed_results = mixed_results._append(add, ignore_index = True)
    
        mixed_results.to_csv(os.path.join(results_dir, model_config.model, f'mixed_l2_{model_config.weight_decay}_results.csv'), index = False)

if __name__ == "__main__":
    fire.Fire(main)

# if __name__ == "__main__":
#     parser = argparse.ArgumentParser()
#     parser.add_argument("--model", type = str)
#     parser.add_argument("--probe_type", type = str)
#     parser.add_argument("--weight_decay", type = float)

#     args = parser.parse_args()

#     main(args.model,
#          args.probe_type,
#          float(args.weight_decay))



