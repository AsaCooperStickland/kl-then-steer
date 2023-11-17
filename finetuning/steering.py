import json
import numpy as np
import random
import os
import torch
from tqdm import tqdm

from transformers import AutoTokenizer, pipeline, AutoModelForCausalLM

from repe import repe_pipeline_registry
repe_pipeline_registry()


def primary_emotions_concept_dataset(data_dir, user_tag='', assistant_tag=''):
    # random.seed(0)

    template_str = '{user_tag} Consider the {emotion} of the following scenario:\nScenario: {scenario}\nAnswer: {assistant_tag} '
    emotions = ["happiness", "sadness", "anger", "fear", "disgust", "surprise"]
    raw_data = {}
    for emotion in emotions:
        with open(os.path.join(data_dir, 'emotions', f'{emotion}.json')) as file:
            # raw_data[emotion] = json.load(file)
            raw_data[emotion] = list(set(json.load(file)))[:200]

    formatted_data = {}
    for emotion in emotions:
        c_e, o_e = raw_data[emotion], np.concatenate([v for k,v in raw_data.items() if k != emotion])
        random.shuffle(o_e)

        data = [[c,o] for c,o in zip(c_e, o_e)]
        train_labels = []
        for d in data:
            true_s = d[0]
            random.shuffle(d)
            train_labels.append([s == true_s for s in d])
        
        data = np.concatenate(data).tolist()
        data_ = np.concatenate([[c,o] for c,o in zip(c_e, o_e)]).tolist()
        
        emotion_test_data = [template_str.format(emotion=emotion, scenario=d, user_tag=user_tag, assistant_tag=assistant_tag) for d in data_]
        emotion_train_data = [template_str.format(emotion=emotion, scenario=d, user_tag=user_tag, assistant_tag=assistant_tag) for d in data]

        formatted_data[emotion] = {
            'train': {'data': emotion_train_data, 'labels': train_labels},
            'test': {'data': emotion_test_data, 'labels': [[1,0]* len(emotion_test_data)]}
        }
    return formatted_data


class Steering:
    def __init__(self, model, tokenizer, data_dir):
        self.model = model
        self.tokenizer = tokenizer
        self.rep_token = -1
        self.hidden_layers = list(range(-1, -model.config.num_hidden_layers, -1))
        self.n_difference = 1
        self.direction_method = 'pca'
        self.rep_reading_pipeline = pipeline("rep-reading", model=model, tokenizer=tokenizer, device='cuda')

        self.train_data = primary_emotions_concept_dataset(data_dir, user_tag="[INST]", assistant_tag="[/INST]")
        self.categories = ["happiness", "sadness", "anger", "fear", "disgust", "surprise"]

    def sample_pairs(self, data, num_pairs):
        """
        Sample num_pairs of pairs from data, maintaining the same format.

        :param data: Dictionary with 'labels' and 'data' as keys.
        :param num_pairs: Number of pairs to sample.
        :return: A dictionary with the sampled pairs.
        """

        # Total number of pairs in the original data
        total_pairs = len(data['labels'])

        # Generating random indices for the pairs
        sampled_indices = random.sample(range(total_pairs), num_pairs)

        # Extracting the sampled pairs
        sampled_data = {'labels': [], 'data': []}
        for index in sampled_indices:
            sampled_data['labels'].append(data['labels'][index])

            # Each pair consists of two consecutive elements in the 'data' list
            pair_index = index * 2
            sampled_data['data'].extend(data['data'][pair_index:pair_index+2])

        return sampled_data
    
    def get_shift(self, coeff, layer_id, num_pairs=10):
        category = random.choice(self.categories)
        orig_data = self.train_data[category]['train']
        subsample = self.sample_pairs(orig_data, num_pairs)
        rep_reader = self.rep_reading_pipeline.get_directions(
            subsample['data'],
            rep_token=self.rep_token, 
            hidden_layers=self.hidden_layers, 
            n_difference=self.n_difference, 
            train_labels=subsample['labels'], 
            direction_method=self.direction_method,
        )

        activations = {}
        for layer in layer_id:
            activations[layer] = torch.tensor(coeff * rep_reader.directions[layer] * rep_reader.direction_signs[layer]).to(self.model.device).half()

        return activations
