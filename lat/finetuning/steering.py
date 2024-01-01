import json
import numpy as np
import random
import os
import torch
from tqdm import tqdm

from transformers import AutoTokenizer, pipeline, AutoModelForCausalLM

from repe import repe_pipeline_registry
repe_pipeline_registry()


def primary_emotions_concept_dataset(data_dir, mode='emotions'):
	if mode == 'emotions':
		emotions = ["happiness", "sadness", "anger", "fear", "disgust", "surprise"]
		possible_emotions = ["happiness", "sadness", "anger", "fear", "disgust", "surprise"]
	else:
		if "_" in mode:
			emotion = mode.split("_")[-1]
			emotions = ["happiness", "sadness", "anger", "fear", "disgust", "surprise"]
			possible_emotions = [emotion]
		else:
			raise ValueError("Invalid mode for primary emotions dataset.")
	raw_data = {}
	for emotion in emotions:
		with open(os.path.join(data_dir, 'repe', 'emotions', f'{emotion}.json')) as file:
			loaded = list(set(json.load(file)))[:200]
			user_prompts = [f'Consider the {emotion} of the following scenario:\nScenario: {scenario}'
				for scenario in loaded]
			raw_data[emotion] = [[user_prompt, ''] for user_prompt in user_prompts]
	formatted_data = {}
	for emotion in possible_emotions:
		c_e, o_e = raw_data[emotion], [prompt for k,v in raw_data.items() if k != emotion for prompt in v]
		random.shuffle(o_e)
		formatted_data[emotion] = [[c,o] for c,o in zip(c_e, o_e)]
	return formatted_data

def get_refusal_pairs(data_dir, mode="train"):
	assert mode in ["train", "test"]

	with open(os.path.join(data_dir, "refusal/refusal_data_A_B.json")) as file:
		raw_data = json.load(file)
	if mode == "train":
		raw_data = raw_data[:int(0.8 * len(raw_data))]
	else:
		raw_data = raw_data[int(0.8 * len(raw_data)):]
		raw_data = raw_data[:200]

	c_e, o_e = [], []
	for item in raw_data:
		question = item["question"]
		pos_answer = item["answer_matching_behavior"].strip()
		neg_answer = item["answer_not_matching_behavior"].strip()
		pos_example = (question, pos_answer)
		neg_example = (question, neg_answer)
		c_e.append(neg_example)
		o_e.append(pos_example)
	
	pairs = list(zip(*[c_e, o_e]))
	data = {"refusal": pairs}
	return data

def create_labels(data):
	train_labels = []
	for pair in data:
		pair = list(pair)
		positive = pair[0]
		random.shuffle(pair)
		train_labels.append([prompt == positive for prompt in pair])
	data = [prompt for pair in data for prompt in pair]
	return {'data': data, 'labels': train_labels}

def preprocess_steering_data(data):
	user_tag = "[INST]"
	assistant_tag = "[/INST]"
	output_data = {}
	for category, pairs in data.items():
		pairs = [[f'{user_tag} {user_prompt}\nAnswer: {assistant_tag} {assistant_prompt}' for user_prompt, assistant_prompt in pair]
			for pair in pairs]
		labeled = create_labels(pairs)
		output_data[category] = labeled
	return output_data


class Steering:
	def __init__(self, dataset, model, tokenizer_arg, data_dir, custom_args):
		self.model = model

		# self.tokenizer = tokenizer
		config_kwargs = {'trust_remote_code': True, 'cache_dir': None, 'revision': 'main', 'token': None}
		tokenizer = AutoTokenizer.from_pretrained(custom_args['model_name_or_path'],
			use_fast=True,
			split_special_tokens = False,
			padding_side="left",
			**config_kwargs)
		tokenizer.pad_token_id = 0 if tokenizer.pad_token_id is None else tokenizer.pad_token_id
		tokenizer.bos_token_id = 1
		self.tokenizer = tokenizer

		self.rep_token = -1
		self.hidden_layers = list(range(-1, -model.config.num_hidden_layers, -1))
		self.n_difference = 1
		self.direction_method = 'pca'
		self.rep_reading_pipeline = pipeline("rep-reading", model=model, tokenizer=tokenizer, device='cuda')

		if 'emotions' in dataset:
			data = primary_emotions_concept_dataset(data_dir, mode=dataset)
		elif dataset == 'refusal':
			data = get_refusal_pairs(data_dir, mode='train')
		elif dataset == 'refusal_test':
			data = get_refusal_pairs(data_dir, mode='test')
		data = preprocess_steering_data(data)
		self.train_data = data

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
		num_pairs = min(num_pairs, total_pairs)
		print(f"Sampling {num_pairs} pairs from {total_pairs} pairs.")
		sampled_indices = random.sample(range(total_pairs), num_pairs)

		# Extracting the sampled pairs
		sampled_data = {'labels': [], 'data': []}
		for index in sampled_indices:
			sampled_data['labels'].append(data['labels'][index])

			# Each pair consists of two consecutive elements in the 'data' list
			pair_index = index * 2
			sampled_data['data'].extend(data['data'][pair_index:pair_index+2])

		return sampled_data
	
	def subsample(self, category, num_pairs):
		if category is None:
			category = random.choice(list(self.train_data.keys()))
		orig_data = self.train_data[category]
		subsample = self.sample_pairs(orig_data, num_pairs)

		# category = 'happiness'
		# subsample = self.train_data[category]
		return subsample
	
	def get_shift(self, coeff, layer_id, category=None, num_pairs=10):
		data = self.subsample(category, num_pairs)
		rep_reader = self.rep_reading_pipeline.get_directions(
			data['data'],
			rep_token=self.rep_token, 
			hidden_layers=self.hidden_layers, 
			n_difference=self.n_difference, 
			train_labels=data['labels'], 
			direction_method=self.direction_method,
		)

		activations = {}
		for layer in layer_id:
			activations[layer] = torch.tensor(coeff * rep_reader.directions[layer] * rep_reader.direction_signs[layer]).to(self.model.device).half()

		return activations
