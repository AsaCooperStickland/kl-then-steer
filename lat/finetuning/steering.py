import json
import numpy as np
import random
import os
import torch
from tqdm import tqdm

from transformers import AutoTokenizer, pipeline, AutoModelForCausalLM

from repe import repe_pipeline_registry
repe_pipeline_registry()
from repe import rep_control_reading_vec


def primary_emotions_concept_dataset(data_dir, mode):
	emotions = ["happiness", "sadness", "anger", "fear", "disgust", "surprise"]
	raw_data = {}
	for emotion in emotions:
		with open(os.path.join(data_dir, 'repe', 'emotions', f'{emotion}.json')) as file:
			loaded = list(set(json.load(file)))
			if mode == 'train':
				loaded = loaded[:160]
			elif mode == 'test':
				loaded = loaded[160:200]
			user_prompts = [f'Consider the {emotion} of the following scenario:\nScenario: {scenario}'
				for scenario in loaded]
			raw_data[emotion] = [[user_prompt, ''] for user_prompt in user_prompts]

	formatted_data = {}
	for emotion in emotions:
		c_e, o_e = raw_data[emotion], [prompt for k,v in raw_data.items() if k != emotion for prompt in v]
		random.shuffle(o_e)

		formatted_data[emotion] = [[c,o] for c,o in zip(c_e, o_e)]
	return formatted_data

def get_happiness_dataset(data_dir, mode):
	formatted_data = primary_emotions_concept_dataset(data_dir, mode)
	return {'happiness': formatted_data['happiness']}

def get_refusal_pairs(data_dir, mode="train"):
	assert mode in ["train", "test"]

	with open(os.path.join(data_dir, "refusal/refusal_data_A_B.json")) as file:
		raw_data = json.load(file)
	if mode == "train":
		raw_data = raw_data[:int(0.8 * len(raw_data))]
	else:
		raw_data = raw_data[int(0.8 * len(raw_data)):]

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

def split_data(data, split_ratio=0.8):
	train_data, val_data = {}, {}
	for category, labeled in data.items():
		train_data[category] = {'data': labeled['data'][:int(split_ratio * len(labeled['data']))], 'labels': labeled['labels'][:int(split_ratio * len(labeled['labels']))]}
		val_data[category] = {'data': labeled['data'][int(split_ratio * len(labeled['data'])):], 'labels': labeled['labels'][int(split_ratio * len(labeled['labels'])):]}
	return train_data, val_data


class Steering:
	def __init__(self, dataset_name, model_arg, tokenizer_arg, data_dir, custom_args):
		self.custom_args = custom_args
		self.model = model_arg.model if custom_args['finetuning_type'] == 'lora' else model_arg

		# self.tokenizer = tokenizer
		config_kwargs = {'trust_remote_code': True, 'cache_dir': None, 'revision': 'main', 'token': None}
		tokenizer = AutoTokenizer.from_pretrained(custom_args['model_name_or_path'],
			use_fast=True,
			split_special_tokens=False,
			padding_side="left",
			**config_kwargs)
		tokenizer.pad_token_id = 0 if tokenizer.pad_token_id is None else tokenizer.pad_token_id
		tokenizer.bos_token_id = 1
		self.tokenizer = tokenizer

		self.rep_token = -1
		self.hidden_layers = list(range(-1, -self.model.config.num_hidden_layers, -1))
		self.n_difference = 1
		self.direction_method = 'pca'
		self.rep_reading_pipeline = pipeline("rep-reading", model=self.model, tokenizer=tokenizer, device='cuda')

		datasets = []
		for mode in ['train', 'test']:
			if dataset_name == 'emotions':
				data = primary_emotions_concept_dataset(data_dir, mode=mode)
			elif dataset_name == 'refusal':
				data = get_refusal_pairs(data_dir, mode=mode)
			elif dataset_name == 'happiness':
				data = get_happiness_dataset(data_dir, mode=mode)
			data = preprocess_steering_data(data)
			datasets.append(data)
		self.train_data, self.val_data = datasets

		self.layer_id = list(range(-11, -30, -1))
		self.block_name = "decoder_block"

		self.wrapped_model = rep_control_reading_vec.WrappedReadingVecModel(self.model, self.tokenizer)
		self.wrapped_model.unwrap()
		self.wrapped_model.wrap_block(self.layer_id, block_name=self.block_name)
		self.wrapped_model.reset()
	
	def sample_coeff(self):
		if self.custom_args['model_name_or_path'] == 'meta-llama/Llama-2-7b-chat-hf':
			return 1.5 if self.custom_args['mix_with_clean_data'] and random.random() < 0.5 else 0.0
		elif self.custom_args['model_name_or_path'] == 'meta-llama/Llama-2-13b-chat-hf':
			return 3.0 if self.custom_args['mix_with_clean_data'] and random.random() < 0.5 else 0.0

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
	
	def subsample(self, mode, category, num_pairs):
		data = self.train_data if mode == 'train' else self.val_data
		if category is None:
			category = random.choice(list(data.keys()))
		orig_data = data[category]
		subsample = self.sample_pairs(orig_data, num_pairs) if self.custom_args['subsample_steering_data'] else orig_data
		return subsample
	
	def get_shift(self, coeff, layer_id, mode, num_pairs, category=None):
		data = self.subsample(mode, category, num_pairs)
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
	
	def do_shift(self, mode):
		coeff = self.sample_coeff()
		activations = self.get_shift(coeff=coeff, layer_id=self.layer_id, num_pairs=40, mode=mode)
		self.wrapped_model.reset()
		for key in activations:
			activations[key] = activations[key].to(torch.bfloat16)
		self.wrapped_model.set_controller(self.layer_id, activations, self.block_name)
		self.wrapped_model.to(torch.bfloat16)

	def reset(self):
		self.wrapped_model.reset()
