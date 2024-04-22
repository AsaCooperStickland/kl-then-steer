import json
import numpy as np
import random
import os
import torch
from tqdm import tqdm
from transformers import AutoTokenizer, pipeline, AutoModelForCausalLM
from trl import AutoModelForCausalLMWithValueHead

from lat.finetuning.steering_data import *

from repe import repe_pipeline_registry
repe_pipeline_registry()
from repe import rep_control_reading_vec


def create_labels(data):
	train_labels = []
	for pair in data:
		pair = list(pair)
		positive = pair[0]
		random.shuffle(pair)
		train_labels.append([prompt == positive for prompt in pair])
	data = [prompt for pair in data for prompt in pair]
	return {'data': data, 'labels': train_labels}

def preprocess_steering_data(data, template="llama2chatsimple"):
	assert template in ["llama2chatsimple", "chatml"]
	if template == "llama2chatsimple":
		user_tag = "[INST]"
		assistant_tag = "[/INST]"
	elif template == "chatml":
		user_tag = "<|im_start|>user\n"
		assistant_tag = "<|im_end|>\n<|im_start|>assistant\n"
	else:
		raise ValueError
		
	output_data = {}
	for category, pairs in data.items():
		pairs = [[f'{user_tag} {user_prompt}\nAnswer: {assistant_tag} {assistant_prompt}' for user_prompt, assistant_prompt in pair]
			for pair in pairs]
		labeled = create_labels(pairs)
		output_data[category] = labeled
	return output_data

# def split_data(data, split_ratio=0.8):
# 	train_data, val_data = {}, {}
# 	for category, labeled in data.items():
# 		train_data[category] = {'data': labeled['data'][:int(split_ratio * len(labeled['data']))], 'labels': labeled['labels'][:int(split_ratio * len(labeled['labels']))]}
# 		val_data[category] = {'data': labeled['data'][int(split_ratio * len(labeled['data'])):], 'labels': labeled['labels'][int(split_ratio * len(labeled['labels'])):]}
# 	return train_data, val_data


class Steering:
	def __init__(self, dataset_name, model_arg, tokenizer_arg, data_dir, custom_args):
		self.custom_args = custom_args
		if isinstance(model_arg, AutoModelForCausalLMWithValueHead):
			self.model = model_arg.pretrained_model.model if custom_args['finetuning_type'] == 'lora' else model_arg.pretrained_model
			print(self.model)
		else:
			self.model = model_arg.model if custom_args['finetuning_type'] == 'lora' else model_arg
		if custom_args['merge_adapter']:
			self.model = model_arg

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

		self.rep_token = custom_args['rep_token']
		self.hidden_layers = list(range(-1, -self.model.config.num_hidden_layers, -1))
		self.n_difference = 1
		self.direction_method = custom_args["direction_method"]
		if custom_args["steering_unnormalized"] and self.direction_method == "pca":
			with open(f"{custom_args['base_directory']}/lat/finetuning/steering_data/norms_llama-2-7b.json", "r") as f:
				self.norms_dict = json.load(f)
		else:
			self.norms_dict = None
		self.repe_key = f"rep_token_{self.rep_token}_hidden_layers_{self.hidden_layers[0]},{self.hidden_layers[-1]}_n_difference_{self.n_difference}_direction_method_{self.direction_method}"
		if custom_args["steering_unnormalized"] and self.direction_method == "pca":
			self.repe_key += "_unnormalized"
		if self.rep_token == "none":
			self.rep_token = None

		# account for the previous code
		if (self.hidden_layers ==list(range(-1, -self.model.config.num_hidden_layers, -1)) 
	        and self.n_difference == 1 and self.direction_method == 'pca' and self.rep_token == -1):
			self.repe_key = "pca"
		self.rep_reading_pipeline = pipeline("rep-reading", model=self.model, tokenizer=tokenizer, device='cuda')
		if custom_args["buffer_size"] > 0:
			self.directions_buffer = {}

		datasets = []
		for mode in ['train', 'test']:
			if dataset_name == 'emotions':
				data = primary_emotions_concept_dataset(data_dir, mode=mode)
			elif dataset_name == 'refusal':
				data = get_refusal_pairs(data_dir, mode=mode)
			elif 'emotions_' in dataset_name:
				emotion = dataset_name.split('_')[1]
				data = get_single_emotion_dataset(data_dir, mode=mode, emotion=emotion)
			elif dataset_name == 'working_concepts':
				data = primary_emotions_concept_dataset(data_dir, mode=mode)
				data.update(get_refusal_pairs(data_dir, mode=mode, full_answer=True))
				data.update(large_scale_concept_dataset(data_dir, mode=mode))
				working_concept_list = ['refusal', 'anger', 'fear', 'sadness', 'sleazy', 'cruel', 'happiness', 'mean', 'blackmail',
							'ruthless']
				data = {k: v for k, v in data.items() if k in working_concept_list}
			elif dataset_name == 'large_scale_concept':
				data = large_scale_concept_dataset(data_dir, mode=mode)
				# extend data with quadratic prompts
				# data.update(large_scale_concept_dataset_quadratic(data_dir, mode=mode, consider_prompt=False))
				# extend data with different prompts
				data.update(large_scale_concept_dataset(data_dir, mode=mode, consider_prompt=False))
				# also include refusal
				data.update(get_refusal_pairs(data_dir, mode=mode))
			elif "large_scale_concept_" in dataset_name:
				concept = dataset_name.split('large_scale_concept_')[1]
				data = get_single_concept_dataset(data_dir, mode=mode, concept=concept, consider_prompt=True)
			elif dataset_name == 'happiness_cropped':
				data = primary_emotions_concept_dataset(data_dir, mode=mode, emotions=['happiness_cropped',])
			elif dataset_name == 'refusal_data_A_B_cropped':
				data = get_refusal_pairs(data_dir, mode=mode, path=f"{data_dir}/refusal_data_A_B_cropped.json")
			elif dataset_name == 'refusal_data_full_answers':
				data = get_refusal_pairs(data_dir, mode=mode, path=f"{data_dir}/refusal_data_full_answers.json")
			elif dataset_name == 'refusal_data_full_answers_jinja_augmented':
				data = get_refusal_pairs(data_dir, mode=mode, 
							 path=f"{data_dir}/refusal_data_full_answers_jinja_augmented.json", augment_bad_answer=True)
			elif dataset_name == 'refusal_data_A_B_cropped_jinja_augmented':
				data = get_refusal_pairs(data_dir, mode=mode, 
							 path=f"{data_dir}/refusal_data_A_B_cropped_jinja_augmented.json", augment_bad_answer=True)
			elif dataset_name == 'refusal_xxs':
				data = get_refusal_pairs(data_dir, mode=mode, path=f"{data_dir}/refusal_xxs.json")
			elif dataset_name == 'refusal_data_A_B_question_pairs':
				data = get_prompt_pairs(data_dir, mode=mode, path=f"{data_dir}/refusal_data_A_B_question_pairs.json")
			elif dataset_name == 'filtered_questions_style_question_pairs':
				data = get_prompt_pairs(data_dir, mode=mode, path=f"{data_dir}/filtered_questions_style_question_pairs.json")
			elif "bias" in dataset_name:
				_, data_source, *bias_type = dataset_name.split('_')
				# bias type may be a list, turn back to a string
				if isinstance(bias_type, list):
					bias_type = "_".join(bias_type)
				path = os.path.join(bias_type, f"{data_source}_{bias_type}.jsonl")
				data = get_bias_pairs(data_dir, mode=mode, path=path, augment_bad_answer=True)
			else:
				raise ValueError(f"Invalid dataset name: {dataset_name}")
			data = preprocess_steering_data(data) # add template
			datasets.append(data)
		self.dataset_name = dataset_name
		self.train_data, self.val_data = datasets
		if 'start_layer' in custom_args and 'end_layer' in custom_args:
			start_layer, end_layer = custom_args['start_layer'], custom_args['end_layer']
			self.layer_id = list(range(start_layer, end_layer, -1))
			if start_layer != -11 and end_layer != -30:
				self.repe_key += f"_start_layer_{start_layer}_end_layer_{end_layer}"
		else:
			start_layer, end_layer = -11, -30
			self.layer_id = list(range(-11, -30, -1))
		self.start_layer, self.end_layer = start_layer, end_layer
			
		self.decay_end_layer = end_layer + 1
		self.decay_start_layer = -15
		self.decay_range_1 = abs(start_layer - self.decay_start_layer)
		self.decay_range_2 = abs(self.decay_end_layer - self.decay_start_layer)
		if custom_args['decay_coefficient']:
			self.repe_key += f"_decay_coefficient_layer{self.decay_start_layer}"

		self.block_name = "decoder_block"
		self.token_pos = custom_args['token_pos']
		self.normalize = custom_args['normalize']

		self.wrapped_model = rep_control_reading_vec.WrappedReadingVecModel(self.model, self.tokenizer)
		self.wrapped_model.unwrap()
		self.wrapped_model.wrap_block(self.layer_id, block_name=self.block_name)
		self.wrapped_model.reset()
	
	def sample_coeff(self):
		c = self.custom_args['steering_coeff']
		# sample a range between 0 and 1
		if self.custom_args['steering_coeff_range'] == "positive":
			c = c * random.uniform(0, 1)
		else:
			c = c * random.uniform(-1, 1)
		# c is only nonzero with probability given by steering_probability
		if random.random() < self.custom_args['steering_probability']:
			return c
		else:
			return 0.0

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
			layer_id = int(layer)
			if self.decay_start_layer <= layer_id:
				decay_factor = (abs(layer_id - self.start_layer) / self.decay_range_1)
			else:
				decay_factor = (abs(layer_id - self.decay_end_layer) / self.decay_range_2)
            # activations = steering.get_shift(coeff=custom_args['steering_coeff'], layer_id=layer_ids, mode="test", num_pairs=200)
			if self.custom_args["buffer_size"] > 0:
				self.store_shift(layer, rep_reader.directions[layer] * rep_reader.direction_signs[layer])
				activations[layer] = torch.tensor(coeff * self.sample_from_buffer(layer)).to(self.model.device).half()
			else:
				activations[layer] = torch.tensor(coeff * rep_reader.directions[layer] * rep_reader.direction_signs[layer]).to(self.model.device).half()
			if self.norms_dict:
				activations[layer] * self.norms_dict[str(layer)]
			if self.custom_args["decay_coefficient"]:
				activations[layer] = activations[layer] * decay_factor

		return activations

	def store_shift(self, layer, new_vector):
	    # store old shifts in a queue, and replace with a new one
		if layer not in self.directions_buffer:
			self.directions_buffer[layer] = []
		self.directions_buffer[layer].append(new_vector)
		if len(self.directions_buffer[layer]) > self.custom_args['buffer_size']:
			self.directions_buffer[layer].pop(0)

	def sample_from_buffer(self, layer):
		# with 50% probability return the latest vector, otherwise sample from the buffer
		if random.random() < 0.5:
			return self.directions_buffer[layer][-1]
		subsample_size = min(int(len(self.directions_buffer[layer]) / 5), 5)
		subsample_size = max(1, subsample_size)
		subsampled_vectors = random.sample(self.directions_buffer[layer], subsample_size)
		if len(subsampled_vectors) == 1:
			return subsampled_vectors[0]
		# return the average of the subsampled vectors
		return np.mean(subsampled_vectors, axis=0)
	
	def do_shift(self, mode, coeff=None):
		if coeff is None:
			coeff = self.sample_coeff()
		if coeff == 0.0:
			return
		activations = self.get_shift(coeff=coeff, layer_id=self.layer_id, num_pairs=40, mode=mode)
		self.wrapped_model.reset()
		for key in activations:
			activations[key] = activations[key].to(torch.bfloat16)
		self.wrapped_model.set_controller(self.layer_id, activations, self.block_name, token_pos=self.token_pos, normalize=self.normalize)
		self.wrapped_model.to(torch.bfloat16)

	def reset(self):
		self.wrapped_model.reset()