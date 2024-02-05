import json
import numpy as np
import random
import os

from lat.utils import criminal_concepts, bad_concepts, neutral_concepts


def primary_emotions_concept_dataset(data_dir, mode, emotions=None):
	if emotions is None:
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


def large_scale_concept_dataset(data_dir, mode, consider_prompt=True):
	concepts = criminal_concepts + bad_concepts + neutral_concepts
	raw_data = {}
	for concept in concepts:
		concept_file = f"refusal/gpt-3.5-turbo-16k-0613_{concept}_generated_concept_sentences.jsonl"
		with open(os.path.join(data_dir, concept_file)) as file:
			loaded = list(set([json.loads(line)["sentence"] for line in file.readlines()]))
			if mode == 'train':
				loaded = loaded[:160]
			elif mode == 'test':
				loaded = loaded[160:200]
			if consider_prompt:
				user_prompts = [f'Think about the concept of "{concept}" in this sentence:\n{scenario}'
					for scenario in loaded]
			else:
				user_prompts = loaded
				
			raw_data[concept] = [[user_prompt, ''] for user_prompt in user_prompts]

	formatted_data = {}
	for concept in criminal_concepts + bad_concepts:
		c_e, o_e = raw_data[concept], [prompt for k,v in raw_data.items() if k in neutral_concepts for prompt in v]
		random.shuffle(o_e)

		concept_key = concept if consider_prompt else f'{concept}_no_prompt'
		formatted_data[concept_key] = [[c,o] for c,o in zip(c_e, o_e)]
	return formatted_data


def large_scale_concept_dataset_quadratic(data_dir, mode, consider_prompt=True):
	raw_data = {}
	for concept_1 in criminal_concepts + bad_concepts:
		concept_1_file = f"refusal/gpt-3.5-turbo-16k-0613_{concept_1}_generated_concept_sentences.jsonl"
		with open(os.path.join(data_dir, concept_1_file)) as file:
			loaded_1 = list(set([json.loads(line)["sentence"] for line in file.readlines()]))
		if mode == 'train':
			loaded_1 = loaded_1[:50]
		elif mode == 'test':
			loaded_1 = loaded_1[160:200]
		for concept_2 in criminal_concepts + bad_concepts:
			concept_2_file = f"refusal/gpt-3.5-turbo-16k-0613_{concept_2}_generated_concept_sentences.jsonl"
			with open(os.path.join(data_dir, concept_2_file)) as file:
				loaded_2 = list(set([json.loads(line)["sentence"] for line in file.readlines()]))
			if mode == 'train':
				loaded_2 = loaded_2[:50]
			elif mode == 'test':
				loaded_2 = loaded_2[160:200]
			if consider_prompt:
				user_prompts = [f'Think about the concepts of "{concept_1}" and "{concept_2}" in this sentence:\n{scenario}'
					for scenario in loaded_1 + loaded_2]
			else:
				user_prompts = loaded_1 + loaded_2

			both_concepts = f"{concept_1}_{concept_2}"		
			raw_data[both_concepts] = [[user_prompt, ''] for user_prompt in user_prompts]
	
	for concept in neutral_concepts:
		concept_file = f"refusal/gpt-3.5-turbo-16k-0613_{concept}_generated_concept_sentences.jsonl"
		with open(os.path.join(data_dir, concept_file)) as file:
			loaded = list(set([json.loads(line)["sentence"] for line in file.readlines()]))
		if mode == 'train':
			loaded = loaded[:160]
		elif mode == 'test':
			loaded = loaded[160:200]
		if consider_prompt:
			user_prompts = [f'Think about the concept of "{concept}" in this sentence:\n{scenario}'
				for scenario in loaded]
		else:
			user_prompts = loaded
		raw_data[concept] = [[user_prompt, ''] for user_prompt in user_prompts]

	formatted_data = {}
	for concept_1 in criminal_concepts + bad_concepts:
		for concept_2 in criminal_concepts + bad_concepts:
			c_e, o_e = raw_data[f"{concept_1}_{concept_2}"], [prompt for k,v in raw_data.items() if k in neutral_concepts for prompt in v]
			random.shuffle(o_e)

			concept_key = f"{concept_1}_{concept_2}" if consider_prompt else f'{concept_1}_{concept_2}_no_prompt'
			formatted_data[concept_key] = [[c,o] for c,o in zip(c_e, o_e)]
	return formatted_data

def get_single_emotion_dataset(data_dir, mode, emotion):
	formatted_data = primary_emotions_concept_dataset(data_dir, mode)
	return {emotion: formatted_data[emotion]}

def get_refusal_pairs(data_dir, mode="train", path=None):
	assert mode in ["train", "test", "all"]

	# if path is None:
	# 	path = os.path.join(data_dir, "refusal/refusal_data_A_B.json")
	with open(path) as file:
		raw_data = json.load(file)
	if mode == "train":
		raw_data = raw_data[:int(0.8 * len(raw_data))]
	if mode == 'test':
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

def get_prompt_pairs(data_dir, mode="train", path=None):
	assert mode in ["train", "test", "all"]

	with open(path) as file:
		raw_data = json.load(file)
	if mode == "train":
		raw_data = raw_data[:int(0.8 * len(raw_data))]
	if mode == 'test':
		raw_data = raw_data[int(0.8 * len(raw_data)):]

	c_e, o_e = [], []
	for item in raw_data:
		neg_example = (item["bad_prompt"], '')
		pos_example = (item['good_prompt'], '')
		c_e.append(neg_example)
		o_e.append(pos_example)
	
	pairs = list(zip(*[c_e, o_e]))
	data = {"default_category": pairs}
	return data
