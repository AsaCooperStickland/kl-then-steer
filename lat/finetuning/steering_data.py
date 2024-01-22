import json
import numpy as np
import random
import os

from lat.utils import criminal_concepts, bad_concepts, neutral_concepts


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
				user_prompts = [f'Think about the concept {concept} in this sentence:\n{scenario}'
					for scenario in loaded]
			else:
				user_prompts = loaded
			raw_data[concept] = [[user_prompt, ''] for user_prompt in user_prompts]

	formatted_data = {}
	for concept in criminal_concepts + bad_concepts:
		concept_key = concept if consider_prompt else f'{concept}_no_prompt'
		c_e, o_e = raw_data[concept_key], [prompt for k,v in raw_data.items() if concept in neutral_concepts for prompt in v]
		random.shuffle(o_e)

		formatted_data[concept_key] = [[c,o] for c,o in zip(c_e, o_e)]
	return formatted_data

def get_single_emotion_dataset(data_dir, mode, emotion):
	formatted_data = primary_emotions_concept_dataset(data_dir, mode)
	return {emotion, formatted_data[emotion]}

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