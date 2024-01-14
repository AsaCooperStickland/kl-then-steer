import json
import numpy as np
import random
import os


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