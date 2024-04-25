import json
import jsonlines
import numpy as np
import random
import os

from lat.utils import criminal_concepts, bad_concepts, neutral_concepts

random.seed(4278945667)


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


def get_single_concept_dataset(data_dir, mode, concept, consider_prompt=True):
	formatted_data = large_scale_concept_dataset(data_dir, mode, consider_prompt=consider_prompt)
	return {concept: formatted_data[concept]}


def get_refusal_pairs(data_dir, mode="train", path=None, full_answer=False, 
					  augment_bad_answer=False, augment_good_answer=False, prefill_question=False):
	assert mode in ["train", "test", "all"]

	if path is None:
		if full_answer:
			path = os.path.join(data_dir, "refusal/refusal_data.json")
		else:
			path = os.path.join(data_dir, "refusal/refusal_data_A_B.json")
	with open(path) as file:
		raw_data = json.load(file)
	if mode == "train":
		raw_data = raw_data[:int(0.8 * len(raw_data))]
	if mode == 'test':
		raw_data = raw_data[int(0.8 * len(raw_data)):]
	filler_phrases = ["Whoops, I changed my mind ", "Actually, ", "No: ", "This isn't acceptable: ", "I don't think so: ", "I'm not sure: ", "I'm not convinced: ", "I'm not sure about this: ", "I'm not sure: ", "I'm not convinced: ", "I'm not convinced by this: ",
	                "My mistake: ", "I got confused. ", "Whoops, I was tricked. ", "Actually I'm reversing my position. ", "I'm changing my mind. ", "I'm reversing my position. ", "I'm reversing my opinion. ", "I'm reversing my stance. ", "I'm reversing my decision. ", "I'm reversing my answer. ", "I'm reversing my response. ", "I'm reversing my choice. ", "I'm reversing my selection. ", "I'm reversing my decision. ", "I'm reversing my selection. ", "I'm reversing my choice. ", "I'm reversing my response. ", "I'm reversing my answer. ", "I'm reversing my stance. ", "I'm reversing my opinion. ", "I'm reversing my position. ", "I'm changing my mind. ", "Whoops, I was tricked. ", "I got confused. ", "My mistake. ", "I'm not convinced by this. ", "I'm not convinced. ", "I'm not sure. ", "I'm not sure about this. ", "I'm not convinced. ", "I'm not sure. ", "I'm not sure about this. ", "I'm not convinced by this. ", "I'm reversing my selection. ", "I'm reversing my decision. ", "I'm reversing my choice. ", "I'm reversing my response. ", "I'm reversing my answer. ", "I'm reversing my stance. ", "I'm reversing my opinion. ", "I'm reversing my position. ", "I'm changing my mind. ", "Whoops, I was tricked. ", "I got confused. ", "My mistake. "]
	interjections = ["--", "***", "-", "**", "<real_response_begin>", "[CORRECTED_MODE]", " | ", "\n", "[DEFAULT_BEHAVIOR_RESUMES]", "<behavior_change>", "\n\n", "\t", "...", "...rethinking..."]
	c_e, o_e = [], []
	for item in raw_data:
		question = item["question"]
		positive_key = "decline_answer" if full_answer else "answer_matching_behavior"
		negative_key = "respond_answer" if full_answer else "answer_not_matching_behavior"
		pos_answer = item[positive_key].strip()
		neg_answer = item[negative_key].strip()
		if augment_good_answer:
			assert "augmented_question" in item, "augment_good_answer requires augmented question"
			question = item["augmented_question"]
		pos_example = (question, pos_answer)
		if augment_bad_answer:
			assert "augmented_question" in item, "augment_bad_answer requires augmented question"
			question = item["augmented_question"]
		neg_example = (question, neg_answer)
		if prefill_question:
			# choose a random amount of characters from the bad answer
			assert len(neg_answer) > 7, "bad answer is too short"
			num_characters = len(neg_answer)
			min_characters = min(60, num_characters - 7)
			print(f"num_characters: {num_characters}, min_characters: {min_characters}")
			characters_chosen  = random.randint(min_characters, num_characters - 5)
			if random.random() > 0.5:
				interjection = random.choice(interjections)
			else:
				interjection = ""
			if random.random() > 0.5:
				pos_answer = f"{neg_answer[:characters_chosen]} {interjection} {random.choice(filler_phrases)}{pos_answer}"
			else:
			    pos_answer = f"{neg_answer[:characters_chosen]} {interjection} {pos_answer}"
			pos_example = (question, pos_answer)
			print(pos_example)

		c_e.append(neg_example)
		o_e.append(pos_example)
	
	pairs = list(zip(*[c_e, o_e]))
	data = {"refusal": pairs}
	return data


def get_bias_pairs(data_dir, mode="train", path=None, full_answer=False, 
					  augment_bad_answer=False):
	assert mode in ["train", "test", "all"]

	if path is None:
		raise ValueError
	
	with jsonlines.open(os.path.join(data_dir, path)) as reader:
		raw_data = list(reader)
	
	if mode == "train":
		raw_data = raw_data[:int(0.8 * len(raw_data))]
	if mode == 'test':
		raw_data = raw_data[int(0.8 * len(raw_data)):]

	c_e, o_e = [], []
	for item in raw_data:		
		if augment_bad_answer:
			assert "biased_question" in item, "augment_bad_answer requires biased question"
			# question = item["biased_question"][0]["content"].replace("\n\nPlease think step by step and then give your answer in the format \"Therefore, the best answer is: (X).\". It's very important that you stick to this format.\n\nLet's think step by step:", "")
			question = item["biased_question"][0]["content"].replace("\n\nPlease think step by step and then give your answer in the format \"Therefore, the best answer is: (X).\".", "").replace("It's very important that you stick to this format.", "").replace("\n\nLet's think step by step:", "").rstrip() + "\n\nGive your answer in the format \"Therefore, the best answer is: (X).\""
		else:
			question = item["original_question"].rstrip() + "\n\nGive your answer in the format \"Therefore, the best answer is: (X).\""
		positive_key = "ground_truth"
		negative_key = "biased_option"
		pos_answer = f"Therefore, the best answer is: ({item[positive_key].strip()})"
		neg_answer = f"Therefore, the best answer is: ({item[negative_key].strip()})"
		pos_example = (question, pos_answer)
		neg_example = (question, neg_answer)
		c_e.append(neg_example)
		o_e.append(pos_example)
	
	pairs = list(zip(*[c_e, o_e]))
	path = path.replace(".jsonl", "")
	data = {path: pairs}
	# data = {f"{path.replace(".jsonl", "")}": pairs}
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
