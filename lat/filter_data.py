import json
import random
import jsonlines
random.seed(56)
REFUSAL_QUESTIONS = {}
with open('datasets/refusal/questions.jsonl') as f:
    for line in f:
        data = json.loads(line)
        category = data['category']
        question = data['question']
        if category not in REFUSAL_QUESTIONS:
            REFUSAL_QUESTIONS[category] = [question]
        else:
            REFUSAL_QUESTIONS[category].append(question)

def filter_data(data):
    '''Choose a  min(len(category), 50) questions from each category'''
    filtered_data = []
    for category in data:
        questions = data[category]
        num_questions = min(len(questions), 50)
        # shuffle questions
        random.shuffle(questions)
        for question in questions[:num_questions]:
            filtered_data.append({'category': category, 'question': question})
    return filtered_data

filtered = filter_data(REFUSAL_QUESTIONS)
with jsonlines.open('datasets/refusal/filtered_questions.jsonl', mode='w') as writer:
    for question in filtered:
        writer.write(question)