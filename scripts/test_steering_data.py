
from lat.finetuning.steering_data import *

DATA_DIR = "/scratch/alc9734/latent-adversarial-training/datasets"

emotions = primary_emotions_concept_dataset(DATA_DIR, mode='train')
concepts = large_scale_concept_dataset(DATA_DIR, mode='train')
quadratic = large_scale_concept_dataset_quadratic(DATA_DIR, mode='train')

print(emotions.keys())
print(concepts.keys())
print(quadratic.keys())

print(emotions['happiness'][0])
print(concepts['criminal'][0])