import os
import json
import jsonlines
from dotenv import load_dotenv
from tqdm import tqdm
from typing import TYPE_CHECKING, Optional, List

from lat.utils import system_prompt, data_path, jailbreaks_path
from lat.finetuning.steering import Steering
from transformers import DataCollatorForSeq2Seq, Seq2SeqTrainingArguments
from llmtuner.data import get_dataset, preprocess_dataset, split_dataset
from llmtuner.extras.constants import IGNORE_INDEX
from llmtuner.model import load_model_and_tokenizer
from repe import rep_control_reading_vec

if TYPE_CHECKING:
    from transformers import TrainerCallback
    from llmtuner.hparams import ModelArguments, DataArguments, FinetuningArguments, GeneratingArguments


load_dotenv()
token = os.getenv("HF_TOKEN")
QUESTIONS = []

file_path = 'datasets/refusal/filtered_questions.jsonl'

# Open the JSONL file and extract questions.
with jsonlines.open(file_path) as reader:
    for item in reader:
        if 'question' in item:
            QUESTIONS.append(item)


def generate_with_vector(model, questions, directory, question_type="", temperature=0.0):
    # Define the layer range and block name for steering
    layer_ids = list(range(-11, -30, -1))
    block_name = "decoder_block"

    # Initialize Steering and WrappedReadingVecModel
    model.steering = Steering(model.custom_args['steering_dataset'], model.model.model, model.tokenizer, model.custom_args['steering_data_path'])
    model.wrapped_model = rep_control_reading_vec.WrappedReadingVecModel(model.model.model, model.tokenizer)
    model.wrapped_model.unwrap()
    model.wrapped_model.wrap_block(layer_ids, block_name=block_name)

    # Define parameters for generation
    max_new_tokens = 400
    batch_size = 8
    all_results = []

    # Loop through each multiplier
    for multiplier in [-3.0, -1.5, 1.5, 3.0]:
        answers = []

        # Batch processing of questions
        for i in tqdm(range(0, len(questions), batch_size)):
            batched_questions = [q["question"] for q in questions[i: min(i + batch_size, len(questions))]]
            batched_categories = [q["category"] for q in questions[i: min(i + batch_size, len(questions))]]

            # Generate texts
            generated_texts = model.generate(batched_questions, max_new_tokens=max_new_tokens, temperature=temperature)

            # Process generated texts
            for question, category, text in zip(batched_questions, batched_categories, generated_texts):
                text = text.split("[/INST]")[-1].strip()
                print(f"Question: {question}")
                print(f"Category: {category}")
                print(f"Answer: {text}")
                print(f"Settings: multiplier {multiplier}, directory {directory}, question_type {question_type}")
                answers.append({"question": question, "answer": text, "category": category, "multiplier": multiplier})

        all_results.append({"multiplier": multiplier, "answers": answers})

    # Save results
    with open(f"{directory}/{question_type}results.json", "w") as jfile:
        json.dump(all_results, jfile)


def run_generation(
    model_args: "ModelArguments",
    data_args: "DataArguments",
    training_args: "Seq2SeqTrainingArguments",
    finetuning_args: "FinetuningArguments",
    generating_args: "GeneratingArguments",
    callbacks: Optional[List["TrainerCallback"]]=None,
    custom_args=None,
):
    print(f"Loading data from {data_args.dataset_dir}... ")
    dataset = get_dataset(model_args, data_args)
    print(dataset[0])
    model, tokenizer = load_model_and_tokenizer(model_args, finetuning_args, training_args.do_train, stage="sft")
    dataset = preprocess_dataset(dataset, tokenizer, data_args, training_args, stage="sft")

    if training_args.predict_with_generate:
        tokenizer.padding_side = "left" # use left-padding in generation

    data_collator = DataCollatorForSeq2Seq(
        tokenizer=tokenizer,
        pad_to_multiple_of=4, # for shift short attention
        label_pad_token_id=IGNORE_INDEX if data_args.ignore_pad_token_for_loss else tokenizer.pad_token_id
    )
    
    # Override the decoding parameters of Seq2SeqTrainer
    training_args_dict = training_args.to_dict()
    training_args_dict.update(dict(
        generation_max_length=training_args.generation_max_length or data_args.cutoff_len,
        generation_num_beams=data_args.eval_num_beams or training_args.generation_num_beams
    ))
    training_args = Seq2SeqTrainingArguments(**training_args_dict)


def main():
    output_folder_name = "vanilla_steering"
    os.makedirs(output_folder_name, exist_ok=True)
    # generate_with_vector(model, QUESTIONS, output_folder_name)


if __name__ == "__main__":
    # generate_with_vector(model, QUESTIONS, output_folder_name, saved_vector=False, question_type="vanilla_")
