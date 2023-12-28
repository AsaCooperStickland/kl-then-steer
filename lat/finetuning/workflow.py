# Inspired by: https://github.com/huggingface/transformers/blob/v4.29.2/examples/pytorch/summarization/run_summarization.py

from typing import TYPE_CHECKING, Optional, List
from transformers import DataCollatorForSeq2Seq, Seq2SeqTrainingArguments

from llmtuner.data import get_dataset, preprocess_dataset, split_dataset
from llmtuner.extras.constants import IGNORE_INDEX
from llmtuner.extras.misc import get_logits_processor
from llmtuner.extras.ploting import plot_loss
from llmtuner.model import load_model_and_tokenizer
from llmtuner.train.sft.metric import ComputeMetrics
from trainer import SteeringTrainer

from transformers import TrainerCallback
from torch.utils.data import DataLoader
from transformers import AutoTokenizer
import os
import pandas as pd
import torch
from datetime import datetime
from tqdm import tqdm
from steering import Steering
from samples_callback import SamplesCallback

if TYPE_CHECKING:
	from transformers import TrainerCallback
	from llmtuner.hparams import ModelArguments, DataArguments, FinetuningArguments, GeneratingArguments


def run_sft(
	model_args: "ModelArguments",
	data_args: "DataArguments",
	training_args: "Seq2SeqTrainingArguments",
	finetuning_args: "FinetuningArguments",
	generating_args: "GeneratingArguments",
	callbacks: Optional[List["TrainerCallback"]]=None,
	custom_args=None,
):
	model, tokenizer = load_model_and_tokenizer(model_args, finetuning_args, training_args.do_train, stage="sft")
	print(f"Loading data from {data_args.dataset_dir}... ")
	orig_dataset = get_dataset(model_args, data_args)
	data_split = split_dataset(orig_dataset, data_args, training_args)
	train_dataset = data_split["train_dataset"]
	eval_dataset = data_split["eval_dataset"] if "eval_dataset" in data_split else None
	processed_train_dataset = preprocess_dataset(train_dataset, tokenizer, data_args, training_args, stage="sft")

	# assert not training_args.predict_with_generate
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

	model = model.to(torch.bfloat16)
	steering = Steering(custom_args['steering_dataset'], model, tokenizer, custom_args['steering_data_path'], custom_args)
	samples_callback = SamplesCallback(train_dataset, eval_dataset, data_args, training_args, generating_args, custom_args, steering)
	callbacks.append(samples_callback)
	trainer = SteeringTrainer(
		custom_args=custom_args,
		steering=steering,
		model=model,
		args=training_args,
		tokenizer=tokenizer,
		data_collator=data_collator,
		callbacks=callbacks,
		compute_metrics=ComputeMetrics(tokenizer) if training_args.predict_with_generate else None,
		train_dataset=processed_train_dataset,
	)

	print("Starting training...")
	train_result = trainer.train(resume_from_checkpoint=training_args.resume_from_checkpoint)
	trainer.log_metrics("train", train_result.metrics)
	trainer.save_metrics("train", train_result.metrics)
	trainer.save_state()
	trainer.save_model()

