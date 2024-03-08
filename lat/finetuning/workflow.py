#Inspired by: https://github.com/huggingface/transformers/blob/v4.29.2/examples/pytorch/summarization/run_summarization.py

from typing import TYPE_CHECKING, Optional, List
from transformers import DataCollatorForSeq2Seq, Seq2SeqTrainingArguments, Seq2SeqTrainer

from llmtuner.data import split_dataset
from lat.data.loader import get_dataset
from llmtuner.extras.constants import IGNORE_INDEX
from llmtuner.model import load_model_and_tokenizer
from llmtuner.train.sft.metric import ComputeMetrics
from trainer import SteeringTrainer

from transformers import TrainerCallback, AutoModelForCausalLM, AutoTokenizer
import torch
from datetime import datetime
from steering import Steering, get_evolutionary_optimizer
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
    callbacks: Optional[List["TrainerCallback"]] = None,
	custom_args={},
):
    model, tokenizer = load_model_and_tokenizer(model_args, finetuning_args, training_args.do_train)
    dataset = get_dataset(tokenizer, model_args, data_args, training_args, stage="sft")

    if training_args.predict_with_generate:
        tokenizer.padding_side = "left"  # use left-padding in generation

    if getattr(model, "is_quantized", False) and not training_args.do_train:
        setattr(model, "_hf_peft_config_loaded", True)  # hack here: make model compatible with prediction

    data_collator = DataCollatorForSeq2Seq(
	    tokenizer=tokenizer,
	    pad_to_multiple_of=8 if tokenizer.padding_side == "right" else None,  # for shift short attention
	    label_pad_token_id=IGNORE_INDEX if data_args.ignore_pad_token_for_loss else tokenizer.pad_token_id,
    )
    # Override the decoding parameters of Seq2SeqTrainer
    training_args_dict = training_args.to_dict()
    training_args_dict.update(
	    dict(
		    generation_max_length=training_args.generation_max_length or data_args.cutoff_len,
		    generation_num_beams=data_args.eval_num_beams or training_args.generation_num_beams,
	    )
    )
    model = model.to(torch.bfloat16)
    if custom_args['do_steer']:
        if custom_args['optimize_steering']:
            hermes = AutoModelForCausalLM.from_pretrained('teknium/OpenHermes-2.5-Mistral-7B', cache_dir='/scratch/jp6263/slackV2/hf/models/', torch_dtype=torch.float16).to('cuda')
            hermes_tokenizer = AutoTokenizer.from_pretrained('teknium/OpenHermes-2.5-Mistral-7B', cache_dir='/scratch/jp6263/slackV2/hf/models/') #TODO change cache_dir via args
            optimizer = get_evolutionary_optimizer(hermes, hermes_tokenizer, weight_percentage=0.5)
            steering = Steering(custom_args['steering_dataset'], model, tokenizer, custom_args['steering_data_path'], custom_args,
                                 optimizer=optimizer, expunge=True, preserve_categories=True, n_new=None, optimizer_frequency=None)
        else:
            steering = Steering(custom_args['steering_dataset'], model, tokenizer, custom_args['steering_data_path'], custom_args)
        print(f"Steering dataset: '{custom_args['steering_dataset']}' found at: {custom_args['steering_data_path']}")
    else:
        steering = None
        print("Warning: steering is not enabled.")
    # samples_callback = SamplesCallback(train_dataset, eval_dataset, data_args, training_args, generating_args, custom_args, steering)
    # callbacks.append(samples_callback)
    if custom_args['do_steer']:
        trainer = SteeringTrainer(
			custom_args=custom_args,
			steering=steering,
			model=model,
			args=training_args,
			tokenizer=tokenizer,
			data_collator=data_collator,
			callbacks=callbacks,
		    compute_metrics=ComputeMetrics(tokenizer) if training_args.predict_with_generate else None,
			**split_dataset(dataset, data_args, training_args),
		)
    else:
        trainer = Seq2SeqTrainer(
			model=model,
			args=training_args,
			tokenizer=tokenizer,
			data_collator=data_collator,
			callbacks=callbacks,
			compute_metrics=ComputeMetrics(tokenizer) if training_args.predict_with_generate else None,
			**split_dataset(dataset, data_args, training_args),
		)

    print("Starting training...")
    train_result = trainer.train(resume_from_checkpoint=training_args.resume_from_checkpoint)
    trainer.log_metrics("train", train_result.metrics)
    trainer.save_metrics("train", train_result.metrics)
    trainer.save_state()
    trainer.save_model()

