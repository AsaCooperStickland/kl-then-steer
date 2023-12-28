
from transformers import DataCollatorForSeq2Seq, Seq2SeqTrainingArguments
from llmtuner.data import get_dataset, preprocess_dataset, split_dataset
from llmtuner.extras.constants import IGNORE_INDEX
from llmtuner.extras.misc import get_logits_processor

from transformers import TrainerCallback
from torch.utils.data import DataLoader
from transformers import AutoTokenizer
import os
import pandas as pd
import torch
from datetime import datetime
from tqdm import tqdm

class SamplesCallback(TrainerCallback):
	def __init__(self, dataset, data_args, training_args, generating_args, custom_args, steering):
		config_kwargs = {'trust_remote_code': True, 'cache_dir': None, 'revision': 'main', 'token': None}
		tokenizer = AutoTokenizer.from_pretrained(custom_args['model_name_or_path'],
					use_fast=True,
					split_special_tokens = False,
					padding_side="left",
					**config_kwargs)
		tokenizer.pad_token_id = 0 if tokenizer.pad_token_id is None else tokenizer.pad_token_id
		tokenizer.bos_token_id = 1
		self.tokenizer = tokenizer
		self.steering = steering
		self.custom_args = custom_args

		dataset = preprocess_dataset(dataset, tokenizer, data_args, training_args, stage="sft")

		data_collator = DataCollatorForSeq2Seq(
			tokenizer=tokenizer,
			pad_to_multiple_of=4, # for shift short attention
			label_pad_token_id=IGNORE_INDEX if data_args.ignore_pad_token_for_loss else tokenizer.pad_token_id
		)

		# # could also use that code for train dataloader?
		# # eval_dataloader = state.get_eval_dataloader(self.dataset)
		dataloader_params = {
			"batch_size": 4,
			"collate_fn": data_collator,
			"num_workers": training_args.dataloader_num_workers,
			"pin_memory": training_args.dataloader_pin_memory,
		}
		self.loader = DataLoader(dataset, **dataloader_params)

		# Keyword arguments for `model.generate`
		gen_kwargs = generating_args.to_dict()
		gen_kwargs["eos_token_id"] = [tokenizer.eos_token_id] + tokenizer.additional_special_tokens_ids
		gen_kwargs["pad_token_id"] = tokenizer.pad_token_id
		gen_kwargs["logits_processor"] = get_logits_processor()
		self.gen_kwargs = gen_kwargs

	def on_step_begin(self, args, state, control, model, **kwargs):
		print(f"Generating samples at step {state.global_step}")

		if self.custom_args['samples_freq'] != -1 and (state.global_step % self.custom_args['samples_freq'] == 0):
			self.sample_and_save(state, model, 'none')

			self.steering.do_shift(mode='train')
			self.sample_and_save(state, model, 'train')
			self.steering.reset()

			self.steering.do_shift(mode='test')
			self.sample_and_save(state, model, 'test')
			self.steering.reset()

	def sample_and_save(self, state, model, steering_mode):
		records = []
		for inputs in self.loader:
			# for left-padded tensors for generation
			gen_input_ids = []
			for i in range(inputs['input_ids'].size(0)):
				# Find the index of the first non -100 label
				label_idx = (inputs['labels'][i] != -100).nonzero(as_tuple=True)[0]
				if len(label_idx) > 0:
					# If there are labels for generation, truncate input_ids before this label
					gen_input_ids.append(inputs['input_ids'][i, :label_idx[0]])
				else:
					# If there are no labels for generation, take the whole input
					gen_input_ids.append(inputs['input_ids'][i])
			max_length = max(x.size(0) for x in gen_input_ids)
			padded_tensors = [torch.nn.functional.pad(x, (max_length - x.size(0), 0), value=self.tokenizer.pad_token_id) for x in gen_input_ids]
			gen_input_ids = torch.stack(padded_tensors)

			decoded_input_texts = []
			for input_id_tensor in gen_input_ids:
				# Decode each input ID tensor to a string
				decoded_input_text = self.tokenizer.decode(input_id_tensor, skip_special_tokens=True)
				decoded_input_texts.append(decoded_input_text)

			num_return_sequences = 4
			self.gen_kwargs['num_return_sequences'] = num_return_sequences
			gen = model.generate(input_ids=gen_input_ids.cuda(), **self.gen_kwargs)
			generated_texts = []
			for decoded_input_text_i, decoded_input_text in enumerate(decoded_input_texts):
				for sample_i in range(num_return_sequences):
					gen_i = decoded_input_text_i * num_return_sequences + sample_i
					gen_decoded = self.tokenizer.decode(gen[gen_i], skip_special_tokens=True)
					# generated_texts.append(decoded_text)
					gen_only = gen_decoded[len(decoded_input_text):]
					records.append({'step': state.global_step, 'steering_mode': steering_mode, 'input': decoded_input_text, 'sample_i': sample_i, 'generation': gen_only})

			# generated_texts = [generated_texts[i][len(decoded_input_texts[i]):] for i in range(len(decoded_input_texts))]

			# for input_text, generated_text in zip(decoded_input_texts, generated_texts):
			# 	records.append({'step': state.global_step, 'input': input_text, 'generation': generated_text, 'steering_mode': steering_mode})
		df = pd.DataFrame(records)
		# time = datetime.now().strftime("%Y-%m-%d_%H:%M")
		# csv_path = os.path.join(self.output_dir, f'samples_step_{state.global_step}_time_{time}.csv')
		csv_path = os.path.join(self.custom_args['samples_dir'], f'samples_{self.custom_args["run_name"]}.csv')
		os.makedirs(self.custom_args['samples_dir'], exist_ok=True)
		df.to_csv(csv_path, mode='a', header=not os.path.exists(csv_path), index=False)