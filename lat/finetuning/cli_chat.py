from llmtuner import ChatModel
from llmtuner.extras.misc import torch_gc

import argparse
import os
import sys

try:
	import platform
	if platform.system() != "Windows":
		import readline
except ImportError:
	print("Install `readline` for a better experience.")


def main():
	script_dir = os.path.dirname(os.path.abspath(sys.argv[0]))
	os.chdir(script_dir)
	parser = argparse.ArgumentParser()
	parser.add_argument('--flash_attn', action='store_true')
	parser.add_argument('--checkpoint_dir', default=None)
	cmd_args = parser.parse_args()
	
	input_args = {
		"stage": "sft",
		"model_name_or_path": "meta-llama/Llama-2-7b-chat-hf",
		'dataset_dir': 'finetuning_data',
		"dataset": "16_examples",
		"template": "default",
		"finetuning_type": "lora",
		"lora_target": "q_proj,v_proj",
		"overwrite_cache": True,
		"plot_loss": True,
		"flash_attn": cmd_args.flash_attn,
		"checkpoint_dir": cmd_args.checkpoint_dir,
	}

	chat_model = ChatModel(input_args)
	history = []
	print("Welcome to the CLI application, use `clear` to remove the history, use `exit` to exit the application.")

	while True:
		try:
			query = input("\nUser: ")
		except UnicodeDecodeError:
			print("Detected decoding error at the inputs, please set the terminal encoding to utf-8.")
			continue
		except Exception:
			raise

		if query.strip() == "exit":
			break

		if query.strip() == "clear":
			history = []
			torch_gc()
			print("History has been removed.")
			continue

		print("Assistant: ", end="", flush=True)

		response = ""
		for new_text in chat_model.stream_chat(query, history):
			print(new_text, end="", flush=True)
			response += new_text
		print()

		history = history + [(query, response)]


if __name__ == "__main__":
	main()
