# Finetuning with steering

## Installation
```

Clone the repo, run `pip install -e .`, and also install `python-dotenv` and `matplotlib`.
Add your huggingface token (assuming you have access to llama2) to a `.env` file as `HF_TOKEN`.
pip install wandb scikit-learn
pip install llmtuner==0.5.0 git+https://github.com/andyzoujm/representation-engineering.git@97ec903#egg=repe

huggingface-cli login
# provide your huggingface token

wandb login
# provide your wandb API key
```

You may also need to convert your old cache `cache.json` to an sqlite database with `scripts/convert_cache.py`.

## Usage
```
python lat/finetuning/finetune.py [--flash_attn]
# track progress on https://wandb.ai/alexlyzhov_team/lat

# chat after finetuning
python lat/finetuning/cli_chat.py [--checkpoint_dir=results/tmp/checkpoint-10] [--flash_attn]
```

