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

To install the `representation-engineering` submodule, run `git submodule update --init --recursive`, then `cd representation-engineering` and `pip install -e .`.

## Usage
```
python lat/finetuning/finetune.py [--flash_attn]
# track progress on https://wandb.ai/alexlyzhov_team/lat

# chat after finetuning
python lat/finetuning/cli_chat.py [--checkpoint_dir=results/tmp/checkpoint-10] [--flash_attn]
```

## Evaluation Usage

You can evaluate a llama-7b or mistral on various steering vectors with the following:

 ```
 if [ "$1" == "NousResearch/Nous-Hermes-2-Mistral-7B-DPO" ] ; then
 TEMPLATE="chatml"
 else 
 TEMPLATE="llama2chatsimple"
 fi
 # PCA steering with the fix so it's no longer normalized
 python scripts/test_repe.py --flash_attn --model_name_or_path $1 --steering_dataset $2 --output_dir /scratch/alc9734/latent-adversarial-training/results/ --steering_data_path $3 --test_setting vanilla --template $TEMPLATE --steering_unnormalized
 # Difference of two means steering
 python scripts/test_repe.py --flash_attn --model_name_or_path $1 --steering_dataset $2 --output_dir /scratch/alc9734/latent-adversarial-training/results/ --steering_data_path $3 --test_setting vanilla --template $TEMPLATE --steering_unnormalized --direction_method cluster_mean --start_layer $4 --end_layer $5
 ```

 Or to evaluate a trained lora model:
  ```
 if [ "$1" == "NousResearch/Nous-Hermes-2-Mistral-7B-DPO" ] ; then
 TEMPLATE="chatml"
 else 
 TEMPLATE="llama2chatsimple"
 fi
 # PCA steering with the fix so it's no longer normalized
 python scripts/test_repe.py --flash_attn--steering_dataset $2 --output_dir /scratch/alc9734/latent-adversarial-training/results/ --steering_data_path $3 --test_setting vanilla --template $TEMPLATE --steering_unnormalized --model_name_or_path /vast/work/public/ml-datasets/llama-2/Llama-2-7b-chat-hf --adapter_name_or_path $1 --finetuning_type lora
 # Difference of two means steering
 python scripts/test_repe.py --flash_attn --steering_dataset $2 --output_dir /scratch/alc9734/latent-adversarial-training/results/ --steering_data_path $3 --test_setting vanilla --template $TEMPLATE --steering_unnormalized --direction_method cluster_mean --start_layer $4 --end_layer $5 --model_name_or_path /vast/work/public/ml-datasets/llama-2/Llama-2-7b-chat-hf --adapter_name_or_path $1 --finetuning_type lora
 ```
Set start_layer as -11 and  end_layer as -30 for the default setting
`python scripts/evaluate_results.py --steering_unnormalized`
`python scripts/evaluate_results.py --steering_unnormalized --direction_method cluster_mean`
Are the equivalent evaluation scripts

## MT-Bench Evaluation
You need to install a new environment for the `FastChat` repository
`cd FastChat`. Then install with `pip install -e .`, you also need to reinstall the `lat` module by `cd ../ ; pip install -e .`.
For evaluation on MT-Bench first navigate to the `llm_judge` directory:
`cd FastChat/fastchat/llm_judge/`.
Example answer generation:
```
# for cluster_mean
for m in /vast/work/public/ml-datasets/llama-2/Llama-2-7b-chat-hf ; do bash eval_steering.sh $m llama-2-chat7b mean ; done;
# for unnormalized pca
for m in /vast/work/public/ml-datasets/llama-2/Llama-2-7b-chat-hf ; do bash eval_steering.sh $m llama-2-chat7b pca_unnorm ; done;
```
Then set your `OPENAI_API_KEY` and run `judge.sh` to judge the answers using to gpt-4. Finally run `python show_result.py`.


