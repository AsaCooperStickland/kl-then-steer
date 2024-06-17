# KL-Then-Steer (KTS)

## Installation
```

# Clone the repo, run `pip install -e .`, and also install `python-dotenv` and `matplotlib`.
# Add your huggingface token (assuming you have access to llama2) to a `.env` file as `HF_TOKEN`.
pip install wandb scikit-learn
pip install llmtuner==0.5.0 git+https://github.com/andyzoujm/representation-engineering.git@97ec903#egg=repe

huggingface-cli login
# provide your huggingface token

wandb login
# provide your wandb API key
```

You may need to use `pip install -U [LIB] --no-build-isolatio`. You may also need to convert your old cache `cache.json` to an sqlite database with `scripts/convert_cache.py`.

To install the `representation-engineering` submodule, run `git submodule update --init --recursive`, then `cd representation-engineering` and `pip install -e .`.

## Finetuning
You need to use [Git LFS](https://git-lfs.com/) to access the training datasets.
```
# KTS training
python lat/finetuning/finetune.py --flash_attn --batch_size 16 --finetuning_type lora --output_dir $1 --steering_dataset large_scale_concept --steering_probability 0.125 --loss_function kl --learning_rate 1e-5 --steering_coeff 1.5 --steering_coeff_range both --direction_method cluster_mean
# track progress on https://wandb.ai/alexlyzhov_team/lat

# DPO training
python lat/finetuning/finetune.py --batch_size 8 --finetuning_type lora --output_dir $1 --learning_rate 1e-5 --stage dpo --dataset training_dpo_refusal0.50 --batch_lora --num_train_epochs 4

# chat after finetuning
python lat/finetuning/cli_chat.py --flash_attn --checkpoint_dir $1
```

## Evaluation

### Inference
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

### Toxicity Evaluation
Set start_layer as -11 and  end_layer as -30 for the default setting
`python scripts/evaluate_results.py --steering_unnormalized`
`python scripts/evaluate_results.py --steering_unnormalized --direction_method cluster_mean`
are the equivalent evaluation scripts.
This can be adapted for other desired safety metrics.

### MT-Bench Evaluation
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


