# latent-adversarial-training

## Installation
Clone the repo, run `pip install -e .`, and also install `python-dotenv` and `matplotlib`.
Add your huggingface token (assuming you have access to llama2) to a `.env` file as `HF_TOKEN`.
## Usage
Generate steering vectors as follows:
```
python lat/generate_steering_vectors.py
```
Test them on a small set of questions
```
python lat/test_steering_vectors.py
```

## Finetuning: Installation & Usage

```
pip install wandb scikit-learn
pip install llmtuner==0.2.3 git+https://github.com/andyzoujm/representation-engineering.git@97ec903#egg=repe

huggingface-cli login
# provide your huggingface token

wandb login
# provide your wandb API key

python finetuning/train_bash.py
```