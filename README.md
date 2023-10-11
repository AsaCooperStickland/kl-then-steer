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