from typing import TYPE_CHECKING, Optional, Union
import os
import torch
from torch import nn
from transformers import AutoModelForCausalLM, AutoTokenizer
from huggingface_hub import snapshot_download

from llmtuner.extras.logging import get_logger
from llmtuner.hparams import FinetuningArguments, ModelArguments
from llmtuner.model import load_model_and_tokenizer, load_valuehead_params


if TYPE_CHECKING:
    from trl import AutoModelForCausalLMWithValueHead


logger = get_logger(__name__)


class GPTRewardModel(nn.Module):
    def __init__(self, model_path):
        super().__init__()
        model = AutoModelForCausalLM.from_pretrained(model_path)
        self.config = model.config
        self.config.n_embd = self.config.hidden_size if hasattr(self.config, "hidden_size") else self.config.n_embd
        self.model = model
        self.transformer = model.model
        self.v_head = nn.Linear(self.config.n_embd, 1, bias=False)
        self.tokenizer = AutoTokenizer.from_pretrained(model_path)
        self.tokenizer.pad_token = self.tokenizer.unk_token
        self.PAD_ID = self.tokenizer(self.tokenizer.pad_token)["input_ids"][0]

    def get_device(self):
        return self.model.device

    def forward(
        self,
        input_ids=None,
        past_key_values=None,
        attention_mask=None,
        **kwargs,
    ):
        """
        input_ids, attention_mask: torch.Size([bs, seq_len])
        return: scores: List[bs]
        """
        base_model_output = self.transformer(
            input_ids=input_ids,
            attention_mask=attention_mask,
            **kwargs,
        )

        last_hidden_state = base_model_output.hidden_states[-1]
        lm_logits = base_model_output.logits
        loss = base_model_output.loss

        if last_hidden_state.device != self.v_head.summary.weight.device:
            last_hidden_state = last_hidden_state.to(self.v_head.summary.weight.device)

        value = self.v_head(last_hidden_state).squeeze(-1)

        # force upcast in fp32 if logits are in half-precision
        if lm_logits.dtype != torch.float32:
            lm_logits = lm_logits.float()

        return (lm_logits, loss, value)


## Load the model and tokenizer
def get_starling_reward_model(base_model):
    assert base_model in ["meta-llama/Llama-2-7b-hf", "/vast/work/public/ml-datasets/llama-2/Llama-2-7b-chat-hf"]
    reward_model = GPTRewardModel(base_model)
    reward_tokenizer = reward_model.tokenizer
    reward_tokenizer.truncation_side = "left"
    
    directory = snapshot_download("berkeley-nest/Starling-RM-7B-alpha")
    for fpath in os.listdir(directory):
        if fpath.endswith(".pt") or fpath.endswith("model.bin"):
            checkpoint = os.path.join(directory, fpath)
            break
       
    reward_model.load_state_dict(torch.load(checkpoint), strict=False)
    reward_model.eval().requires_grad_(False)
    return reward_model

    
def create_reward_model(
    model: "AutoModelForCausalLMWithValueHead", model_args: "ModelArguments", finetuning_args: "FinetuningArguments"
) -> "AutoModelForCausalLMWithValueHead":
    r"""
    Creates reward model for PPO training.
    """
    if finetuning_args.reward_model_type == "api":
        assert finetuning_args.reward_model.startswith("http"), "Please provide full url."
        logger.info("Use reward server {}".format(finetuning_args.reward_model))
        return finetuning_args.reward_model
    elif finetuning_args.reward_model_type == "lora":
        model.pretrained_model.load_adapter(finetuning_args.reward_model, "reward")
        for name, param in model.named_parameters():  # https://github.com/huggingface/peft/issues/1090
            if "default" in name:
                param.data = param.data.to(torch.float32)  # trainable params should in fp32
        vhead_params = load_valuehead_params(finetuning_args.reward_model, model_args)
        assert vhead_params is not None, "Reward model is not correctly loaded."
        model.register_buffer("reward_head_weight", vhead_params["v_head.summary.weight"], persistent=False)
        model.register_buffer("reward_head_bias", vhead_params["v_head.summary.bias"], persistent=False)
        model.register_buffer(
            "default_head_weight", torch.zeros_like(vhead_params["v_head.summary.weight"]), persistent=False
        )
        model.register_buffer(
            "default_head_bias", torch.zeros_like(vhead_params["v_head.summary.bias"]), persistent=False
        )
        logger.info("Loaded adapter weights of reward model from {}".format(finetuning_args.reward_model))
        return None
    elif finetuning_args.reward_model_type == "starling":
        reward_model = get_starling_reward_model(model_args.model_name_or_path)
        reward_model = reward_model.to(model_args.compute_dtype) if not getattr(reward_model, "quantization_method", None) else reward_model
        return reward_model
    else:
        reward_model_args_dict = model_args.to_dict()
        reward_model_args_dict.update(
            dict(
                model_name_or_path=finetuning_args.reward_model,
                adapter_name_or_path=finetuning_args.reward_model_adapters,
                quantization_bit=finetuning_args.reward_model_quantization_bit,
            )
        )
        reward_model_args = ModelArguments(**reward_model_args_dict)
        reward_finetuning_args = FinetuningArguments(finetuning_type="lora")
        reward_model, _ = load_model_and_tokenizer(
            reward_model_args, reward_finetuning_args, is_trainable=False, add_valuehead=True
        )
        logger.info("Loaded full weights of reward model from {}".format(finetuning_args.reward_model))
        logger.warning("Please ensure the ppo model and reward model share SAME tokenizer and vocabulary.")
        return reward_model
