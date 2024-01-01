import os
import json
import torch
import numpy as np
import torch.nn as nn
from typing import TYPE_CHECKING, Any, Dict, List, Optional, Tuple, Union

from llmtuner.extras.constants import IGNORE_INDEX
from llmtuner.extras.logging import get_logger

if TYPE_CHECKING:
    from transformers.trainer import PredictionOutput

from transformers.modeling_utils import unwrap_model
from transformers.models.auto.modeling_auto import MODEL_FOR_CAUSAL_LM_MAPPING_NAMES

from tqdm import tqdm
import random

from repe import rep_control_reading_vec
from llmtuner.train.sft.trainer import CustomSeq2SeqTrainer
from lat.finetuning.steering import Steering


logger = get_logger(__name__)


class SteeringTrainer(CustomSeq2SeqTrainer):
    def __init__(self, custom_args, **kwargs):
        super().__init__(**kwargs)
        self.custom_args = custom_args

        self.layer_id = list(range(-11, -30, -1))
        self.block_name = "decoder_block"

        # self.model:
        # PeftModelForCausalLM(
        #   (base_model): LoraModel(
        #   (model): LlamaForCausalLM(
        #     (model): LlamaModel(
        self.model.to(torch.bfloat16)

        model_to_steer = self.model.model if self.custom_args['finetuning_type'] == 'lora' else self.model

        self.steering = Steering(self.custom_args['steering_dataset'], model_to_steer, self.tokenizer, self.custom_args['steering_data_path'], self.custom_args)

        self.wrapped_model = rep_control_reading_vec.WrappedReadingVecModel(model_to_steer, self.tokenizer)
        self.wrapped_model.unwrap()
        self.wrapped_model.wrap_block(self.layer_id, block_name=self.block_name)
        self.wrapped_model.reset()
    
    def sample_coeff(self):
        if self.custom_args['model_name_or_path'] == 'meta-llama/Llama-2-7b-chat-hf':
            return 1.5 if random.random() < 0.5 else 0.0
            # return 1.5
        elif self.custom_args['model_name_or_path'] == 'meta-llama/Llama-2-13b-chat-hf':
            return 3.0 if random.random() < 0.5 else 0.0
            # return 3.0
        else:
            # uniform random between 0 and 1.5
            scale = random.random() * 1.5 
            return scale if random.random() < 0.5 else 0.0

    def compute_loss(self, model, inputs, return_outputs=False):
        """
        How the loss is computed by Trainer. By default, all models return the loss in the first element.

        Subclass and override for custom behavior.
        """
        coeff = self.sample_coeff()
        activations = self.steering.get_shift(coeff=coeff, layer_id=self.layer_id, num_pairs=10)
        self.wrapped_model.reset()
        for key in activations:
            activations[key] = activations[key].to(torch.bfloat16)
        self.wrapped_model.set_controller(self.layer_id, activations, self.block_name)
        self.wrapped_model.to(torch.bfloat16)
        
        if self.label_smoother is not None and "labels" in inputs:
            labels = inputs.pop("labels")
        else:
            labels = None
        outputs = model(**inputs)

        self.wrapped_model.reset()

        # Save past state if it exists
        # TODO: this needs to be fixed and made cleaner later.
        if self.args.past_index >= 0:
            self._past = outputs[self.args.past_index]

        if labels is not None:
            if unwrap_model(model)._get_name() in MODEL_FOR_CAUSAL_LM_MAPPING_NAMES.values():
                loss = self.label_smoother(outputs, labels, shift_labels=True)
            else:
                loss = self.label_smoother(outputs, labels)
        else:
            if isinstance(outputs, dict) and "loss" not in outputs:
                raise ValueError(
                    "The model did not return a loss from the inputs, only the following keys: "
                    f"{','.join(outputs.keys())}. For reference, the inputs it received are {','.join(inputs.keys())}."
                )
            # We don't use .loss here since the model may return tuples instead of ModelOutput.
            loss = outputs["loss"] if isinstance(outputs, dict) else outputs[0]

        return (loss, outputs) if return_outputs else loss
    
    def save_model(self, output_dir: Optional[str] = None, _internal_call: bool = False):
        """
        Will save the model, so you can reload it using `from_pretrained()`.

        Will only save from the main process.
        """
        
        self.wrapped_model.reset()
        self.wrapped_model.unwrap()
        super().save_model(output_dir, _internal_call)
        self.wrapped_model.wrap_block(self.layer_id, block_name=self.block_name)

