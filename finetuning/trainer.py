import os
import json
import torch
import numpy as np
import torch.nn as nn
from typing import TYPE_CHECKING, Any, Dict, List, Optional, Tuple, Union
from transformers import Seq2SeqTrainer

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
from steering import Steering


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

        self.steering = Steering(self.custom_args['steering_dataset'], self.model.model, self.tokenizer, self.custom_args['steering_data_path'])

        self.wrapped_model = rep_control_reading_vec.WrappedReadingVecModel(self.model.model, self.tokenizer)
        self.wrapped_model.unwrap()
        self.wrapped_model.wrap_block(self.layer_id, block_name=self.block_name)
        self.wrapped_model.reset()
    
    def sample_coeff(self):
        return 3.0 if random.random() < 0.5 else 0.0

    def compute_loss(self, model, inputs, return_outputs=False):
        """
        How the loss is computed by Trainer. By default, all models return the loss in the first element.

        Subclass and override for custom behavior.
        """
        coeff = self.sample_coeff()
        activations = self.steering.get_shift(coeff=coeff, layer_id=self.layer_id)
        self.wrapped_model.reset()
        self.wrapped_model.set_controller(self.layer_id, activations, self.block_name)
        
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