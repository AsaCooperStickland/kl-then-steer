from typing import TYPE_CHECKING, Any, Dict, List, Optional, Tuple, Union
from contextlib import nullcontext

from llmtuner.extras.constants import IGNORE_INDEX
from llmtuner.extras.logging import get_logger

if TYPE_CHECKING:
    from transformers.trainer import PredictionOutput

import torch.nn.functional as F
import torch
from transformers.modeling_utils import unwrap_model
from accelerate.utils import is_deepspeed_available
from transformers.models.auto.modeling_auto import MODEL_FOR_CAUSAL_LM_MAPPING_NAMES
from peft import PeftModel
from trl.core import logprobs_from_logits

from llmtuner.train.sft.trainer import CustomSeq2SeqTrainer


logger = get_logger(__name__)

if is_deepspeed_available():
    import deepspeed

class SteeringTrainer(CustomSeq2SeqTrainer):
    def __init__(self, ref_model, custom_args, steering, **kwargs):
        self.ref_model = ref_model
        self.args = kwargs["args"]
        self.create_accelerator_and_postprocess()
        if ref_model:
            self.ref_model.eval()
            # self.ref_model = self.accelerator.prepare_model(self.ref_model, evaluation_mode=True)
            self.ref_model = self._prepare_deepspeed_inference(self.ref_model)
            # self.ref_model = (
            #     self.accelerator.prepare(self.ref_model)
            #     if self.is_deepspeed_enabled
            #     else self.accelerator.prepare_model(self.ref_model, evaluation_mode=True)
        super().__init__(**kwargs)
        self.custom_args = custom_args
        self.steering = steering
        self.kl_loss = custom_args["loss_function"] == "kl"
        # print(self.model)
        # print(self.accelerator.unwrap_model(self.model))
        self.optional_peft_ctx = (
            self.model.disable_adapter
            if (isinstance(self.model, PeftModel) and self.kl_loss)
            else nullcontext
        )

    def _prepare_deepspeed_inference(self, model):
        # Adapted from accelerate: https://github.com/huggingface/accelerate/blob/739b135f8367becb67ffaada12fe76e3aa60fefd/src/accelerate/accelerator.py#L1473
        deepspeed_plugin = self.accelerator.state.deepspeed_plugin
        config_kwargs = deepspeed_plugin.deepspeed_config
        if model is not None:
            if hasattr(model, "config"):
                hidden_size = (
                    max(model.config.hidden_sizes)
                    if getattr(model.config, "hidden_sizes", None)
                    else getattr(model.config, "hidden_size", None)
                )
                if hidden_size is not None and config_kwargs["zero_optimization"]["stage"] == 3:
                    # Note that `stage3_prefetch_bucket_size` can produce DeepSpeed messages like: `Invalidate trace cache @ step 0: expected module 1, but got module 0`
                    # This is expected and is not an error, see: https://github.com/microsoft/DeepSpeed/discussions/4081
                    config_kwargs.update(
                        {
                            "zero_optimization.reduce_bucket_size": hidden_size * hidden_size,
                            "zero_optimization.stage3_param_persistence_threshold": 10 * hidden_size,
                            "zero_optimization.stage3_prefetch_bucket_size": 0.9 * hidden_size * hidden_size,
                        }
                    )

        # If ZeRO-3 is used, we shard both the active and reference model.
        # Otherwise, we assume the reference model fits in memory and is initialized on each device with ZeRO disabled (stage 0)
        if config_kwargs["zero_optimization"]["stage"] != 3:
            config_kwargs["zero_optimization"]["stage"] = 0
        del config_kwargs["scheduler"]
        del config_kwargs["optimizer"]
        # config_kwargs["scheduler"]["params"]["warmup_num_steps"] = 0
        # config_kwargs["scheduler"]["params"]["total_num_steps"] = 1
        model, *_ = deepspeed.initialize(model=model, config=config_kwargs)
        model.eval()
        return model

    def compute_loss(self, model, inputs, return_outputs=False):
        """
        How the loss is computed by Trainer. By default, all models return the loss in the first element.

        Subclass and override for custom behavior.
        """
        self.steering.do_shift(mode='train')
        
        if self.label_smoother is not None and "labels" in inputs:
            labels = inputs.pop("labels")
        else:
            labels = None
        outputs = model(**inputs)

        self.steering.reset()
        if self.kl_loss:
            logprobs = logprobs_from_logits(outputs.logits, None, gather=False)
            if self.ref_model:
                with torch.no_grad():
                    self.ref_model.eval()
                    original_outputs = self.ref_model(**inputs)
                    original_logprobs = logprobs_from_logits(original_outputs.logits, None, gather=False)
            else:
                model.eval()
                with self.optional_peft_ctx():
                    original_outputs = model(**inputs)
                    original_logprobs = logprobs_from_logits(original_outputs.logits, None, gather=False)
                model.train()

        # Save past state if it exists
        # TODO: this needs to be fixed and made cleaner later.
        if self.args.past_index >= 0:
            self._past = outputs[self.args.past_index]
        
        if self.kl_loss:
                loss = F.kl_div(original_logprobs, logprobs, log_target=True, reduction="none").sum(-1)
                print(loss)
                loss = loss.mean(-1).mean(-1)
                print(loss)
        elif labels is not None:
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
        
        self.steering.reset()
        self.steering.wrapped_model.unwrap()
        super().save_model(output_dir, _internal_call)
        self.steering.wrapped_model.wrap_block(self.steering.layer_id, block_name=self.steering.block_name)