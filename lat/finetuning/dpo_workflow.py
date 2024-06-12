# Inspired by: https://github.com/huggingface/trl/blob/main/examples/research_projects/stack_llama_2/scripts/dpo_llama2.py

from typing import TYPE_CHECKING, List, Optional

from transformers import Seq2SeqTrainingArguments
from peft import LoraConfig, PeftModel, TaskType, get_peft_model
from lat.data.loader import get_dataset
from lat.utils import de_lorafy_

from llmtuner.data import split_dataset
from llmtuner.extras.constants import IGNORE_INDEX
from llmtuner.extras.ploting import plot_loss
from llmtuner.hparams import ModelArguments
from llmtuner.model import load_model_and_tokenizer
from llmtuner.model.utils import find_all_linear_modules
from llmtuner.train.dpo.collator import DPODataCollatorWithPadding
from llmtuner.train.dpo.trainer import CustomDPOTrainer
from llmtuner.train.utils import create_modelcard_and_push, create_ref_model


if TYPE_CHECKING:
    from transformers import TrainerCallback

    from llmtuner.hparams import DataArguments, FinetuningArguments


def run_dpo(
    model_args: "ModelArguments",
    data_args: "DataArguments",
    training_args: "Seq2SeqTrainingArguments",
    finetuning_args: "FinetuningArguments",
    generating_args: "GeneratingArguments",
    callbacks: Optional[List["TrainerCallback"]] = None,
    custom_args=None,
):
    # if custom_args["bias"] == "all":
    #     finetuning_args.finetuning_type = "full"
    existing_adapter = model_args.adapter_name_or_path is not None
    model, tokenizer = load_model_and_tokenizer(model_args, finetuning_args, 
                                                training_args.do_train)
    if existing_adapter:
        model = model.merge_and_unload()
    print(model)
    if custom_args["bias"] == "all":
        target_modules = find_all_linear_modules(model.base_model.model if not existing_adapter else model)
        peft_kwargs = {
                "r": finetuning_args.lora_rank,
                "target_modules": target_modules,
                "lora_alpha": finetuning_args.lora_alpha,
                "lora_dropout": finetuning_args.lora_dropout,
                # "bias": "lora_only",
            }
        lora_config = LoraConfig(
                    task_type=TaskType.CAUSAL_LM,
                    inference_mode=False,
                    modules_to_save=finetuning_args.additional_target,
                    **peft_kwargs,
                )
        print("Reseting lora config to include bias parameters")
        model = get_peft_model(model.base_model.model if not existing_adapter else model, 
                               lora_config)
        print(model)
    if custom_args["batch_lora"]:
        layer_list = range(0, 6)
        de_lorafy_(model, layer_list)
        print(model)
        model.print_trainable_parameters()
    dataset = get_dataset(tokenizer, model_args, data_args, training_args, stage="rm")
    data_collator = DPODataCollatorWithPadding(
        tokenizer=tokenizer,
        pad_to_multiple_of=8,
        label_pad_token_id=IGNORE_INDEX if data_args.ignore_pad_token_for_loss else tokenizer.pad_token_id,
    )

    # Create reference model
    if finetuning_args.ref_model is None and (not training_args.do_train):  # use the model itself
        ref_model = model
    else:
        ref_model = create_ref_model(model_args, finetuning_args)

    # Update arguments
    training_args_dict = training_args.to_dict()
    training_args_dict.update(dict(remove_unused_columns=False))  # important for pairwise dataset
    training_args = Seq2SeqTrainingArguments(**training_args_dict)

    # Initialize our Trainer
    trainer = CustomDPOTrainer(
        beta=finetuning_args.dpo_beta,
        loss_type=finetuning_args.dpo_loss,
        ftx_gamma=finetuning_args.dpo_ftx,
        model=model,
        ref_model=ref_model,
        args=training_args,
        tokenizer=tokenizer,
        data_collator=data_collator,
        callbacks=callbacks,
        **split_dataset(dataset, data_args, training_args),
    )
    trainer.reference_free = False

    # Training
    if training_args.do_train:
        train_result = trainer.train(resume_from_checkpoint=training_args.resume_from_checkpoint)
        trainer.save_model()
        trainer.log_metrics("train", train_result.metrics)
        trainer.save_metrics("train", train_result.metrics)
        trainer.save_state()
        if trainer.is_world_process_zero() and finetuning_args.plot_loss:
            plot_loss(training_args.output_dir, keys=["loss", "eval_loss"])

    # Evaluation
    if training_args.do_eval:
        metrics = trainer.evaluate(metric_key_prefix="eval")
        if id(model) == id(ref_model):  # unable to compute rewards without a reference model
            remove_keys = [key for key in metrics.keys() if "rewards" in key]
            for key in remove_keys:
                metrics.pop(key)
        trainer.log_metrics("eval", metrics)
        trainer.save_metrics("eval", metrics)

    # Create model card
    create_modelcard_and_push(trainer, model_args, data_args, training_args, finetuning_args)