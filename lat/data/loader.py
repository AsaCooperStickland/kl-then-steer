import os
from typing import TYPE_CHECKING, Literal, Union

from datasets import load_from_disk

from llmtuner.extras.logging import get_logger
from llmtuner.data.parser import get_dataset_list
from llmtuner.data.preprocess import get_preprocess_and_print_func
from llmtuner.data.loader import load_single_dataset, merge_dataset
from lat.data.template import get_template_and_fix_tokenizer


if TYPE_CHECKING:
    from datasets import Dataset, IterableDataset
    from transformers import Seq2SeqTrainingArguments
    from transformers.tokenization_utils import PreTrainedTokenizer

    from llmtuner.hparams import DataArguments, ModelArguments


logger = get_logger(__name__)


def get_dataset(
    tokenizer: "PreTrainedTokenizer",
    model_args: "ModelArguments",
    data_args: "DataArguments",
    training_args: "Seq2SeqTrainingArguments",
    stage: Literal["pt", "sft", "rm", "ppo"],
    # split: Optional[str] = "train", # TODO: add split
) -> Union["Dataset", "IterableDataset"]:
    template = get_template_and_fix_tokenizer(tokenizer, data_args.template)
    if data_args.train_on_prompt and template.efficient_eos:
        raise ValueError("Current template does not support `train_on_prompt`.")

    # Load from cache
    if data_args.cache_path is not None:
        if os.path.exists(data_args.cache_path):
            logger.warning("Loading dataset from disk will ignore other data arguments.")
            dataset = load_from_disk(data_args.cache_path)
            if data_args.streaming:
                dataset = dataset.to_iterable_dataset()
            return dataset

    with training_args.main_process_first(desc="load dataset"):
        all_datasets = []
        for dataset_attr in get_dataset_list(data_args):
            all_datasets.append(load_single_dataset(dataset_attr, model_args, data_args))
        dataset = merge_dataset(all_datasets, data_args, training_args)

    with training_args.main_process_first(desc="pre-process dataset"):
        preprocess_func, print_function = get_preprocess_and_print_func(
            tokenizer, template, data_args, training_args, stage
        )
        column_names = list(next(iter(dataset)).keys())
        kwargs = {}
        if not data_args.streaming:
            kwargs = dict(
                num_proc=data_args.preprocessing_num_workers,
                load_from_cache_file=(not data_args.overwrite_cache),
                desc="Running tokenizer on dataset",
            )

        dataset = dataset.map(preprocess_func, batched=True, remove_columns=column_names, **kwargs)

        if data_args.cache_path is not None and not os.path.exists(data_args.cache_path):
            if training_args.should_save:
                dataset.save_to_disk(data_args.cache_path)
                logger.info("Dataset cache saved at {}.".format(data_args.cache_path))

        if training_args.should_log:
            try:
                print_function(next(iter(dataset)))
            except StopIteration:
                raise RuntimeError("Cannot find valid samples, check `data/README.md` for the data format.")

        return dataset