"""Loading datasets."""

import structlog
import syntaxi
import itertools

from datasets import load_dataset, Dataset
from tokenizers import Tokenizer
from dataclasses import dataclass

from accelerate import Accelerator
from transformers import default_data_collator

from training.config import DatasetConfig
from torch.utils.data import DataLoader


logger = structlog.get_logger(__name__)


@dataclass
class DatasetSplit:
    train: DataLoader
    validation: DataLoader
    test: DataLoader


def tokenizer_from_dataset_config(dataset_config: DatasetConfig) -> Tokenizer:
    tokenizer = Tokenizer.from_pretrained(dataset_config.tokenizer_id)

    # Patch using Syntaxi.
    if dataset_config.use_syntaxi:
        tokenizer = syntaxi.patched_tokenizer(tokenizer)

    return tokenizer

def datasplit_from_dataset_config(
    dataset_config: DatasetConfig,
    main_process_first,
    num_processing_workers: int | None = None,
) -> DatasetSplit:
    """Get split dataloaders from dataset config.

    Args:
        dataset_config (DatasetConfig): Dataset loading and processing configuration.
        main_process_first (...): Context manager for doing things on main process first.
        num_processing_workers (int | None, optional): Number of preprocessing workers.
            Defaults to None.

    Returns:
        DatasetSplit: Dataset split into train, validation and test dataloaders.
    """
    # Load tokenizer.
    tokenizer = tokenizer_from_dataset_config(dataset_config=dataset_config)

    raw_datasets = load_dataset(
        path=dataset_config.dataset_id, name=dataset_config.dataset_config_name
    )


    if list(raw_datasets) == ["train"]:
        logger.info("Only training set available. Slicing it up.")
        raw_datasets["validation"] = load_dataset(
            path=dataset_config.dataset_id,
            name=dataset_config.dataset_config_name,
            split=(
                f"train[:{int(dataset_config.validation_percentage * 100)}%]"
            ),
        )

        raw_datasets["train"] = load_dataset(
            path=dataset_config.dataset_id,
            name=dataset_config.dataset_config_name,
            split=f"train[{int(dataset_config.validation_percentage * 100)}%:]",
        )
    else:
        raw_datasets["train"] = load_dataset(
            path=dataset_config.dataset_id,
            name=dataset_config.dataset_config_name,
            split="train",
        )

        raw_datasets["validation"] = load_dataset(
            path=dataset_config.dataset_id,
            name=dataset_config.dataset_config_name,
            split="validation",
        )

    def tokenize(examples, text_key: str = dataset_config.dataset_text_key):
        return {
            "input_ids": [tokenizer.encode(text).ids for text in examples[text_key]]
        }

    with main_process_first:
        tokenized_datasets = raw_datasets.map(
            function=tokenize,
            batched=True,
            num_proc=num_processing_workers,
            remove_columns=raw_datasets["train"].column_names,
            desc="Tokenizing all of the data",
        )

    block_size: int = dataset_config.block_size

    def block_concatenate_texts(examples):
        concatenated_examples = {
            k: list(itertools.chain(*examples[k])) for k in examples
        }

        first_key = next(iter(examples))
        total_length = len(concatenated_examples[first_key])
        total_length = min(total_length, total_length // block_size * block_size)

        result = {
            k: [t[i : i + block_size] for i in range(0, total_length, block_size)]
            for k, t in concatenated_examples.items()
        }

        result["labels"] = result["input_ids"].copy()

        return result

    with accelerator.main_process_first():
        blocked_datasets = tokenized_datasets.map(
            function=block_concatenate_texts,
            batched=True,
            num_proc=num_processing_workers,
            remove_columns=tokenized_datasets["train"].column_names,
            desc=f"Chunking tokenized dataset into chunks of {block_size} tokens",
        )

    train_val_test: tuple[Dataset] = (
        blocked_datasets["train"],
        blocked_datasets["validation"],
        blocked_datasets["test"],
    )

    dataloaders: DataLoader = [
        DataLoader(
            dataset=dataset,
            shuffle=(i == 0),  # Only train.
            collate_fn=default_data_collator,
            batch_size=dataset_config.batch_size,
        )
        for i, dataset in enumerate(train_val_test)
    ]

    train, validation, test = dataloaders
    return DatasetSplit(train=train, validation=validation, test=test)
