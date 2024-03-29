"""Train autoregressive language model using Trainer."""

import os
import math
import evaluate
import typer
import wandb
import torch

from datasets import Dataset
from transformers import (
    AutoModelForCausalLM,
    CONFIG_MAPPING,
    Trainer,
    default_data_collator,
    set_seed,
    PreTrainedTokenizerFast,
)

from training.transformer.config import TrainingConfig
from training.transformer.data import (
    datasplit_from_dataset_config,
    tokenizer_from_dataset_config,
)
from transformers.trainer_utils import get_last_checkpoint

from typing import Any


def train(config_path: str) -> None:
    """Train from config path.

    Args:
        config_path (str): Path to training config.
    """
    training_config = TrainingConfig.from_toml_path(file_path=config_path)
    set_seed(training_config.seed)

    model_config = CONFIG_MAPPING[training_config.model_config.model_type]()
    if overrides := training_config.model_config.model_config_overrides:
        model_config.update(config_dict=overrides)

    torch_dtype = (
        model_config.torch_dtype
        if model_config.torch_dtype in ["auto", None]
        else getattr(torch, model_config.torch_dtype)
    )

    if base_model := training_config.model_config.base_model:
        model = AutoModelForCausalLM.from_pretrained(
            base_model,
            config=model_config,
            torch_dtype=torch_dtype,
            trust_remote_code=True,
        )
    else:
        model = AutoModelForCausalLM.from_config(
            config=model_config, trust_remote_code=True, torch_dtype=torch_dtype
        )

    tokenizer = PreTrainedTokenizerFast(
        tokenizer_object=tokenizer_from_dataset_config(training_config.dataset_config)
    )
    num_params: int = sum(p.numel() for p in model.parameters())

    training_dataset, validation_dataset = datasplit_from_dataset_config(
        dataset_config=training_config.dataset_config,
        training_config=training_config,
    )

    if training_config.wandb:
        with open(config_path, "rb") as file:
            training_config_toml_data = __import__("tomllib").load(file)["training"]

        wandb.init(
            project=training_config.wandb.project,
            notes=training_config.wandb.notes,
            tags=training_config.wandb.tags + [f"{num_params / 1e6:.1f}M"],
            config=training_config_toml_data,
        )

        os.environ["WANDB_PROJECT"] = training_config.wandb.project
        os.environ["WANDB_LOG_MODEL"] = "checkpoint"

        training_config.report_to = ["wandb"]

    # Helper for processing raw logits.
    def preprocess_logits_for_metrics(logits, labels):
        if isinstance(logits, tuple):
            logits = logits[0]
        return logits.argmax(dim=-1)

    # Everyone's favourite metric.
    metric = evaluate.load("accuracy")

    # This is how we compute metrics during training.
    def compute_metrics(eval_preds):
        preds, labels = eval_preds
        labels = labels[:, 1:].reshape(-1)
        preds = preds[:, :-1].reshape(-1)
        return metric.compute(predictions=preds, references=labels)

    trainer = Trainer(
        model=model,
        args=training_config,
        train_dataset=training_dataset,
        eval_dataset=validation_dataset,
        tokenizer=tokenizer,
        data_collator=default_data_collator,
        compute_metrics=compute_metrics,
        preprocess_logits_for_metrics=preprocess_logits_for_metrics,
    )

    last_checkpoint_maybe = (
        get_last_checkpoint(training_config.output_dir)
        if training_config.resume_from_checkpoint
        else None
    )
    train_result = trainer.train(resume_from_checkpoint=last_checkpoint_maybe)
    trainer.save_model()

    metrics_training = train_result.metrics
    metrics_training["train_samples"] = len(training_dataset)

    trainer.log_metrics("train", metrics_training)
    trainer.save_state()

    metrics_eval = trainer.evaluate()
    metrics_eval["eval_samples"] = len(validation_dataset)

    try:
        perplexity = math.exp(metrics_eval["eval_loss"])
    except OverflowError:
        perplexity = float("inf")

    metrics_eval["perplexity"] = perplexity

    trainer.log_metrics("eval", metrics_eval)
    trainer.save_metrics("eval", metrics_eval)

    if training_config.push_to_hub:
        trainer.push_to_hub(
            dataset_tags=training_config.dataset_config.dataset_id,
            dataset=f"{training_config.dataset_config.dataset_id}",
            tasks="text-generation",
            finetuned_from=training_config.model_config.base_model,
        )


if __name__ == "__main__":
    typer.run(train)
