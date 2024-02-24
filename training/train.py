"""Train a model."""

import typer
import torch
import math
import wandb
import numpy as np
import tempfile

from accelerate import Accelerator
from torch.utils.data import DataLoader
from torch.optim import Optimizer
from typing import Any

from models.transformer import Transformer
from training.artifacts import wandb_save_transformer_states

from training.data import datasplit_from_dataset_config
from training.config import TrainingConfig

from transformers import get_scheduler

app = typer.Typer()


@torch.no_grad
def evaluate(
    model: torch.nn.Module,
    accelerator: Accelerator,
    data_loader: DataLoader,
) -> float:
    losses: list[float] = []
    criterion = torch.nn.CrossEntropyLoss()

    for batch in data_loader:
        input_ids = batch["input_ids"]
        labels = input_ids[:, 1:]
        input_ids = input_ids[:, :-1]

        logits = model(input_ids)
        loss = criterion(logits.reshape(-1, logits.size(-1)), labels.reshape(-1))
        losses.append(accelerator.gather_for_metrics(loss.repeat(input_ids.size(0))))

        break

    mean_loss = torch.cat(losses).mean().item()

    try:
        perplexity = math.exp(mean_loss)
    except OverflowError:
        perplexity = float("inf")

    return perplexity


def train_epoch(
    model: torch.nn.Module,
    accelerator: Accelerator,
    data_loader: DataLoader,
    optimizer: Optimizer,
    learning_rate_scheduler,
    completed_steps: int,
    max_train_steps: int,
):
    total_loss: float = 0
    criterion = torch.nn.CrossEntropyLoss()

    for batch in data_loader:
        with accelerator.accumulate(model):
            input_ids = batch["input_ids"]
            labels = input_ids[:, 1:]
            input_ids = input_ids[:, :-1]

            logits = model(input_ids)
            loss = criterion(logits.reshape(-1, logits.size(-1)), labels.reshape(-1))

            total_loss += loss.detach().float()

            accelerator.backward(loss)
            optimizer.step()
            learning_rate_scheduler.step()
            optimizer.step()

        if accelerator.sync_gradients:
            completed_steps += 1

        if completed_steps >= max_train_steps:
            break

    return completed_steps, total_loss


def train(
    accelerator: Accelerator, model: torch.nn.Module, training_config: TrainingConfig
):
    if training_config.wandb:
        wandb.init(
            project=training_config.wandb.project,
            notes=training_config.wandb.notes,
            tags=training_config.wandb.tags,
            config=training_config.dict(),
        )

    dataset_split: DatasetSplit = datasplit_from_dataset_config(
        dataset_config=training_config.dataset_config, accelerator=accelerator
    )

    accumulation_steps_per_epoch: int = math.ceil(
        len(dataset_split.train) / training_config.gradient_accumulation_steps
    )
    max_train_steps = training_config.epochs * accumulation_steps_per_epoch

    # Lion is too hard.
    optimizer = torch.optim.AdamW(
        params=model.parameters(), lr=training_config.learning_rate
    )

    learning_rate_scheduler = get_scheduler(
        name=training_config.learning_rate_scheduler,
        optimizer=optimizer,
        num_warmup_steps=training_config.warmup_steps
        * training_config.gradient_accumulation_steps,
        num_training_steps=max_train_steps
        * training_config.gradient_accumulation_steps,
    )

    (
        model,
        optimizer,
        dataset_split.train,
        dataset_split.validation,
        learning_rate_scheduler,
    ) = accelerator.prepare(
        model,
        optimizer,
        dataset_split.train,
        dataset_split.validation,
        learning_rate_scheduler,
    )

    starting_epoch: int = 0
    completed_steps: int = 0

    for epoch in range(starting_epoch, training_config.epochs):
        model.train()

        completed_steps, total_loss = train_epoch(
            model=model,
            accelerator=accelerator,
            data_loader=dataset_split.train,
            optimizer=optimizer,
            learning_rate_scheduler=learning_rate_scheduler,
            completed_steps=completed_steps,
            max_train_steps=max_train_steps,
        )

        validation_perplexity = evaluate(
            model=model,
            accelerator=accelerator,
            data_loader=dataset_split.validation,
        )

        if training_config.wandb:
            average_training_loss = total_loss / len(dataset_split.train)
            current_learning_rate = learning_rate_scheduler.get_last_lr()[0]

            wandb.log(
                {
                    "train_mean_loss": average_training_loss,
                    "validation_perplexity": validation_perplexity,
                    "learning_rate": current_learning_rate,
                    "epoch": epoch,
                }
            )

            # Every epoch, save the MLP and attention weights.
            wandb_save_transformer_states(model=model, epoch=epoch)

    if training_config.wandb:
        accelerator.wait_for_everyone()
        unwrapped_model = accelerator.unwrap_model(model)

        with tempfile.NamedTemporaryFile(delete=False, suffix=".pt") as tmp_file:
            torch.save(unwrapped_model.state_dict(), tmp_file.name)
            model_artifact = wandb.Artifact("trained_model", type="model")
            model_artifact.add_file(tmp_file.name)
            wandb.log_artifact(model_artifact)

        wandb.finish()


@app.command()
def start_training_run(file_path: str) -> None:
    accelerator = Accelerator()
    training_config = TrainingConfig.from_toml_path(file_path=file_path)

    model = Transformer(
        vocab_dim=training_config.transformer_config.vocab_dim,
        embedding_dim=training_config.transformer_config.embedding_dim,
        max_seq_len=training_config.transformer_config.max_seq_len,
        num_layers=training_config.transformer_config.num_layers,
        num_heads=training_config.transformer_config.num_heads,
    )

    total_params = sum(p.numel() for p in model.parameters())
    print(f"Total parameters: {total_params}")

    train(accelerator=accelerator, model=model, training_config=training_config)


if __name__ == "__main__":
    app()
