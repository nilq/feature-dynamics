"""Train a model."""

import typer
import torch
import math

from accelerate import Accelerator
from torch.utils.data import DataLoader
from torch.optim import Optimizer

from models.transformer import Transformer

from training.data import datasplit_from_dataset_config
from training.config import TrainingConfig, load_training_config_from_toml

from transformers import get_scheduler

app = typer.Typer()


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

            print(loss)

        if accelerator.sync_gradients:
            completed_steps += 1

        if completed_steps >= max_train_steps:
            break

    return completed_steps


def train(
    accelerator: Accelerator, model: torch.nn.Module, training_config: TrainingConfig
):
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
        num_warmup_steps=training_config.warmup_steps * training_config.gradient_accumulation_steps,
        num_training_steps=max_train_steps * training_config.gradient_accumulation_steps
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

        completed_steps = train_epoch(
            accelerator=accelerator,
            model=model,
            data_loader=dataset_split.train,
            optimizer=optimizer,
            learning_rate_scheduler=learning_rate_scheduler,
            completed_steps=completed_steps,
            max_train_steps=max_train_steps,
        )


@app.command()
def start_training_run(file_path: str) -> None:
    accelerator = Accelerator()
    training_config = load_training_config_from_toml(file_path=file_path)

    model = Transformer(
        vocab_dim=training_config.transformer_config.vocab_dim,
        embedding_dim=training_config.transformer_config.embedding_dim,
        max_seq_len=training_config.transformer_config.max_seq_len,
        num_layers=training_config.transformer_config.num_layers,
        num_heads=training_config.transformer_config.num_heads,
    )

    train(accelerator=accelerator, model=model, training_config=training_config)


if __name__ == "__main__":
    app()
