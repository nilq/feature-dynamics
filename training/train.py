"""Train a model."""

import os

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

from training.data import datasplit_from_dataset_config, tokenizer_from_dataset_config
from training.config import TrainingConfig

from tokenizers import Tokenizer
from transformers import get_scheduler

app = typer.Typer()


@torch.no_grad()
def sample_from_model(
    model: torch.nn.Module,
    accelerator: Accelerator,
    tokenizer: Tokenizer,
    prompt: str = "Once upon a time",
    max_length: int = 10,
    top_k: int = 50
):
    input_ids = tokenizer.encode(prompt).ids
    input_ids = torch.tensor(input_ids, dtype=torch.long).unsqueeze(0)
    input_ids = input_ids.to(accelerator.device)

    model.eval()
    for _ in range(max_length - len(input_ids)):
        logits, *_ = model(input_ids)
        logits = logits[:, -1, :]

        top_k_logits, top_k_indices = torch.topk(logits, top_k)
        probabilities = torch.nn.functional.softmax(top_k_logits, dim=-1)

        next_token = torch.multinomial(probabilities, num_samples=1)
        input_ids = torch.cat([input_ids, next_token], dim=-1)

    generated_sequence = tokenizer.decode(input_ids.squeeze().tolist())
    return generated_sequence


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

        logits, *_ = model(input_ids)
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
    use_wandb: bool,
    log_interval: int = 5000,
):
    total_loss: float = 0
    criterion = torch.nn.CrossEntropyLoss()

    for batch in data_loader:
        with accelerator.accumulate(model):
            input_ids = batch["input_ids"]
            labels = input_ids[:, 1:]
            input_ids = input_ids[:, :-1]

            logits, *_ = model(input_ids)
            loss = criterion(logits.reshape(-1, logits.size(-1)), labels.reshape(-1))

            total_loss += loss.item()

            accelerator.backward(loss)
            optimizer.zero_grad()
            optimizer.step()
            learning_rate_scheduler.step()

        if accelerator.sync_gradients:
            completed_steps += 1

            if completed_steps % log_interval == 0 and accelerator.is_main_process:
                current_loss = total_loss / completed_steps
                current_learning_rate = learning_rate_scheduler.get_last_lr()[0]

                if use_wandb:
                    wandb.log({
                        "step_loss": current_loss,
                        "learning_rate": current_learning_rate,
                        "step": completed_steps,
                    })

                    with tempfile.TemporaryDirectory() as checkpoint_dir:
                        accelerator.save_state(checkpoint_dir)
                        artifact = wandb.Artifact(f"model-checkpoint-{completed_steps}", type='model')
                        artifact.add_dir(checkpoint_dir)
                        wandb.log_artifact(artifact)

        if completed_steps >= max_train_steps:
            break
        
        if os.getenv("TEST"):
            break

    mean_loss = total_loss / len(data_loader)
    return completed_steps, mean_loss


def train(
    accelerator: Accelerator, model: torch.nn.Module, training_config: TrainingConfig
):
    total_params = sum(p.numel() for p in model.parameters())
    print(f"Total parameters: {total_params:_}")

    if training_config.wandb and accelerator.is_main_process:
        wandb.init(
            project=training_config.wandb.project,
            notes=training_config.wandb.notes,
            tags=training_config.wandb.tags + [f"{total_params / 1e6:.1f}M"],
            config=training_config.dict(),
        )

    dataset_split: DatasetSplit = datasplit_from_dataset_config(
        dataset_config=training_config.dataset_config, accelerator=accelerator
    )

    accumulation_steps_per_epoch: int = math.ceil(
        len(dataset_split.train) / training_config.gradient_accumulation_steps
    )
    max_train_steps = training_config.epochs * accumulation_steps_per_epoch

    print("Max training steps:", max_train_steps)

    # Lion is too hard.
    optimizer = torch.optim.AdamW(
        params=model.parameters(),
        lr=training_config.learning_rate,
        betas=tuple(training_config.adam_betas),
        weight_decay=training_config.weight_decay,
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

    # For sample generation.
    tokenizer = tokenizer_from_dataset_config(dataset_config=training_config.dataset_config)

    for epoch in range(starting_epoch, training_config.epochs):
        model.train()
        completed_steps, mean_training_loss = train_epoch(
            model=model,
            accelerator=accelerator,
            data_loader=dataset_split.train,
            optimizer=optimizer,
            learning_rate_scheduler=learning_rate_scheduler,
            completed_steps=completed_steps,
            max_train_steps=max_train_steps,
            use_wandb=training_config.wandb is not None,
            log_interval=training_config.log_interval
        )

        model.eval()
        validation_perplexity = evaluate(
            model=model,
            accelerator=accelerator,
            data_loader=dataset_split.validation,
        )


        if accelerator.is_main_process:
            generated_sequence = sample_from_model(
                model=model,
                accelerator=accelerator,
                tokenizer=tokenizer,
                prompt=training_config.sample_prompt,
                max_length=training_config.transformer_config.max_seq_len
            )

            print(f"Sample at {epoch} (val-perplexity {validation_perplexity}):", generated_sequence)

            if training_config.wandb:
                current_learning_rate = learning_rate_scheduler.get_last_lr()[0]

                wandb.log(
                    {
                        "train_mean_loss": mean_training_loss,
                        "validation_perplexity": validation_perplexity,
                        "learning_rate": current_learning_rate,
                        "epoch": epoch,
                        "sampled_sequence": generated_sequence,
                    }
                )

                # Every epoch, save the MLP and attention weights.

                accelerator.wait_for_everyone()
                unwrapped_model = accelerator.unwrap_model(model)
                wandb_save_transformer_states(model=unwrapped_model, epoch=epoch)

    if training_config.wandb and accelerator.is_main_process:
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

    train(accelerator=accelerator, model=model, training_config=training_config)


if __name__ == "__main__":
    app()
