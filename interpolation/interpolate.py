"""Interpolating weight merging with continuous evaluation."""

import json
import torch
import typer
import tomllib
import tempfile

from interpolation.config import InterpolationConfig
from mergekit.config import InputModelDefinition, MergeConfiguration
from mergekit.merge import run_merge
from mergekit.options import MergeOptions
from accelerate import Accelerator
from transformers import AutoModelForCausalLM
from torch.utils.data import DataLoader

from training.transformer.data import datasplit_from_dataset_config

app = typer.Typer()

def evaluate_model_on_dataset(model: AutoModelForCausalLM, dataset: DataLoader, device="cpu") -> dict[str, float]:
    """Evaluate model on dataset.
    """
    model.to(device)
    model.eval()

    total_loss = 0.0
    total_samples = 0

    with torch.no_grad():
        for batch in dataset:
            inputs = torch.tensor(batch["input_ids"])
            labels = torch.tensor(batch["labels"])

            outputs = model(input_ids=inputs.unsqueeze(-1), labels=labels.unsqueeze(-1))
            loss = outputs.loss

            total_loss += loss.item() * inputs.size(0)
            total_samples += inputs.size(0)

    average_loss = total_loss / total_samples
    perplexity = torch.exp(torch.tensor(average_loss))

    return {
        "perplexity": perplexity.item(),
    }


def slerp(
    config_path: str
) -> None:
    interpolation_config = InterpolationConfig(**tomllib.load(open(config_path, "rb"))["slerp"])
    training_dataset_a, validation_dataset_a = datasplit_from_dataset_config(
        dataset_config=interpolation_config.dataset_a,
        training_config=Accelerator(),  # Duck-typing trick.
    )

    training_dataset_b, validation_dataset_b = datasplit_from_dataset_config(
        dataset_config=interpolation_config.dataset_a,
        training_config=Accelerator(),  # Duck-typing trick.
    )

    interpolation_metrics: dict[int, str] = {}

    for step in range(0, 100, int(interpolation_config.stride * 100)):
        merge_config = MergeConfiguration(
            merge_method="slerp",
            base_model=interpolation_config.base_model or interpolation_config.model_a,
            models=[
                InputModelDefinition(
                    model=interpolation_config.model_a,
                ),
                InputModelDefinition(
                    model=interpolation_config.model_b,
                )
            ],
            dtype=interpolation_config.dtype,
            parameters={
                "t": step / 100
            }
        )

        merge_output_directory = tempfile.TemporaryDirectory()
        run_merge(
            merge_config=merge_config,
            out_path=merge_output_directory.name,
            options=MergeOptions(seed=interpolation_config.seed)
        )

        model = AutoModelForCausalLM.from_pretrained(merge_output_directory.name)
        metrics_a = evaluate_model_on_dataset(model, validation_dataset_a)

        metrics_b = evaluate_model_on_dataset(model, validation_dataset_a)

        interpolation_metrics[step / 100] = {}
        interpolation_metrics[step / 100]["validation_a"] = metrics_a
        interpolation_metrics[step / 100]["validation_b"] = metrics_b

    open("metric_test.json", "w+").write(json.dumps(interpolation_metrics))

if __name__ == "__main__":
    typer.run(slerp)

