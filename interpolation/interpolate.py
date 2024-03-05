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

import torch.nn.functional as F
import torch

from training.transformer.data import datasplit_from_dataset_config

from transformers import Trainer, TrainingArguments, default_data_collator, AutoTokenizer
import evaluate

app = typer.Typer()

def evaluate_model_on_dataset(model: AutoModelForCausalLM, dataset: DataLoader, device="cuda") -> dict[str, float]:
    """Evaluate model on dataset.
    """
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

    training_args = TrainingArguments(output_dir="./results", do_train=False, do_eval=True)
    trainer = Trainer(
        model=model,
        args=training_args,
        eval_dataset=dataset,
        tokenizer=AutoTokenizer.from_pretrained("nilq/mistral-1L-tiny"),
        data_collator=default_data_collator,
        compute_metrics=compute_metrics,
        preprocess_logits_for_metrics=preprocess_logits_for_metrics,
    )

    eval_results = trainer.evaluate()

    result = {
        "loss": eval_results["eval_loss"],
        "accuracy": eval_results["eval_accuracy"]
    }

    return result


def slerp(
    config_path: str
) -> None:
    interpolation_config = InterpolationConfig(**tomllib.load(open(config_path, "rb"))["slerp"])
    training_dataset_a, validation_dataset_a = datasplit_from_dataset_config(
        dataset_config=interpolation_config.dataset_a,
        training_config=Accelerator(),  # Duck-typing trick.
    )

    training_dataset_b, validation_dataset_b = datasplit_from_dataset_config(
        dataset_config=interpolation_config.dataset_b,
        training_config=Accelerator(),  # Duck-typing trick.
    )

    interpolation_metrics: dict[int, str] = {}

    print(f"Interpolating: {interpolation_config.model_a} -> {interpolation_config.model_b}.")
    print(f"Evaluating: {interpolation_config.dataset_a.dataset_id} -> {interpolation_config.dataset_b.dataset_id}.")

    print("Pre-slerp scores:")
    model_a = AutoModelForCausalLM.from_pretrained(interpolation_config.model_a)
    model_b = AutoModelForCausalLM.from_pretrained(interpolation_config.model_b)

    metrics_a = evaluate_model_on_dataset(model_a, validation_dataset_a)
    metrics_b = evaluate_model_on_dataset(model_b, validation_dataset_b)

    print(f"  - A: {metrics_a}")
    print(f"  - B: {metrics_b}")

    for step in range(0, 100 + int(interpolation_config.stride * 100), int(interpolation_config.stride * 100)):
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
        metrics_b = evaluate_model_on_dataset(model, validation_dataset_b)

        interpolation_metrics[step / 100] = {}
        interpolation_metrics[step / 100]["validation_a"] = metrics_a
        interpolation_metrics[step / 100]["validation_b"] = metrics_b

    open("metric_test.json", "w+").write(json.dumps(interpolation_metrics))

if __name__ == "__main__":
    typer.run(slerp)

