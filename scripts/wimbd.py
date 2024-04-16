import os
import gzip
import json
import typer

from tqdm import tqdm
from pathlib import Path
from datasets import load_dataset
from typing import Optional


app = typer.Typer()


def save_jsonl_gz(data, output_path: Path) -> None:
    """Save a list of dictionaries as a compressed JSONL file.

    Args:
        data: List of dictionaries to save.
        output_path (Path): The path to save the compressed JSONL file.
    """
    with gzip.open(output_path, "wt") as f:
        for item in data:
            item = {"text": item}
            f.write(json.dumps(item) + "\n")


@app.command()
def download_and_convert(
    dataset_id: str, output_path: str, dataset_config: Optional[str] = None
) -> None:
    """Download and convert dataset to be ready for Wimbd.

    Args:
        dataset_id (str): Name of the dataset.
        output_path (str): The path to save the converted dataset.
        dataset_config (str | None, optional): Name of subset/config. Defaults to None.
    """

    # Load dataset from HuggingFace
    dataset = load_dataset(path=dataset_id, name=dataset_config)

    # Define output directory based on the specified output path
    output_dir = Path(output_path)
    output_dir.mkdir(parents=True, exist_ok=True)

    for split in dataset.keys():
        split_dir = output_dir / split
        split_dir.mkdir(exist_ok=True)

        total_samples = len(dataset[split])

        for start_idx in tqdm(
            range(0, total_samples, 1000), desc=f"Processing {split}"
        ):
            end_idx = min(start_idx + 1000, total_samples)
            data_chunk = dataset[split].select(range(start_idx, end_idx))["content"]

            output_file = split_dir / f"{split}_{start_idx//1000}.json.gz"
            save_jsonl_gz(data_chunk, output_file)


if __name__ == "__main__":
    app()
