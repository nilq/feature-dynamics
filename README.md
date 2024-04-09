# feature-dynamics
🪴 Planting language models, seeing how they grow etc.

## Getting started

**Conda 🐍**
```
conda env create -f conda.yaml
conda activate feature-dynamics
```

**Dependencies 📦**

```
pip install pipx
pipx install poetry
poetry install
```

## End-to-end Toolkit

### Training

#### Transformers 

Train decoder-only models.

```
poetry run python training/transformer/train.py <experiment.toml>
```

#### Autoencoder

Train sparse autoencoder.

```
poetry run python training/autoencoder/train.py <experiment.toml>
```

Using TransformerLens's `HookedTransformer` (specifically via [my hacked fork](https://github.com/nilq/TransformerLens)*) to train sparse autoencoders.

\* This one is required to hook custom Mistral models.

### Evaluation

#### Autoencoder

Evaluation of pretrained autoencoders.

This module contains functionality to make target models use autoencoder reconstructions in place of existing activations, by using a forward pass hook. 

### Interpolation

Interpolate model weights using Mergekit.

```
poetry run python interpolation/interpolate.py <experiment.toml>
```

### Merging

Merge models using Mergekit. This part is just Mergekit.
