# feature-dynamics

This repository contains the code and experiment configurations for reproducing the results in [Tracking Universal Features Through Fine-Tuning and Model Merging](https://arxiv.org/abs/2410.12391).

---

In the paper, we study the following toy models:

1. [BabyPython](https://huggingface.co/nilq/baby-python-mistral-1L-tiny-base): single-layer Mistral model trained on a mixture of BabyML and the Python subset of the stack.
2. [TinyStories](https://huggingface.co/nilq/baby-python-mistral-1L-tiny-TinyStories-ft): BabyPython fine-tuned on TinyStories.
3. [Lua](https://huggingface.co/nilq/baby-python-mistral-1L-tiny-lua-ft): BabyPython fine-tuned on Lua subset of TheStack.
4. [LuaStories-merge](https://huggingface.co/nilq/baby-python-1L-mistral-lua-stories-slerp): spherical linear interpolation of Lua and TinyStories models at `t = 0.58`.

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

This repository contains all of the components built throughout the iteration on the research and [paper](https://arxiv.org/abs/2410.12391).

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

Using TransformerLens's `HookedTransformer` (specifically via [this fork](https://github.com/nilq/TransformerLens) to add Mistral hooks) to train sparse autoencoders.

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

Merge models using [Mergekit](https://github.com/arcee-ai/mergekit).
