# feature-dynamics
🪴 Planting language models, seeing how they grow etc.

## End-to-end Toolkit

### Training

#### Transformers 

Train decoder-only models.

```
poetry run python training/transformer/train.py <experiment.toml>
```

#### Sparse autoencoder

Using TransformerLens's `HookedTransformer` (specifically via [my hacked fork](https://github.com/nilq/TransformerLens)*) to train sparse autoencoders.

\* This one is required to hook custom Mistral models.

### Interpolation

Interpolate model weights using Mergekit.

```
poetry run python interpolation/interpolate.py <experiment.toml>
```

### Merging

Merge models using Mergekit. This part is just Mergekit.
