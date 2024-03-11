## Wimbd directory

This is where the Wimbd-ready datasets go. Use the `scripts/wimbd.py` converter to populate:

```
poetry run python scripts/wimbd.py download_and_convert [DATASET_ID] [OUTPUT_PATH] --dataset_config [CONFIG]
```

> Note: `--dataset_config` is optional.