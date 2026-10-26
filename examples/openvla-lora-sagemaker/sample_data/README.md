# Sample data

`bridge-300/0000.parquet` holds the first 300 episodes of
[VyoJ/BridgeData-V2-Scripted-Images](https://huggingface.co/datasets/VyoJ/BridgeData-V2-Scripted-Images)
(from the Hub's auto-converted Parquet branch; images downscaled from 640x480 to 320x240 JPEG to keep the
file at 19 MB, all other columns unchanged), which is itself derived from
[BridgeData V2](https://rail-berkeley.github.io/bridgedata/) by RAIL Berkeley.

License: [CC BY 4.0](https://creativecommons.org/licenses/by/4.0/), as stated by the source dataset.

It lets notebook 1 run without contacting Hugging Face (anonymous Hub access is rate-limited per IP, which
SageMaker Studio users share). For larger runs, set `config.SOURCE_PARQUET_DIR = None` and log in with a
Hugging Face token, or point it at a folder of Parquet shards you downloaded yourself.
