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

`libero-120/0000.parquet` holds 120 demonstrations (3 per task, 40 tasks) of
[physical-intelligence/libero](https://huggingface.co/datasets/physical-intelligence/libero) (LeRobot format of the
[LIBERO](https://libero-project.github.io/) benchmark): every 5th frame of the agent-view camera, re-encoded as
224x224 JPEG, with the frame's 7-DoF action, the task instruction and the episode id (18 MB). Built with
`libero_data.make_sample("sample_data/libero-120/0000.parquet", episodes_per_task=3, frame_stride=5)`.

License: [CC BY 4.0](https://creativecommons.org/licenses/by/4.0/), as stated by the source dataset.
