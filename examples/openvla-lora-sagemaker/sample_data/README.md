# Sample data

Two small Parquet samples so notebook 1 runs without contacting Hugging Face (anonymous Hub access is
rate-limited per IP, which SageMaker Studio users share). For larger runs, set `config.SOURCE_PARQUET_DIR = None`
and log in with a Hugging Face token, or point it at a folder of Parquet shards you downloaded yourself.

Both samples are third-party data, redistributed under their original license, CC BY 4.0, with the changes listed
below. They are not covered by this repository's license.

## `libero-120/0000.parquet` (18 MB, default dataset)

- **Work:** LIBERO: Benchmarking Knowledge Transfer for Lifelong Robot Learning
- **Authors:** Bo Liu, Yifeng Zhu, Chongkai Gao, Yihao Feng, Qiang Liu, Yuke Zhu, Peter Stone
- **Paper:** [arXiv:2306.03310](https://arxiv.org/abs/2306.03310) (NeurIPS 2023 Datasets and Benchmarks);
  project page <https://libero-project.github.io/>
- **Taken from:** [physical-intelligence/libero](https://huggingface.co/datasets/physical-intelligence/libero),
  Physical Intelligence's LeRobot-format conversion of the LIBERO demonstrations
- **License:** [CC BY 4.0](https://creativecommons.org/licenses/by/4.0/), as stated on that dataset card
- **Changes made:** a subset of 120 demonstrations (3 per task, 40 tasks, chosen with a fixed seed); every 5th
  frame of the agent-view camera only; images re-encoded from PNG to 224x224 JPEG (quality 85); kept columns:
  image, 7-DoF action, task instruction, episode id. Built with
  `libero_data.make_sample("sample_data/libero-120/0000.parquet", episodes_per_task=3, frame_stride=5)`.

```bibtex
@article{liu2023libero,
  title={LIBERO: Benchmarking Knowledge Transfer for Lifelong Robot Learning},
  author={Liu, Bo and Zhu, Yifeng and Gao, Chongkai and Feng, Yihao and Liu, Qiang and Zhu, Yuke and Stone, Peter},
  journal={arXiv preprint arXiv:2306.03310},
  year={2023}
}
```

## `bridge-300/0000.parquet` (19 MB, `DATASET = "bridge"`)

- **Work:** BridgeData V2: A Dataset for Robot Learning at Scale
- **Authors:** Homer Walke, Kevin Black, Abraham Lee, Moo Jin Kim, Max Du, Chongyi Zheng, Tony Zhao,
  Philippe Hansen-Estruch, Quan Vuong, Andre He, Vivek Myers, Kuan Fang, Chelsea Finn, Sergey Levine (RAIL, UC
  Berkeley)
- **Paper:** BridgeData V2, Conference on Robot Learning (CoRL) 2023; project page
  <https://rail-berkeley.github.io/bridgedata/>
- **Taken from:** [VyoJ/BridgeData-V2-Scripted-Images](https://huggingface.co/datasets/VyoJ/BridgeData-V2-Scripted-Images)
  (images from the scripted part of BridgeData V2), via the Hub's auto-converted Parquet branch
- **License:** [CC BY 4.0](https://creativecommons.org/licenses/by/4.0/), as stated on that dataset card
- **Changes made:** the first 300 episodes only; images downscaled from 640x480 to 320x240 and re-encoded as JPEG;
  all other columns unchanged. The x/y action labels used by this lab are computed later, in notebook 1, from
  optical flow between frames (pseudo-labels, as in the AWS blog post); they are not part of the source data.

```bibtex
@inproceedings{walke2023bridgedata,
  title={BridgeData V2: A Dataset for Robot Learning at Scale},
  author={Walke, Homer and Black, Kevin and Lee, Abraham and Kim, Moo Jin and Du, Max and Zheng, Chongyi and Zhao, Tony and Hansen-Estruch, Philippe and Vuong, Quan and He, Andre and Myers, Vivek and Fang, Kuan and Finn, Chelsea and Levine, Sergey},
  booktitle={Conference on Robot Learning (CoRL)},
  year={2023}
}
```
