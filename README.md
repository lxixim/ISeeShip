# ISeeShip and SVI-R1

Official project repository for **ISeeShip: A Multi-Task Benchmark with Reinforcement Learning for Semantically Explainable Ship Visual Intelligence**.

ISeeShip covers fine-grained ship classification, detection, and open-set recognition. SVI-R1 applies task-specific rewards to structured model responses with `<think>...</think><answer>...</answer>`.

<p align="center">
  <img src="assets/overall.png" alt="SVI-R1 overview" width="100%">
</p>

## Dataset

- **Images**: 8,964 camera-captured ship images spanning 30 categories.
- **Tasks**: classification, detection, and open-set recognition.
- **Image download**: [Ship30 dataset](https://github.com/lxixim/Ship30/tree/main/dataset).
- **Data formats and preparation**: [dataset guide](dataset/README.md).

## Main Results

Four-shot benchmark results reported in the paper:

| Method | Backbone | Classification Accuracy (%) | Detection AP50 (%) | Open-Set Overall Accuracy (%) |
| --- | --- | ---: | ---: | ---: |
| SVI-SFT | Qwen2-VL-2B | 47.6 | 57.0 | 58.8 |
| SVI-R1 | Qwen2-VL-2B | **70.0** | **86.1** | **72.3** |

## Installation

The training environment targets Linux and CUDA:

```bash
conda env create -f src/visual.yml
conda activate ISeeShip
```

Flash-Attention is installed as a dependency rather than bundled as project source. See the [upstream installation instructions](https://github.com/Dao-AILab/flash-attention).

## Training

| Setting | Value |
| --- | --- |
| Backbone | Qwen2-VL-2B |
| LoRA rank / alpha | 128 / 256 |
| GRPO group size | 6 |
| Maximum updates | 200 |
| Optimizer | AdamW |
| Main benchmark seed | 100 |
| Main support sizes | 4-shot and 8-shot |
| Main hardware | 2 NVIDIA A800 GPUs |

Choose the corresponding prepared support set and configure the source, model, output, and allocated GPUs:

```bash
export TRAINING_ROOT="/path/to/task_training_source"
export DATA_PATH="/path/to/prepared/classification_4shot"
export CKPT_PATH="/path/to/Qwen2-VL-2B-Instruct"
export SAVE_PATH="/path/to/output"
export CUDA_VISIBLE_DEVICES="<allocated_gpu_id_a>,<allocated_gpu_id_b>"
export NPROC_PER_NODE=2
export SHOT=4
export SEED=100
bash src/scripts/classification/train_grpo.sh
```

The source layout and optional settings are described in the [training guide](src/scripts/README.md). GPU IDs in the example are placeholders; select your allocated devices explicitly. `SHOT` labels the run; `DATA_PATH` must point to the prepared dataset for that support size.

| Task | SVI-R1 | SVI-SFT |
| --- | --- | --- |
| Classification | `src/scripts/classification/train_grpo.sh` | `src/scripts/classification/train_sft.sh` |
| Detection | `src/scripts/detection/train_grpo.sh` | `src/scripts/detection/train_sft.sh` |
| Open-set recognition | `src/scripts/ood/train_grpo.sh` | `src/scripts/ood/train_sft.sh` |

For the controlled classification/detection comparisons, use the same support images and evaluation lists across methods, seeds 42, 43, and 44, and the final checkpoint. Their learning rates are `2e-5` for classification and `1e-6` for detection, as configured in the corresponding launchers.

## Evaluation

- `classification/classification_ship30_infere.py`: external-API classification baseline inference. Configure the provider, credentials, images, and local evaluation manifest before use.
- `coco_evaluation/coco_evaluation.py`: `CocoDetectionEvaluator` for existing COCO predictions and annotations.
- `coco_evaluation/evaluation.ipynb`: detection-scoring notebook.

Detection scoring uses COCO image/category IDs and pixel-coordinate `[x, y, width, height]` boxes. Convert normalized model output coordinates before evaluation.

## Repository Structure

```text
assets/                 project figures
dataset/                data access and formats
ship_data_build/        annotation conversion and construction tools
scripts/                data-preparation helpers
src/scripts/            training launchers and configuration
src/visual.yml          training environment
classification/         API baseline inference
coco_evaluation/        detection evaluation
tests/                  launcher configuration tests
```

Run launcher tests without starting models or GPU jobs:

```bash
python -m unittest discover -s tests -v
```

## Citation

If this project supports your research, please cite **ISeeShip: A Multi-Task Benchmark with Reinforcement Learning for Semantically Explainable Ship Visual Intelligence**.
