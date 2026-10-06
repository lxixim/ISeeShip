# Training Configuration

The launchers cover SVI-R1 (GRPO) and SVI-SFT for classification, detection, and open-set recognition.

## Environment

Set the following variables before launching:

| Variable | Purpose |
| --- | --- |
| `TRAINING_ROOT` | Task-specific training source tree |
| `DATA_PATH` | Prepared support dataset |
| `CKPT_PATH` | Initial model or merged checkpoint |
| `SAVE_PATH` | Output directory for this run |
| `CUDA_VISIBLE_DEVICES` | Explicitly allocated GPU IDs |
| `NPROC_PER_NODE` | Number of processes; default 2 |
| `SHOT` | Main support size, 4 or 8; default 4 |
| `SEED` | Training seed; default 100 |
| `MASTER_PORT` | Optional distributed-training port override |
| `DEEPSPEED_CONFIG` | Optional path to the DeepSpeed configuration |

The training source tree used by the launchers has this layout:

```text
TRAINING_ROOT/
  src/open_r1/grpo_classification_reward.py
  src/open_r1/sft_vision.py
  src/open_r1/ship_detection.py
  src/open_r1/sft_vision_detection.py
  src/open_r1/grpo_classification_ood_openset.py
  src/open_r1/sft_vision_ood.py
  local_scripts/zero3.json
```

Install the dependencies of that source tree in the training environment. `TRAINING_ROOT` defaults to `<repository>/src/virft`; `DEEPSPEED_CONFIG` defaults to `TRAINING_ROOT/local_scripts/zero3.json`. The launchers check entry-point, data-directory, configuration, and GPU-count settings before starting `torchrun`.

## Manuscript Settings

Shared settings in `common.sh` follow **Implementation Details** in the manuscript:

| Parameter | Value |
| --- | --- |
| Model | Qwen2-VL-2B |
| LoRA rank | 128 |
| LoRA alpha | 256 |
| GRPO generations per group | 6 |
| Maximum optimizer updates | 200 |
| Optimizer | AdamW |
| Main benchmark seed | 100 |
| Main GPU processes | 2 |
| Main support sizes | 4 and 8 |

Classification and detection learning rates follow **Controlled Comparisons**: `2e-5` and `1e-6`, respectively. With two processes, per-device batch size 1 and gradient accumulation 2 give an effective batch size of four. `SHOT` only labels the run: choose the corresponding prepared dataset through `DATA_PATH`.

Other CLI settings remain explicit in each launcher. The open-set learning-rate settings use the training entry point's default for GRPO and the existing `5e-5` setting for SFT; the manuscript does not specify an open-set learning rate.

## Commands

After exporting the required environment:

```bash
# Classification
bash src/scripts/classification/train_grpo.sh
bash src/scripts/classification/train_sft.sh

# Detection
bash src/scripts/detection/train_grpo.sh
bash src/scripts/detection/train_sft.sh

# Open-set recognition
bash src/scripts/ood/train_grpo.sh
bash src/scripts/ood/train_sft.sh
```

Use different output directories for each method, support size, and seed. To select the eight-shot setting, set `SHOT=8` and use its prepared support set. For repeated controlled comparisons, use `SEED=42`, `SEED=43`, and `SEED=44`, keeping support/evaluation memberships fixed.

Checkpoint initialization is supplied explicitly through `CKPT_PATH`. The launchers do not automatically warm up, resume, or choose checkpoints.
