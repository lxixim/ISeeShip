#!/usr/bin/env bash
set -euo pipefail

ISEESHIP_ROOT="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/../.." && pwd)"

# Shared settings from the manuscript's Implementation Details.
LORA_R=128
LORA_ALPHA=256
MAX_STEPS=200
GRPO_NUM_GENERATIONS=6

prepare_training() {
    local variable
    for variable in DATA_PATH CKPT_PATH SAVE_PATH CUDA_VISIBLE_DEVICES; do
        if [[ -z "${!variable:-}" ]]; then
            printf 'Set %s before launching training. See src/scripts/README.md.\n' "$variable" >&2
            exit 2
        fi
    done

    NPROC_PER_NODE="${NPROC_PER_NODE:-2}"
    SHOT="${SHOT:-4}"
    TRAINING_SEED="${SEED:-100}"
    if [[ "$SHOT" != 4 && "$SHOT" != 8 ]]; then
        printf 'Set SHOT to 4 or 8 and supply the corresponding prepared dataset.\n' >&2
        exit 2
    fi
    if [[ ! "$TRAINING_SEED" =~ ^[0-9]+$ ]]; then
        printf 'SEED must be a nonnegative integer.\n' >&2
        exit 2
    fi
    if [[ ! "$NPROC_PER_NODE" =~ ^[1-9][0-9]*$ ]]; then
        printf 'NPROC_PER_NODE must be a positive integer.\n' >&2
        exit 2
    fi
    local -a devices
    IFS=',' read -r -a devices <<< "$CUDA_VISIBLE_DEVICES"
    if [[ "$CUDA_VISIBLE_DEVICES" == ,* || "$CUDA_VISIBLE_DEVICES" == *, || "$CUDA_VISIBLE_DEVICES" == *,,* || ${#devices[@]} -ne "$NPROC_PER_NODE" ]]; then
        printf 'Select exactly %s GPU IDs in CUDA_VISIBLE_DEVICES.\n' "$NPROC_PER_NODE" >&2
        exit 2
    fi

    TRAINING_ROOT="${TRAINING_ROOT:-$ISEESHIP_ROOT/src/virft}"
    TRAIN_ENTRY="$TRAINING_ROOT/$1"
    if [[ ! -f "$TRAIN_ENTRY" ]]; then
        printf 'Training entry not found: %s\nSet TRAINING_ROOT to your task-specific training source tree.\n' "$TRAIN_ENTRY" >&2
        exit 2
    fi
    if [[ ! -d "$DATA_PATH" ]]; then
        printf 'DATA_PATH must be an existing prepared dataset directory: %s\n' "$DATA_PATH" >&2
        exit 2
    fi
    if ! command -v torchrun >/dev/null 2>&1; then
        printf 'torchrun is unavailable. Activate the training environment first.\n' >&2
        exit 2
    fi
    export DATA_PATH CKPT_PATH SAVE_PATH CUDA_VISIBLE_DEVICES
}

require_deepspeed() {
    DEEPSPEED_CONFIG="${DEEPSPEED_CONFIG:-$TRAINING_ROOT/local_scripts/zero3.json}"
    if [[ ! -f "$DEEPSPEED_CONFIG" ]]; then
        printf 'Missing DeepSpeed configuration: %s\n' "$DEEPSPEED_CONFIG" >&2
        exit 2
    fi
}
