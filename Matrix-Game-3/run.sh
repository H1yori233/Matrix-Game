#!/usr/bin/env bash
set -euo pipefail

CONDA_ROOT="${CONDA_ROOT:-$HOME/miniforge3}"
CONDA_BIN="$CONDA_ROOT/bin/conda"
ENV_NAME="${ENV_NAME:-FastVideo_kaiqin}"

if [ -x "$CONDA_BIN" ]; then
    eval "$("$CONDA_BIN" shell.bash hook)"
    conda activate "$ENV_NAME"
    echo "conda activated"
else
    echo "conda not found at $CONDA_BIN" >&2
    exit 1
fi

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$SCRIPT_DIR"

NUM_GPUS="${NUM_GPUS:-1}"
CKPT_DIR="${CKPT_DIR:-Matrix-Game-3.0}"
OUTPUT_DIR="${OUTPUT_DIR:-./output}"
IMAGE_PATH="${IMAGE_PATH:-demo_images/001/image.png}"
PROMPT="${PROMPT:-A colorful, animated cityscape with a gas station and various buildings.}"
SAVE_NAME="${SAVE_NAME:-test}"
SEED="${SEED:-42}"
NUM_ITERATIONS="${NUM_ITERATIONS:-12}"
NUM_INFERENCE_STEPS="${NUM_INFERENCE_STEPS:-3}"
SIZE="${SIZE:-704*1280}"
VAE_TYPE="${VAE_TYPE:-mg_lightvae}"
LIGHTVAE_PRUNING_RATE="${LIGHTVAE_PRUNING_RATE:-0.5}"

ARGS=(
  --size "$SIZE"
  --dit_fsdp
  --t5_fsdp
  --ckpt_dir "$CKPT_DIR"
  --fa_version 3
  --use_int8
  --num_iterations "$NUM_ITERATIONS"
  --num_inference_steps "$NUM_INFERENCE_STEPS"
  --image "$IMAGE_PATH"
  --prompt "$PROMPT"
  --save_name "$SAVE_NAME"
  --seed "$SEED"
  --compile_vae
  --lightvae_pruning_rate "$LIGHTVAE_PRUNING_RATE"
  --vae_type "$VAE_TYPE"
  --output_dir "$OUTPUT_DIR"
)

if [[ -n "${EXTRA_ARGS:-}" ]]; then
  # Allow quick local overrides without editing this script.
  # shellcheck disable=SC2206
  EXTRA_ARGS_ARRAY=( ${EXTRA_ARGS} )
  ARGS+=( "${EXTRA_ARGS_ARRAY[@]}" )
fi

exec torchrun \
  --nproc_per_node="$NUM_GPUS" \
  "$SCRIPT_DIR/generate.py" \
  "${ARGS[@]}"
