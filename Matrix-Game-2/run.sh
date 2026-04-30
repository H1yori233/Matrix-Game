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

CONFIG_PATH="${CONFIG_PATH:-configs/inference_yaml/inference_universal.yaml}"
CHECKPOINT_PATH="${CHECKPOINT_PATH:-/home/hal-kaiqin/models/Matrix-Game-2.0/base_distilled_model/base_distill.safetensors}"
IMAGE_PATH="${IMAGE_PATH:-demo_images/universal/0000.png}"
OUTPUT_DIR="${OUTPUT_DIR:-./outputs}"
NUM_OUTPUT_FRAMES="${NUM_OUTPUT_FRAMES:-150}"
SEED="${SEED:-42}"
PRETRAINED_MODEL_PATH="${PRETRAINED_MODEL_PATH:-/home/hal-kaiqin/models/Matrix-Game-2.0}"

if [[ ! -f "$CHECKPOINT_PATH" ]]; then
    echo "Checkpoint not found: ${CHECKPOINT_PATH}" >&2
    echo "Override with: CHECKPOINT_PATH=/path/to/model.safetensors ./run.sh" >&2
    exit 1
fi

ARGS=(
  --config_path "$CONFIG_PATH"
  --checkpoint_path "$CHECKPOINT_PATH"
  --img_path "$IMAGE_PATH"
  --output_folder "$OUTPUT_DIR"
  --num_output_frames "$NUM_OUTPUT_FRAMES"
  --seed "$SEED"
  --pretrained_model_path "$PRETRAINED_MODEL_PATH"
)

if [[ -n "${EXTRA_ARGS:-}" ]]; then
  # Allow quick local overrides without editing this script.
  # shellcheck disable=SC2206
  EXTRA_ARGS_ARRAY=( ${EXTRA_ARGS} )
  ARGS+=( "${EXTRA_ARGS_ARRAY[@]}" )
fi

exec python "$SCRIPT_DIR/inference.py" "${ARGS[@]}"
