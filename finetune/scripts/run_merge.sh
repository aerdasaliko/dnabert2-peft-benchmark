#!/usr/bin/env bash
set -euo pipefail

# -----------------------------
# User-configurable paths
# -----------------------------
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
FINETUNE_DIR="$(cd "${SCRIPT_DIR}/.." && pwd)"
REPO_ROOT="$(cd "${FINETUNE_DIR}/../.." && pwd)"

TRAIN_SCRIPT="${FINETUNE_DIR}/finetune.py"
MODEL_PATH="${REPO_ROOT}/DNABERT-2-117M-adapted"

OUTPUT_DIR="${FINETUNE_DIR}/output/merge"

TRAIN_FILE="${REPO_ROOT}/datasets/data/classification/fus-binding-sites/train.csv"
EVAL_FILE="${REPO_ROOT}/datasets/data/classification/fus-binding-sites/dev.csv"
TEST_FILE="${REPO_ROOT}/datasets/data/classification/fus-binding-sites/test.csv"

# Task: classification | regression
PROBLEM_TYPE="classification"

# LoRA toggle: true | false
USE_LORA="true"


# -----------------------------
# Training hyperparameters
# -----------------------------
MODEL_MAX_LENGTH=128
BATCH_TRAIN=32
BATCH_EVAL=64
GRAD_ACC=1
LR=2e-5
EPOCHS=3
SAVE_STEPS=2000
EVAL_STEPS=2000
LOGGING_STEPS=100
WARMUP_STEPS=2000
SEED=42

# Save controls
SAVE_MODEL="true"
EVAL_AND_SAVE_RESULTS="true"

RUN_NAME="dnabert_${PROBLEM_TYPE}_$(date +%Y%m%d_%H%M%S)"

mkdir -p "${OUTPUT_DIR}"

python "${TRAIN_SCRIPT}" \
  --model_name_or_path "${MODEL_PATH}" \
  --problem_type "${PROBLEM_TYPE}" \
  --use_lora "${USE_LORA}" \
  --train_file "${TRAIN_FILE}" \
  --eval_file "${EVAL_FILE}" \
  --test_file "${TEST_FILE}" \
  --output_dir "${OUTPUT_DIR}" \
  --run_name "${RUN_NAME}" \
  --num_train_epochs "${EPOCHS}" \
  --learning_rate "${LR}" \
  --per_device_train_batch_size "${BATCH_TRAIN}" \
  --per_device_eval_batch_size "${BATCH_EVAL}" \
  --gradient_accumulation_steps "${GRAD_ACC}" \
  --model_max_length "${MODEL_MAX_LENGTH}" \
  --warmup_steps "${WARMUP_STEPS}" \
  --logging_steps "${LOGGING_STEPS}" \
  --eval_steps "${EVAL_STEPS}" \
  --save_steps "${SAVE_STEPS}" \
  --save_strategy steps \
  --eval_strategy steps \
  --save_model "${SAVE_MODEL}" \
  --log_level info \
  --eval_and_save_results "${EVAL_AND_SAVE_RESULTS}"
