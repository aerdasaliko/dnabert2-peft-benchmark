#!/usr/bin/env bash
set -euo pipefail

# =========================
# User config
# =========================
METHODS=("none" "lora" "qlora" "adalora")
# METHODS=("qlora")

# Resolve absolute paths from this script's location
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
FINETUNE_DIR="$(cd "${SCRIPT_DIR}/.." && pwd)"
REPO_ROOT="$(cd "${FINETUNE_DIR}/../.." && pwd)"

TRAIN_SCRIPT="${FINETUNE_DIR}/train-peft-regression.py"
MODEL_PATH="${REPO_ROOT}/DNABERT-2-117M-adapted"
LOG_DIR="${REPO_ROOT}/logs"
OUT_ROOT="${FINETUNE_DIR}/output/peft_bench"
DATASETS=(
    "${REPO_ROOT}/datasets/data/regression/hepg2_log2fc"
    "${REPO_ROOT}/datasets/data/regression/k562_log2fc"
    "${REPO_ROOT}/datasets/data/regression/sknsh_log2fc"
    # "${REPO_ROOT}/datasets/data/regression/rna_expr_brain_hippocampus"
    # "${REPO_ROOT}/datasets/data/regression/rna_expr_liver"
    # "${REPO_ROOT}/datasets/data/regression/rna_expr_cells_transformed_fibroblasts"
)

# Training params
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

# MODEL_MAX_LENGTH=100
# BATCH_TRAIN=16
# BATCH_EVAL=32
# GRAD_ACC=1
# LR=3e-5
# EPOCHS=5
# SAVE_STEPS=200
# EVAL_STEPS=200
# WARMUP_STEPS=50
# LOGGING_STEPS=100

# Optional offline mode
# export TRANSFORMERS_OFFLINE=1

mkdir -p "${LOG_DIR}" "${OUT_ROOT}"

# Preflight checks
[[ -f "${TRAIN_SCRIPT}" ]] || { echo "Missing train script: ${TRAIN_SCRIPT}"; exit 1; }
[[ -d "${MODEL_PATH}" ]] || { echo "Missing model dir: ${MODEL_PATH}"; exit 1; }
[[ -f "${MODEL_PATH}/config.json" ]] || { echo "Model config.json not found in ${MODEL_PATH}"; exit 1; }

timestamp() { date +"%Y%m%d_%H%M%S"; }

start_gpu_logger() {
  local outfile="$1"
  nvidia-smi \
    --query-gpu=timestamp,index,name,utilization.gpu,utilization.memory,memory.used,memory.total,power.draw,temperature.gpu \
    --format=csv -l 1 > "${outfile}" &
  echo $!
}

stop_gpu_logger() {
  local pid="$1"
  kill "${pid}" >/dev/null 2>&1 || true
}

TIME_CSV="${LOG_DIR}/train_times.csv"
if [[ ! -f "${TIME_CSV}" ]]; then
  echo "timestamp,method,elapsed_sec,elapsed_min,run_name,output_dir" > "${TIME_CSV}"
fi

SUMMARY_CSV="${LOG_DIR}/train_times_summary.csv"

for dataset in "${DATASETS[@]}"; do
  DATA_NAME=$(basename "${dataset}")

  for method in "${METHODS[@]}"; do
    RUN_NAME="${DATA_NAME}_${method}"
    TS=$(timestamp)

    OUT_DIR="${OUT_ROOT}/${DATA_NAME}/${method}"

    # overwrite behavior handled here instead of passing --overwrite_output_dir
    rm -rf "${OUT_DIR}"
    mkdir -p "${OUT_DIR}"

    GPU_LOG="${LOG_DIR}/gpu_${RUN_NAME}_${TS}.csv"
    TRAIN_LOG="${LOG_DIR}/train_${RUN_NAME}_${TS}.log"
    EVAL_COPY="${LOG_DIR}/eval_${RUN_NAME}_${TS}.json"

    echo "========================================="
    echo "Starting ${RUN_NAME}"
    echo "Method: ${method}"
    echo "Model:  ${MODEL_PATH}"
    echo "Data:   ${dataset}"
    echo "Output: ${OUT_DIR}"
    echo "========================================="

    GPU_PID=$(start_gpu_logger "${GPU_LOG}")
    START_TS=$(date +%s)

    set +e
    python "${TRAIN_SCRIPT}" \
      --model_name_or_path "${MODEL_PATH}" \
      --data_path "${dataset}" \
      --kmer -1 \
      --peft_method "${method}" \
      --run_name "${RUN_NAME}" \
      --model_max_length "${MODEL_MAX_LENGTH}" \
      --per_device_train_batch_size "${BATCH_TRAIN}" \
      --per_device_eval_batch_size "${BATCH_EVAL}" \
      --gradient_accumulation_steps "${GRAD_ACC}" \
      --learning_rate "${LR}" \
      --num_train_epochs "${EPOCHS}" \
      --fp16 \
      --save_steps "${SAVE_STEPS}" \
      --save_strategy steps \
      --output_dir "${OUT_DIR}" \
      --eval_strategy steps \
      --eval_steps "${EVAL_STEPS}" \
      --warmup_steps "${WARMUP_STEPS}" \
      --logging_steps "${LOGGING_STEPS}" \
      --log_level info \
      --find_unused_parameters False \
      2>&1 | tee "${TRAIN_LOG}"
    EXIT_CODE=${PIPESTATUS[0]}
    set -e

    END_TS=$(date +%s)
    ELAPSED_SEC=$((END_TS - START_TS))
    ELAPSED_MIN=$(awk "BEGIN {printf \"%.2f\", ${ELAPSED_SEC}/60}")

    stop_gpu_logger "${GPU_PID}"

    echo "$(date '+%Y-%m-%d %H:%M:%S'),${method},${ELAPSED_SEC},${ELAPSED_MIN},${RUN_NAME},${OUT_DIR}" >> "${TIME_CSV}"

    # Copy eval json if present
    SRC_EVAL="${OUT_DIR}/results/${RUN_NAME}/eval_results.json"
    if [[ -f "${SRC_EVAL}" ]]; then
      cp "${SRC_EVAL}" "${EVAL_COPY}"
      echo "Saved eval copy -> ${EVAL_COPY}"
    else
      echo "WARNING: eval_results.json not found for ${RUN_NAME} at ${SRC_EVAL}" | tee -a "${TRAIN_LOG}"
    fi

    if [[ ${EXIT_CODE} -ne 0 ]]; then
      echo "ERROR: ${RUN_NAME} failed with exit code ${EXIT_CODE}" | tee -a "${TRAIN_LOG}"
    fi

    echo "Finished ${RUN_NAME} in ${ELAPSED_SEC}s (${ELAPSED_MIN} min)"
  done
done

# Build per-method average time summary
python - <<PY
import csv, statistics
from collections import defaultdict

time_csv = r"${TIME_CSV}"
summary_csv = r"${SUMMARY_CSV}"

rows = []
with open(time_csv, newline="") as f:
    reader = csv.DictReader(f)
    for r in reader:
        rows.append(r)

by_method = defaultdict(list)
for r in rows:
    by_method[r["method"]].append(float(r["elapsed_sec"]))

with open(summary_csv, "w", newline="") as f:
    w = csv.writer(f)
    w.writerow(["method", "num_runs", "mean_sec", "std_sec", "mean_min"])
    for m in ["none", "lora", "qlora", "adalora"]:
        vals = by_method.get(m, [])
        if not vals:
            w.writerow([m, 0, "", "", ""])
            continue
        mean = sum(vals)/len(vals)
        std = statistics.pstdev(vals) if len(vals) > 1 else 0.0
        w.writerow([m, len(vals), f"{mean:.2f}", f"{std:.2f}", f"{mean/60:.2f}"])
print(f"Wrote {summary_csv}")
PY

echo "All benchmark runs completed."
echo "Time log:    ${TIME_CSV}"
echo "Time summary:${SUMMARY_CSV}"
echo "GPU logs:    ${LOG_DIR}/gpu_*.csv"
echo "Eval copies: ${LOG_DIR}/eval_*.json"
