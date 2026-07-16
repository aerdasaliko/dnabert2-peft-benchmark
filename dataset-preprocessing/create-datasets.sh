#!/bin/bash

# Usage: ./create_dataset.sh <VAR> <VAR2>
# VAR: base name for benchmark FASTA files (e.g., VAR)

if [ $# -ne 2 ]; then
    echo "Usage: $0 <new_directory_name> <VAR>"
    exit 1
fi

NEW_DIR=$1
VAR=$2

SCRIPT_DIR="./scripts"
BENCHMARK_DIR="./benchmark"
OUTPUT_DIR="./data/$NEW_DIR"

# Create the new directory for the dataset
mkdir -p "$OUTPUT_DIR"

# Run fasta-to-tsv.py for negatives
python3 "$SCRIPT_DIR/fasta-to-tsv.py" 0 "$BENCHMARK_DIR/${VAR}.negative.fa" "$OUTPUT_DIR/negatives.tsv"

# Run fasta-to-tsv.py for positives
python3 "$SCRIPT_DIR/fasta-to-tsv.py" 1 "$BENCHMARK_DIR/${VAR}.positive.fa" "$OUTPUT_DIR/positives.tsv"

# Run shuffle-and-split.py to create train/validation/test sets
python3 "$SCRIPT_DIR/shuffle-and-split.py" \
    "$OUTPUT_DIR/negatives.tsv" \
    "$OUTPUT_DIR/positives.tsv" \
    "$OUTPUT_DIR" \
    42

echo "Dataset created in $OUTPUT_DIR"
