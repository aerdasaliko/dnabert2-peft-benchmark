#!/usr/bin/env python3
import os
# Must be set before torch/transformers touch legacy .bin checkpoints
os.environ["TORCH_FORCE_NO_WEIGHTS_ONLY_LOAD"] = "1"

import argparse
import json
import logging
import shutil
from pathlib import Path
from typing import Dict, List

import torch
import transformers


def parse_args():
    parser = argparse.ArgumentParser(
        description="Convert DNABERT-2 legacy .bin checkpoint to safetensors."
    )
    parser.add_argument(
        "--src_dir",
        type=str,
        default="./DNABERT-2-117M-adapted",
        help="Source model directory containing config/tokenizer/model weights.",
    )
    parser.add_argument(
        "--dst_dir",
        type=str,
        default="./DNABERT-2-117M-safetensors",
        help="Destination directory for safetensors output.",
    )
    return parser.parse_args()


def find_checkpoint_files(src_dir: Path) -> List[Path]:
    """
    Supports:
      - pytorch_model.bin
      - model.bin
      - sharded checkpoints with pytorch_model.bin.index.json
    """
    index_file = src_dir / "pytorch_model.bin.index.json"
    if index_file.exists():
        with open(index_file, "r") as f:
            index = json.load(f)
        shard_names = sorted(set(index["weight_map"].values()))
        return [src_dir / shard_name for shard_name in shard_names]

    for name in ("pytorch_model.bin", "model.bin"):
        path = src_dir / name
        if path.exists():
            return [path]

    raise FileNotFoundError(
        f"No checkpoint found in {src_dir}. Expected pytorch_model.bin, model.bin, "
        "or pytorch_model.bin.index.json."
    )


def unwrap_state_dict(obj):
    """
    Handle common wrapper formats.
    """
    if not isinstance(obj, dict):
        raise TypeError(f"Expected a dict-like checkpoint, got {type(obj)}")

    for key in ("state_dict", "model_state_dict", "model"):
        if key in obj and isinstance(obj[key], dict):
            return obj[key]

    return obj


def load_legacy_state_dict(files: List[Path]) -> Dict[str, torch.Tensor]:
    """
    Load legacy .bin checkpoint(s) with weights_only=False.
    """
    state_dict: Dict[str, torch.Tensor] = {}

    for path in files:
        logging.info(f"Loading checkpoint shard: {path.name}")
        obj = torch.load(path, map_location="cpu", weights_only=False)
        shard = unwrap_state_dict(obj)
        state_dict.update(shard)

    return state_dict


def copy_custom_code(src_dir: Path, dst_dir: Path):
    """
    Copy custom DNABERT-2 python files so trust_remote_code=True still works
    from the converted directory.
    """
    for fname in ("configuration_bert.py", "bert_padding.py", "bert_layers.py"):
        src = src_dir / fname
        if src.exists():
            shutil.copy2(src, dst_dir / fname)


def main():
    args = parse_args()
    src_dir = Path(args.src_dir).resolve()
    dst_dir = Path(args.dst_dir).resolve()

    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s - %(levelname)s - %(message)s",
    )

    logging.info(f"Source:      {src_dir}")
    logging.info(f"Destination: {dst_dir}")

    if not src_dir.exists():
        raise FileNotFoundError(f"Source directory not found: {src_dir}")

    dst_dir.mkdir(parents=True, exist_ok=True)

    # Tokenizer
    tokenizer = transformers.AutoTokenizer.from_pretrained(
        str(src_dir),
        trust_remote_code=True,
        use_fast=True,
    )
    if tokenizer.pad_token is None:
        tokenizer.pad_token = "[PAD]"

    logging.info(f"Tokenizer pad token: {tokenizer.pad_token}")
    logging.info(f"Tokenizer pad token id: {tokenizer.pad_token_id}")

    # Config
    config = transformers.AutoConfig.from_pretrained(
        str(src_dir),
        trust_remote_code=True,
    )

    config.pad_token_id = tokenizer.pad_token_id if tokenizer.pad_token_id is not None else 3
    if not hasattr(config, "num_labels") or config.num_labels is None:
        config.num_labels = 2

    logging.info(f"Config class: {config.__class__.__name__}")
    logging.info(f"Config pad_token_id: {config.pad_token_id}")
    logging.info(f"Config num_labels: {config.num_labels}")

    # Build model from config only; do not load weights yet.
    model = transformers.AutoModelForSequenceClassification.from_config(
        config,
        trust_remote_code=True,
    )

    # Load legacy checkpoint manually
    ckpt_files = find_checkpoint_files(src_dir)
    state_dict = load_legacy_state_dict(ckpt_files)

    missing_keys, unexpected_keys = model.load_state_dict(state_dict, strict=False)
    logging.info(f"Missing keys: {len(missing_keys)}")
    logging.info(f"Unexpected keys: {len(unexpected_keys)}")

    if missing_keys:
        logging.info(f"First few missing keys: {missing_keys[:10]}")
    if unexpected_keys:
        logging.info(f"First few unexpected keys: {unexpected_keys[:10]}")

    # Save as safetensors
    logging.info("Saving model with safe_serialization=True ...")
    model.save_pretrained(
        str(dst_dir),
        safe_serialization=True,
        max_shard_size="10GB",
    )

    tokenizer.save_pretrained(str(dst_dir))
    config.save_pretrained(str(dst_dir))
    copy_custom_code(src_dir, dst_dir)

    logging.info("Done.")
    logging.info(f"Safetensors model written to: {dst_dir}")


if __name__ == "__main__":
    main()