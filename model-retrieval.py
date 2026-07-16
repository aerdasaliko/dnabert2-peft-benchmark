import argparse
from huggingface_hub import snapshot_download


def main():
    parser = argparse.ArgumentParser(
        description="Download a Hugging Face model repository locally."
    )

    parser.add_argument(
        "--output",
        type=str,
        default="./DNABERT-2-117M",
        help="Local directory to save the model"
    )

    args = parser.parse_args()
    repo_id = "zhihan1996/DNABERT-2-117M"

    print(f"Downloading {repo_id}")
    print(f"Saving to {args.output}")

    snapshot_download(
        repo_id=repo_id,
        local_dir=args.output,
        local_dir_use_symlinks=False,
        allow_patterns=[
            "bert_layers.py",
            "bert_padding.py",
            "config.json",
            "generation_config.json",
            "pytorch_model.bin",
            "tokenizer.json",
            "tokenizer_config.json",
            "configuration_bert.py",
        ],
    )

    print("Download complete.")


if __name__ == "__main__":
    main()