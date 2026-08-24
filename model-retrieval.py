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
    repo_id_dnabert2 = "Taykhoom/DNABERT2"

    print(f"Downloading {repo_id_dnabert2}")
    print(f"Saving to {args.output}")

    snapshot_download(
        repo_id=repo_id_dnabert2,
        local_dir=args.output,
    )

    print("Download complete.")


if __name__ == "__main__":
    main()