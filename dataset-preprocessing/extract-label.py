import argparse
import os
import pandas as pd

def process_split(input_path, output_path, label):
    df = pd.read_csv(input_path)

    if label not in df.columns:
        raise ValueError(f"Label '{label}' not found in {input_path}")

    # Keep only sequence + selected label
    df_new = df[["sequence", label]].rename(columns={label: "label"})

    df_new.to_csv(output_path, index=False)


def main():
    parser = argparse.ArgumentParser(description="Extract single-label datasets")
    parser.add_argument("--label", required=True, help="Label column name")
    parser.add_argument("--input_dir", required=True, help="Folder with train/dev/test CSVs")

    args = parser.parse_args()

    label = args.label
    input_dir = args.input_dir

    folder_name = f"rna_expr_{label.lower().replace(' ', '-')}"

    output_dir = os.path.join(input_dir, folder_name)
    os.makedirs(output_dir, exist_ok=True)

    for split in ["train", "dev", "test"]:
        input_file = os.path.join(input_dir, f"{split}.csv")
        output_file = os.path.join(output_dir, f"{split}.csv")

        print(f"Processing {input_file} → {output_file}")
        process_split(input_file, output_file, label)

    print(f"\nDone. Files saved in: {output_dir}")


if __name__ == "__main__":
    main()
