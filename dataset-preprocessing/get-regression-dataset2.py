from datasets import load_dataset
import os

ds = load_dataset("HuggingFaceBio/malinois-mpra-regression")

split_map = {
    "train": "train",
    "validation": "dev",
    "test": "test"
}

targets = [
    "K562_log2FC_train_zscore",
    "HepG2_log2FC_train_zscore",
    "SKNSH_log2FC_train_zscore"
]

output_dir = "./regression"
os.makedirs(output_dir, exist_ok=True)

for target in targets:
    for split, new_name in split_map.items():
        subset = ds[split]

        if split == "validation":
            subset = subset.filter(lambda row: row["all_se_le_1"])

        subset = subset.select_columns(["sequence", target])

        df = subset.to_pandas()
        df.columns = ["sequence", "target"]

        short_name = target.replace("_train_zscore", "")
        path = os.path.join(output_dir, f"{short_name}_{new_name}.csv")
        df.to_csv(path, index=False)

        print(f"Saved: {path}")