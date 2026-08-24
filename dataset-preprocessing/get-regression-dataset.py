from datasets import load_dataset, Dataset
import pandas as pd

# Use this parameter to download sequences of arbitrary length (see docs below for edge cases)
sequence_length=512

# One of:
# ["variant_effect_causal_eqtl","variant_effect_pathogenic_clinvar",
# "variant_effect_pathogenic_omim","cage_prediction", "bulk_rna_expression",
# "chromatin_features_histone_marks","chromatin_features_dna_accessibility",
# "regulatory_element_promoter","regulatory_element_enhancer"] 

task_name = "bulk_rna_expression"

dataset = load_dataset(
    "InstaDeepAI/genomics-long-range-benchmark",
    task_name=task_name,
    sequence_length=sequence_length,
    # subset = True, if applicable
)

# Split train into train + validation
train_df_full = dataset["train"].to_pandas()

val_mask = train_df_full["chromosome"].isin(["7"])

val_df_raw = train_df_full[val_mask]
train_df_raw = train_df_full[~val_mask]

train_ds = Dataset.from_pandas(train_df_raw, preserve_index=False)
val_ds = Dataset.from_pandas(val_df_raw, preserve_index=False)
test_ds = dataset["test"]

# Load label mapping
label_map = pd.read_csv("label_mapping.csv")
label_names = label_map["Labels"].tolist()

def to_dataframe(ds):
    df = ds.to_pandas()
    labels_expanded = pd.DataFrame(df["labels"].tolist(), columns=label_names)
    final_df = pd.concat([df[["sequence"]], labels_expanded], axis=1)
    return final_df

train_df = to_dataframe(train_ds)
val_df = to_dataframe(val_ds)
test_df = to_dataframe(test_ds)

# Save to CSV
train_df.to_csv("./bulk-rna/all_labels/train.csv", index=False)
val_df.to_csv("./bulk-rna/all_labels/dev.csv", index=False)
test_df.to_csv("./bulk-rna/all_labels/test.csv", index=False)
