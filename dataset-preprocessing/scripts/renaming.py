import os
import glob

base_folder = "../"

for folder in glob.glob(os.path.join(base_folder, "*-binding-sites")):
    for csv_file in glob.glob(os.path.join(folder, "*.csv")):
        filename = os.path.basename(csv_file)
        if filename.endswith("_train.csv"):
            new_name = os.path.join(folder, "train.csv")
        elif filename.endswith("_test.csv"):
            new_name = os.path.join(folder, "test.csv")
        elif filename.endswith("_val.csv"):
            new_name = os.path.join(folder, "dev.csv")
        else:
            continue
        print(f"Renaming {csv_file} → {new_name}")
        os.rename(csv_file, new_name)