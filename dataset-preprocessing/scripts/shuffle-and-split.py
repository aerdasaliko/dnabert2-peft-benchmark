import sys
import csv
import random
import os

def read_tsv(file_path):
    """Read a TSV file and return a list of rows (excluding header)."""
    with open(file_path, 'r') as f:
        reader = csv.reader(f, delimiter='\t')
        header = next(reader)
        rows = [row for row in reader if len(row) >= 2]
    return rows

def write_csv(file_path, rows):
    """Write a list of rows to a CSV file with header."""
    with open(file_path, 'w', newline='') as f:
        writer = csv.writer(f)
        writer.writerow(['sequence', 'label'])
        writer.writerows(rows)

def split_data(data, train_ratio=0.7, val_ratio=0.15, test_ratio=0.15):
    """Split data into train, validation, and test sets."""
    random.shuffle(data)
    n = len(data)
    n_train = int(n * train_ratio)
    n_val = int(n * val_ratio)
    
    train = data[:n_train]
    val = data[n_train:n_train+n_val]
    test = data[n_train+n_val:]
    
    return train, val, test

def main():
    if len(sys.argv) != 5:
        print(f"Usage: python {sys.argv[0]} <label0.tsv> <label1.tsv> <output_dir> <random_seed>")
        sys.exit(1)
    
    label0_file = sys.argv[1]
    label1_file = sys.argv[2]
    output_dir = sys.argv[3]
    seed = int(sys.argv[4])
    
    os.makedirs(output_dir, exist_ok=True)
    random.seed(seed)

    # Read data
    data0 = read_tsv(label0_file)
    data1 = read_tsv(label1_file)
    
    # Combine data
    all_data = data0 + data1
    
    # Split data
    train, val, test = split_data(all_data)
    
    # Write output files
    write_csv(os.path.join(output_dir, "train.csv"), train)
    write_csv(os.path.join(output_dir, "dev.csv"), val)
    write_csv(os.path.join(output_dir, "test.csv"), test)
    
    print(f"Train/Validation/Test datasets created in {output_dir}")
    print(f"Train: {len(train)}, Validation: {len(val)}, Test: {len(test)}")

if __name__ == "__main__":
    main()
