import sys
import csv
import os

def merge_3grams(three_grams):
    if not three_grams:
        return ""
    sequence = three_grams[0]
    for gram in three_grams[1:]:
        sequence += gram[-1]
    return sequence

def main():
    if len(sys.argv) != 3:
        print(f"Usage: python {sys.argv[0]} <input_file.tsv> <output_file.csv>")
        sys.exit(1)
    
    input_file = sys.argv[1]
    output_file = sys.argv[2]

    with open(input_file, newline='') as tsvfile, open(output_file, 'w', newline='') as out_file:
        reader = csv.reader(tsvfile, delimiter='\t')
        writer = csv.writer(out_file)
        
        header = next(reader, None)
        if header:
            writer.writerow(header)
        
        for row in reader:
            if len(row) < 2:
                continue
            three_grams_str, label = row[0], row[1]
            three_grams = three_grams_str.split()
            original_sequence = merge_3grams(three_grams)
            writer.writerow([original_sequence, label])
    
    print(f"Reconstructed sequences written to: {output_file}")

if __name__ == "__main__":
    main()
