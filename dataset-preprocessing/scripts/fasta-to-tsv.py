import sys
import csv

def fasta_to_tsv(fasta_file, label, output_file):
    """
    Converts a FASTA file to a TSV file with sequence and label columns.
    Only includes sequences in the sense (+) direction.
    """
    with open(fasta_file, 'r') as f, open(output_file, 'w', newline='') as out_file:
        writer = csv.writer(out_file, delimiter='\t')
        writer.writerow(['sequence', 'label'])

        include_sequence = False

        for line in f:
            line = line.strip()
            if line.startswith('>'):
                include_sequence = line.endswith('(+)')
            else:
                if include_sequence:
                    writer.writerow([line.upper().replace('U', 'T'), label])

def main():
    if len(sys.argv) != 4:
        print(f"Usage: python {sys.argv[0]} <label> <input.fasta> <output.tsv>")
        sys.exit(1)
    
    label = sys.argv[1]
    fasta_file = sys.argv[2]
    output_file = sys.argv[3]

    fasta_to_tsv(fasta_file, label, output_file)
    print(f"TSV file created: {output_file}")

if __name__ == "__main__":
    main()