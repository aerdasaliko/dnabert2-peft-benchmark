import argparse
import csv
import textwrap

def csv_to_fasta(input_csv, output_fasta, line_width=80):
    with open(input_csv, "r", newline="", encoding="utf-8") as infile:
        reader = csv.DictReader(infile)

        required_columns = {"sequence", "label"}
        if not required_columns.issubset(reader.fieldnames or []):
            raise ValueError(
                f"CSV must contain the columns: {', '.join(required_columns)}"
            )

        with open(output_fasta, "w", encoding="utf-8") as outfile:
            for index, row in enumerate(reader, start=1):
                sequence = row["sequence"].strip()
                if not sequence:
                    raise ValueError(f"Empty sequence found on row {index + 1}")
                outfile.write(f">seq_{index}\n")
                for chunk in textwrap.wrap(sequence, width=line_width):
                    outfile.write(f"{chunk}\n")

def main():
    parser = argparse.ArgumentParser(
        description="Convert a CSV file with sequence and label columns to FASTA."
    )
    parser.add_argument("input_csv", help="Input CSV file")
    parser.add_argument("output_fasta", help="Output FASTA file")
    args = parser.parse_args()
    csv_to_fasta(args.input_csv, args.output_fasta)
    print(f"FASTA file written to: {args.output_fasta}")

if __name__ == "__main__":
    main()