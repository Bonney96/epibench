import json
import pandas as pd
from typing import Optional, List

def fasta_to_json(fasta_path: str, json_path: str) -> bool:
    """
    Converts a FASTA file to a JSON file.

    Args:
        fasta_path: Path to the input FASTA file.
        json_path: Path to the output JSON file.

    Returns:
        True if conversion was successful, False otherwise.
    """
    sequences = {}
    try:
        from Bio import SeqIO
        for record in SeqIO.parse(fasta_path, "fasta"):
            sequences[record.id] = str(record.seq)
        with open(json_path, 'w') as f:
            json.dump(sequences, f, indent=4)
        print(f"Successfully converted {fasta_path} to {json_path}")
        return True
    except ImportError:
        print("Error: Biopython is required for FASTA conversion. Install with 'pip install biopython'")
        return False
    except FileNotFoundError:
        print(f"Error: FASTA file not found at {fasta_path}")
        return False
    except Exception as e:
        print(f"An error occurred during FASTA to JSON conversion: {e}")
        return False

def bed_to_csv(bed_path: str, csv_path: str, column_names: Optional[List[str]] = None) -> bool:
    """
    Converts a BED file (assuming tab-separated) to a CSV file.

    Args:
        bed_path: Path to the input BED file.
        csv_path: Path to the output CSV file.
        column_names: Optional list of column names. If None, uses standard
                      BED columns (chromosome, start, end) plus numbered extras.

    Returns:
        True if conversion was successful, False otherwise.
    """
    try:
        # Read the BED file, handling potential lines with varying number of columns
        # Find the maximum number of columns first
        max_cols = 0
        with open(bed_path, 'r') as f:
            for line in f:
                 if line.strip() and not line.startswith(('#', 'track', 'browser')):
                     max_cols = max(max_cols, len(line.strip().split('\t')))

        # Define default column names if needed
        if column_names is None:
            column_names = ['chromosome', 'start', 'end']
            if max_cols > 3:
                 # BED spec often implies 4=name, 5=score, 6=strand, etc.
                 # Let's try a more standard default set if cols >= 6
                 if max_cols >= 6:
                     default_names = ['name', 'score', 'strand']
                     column_names.extend(default_names[:max_cols - 3])
                 else:
                     # For 4 or 5 columns, just add generic names
                     column_names.extend([f'col_{i+4}' for i in range(max_cols - 3)])
            # Ensure the list length matches max_cols if fewer default names provided
            if len(column_names) < max_cols:
                 column_names.extend([f'col_{i+len(column_names)+1}' for i in range(max_cols - len(column_names))])

        # Read into pandas DataFrame
        df = pd.read_csv(
            bed_path,
            sep='\t',
            header=None,
            comment='#', # Ignore comment lines
            names=column_names[:max_cols], # Use names only up to max detected columns
            low_memory=False
        )

        # Handle empty file case after reading - moved to exception handling
        # if df.empty: ...

        # Basic validation - check if core columns *based on provided names* exist
        core_cols_to_check = ['chromosome', 'start', 'end']
        if column_names is not None:
             # If custom names provided, check if the *first three* exist
             if len(df.columns) < 3:
                  raise ValueError("BED file must contain at least three columns.")
             # Check if the first three columns are numeric for start/end
             if not pd.api.types.is_numeric_dtype(df.iloc[:, 1]) or not pd.api.types.is_numeric_dtype(df.iloc[:, 2]):
                   raise ValueError("BED file 'start' (col 2) and 'end' (col 3) columns must be numeric.")
        else:
             # If default names, check standard names
             if not all(col in df.columns for col in core_cols_to_check):
                  raise ValueError("BED file must contain at least chromosome, start, and end columns.")
             if not pd.api.types.is_numeric_dtype(df['start']) or not pd.api.types.is_numeric_dtype(df['end']):
                  raise ValueError("BED file 'start' and 'end' columns must be numeric.")

        df.to_csv(csv_path, index=False)
        print(f"Successfully converted {bed_path} to {csv_path}")
        return True
    except FileNotFoundError:
        print(f"Error: BED file not found at {bed_path}")
        return False
    except pd.errors.EmptyDataError:
         # Handle empty file specifically
         print(f"Input BED file {bed_path} is empty. Creating empty CSV.")
         header = column_names if column_names is not None else ['chromosome', 'start', 'end']
         pd.DataFrame(columns=header).to_csv(csv_path, index=False)
         return True # Empty file conversion is considered a success
    except pd.errors.ParserError as e:
        print(f"Error parsing BED file {bed_path}: {e}")
        return False
    except ValueError as e:
         print(f"Error validating BED file {bed_path}: {e}")
         return False
    except Exception as e:
        print(f"An error occurred during BED to CSV conversion: {e}")
        return False

# Example Usage (Conceptual)
# if fasta_to_json("input.fasta", "output.json"):
#     print("FASTA conversion done.")

# if bed_to_csv("input.bed", "output.csv", column_names=['chr', 'start', 'end', 'name', 'score', 'strand']):
#      print("BED conversion done.")
# if bed_to_csv("input_simple.bed", "output_simple.csv"):
#      print("Simple BED conversion done.") 