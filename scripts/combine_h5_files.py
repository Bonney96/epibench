import argparse
import h5py
import numpy as np
from pathlib import Path
import sys
import logging

# Set up basic logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

def combine_h5_files(sample_id: str, input_dir: Path, output_dir: Path):
    """
    Combines train, validation, and test HDF5 files for a given sample ID
    into a single HDF5 file.

    Args:
        sample_id: The sample identifier (e.g., '263578').
        input_dir: The base directory where the sample folders are located.
        output_dir: The directory where the combined file will be saved.
    """
    logger.info(f"Starting combination for sample ID: {sample_id}")
    
    # Define file suffixes and dataset names
    file_suffixes = ['train.h5', 'validation.h5', 'test.h5']
    dataset_names = ['features', 'targets', 'coordinates']
    
    combined_data = {name: [] for name in dataset_names}

    # Find the sample directory
    # Assumes a structure like .../pipeline_output/AML_263578_cpgislands/
    sample_glob = f"*{sample_id}*"
    matching_dirs = list(input_dir.glob(sample_glob))
    
    if not matching_dirs:
        logger.error(f"No directory found for sample ID glob '{sample_glob}' in {input_dir}")
        sys.exit(1)
        
    sample_dir = matching_dirs[0]
    if len(matching_dirs) > 1:
        logger.warning(f"Multiple directories found for glob '{sample_glob}'. Using the first one: {sample_dir}")

    logger.info(f"Processing files in directory: {sample_dir}")

    # Loop through train, validation, and test files
    for suffix in file_suffixes:
        file_path = sample_dir / suffix
        if not file_path.exists():
            logger.warning(f"File not found: {file_path}. Skipping.")
            continue

        logger.info(f"Reading data from: {file_path}")
        try:
            with h5py.File(file_path, 'r') as f:
                for name in dataset_names:
                    if name in f:
                        data = f[name][:]
                        if name == 'coordinates':
                            # Coordinates are structured arrays, handle them carefully
                            # Recreate the structured array to ensure compatibility
                            # Assuming dtype is (('chrom', 'S10'), ('start', '<i8'), ('end', '<i8')) or similar
                            # We can just append the arrays and concatenate later
                            combined_data[name].append(data)
                        else:
                            combined_data[name].append(data)
                    else:
                        logger.warning(f"Dataset '{name}' not found in {file_path}. It will be missing from the combined file.")
        except Exception as e:
            logger.error(f"Failed to read {file_path}: {e}")
            sys.exit(1)

    # Check if any data was collected
    if not any(combined_data.values()):
        logger.error(f"No data could be read for sample {sample_id}. Exiting.")
        sys.exit(1)

    # Concatenate the lists of arrays
    final_data = {}
    try:
        logger.info("Concatenating datasets...")
        for name in dataset_names:
            if combined_data[name]:
                if name == 'coordinates':
                    # Special handling for structured arrays
                    final_data[name] = np.concatenate(combined_data[name], axis=0)
                else:
                    final_data[name] = np.vstack(combined_data[name])
                logger.info(f"  - Final shape for '{name}': {final_data[name].shape}")
            else:
                logger.warning(f"No data for dataset '{name}' across all files.")

    except Exception as e:
        logger.error(f"Failed during dataset concatenation: {e}")
        sys.exit(1)

    # Save the combined data to a new HDF5 file
    output_path = output_dir / f"{sample_id}_all_regions.h5"
    output_dir.mkdir(parents=True, exist_ok=True)
    
    logger.info(f"Saving combined data to: {output_path}")
    try:
        with h5py.File(output_path, 'w') as f:
            for name, data in final_data.items():
                if data is not None:
                    f.create_dataset(name, data=data, compression='gzip')
        logger.info("Successfully created combined HDF5 file.")
    except Exception as e:
        logger.error(f"Failed to write to output file {output_path}: {e}")
        sys.exit(1)


def main():
    parser = argparse.ArgumentParser(
        description="Combine train, validation, and test HDF5 files for an EpiBench sample.",
        formatter_class=argparse.RawTextHelpFormatter
    )
    parser.add_argument(
        "--sample-id",
        type=str,
        required=True,
        help="The unique identifier for the sample (e.g., '263578')."
    )
    parser.add_argument(
        "--input-dir",
        type=Path,
        required=True,
        help="The base directory containing the sample folders (e.g., '.../pipeline_output/processed_data/')."
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        required=True,
        help="The directory where the combined HDF5 file will be saved."
    )
    
    args = parser.parse_args()

    combine_h5_files(args.sample_id, args.input_dir, args.output_dir)

if __name__ == '__main__':
    main() 