import argparse
import sys
import os
import numpy as np
import pandas as pd
import pyfaidx
import pyBigWig
import h5py
from tqdm import tqdm
import yaml
import json
from typing import List, Tuple, Optional
import warnings
import random # For shuffling chromosomes
import logging # Import logging module

# Import helper functions and config loading
from epibench.config.config_manager import ConfigManager # Import the class
from epibench.utils.logging import LoggerManager # Import the LoggerManager class

# Get a logger for this module
logger = logging.getLogger(__name__)

def one_hot_encode(sequence: str, dtype=np.float32) -> np.ndarray:
    """One-hot encodes a DNA sequence (A, C, G, T). N maps to all zeros.

    Args:
        sequence: The DNA sequence string.
        dtype: The numpy dtype for the output array.

    Returns:
        A numpy array of shape (len(sequence), 4).
    """
    sequence = sequence.upper()
    seq_len = len(sequence)
    encoding = np.zeros((seq_len, 4), dtype=dtype)
    
    base_map = {'A': 0, 'C': 1, 'G': 2, 'T': 3}
    
    for i, base in enumerate(sequence):
        if base in base_map:
            encoding[i, base_map[base]] = 1
        # N or other characters remain all zeros
            
    return encoding

def get_windows(fasta_handle: pyfaidx.Fasta, window_size: int, step: int) -> List[Tuple[str, int, int]]:
    """Generates genomic windows based on chromosome lengths.

    Args:
        fasta_handle: An open pyfaidx.Fasta handle to the reference genome.
        window_size: The size of each window (e.g., 10000).
        step: The step size between windows (e.g., 10000 for non-overlapping).

    Returns:
        A list of tuples, where each tuple is (chromosome, start, end).
    """
    windows = []
    for chrom_name in fasta_handle.keys():
        chrom_len = len(fasta_handle[chrom_name])
        for start in range(0, chrom_len - window_size + 1, step):
            end = start + window_size
            windows.append((chrom_name, start, end))
        # Optionally handle edge cases like chromosomes shorter than window_size
        # or adding a final partial window if needed by the analysis.
        # For now, only full-sized windows are generated.
    return windows

def setup_process_data_parser(parser):
    """Adds the arguments for the process-data command to the main parser."""
    # Placeholder arguments - these will be refined in later subtasks
    parser.add_argument(
        '--config',
        type=str,
        required=True,
        help='Path to the configuration file (YAML/JSON) defining processing parameters.'
    )
    parser.add_argument(
        '--output-dir',
        type=str,
        required=True,
        help='Directory to save the processed data and splits.'
    )
    # Add more specific arguments as needed, potentially driven by the config file
    # e.g., input file paths if not in config, override parameters, etc.

def split_hdf5_by_chromosome(input_h5_path: str, output_dir: str, train_ratio: float = 0.7, val_ratio: float = 0.15, random_seed: Optional[int] = None):
    """Splits a single HDF5 file into train, validation, and test sets based on chromosomes.

    Args:
        input_h5_path: Path to the combined HDF5 file created by processing.
        output_dir: Directory to save the train.h5, validation.h5, test.h5 files.
        train_ratio: Proportion of chromosomes for the training set.
        val_ratio: Proportion of chromosomes for the validation set.
        random_seed: Optional random seed for reproducible shuffling.
    """
    logger.info("\nStarting chromosome-based data splitting...")
    
    train_path = os.path.join(output_dir, 'train.h5')
    val_path = os.path.join(output_dir, 'validation.h5')
    test_path = os.path.join(output_dir, 'test.h5')
    
    with h5py.File(input_h5_path, 'r') as infile, \
         h5py.File(train_path, 'w') as train_file, \
         h5py.File(val_path, 'w') as val_file, \
         h5py.File(test_path, 'w') as test_file:
        
        # 1. Get chromosome assignments
        if 'chromosomes' not in infile:
            raise ValueError(f"Input HDF5 file {input_h5_path} missing 'chromosomes' dataset for splitting.")
        
        all_sample_chroms_bytes = infile['chromosomes'][:] # Load all chromosome names
        all_sample_chroms = [c.decode('utf-8') for c in all_sample_chroms_bytes]
        unique_chromosomes = sorted(list(set(all_sample_chroms)))
        num_chromosomes = len(unique_chromosomes)
        logger.info(f"Found {infile['chromosomes'].shape[0]} total samples across {num_chromosomes} unique chromosomes: {unique_chromosomes}")

        if num_chromosomes < 3:
             raise ValueError("Need at least 3 unique chromosomes to perform train/validation/test split.")

        # 2. Shuffle and split chromosomes
        if random_seed is not None:
            logger.info(f"Using random seed: {random_seed} for shuffling chromosomes.")
            random.seed(random_seed)
        random.shuffle(unique_chromosomes)
        
        num_train = int(np.ceil(train_ratio * num_chromosomes))
        num_val = int(np.floor(val_ratio * num_chromosomes))
        # Ensure num_val is at least 1 if val_ratio > 0
        if val_ratio > 0 and num_val == 0:
             num_val = 1
        num_test = num_chromosomes - num_train - num_val

        # Ensure test set has at least 1 chromosome if possible
        if num_test < 1 and num_train + num_val < num_chromosomes:
             num_test = 1
             # Adjust train or val down if needed, prioritizing train size
             if num_val > 1:
                  num_val -= 1
             elif num_train > 1:
                  num_train -= 1
             else:
                  # This should be rare, happens if ratios are very small
                  logger.warning("Warning: Cannot guarantee minimum size for all splits due to small chromosome count and ratios.")

        if num_train + num_val + num_test > num_chromosomes:
             num_train = num_chromosomes - num_val - num_test # Prioritize val/test sizes
        
        train_chroms = set(unique_chromosomes[:num_train])
        val_chroms = set(unique_chromosomes[num_train:num_train + num_val])
        test_chroms = set(unique_chromosomes[num_train + num_val:])

        logger.info(f"  Train chromosomes ({len(train_chroms)}): {sorted(list(train_chroms))}")
        logger.info(f"  Validation chromosomes ({len(val_chroms)}): {sorted(list(val_chroms))}")
        logger.info(f"  Test chromosomes ({len(test_chroms)}): {sorted(list(test_chroms))}")

        # 3. Create datasets in output files (copy structure from input)
        output_files = {'train': train_file, 'validation': val_file, 'test': test_file}
        output_counts = {'train': 0, 'validation': 0, 'test': 0}
        
        for name, dset in infile.items():
             # Recreate datasets in each output file with maxshape=None
             for outfile in output_files.values():
                 outfile.create_dataset(name, shape=(0,) + dset.shape[1:], 
                                        maxshape=(None,) + dset.shape[1:], 
                                        dtype=dset.dtype, chunks=dset.chunks, 
                                        compression=dset.compression)

        # 4. Iterate through input data and write to appropriate split file
        total_samples = infile['features'].shape[0]
        chunk_size = 1000 # Process in chunks for memory efficiency
        
        logger.info(f"Writing data to split files (chunk size: {chunk_size})...")
        for i in tqdm(range(0, total_samples, chunk_size), desc="Splitting data"): 
            end_idx = min(i + chunk_size, total_samples)
            
            # Read chunk data
            features_chunk = infile['features'][i:end_idx]
            targets_chunk = infile['targets'][i:end_idx]
            chroms_chunk = [c.decode('utf-8') for c in infile['chromosomes'][i:end_idx]]
            coords_chunk = infile['coordinates'][i:end_idx] if 'coordinates' in infile else None
            
            # Assign rows to splits based on chromosome
            for j in range(len(chroms_chunk)):
                chrom = chroms_chunk[j]
                if chrom in train_chroms:
                    split_name = 'train'
                elif chrom in val_chroms:
                    split_name = 'validation'
                elif chrom in test_chroms:
                    split_name = 'test'
                else:
                    continue # Should not happen if logic is correct
                    
                outfile = output_files[split_name]
                current_size = outfile['features'].shape[0]
                
                # Resize datasets in the target file
                outfile['features'].resize((current_size + 1,) + infile['features'].shape[1:])
                outfile['targets'].resize((current_size + 1,) + infile['targets'].shape[1:])
                outfile['chromosomes'].resize((current_size + 1,))
                if 'coordinates' in outfile:
                     outfile['coordinates'].resize((current_size + 1,) + infile['coordinates'].shape[1:])
                
                # Write data
                outfile['features'][current_size, ...] = features_chunk[j]
                outfile['targets'][current_size, ...] = targets_chunk[j]
                outfile['chromosomes'][current_size] = chrom.encode('utf-8')
                if 'coordinates' in outfile and coords_chunk is not None:
                     outfile['coordinates'][current_size, ...] = coords_chunk[j]
                
                output_counts[split_name] += 1

    logger.info("Splitting complete.")
    logger.info(f"  Train set ({train_path}): {output_counts['train']} samples")
    logger.info(f"  Validation set ({val_path}): {output_counts['validation']} samples")
    logger.info(f"  Test set ({test_path}): {output_counts['test']} samples")

def process_data_main(args):
    """Main function for the process-data command."""
    config_manager = None # Initialize to None
    config = {}
    try:
        # Try to load config first to get logging settings
        config_manager = ConfigManager(args.config) # Instantiate the manager
        config = config_manager.config # Get the loaded config dictionary

        log_settings = config.get('logging', {})
        log_level_str = log_settings.get('level', 'INFO').upper()
        log_level = getattr(logging, log_level_str, logging.INFO)
        log_file = log_settings.get('file') # Can be None
        
        # Setup logger based on config
        # Use the LoggerManager class to setup
        LoggerManager.setup_logger(config_manager=config_manager, 
                                   default_log_level=log_level, 
                                   default_log_file=log_file)
        
        logger.info("Starting data processing...")
        logger.info(f"Configuration file path: {args.config}")
        logger.info(f"Output directory: {args.output_dir}")
        logger.info("Configuration loaded successfully.")
        # Optionally log the config itself at DEBUG level
        # import pprint; logger.debug(f"Loaded configuration:\n{pprint.pformat(config)}")

    except FileNotFoundError as e:
        # If config load fails before logger setup, print to stderr
        print(f"Error: Configuration file not found: {e}", file=sys.stderr)
        sys.exit(1)
    except (ValueError, yaml.YAMLError, json.JSONDecodeError) as e:
        print(f"Error loading or parsing configuration file {args.config} before logging setup: {e}", file=sys.stderr)
        sys.exit(1)
    except Exception as e:
        # Catch any other errors during initial config load/logger setup
        print(f"An unexpected error occurred during initial setup: {e}", file=sys.stderr)
        sys.exit(1)

    # --- Actual Processing Logic --- 
    fasta_handle = None
    histone_handles = []
    hdf5_handle = None
    bed_df = None
    processed_h5_path = None

    try:
        # 2. Extract parameters from config
        ref_genome_path = config.get('reference_genome')
        methylation_bed_path = config.get('methylation_bed')
        histone_bw_paths = config.get('histone_bigwigs', []) # List of histone mark BigWig paths
        output_h5_path = config.get('output_hdf5', os.path.join(args.output_dir, 'processed_data.h5'))
        processed_h5_path = output_h5_path # Store path for splitting later
        window_size = config.get('window_size', 10000)
        step = config.get('step_size', window_size) # Default to non-overlapping
        num_histone_features = len(histone_bw_paths)

        # Validate required config parameters
        if not ref_genome_path or not methylation_bed_path:
            raise ValueError("Configuration must include 'reference_genome' and 'methylation_bed' paths.")
        if num_histone_features == 0:
             warnings.warn("No 'histone_bigwigs' specified in config. Output matrix will only contain sequence data.")
        # Target matrix shape requires 11 features (4 sequence + 7 histone) - enforce?
        if num_histone_features != 7:
             warnings.warn(f"Expected 7 histone BigWigs for (10000, 11) matrix, but found {num_histone_features}. Adjusting output shape.")
        output_feature_dim = 4 + num_histone_features

        # Ensure output directory exists
        try:
            os.makedirs(os.path.dirname(output_h5_path), exist_ok=True)
        except OSError as e:
             logger.error(f"Failed to create output directory {os.path.dirname(output_h5_path)}: {e}", exc_info=True)
             raise # Re-raise to exit

        # 3. Initialize file handlers and load methylation data
        logger.info(f"Opening reference genome: {ref_genome_path}")
        fasta_handle = pyfaidx.Fasta(ref_genome_path)
        
        logger.info("Opening histone BigWig files:")
        for bw_path in histone_bw_paths:
            logger.info(f"  - {bw_path}")
            histone_handles.append(pyBigWig.open(bw_path))
        
        logger.info(f"Loading methylation BED: {methylation_bed_path}")
        # Assuming simple BED: chr, start, end, methylation_level (0-1)
        bed_df = pd.read_csv(methylation_bed_path, sep='\t', header=None, usecols=[0, 1, 2, 3],
                               names=['chromosome', 'start', 'end', 'methylation_level'])
        # Basic validation
        if not all(col in bed_df.columns for col in ['chromosome', 'start', 'end', 'methylation_level']):
            raise ValueError("Methylation BED must contain columns: chromosome, start, end, methylation_level")
        if not pd.api.types.is_numeric_dtype(bed_df['methylation_level']):
             raise ValueError("Methylation level column in BED must be numeric.")

        # 4. Generate genomic windows
        logger.info(f"Generating {window_size}bp windows with step {step}...")
        windows = get_windows(fasta_handle, window_size, step)
        logger.info(f"Generated {len(windows)} windows.")
        if not windows:
             raise ValueError("No valid genomic windows generated. Check reference genome and window size.")

        # 5. Create HDF5 output file (potentially large, use chunking/compression)
        logger.info(f"Creating HDF5 output file: {output_h5_path}")
        hdf5_handle = h5py.File(output_h5_path, 'w')
        # Create resizable datasets with chunking and compression
        # Adjust chunks based on expected data size and access patterns
        chunk_shape_feat = (1, window_size, output_feature_dim)
        chunk_shape_target = (1, 1)
        chunk_shape_chrom = (1,)
        
        dset_features = hdf5_handle.create_dataset('features', shape=(0, window_size, output_feature_dim), 
                                                    maxshape=(None, window_size, output_feature_dim),
                                                    dtype=np.float32, chunks=chunk_shape_feat, compression='gzip')
        dset_targets = hdf5_handle.create_dataset('targets', shape=(0, 1), maxshape=(None, 1), 
                                                  dtype=np.float32, chunks=chunk_shape_target, compression='gzip')
        dset_chromosomes = hdf5_handle.create_dataset('chromosomes', shape=(0,), maxshape=(None,), 
                                                       dtype=h5py.string_dtype(encoding='utf-8'), 
                                                       chunks=chunk_shape_chrom, compression='gzip')
        # Store original window coordinates (optional but useful metadata)
        dset_coords = hdf5_handle.create_dataset('coordinates', shape=(0, 2), maxshape=(None, 2), 
                                                 dtype=np.int64, chunks=(1024, 2), compression='gzip')


        # 6. Process each window
        logger.info("Processing windows...")
        processed_count = 0
        for chrom, start, end in tqdm(windows, desc="Processing windows", file=sys.stdout): 
            # Fetch sequence
            sequence = fasta_handle[chrom][start:end].seq
            if len(sequence) != window_size:
                 warnings.warn(f"Skipping window {chrom}:{start}-{end} - sequence length mismatch ({len(sequence)} != {window_size}).")
                 continue

            # One-hot encode sequence
            seq_encoded = one_hot_encode(sequence)
            
            # Fetch histone signals
            histone_signals = np.zeros((window_size, num_histone_features), dtype=np.float32)
            for i, bw_handle in enumerate(histone_handles):
                 try:
                     # Get values, fill NaNs with 0
                     vals = bw_handle.values(chrom, start, end, numpy=True) 
                     vals = np.nan_to_num(vals) # Replace NaN with 0
                     if len(vals) == window_size:
                          histone_signals[:, i] = vals
                     else:
                          # Handle cases where pyBigWig returns fewer values (e.g., end of chrom)
                          # Pad with zeros or decide how to handle. Padding is simpler.
                          histone_signals[:len(vals), i] = vals
                          if len(vals) < window_size:
                               warnings.warn(f"Histone mark {i+1} ({histone_bw_paths[i]}) signal shorter than window {window_size} at {chrom}:{start}-{end}. Padding with zeros.")
                 except Exception as e:
                      warnings.warn(f"Error fetching signal for histone {i+1} in window {chrom}:{start}-{end}: {e}. Skipping this mark for this window.")
            
            # Combine features: (WindowSize, 4 + NumHistone)
            features_matrix = np.concatenate([seq_encoded, histone_signals], axis=1)
            
            # Calculate target methylation level for the window
            # Simple average methylation level within the window from BED
            window_meth = bed_df[(bed_df['chromosome'] == chrom) & (bed_df['start'] >= start) & (bed_df['end'] <= end)]
            if window_meth.empty:
                 target_methylation = 0.0 # Or np.nan, or skip window? Let's use 0 for now.
                 warnings.warn(f"No methylation sites found in window {chrom}:{start}-{end}. Setting target to 0.")
            else:
                 target_methylation = window_meth['methylation_level'].mean()
                 if np.isnan(target_methylation):
                      target_methylation = 0.0 # Handle case where mean might be NaN

            # Append data to HDF5 datasets
            current_size = dset_features.shape[0]
            dset_features.resize((current_size + 1, window_size, output_feature_dim))
            dset_targets.resize((current_size + 1, 1))
            dset_chromosomes.resize((current_size + 1,))
            dset_coords.resize((current_size + 1, 2))

            dset_features[current_size, :, :] = features_matrix
            dset_targets[current_size, 0] = target_methylation
            dset_chromosomes[current_size] = chrom
            dset_coords[current_size, :] = [start, end]
            processed_count += 1

        logger.info(f"Finished processing. Total windows written: {processed_count}")

    except (FileNotFoundError, ValueError, KeyError, yaml.YAMLError, json.JSONDecodeError, h5py.FileError, pyBigWig.BigWigError, pd.errors.ParserError) as e:
        logger.error(f"Error during data processing: {e}", exc_info=True) # Log traceback for errors
        sys.exit(1)
    except Exception as e:
        logger.critical(f"An unexpected critical error occurred during data processing: {e}", exc_info=True)
        sys.exit(1)
    finally:
        # Ensure all file handles are closed
        if fasta_handle:
            # pyfaidx doesn't have an explicit close method typically
            pass 
        if histone_handles:
            for i, handle in enumerate(histone_handles):
                 try: handle.close()
                 except Exception as e: logger.warning(f"Error closing BigWig handle {i}: {e}")
        if hdf5_handle:
            try: 
                hdf5_handle.close()
                logger.info(f"Closed HDF5 file: {processed_h5_path}")
            except Exception as e: logger.warning(f"Error closing HDF5 file: {e}")

    # --- Chromosome-based Splitting --- 
    if processed_h5_path and os.path.exists(processed_h5_path):
         try:
             split_config = config.get('split_ratios', {})
             train_ratio = split_config.get('train', 0.7)
             val_ratio = split_config.get('validation', 0.15)
             # test_ratio is implicitly 1 - train - val
             random_seed = config.get('random_seed', None)
             
             split_hdf5_by_chromosome(
                 input_h5_path=processed_h5_path,
                 output_dir=args.output_dir,
                 train_ratio=train_ratio,
                 val_ratio=val_ratio,
                 random_seed=random_seed
             )
             # Optional: Delete the combined file after splitting?
             # os.remove(processed_h5_path)
             # print(f"Removed combined file: {processed_h5_path}")

         except Exception as e:
             logger.error(f"Error during data splitting: {e}", exc_info=True)
             logger.warning("Data processing completed, but splitting failed.")
    else:
         logger.warning("Skipping data splitting because processing did not complete or output file is missing.")

    logger.info("Data processing and splitting pipeline complete.")

if __name__ == '__main__':
    # This allows running the script directly for testing
    parser = argparse.ArgumentParser(description='EpiBench Data Processing CLI')
    setup_process_data_parser(parser)
    args = parser.parse_args()
    # Basic logger setup if run directly (can be overridden by config)
    # setup_logger()
    process_data_main(args) 