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
from typing import List, Tuple, Optional, Iterator
import warnings
import random # For shuffling chromosomes
import logging # Import logging module

# Import helper functions and config loading
from epibench.config.config_manager import ConfigManager # Import the class
from epibench.utils.logging import LoggerManager # Import the LoggerManager class

# Get a logger for this module
logger = logging.getLogger(__name__)

def load_bed_regions(bed_path: str, methyl_col_idx: int = 5) -> Iterator[Tuple[str, int, int, float]]:
    """Loads regions and methylation values from a BED file.

    Assumes a BED-like format with at least 4 columns (chrom, start, end, ...)
    and the methylation value in the specified column (defaulting to 6th column, 0-indexed 5).
    Skips lines that don't conform to the expected format.

    Args:
        bed_path (str): Path to the BED file.
        methyl_col_idx (int): 0-based index of the column containing the methylation score.

    Yields:
        Tuple[str, int, int, float]: Chromosome, start, end, methylation value.
    """
    logger.info(f"Loading BED regions from: {bed_path} (Methylation column index: {methyl_col_idx})")
    line_count = 0
    skipped_count = 0
    try:
        with open(bed_path, 'r') as f:
            for line in f:
                line_count += 1
                line = line.strip()
                if not line or line.startswith('#') or line.startswith('track') or line.startswith('browser'):
                    skipped_count += 1
                    continue
                
                parts = line.split('\t')
                if len(parts) <= methyl_col_idx:
                    logger.warning(f"Skipping line {line_count} in {bed_path}: Not enough columns (expected at least {methyl_col_idx + 1}). Line: '{line}'")
                    skipped_count += 1
                    continue
                    
                try:
                    chrom = parts[0]
                    start = int(parts[1])
                    end = int(parts[2])
                    methyl_val = float(parts[methyl_col_idx])
                    
                    if start >= end:
                         logger.warning(f"Skipping line {line_count} in {bed_path}: Start position ({start}) is not less than end position ({end}). Line: '{line}'")
                         skipped_count += 1
                         continue
                         
                    yield chrom, start, end, methyl_val
                except ValueError as e:
                    logger.warning(f"Skipping line {line_count} in {bed_path}: Error parsing values ({e}). Line: '{line}'")
                    skipped_count += 1
                    continue
                    
    except FileNotFoundError:
        logger.error(f"BED file not found: {bed_path}")
        raise
    except Exception as e:
        logger.error(f"Error reading BED file {bed_path}: {e}", exc_info=True)
        raise
        
    logger.info(f"Finished loading BED regions from {bed_path}. Processed {line_count} lines, skipped {skipped_count}.")

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
    processed_h5_path = None
    train_h5_path = None
    val_h5_path = None
    test_h5_path = None
    # HDF5 handles for splits
    h5_handles = {}

    try:
        # 2. Extract parameters from config
        ref_genome_path = config.get('reference_genome')
        methylation_bed_path = config.get('methylation_bed')
        histone_bw_paths = config.get('histone_bigwigs', []) # List of histone mark BigWig paths
        
        # --- New Parameters for Region-Based Processing ---
        target_seq_length = config.get('target_sequence_length', 1000) # Fixed length for model input
        methyl_col_idx = config.get('methylation_bed_column', 5) # 0-based index (6th column default)
        num_histone_features = len(histone_bw_paths)
        output_feature_dim = 4 + num_histone_features # 4 for DNA sequence

        # --- Splitting Parameters ---
        split_config = config.get('split_ratios', {})
        train_ratio = split_config.get('train', 0.7)
        val_ratio = split_config.get('validation', 0.15)
        test_ratio = 1.0 - train_ratio - val_ratio
        if test_ratio < 0:
            raise ValueError("Train and validation ratios sum to more than 1.0")
        random_seed = config.get('random_seed')

        # Validate required config parameters
        if not ref_genome_path or not methylation_bed_path:
            raise ValueError("Configuration must include 'reference_genome' and 'methylation_bed' paths.")
        if num_histone_features == 0:
             warnings.warn("No 'histone_bigwigs' specified in config. Feature matrix will only contain sequence data.")
        # Output feature dim now depends on number of histone files + 4

        # --- Prepare Output Paths --- 
        train_h5_path = os.path.join(args.output_dir, 'train.h5')
        val_h5_path = os.path.join(args.output_dir, 'validation.h5')
        test_h5_path = os.path.join(args.output_dir, 'test.h5')
        output_paths = {'train': train_h5_path, 'validation': val_h5_path, 'test': test_h5_path}

        # Ensure output directory exists
        try:
            os.makedirs(args.output_dir, exist_ok=True)
        except OSError as e:
             logger.error(f"Failed to create output directory {args.output_dir}: {e}", exc_info=True)
             raise # Re-raise to exit

        # 3. Initialize file handlers
        logger.info(f"Opening reference genome: {ref_genome_path}")
        fasta_handle = pyfaidx.Fasta(ref_genome_path)
        
        logger.info("Opening histone BigWig files:")
        for bw_path in histone_bw_paths:
            logger.info(f"  - {bw_path}")
            histone_handles.append(pyBigWig.open(bw_path))
        
        # --- Load and Split BED Regions (Subtask 24.3) ---
        logger.info(f"Loading and splitting BED regions from: {methylation_bed_path}")
        all_regions = list(load_bed_regions(methylation_bed_path, methyl_col_idx=methyl_col_idx))
        num_regions = len(all_regions)
        logger.info(f"Loaded {num_regions} valid regions.")

        if num_regions == 0:
            raise ValueError("No valid regions loaded from the BED file.")

        # Shuffle indices for splitting
        indices = list(range(num_regions))
        if random_seed is not None:
            logger.info(f"Using random seed {random_seed} for region splitting.")
            random.seed(random_seed)
        random.shuffle(indices)

        # Calculate split points
        n_train = int(np.floor(train_ratio * num_regions))
        n_val = int(np.floor(val_ratio * num_regions))
        # n_test = num_regions - n_train - n_val # Remainder goes to test

        split_indices = {
            'train': indices[:n_train],
            'validation': indices[n_train:n_train + n_val],
            'test': indices[n_train + n_val:]
        }

        logger.info(f"Region counts per split: Train={len(split_indices['train'])}, Validation={len(split_indices['validation'])}, Test={len(split_indices['test'])}")

        # --- Create HDF5 output files (Subtask 24.4) --- 
        logger.info("Creating HDF5 output files...")
        chunk_shape_feat = (1, target_seq_length, output_feature_dim) # Fixed sequence length
        chunk_shape_target = (1, 1)
        chunk_shape_coords = (1, 3) # Store original chr, start, end
        
        for split_name, h5_path in output_paths.items():
            try:
                 h5_handles[split_name] = h5py.File(h5_path, 'w')
                 # Create datasets within each file
                 handle = h5_handles[split_name]
                 handle.create_dataset('features', shape=(0, target_seq_length, output_feature_dim), 
                                         maxshape=(None, target_seq_length, output_feature_dim),
                                         dtype=np.float32, chunks=chunk_shape_feat, compression='gzip')
                 handle.create_dataset('targets', shape=(0, 1), maxshape=(None, 1), 
                                       dtype=np.float32, chunks=chunk_shape_target, compression='gzip')
                 # Store original BED coordinates as metadata
                 handle.create_dataset('coordinates', shape=(0, 3), maxshape=(None, 3), 
                                      dtype=h5py.string_dtype(encoding='utf-8'), # Store chr as string
                                      chunks=chunk_shape_coords, compression='gzip') 
                                      
                 # Add metadata (optional)
                 handle.attrs['reference_genome'] = ref_genome_path
                 handle.attrs['methylation_bed'] = methylation_bed_path
                 handle.attrs['histone_bigwigs'] = json.dumps(histone_bw_paths) # Store list as JSON string
                 handle.attrs['target_sequence_length'] = target_seq_length
                 handle.attrs['epibench_version'] = config.get('_package_version', 'unknown') # Pass version from main

                 logger.info(f"Created HDF5 file for {split_name} split: {h5_path}")
            except Exception as e:
                 logger.error(f"Failed to create or initialize HDF5 file {h5_path} for split {split_name}: {e}", exc_info=True)
                 # Clean up already opened handles before raising
                 for h in h5_handles.values(): h.close()
                 raise

        # --- Process Each Region (Subtask 24.2) --- 
        logger.info("Processing regions and writing to HDF5 files...")
        split_counts = {'train': 0, 'validation': 0, 'test': 0}
        
        # Iterate through shuffled indices and process regions
        for split_name, indices_list in split_indices.items():
             logger.info(f"Processing {len(indices_list)} regions for {split_name} split...")
             h5_handle = h5_handles[split_name]
             # Use tqdm for progress within each split
             for idx in tqdm(indices_list, desc=f"Processing {split_name}", leave=False): 
                 chrom, bed_start, bed_end, target_methylation = all_regions[idx]
                 
                 # --- Center region and determine fetch coordinates --- 
                 center = bed_start + (bed_end - bed_start) // 2
                 fetch_start = center - target_seq_length // 2
                 fetch_end = fetch_start + target_seq_length
                 
                 # --- Handle chromosome boundary edge cases --- 
                 chrom_len = len(fasta_handle[chrom])
                 if fetch_start < 0:
                     logger.debug(f"Adjusting fetch_start ({fetch_start}) to 0 for region {chrom}:{bed_start}-{bed_end}")
                     fetch_start = 0
                     fetch_end = target_seq_length # Ensure fixed length
                 if fetch_end > chrom_len:
                      logger.debug(f"Adjusting fetch_end ({fetch_end}) to {chrom_len} for region {chrom}:{bed_start}-{bed_end}")
                      fetch_end = chrom_len
                      fetch_start = max(0, fetch_end - target_seq_length) # Ensure fixed length
                 
                 # Fetch sequence
                 sequence = fasta_handle[chrom][fetch_start:fetch_end].seq
                 actual_len = len(sequence)
                 
                 # One-hot encode sequence - Pad if needed
                 seq_encoded = one_hot_encode(sequence) # (ActualLen, 4)
                 if actual_len < target_seq_length:
                     padding_needed = target_seq_length - actual_len
                     # Pad at the end (or beginning, decide strategy)
                     pad_array = np.zeros((padding_needed, 4), dtype=np.float32)
                     seq_encoded = np.concatenate([seq_encoded, pad_array], axis=0)
                     logger.debug(f"Padded sequence for region {chrom}:{bed_start}-{bed_end} by {padding_needed} bases.")
                 elif actual_len > target_seq_length:
                      # This shouldn't happen with correct fetch logic, but as safeguard:
                      logger.warning(f"Fetched sequence longer ({actual_len}) than target ({target_seq_length}) for {chrom}:{bed_start}-{bed_end}. Truncating.")
                      seq_encoded = seq_encoded[:target_seq_length, :]
                 
                 # Fetch histone signals
                 histone_signals = np.zeros((target_seq_length, num_histone_features), dtype=np.float32)
                 for i, bw_handle in enumerate(histone_handles):
                     try:
                         # Get values, fill NaNs with 0
                         vals = bw_handle.values(chrom, fetch_start, fetch_end, numpy=True) 
                         vals = np.nan_to_num(vals) # Replace NaN with 0
                         actual_signal_len = len(vals)

                         if actual_signal_len == target_seq_length:
                              histone_signals[:, i] = vals
                         elif actual_signal_len < target_seq_length:
                              # Pad histone signal similar to sequence
                              padding_needed = target_seq_length - actual_signal_len
                              histone_signals[:actual_signal_len, i] = vals
                              # Remainder is already zeros
                              logger.debug(f"Padded histone {i+1} for region {chrom}:{bed_start}-{bed_end} by {padding_needed}.")
                         else: # actual_signal_len > target_seq_length (unlikely)
                              logger.warning(f"Fetched histone signal {i+1} longer ({actual_signal_len}) than target ({target_seq_length}) for {chrom}:{bed_start}-{bed_end}. Truncating.")
                              histone_signals[:, i] = vals[:target_seq_length]
                              
                     except Exception as e:
                          warnings.warn(f"Error fetching signal for histone {i+1} in region {chrom}:{bed_start}-{bed_end} (coords {fetch_start}-{fetch_end}): {e}. Skipping this mark for this region.")
                 
                 # Combine features: (TargetSeqLength, 4 + NumHistone)
                 features_matrix = np.concatenate([seq_encoded, histone_signals], axis=1)
                 
                 # Target methylation is already available from the BED region
                 # target_methylation = target_methylation (variable already holds it)
                 
                 # Append data to HDF5 datasets for the correct split
                 current_size = h5_handle['features'].shape[0]
                 h5_handle['features'].resize((current_size + 1, target_seq_length, output_feature_dim))
                 h5_handle['targets'].resize((current_size + 1, 1))
                 h5_handle['coordinates'].resize((current_size + 1, 3))
     
                 h5_handle['features'][current_size, :, :] = features_matrix
                 h5_handle['targets'][current_size, 0] = target_methylation
                 # Store original BED coordinates as bytes
                 h5_handle['coordinates'][current_size, :] = [chrom.encode('utf-8'), str(bed_start).encode('utf-8'), str(bed_end).encode('utf-8')]
                 split_counts[split_name] += 1

        logger.info(f"Finished processing regions.")
        for split_name, count in split_counts.items():
             logger.info(f"  {split_name.capitalize()} split: {count} regions written to {output_paths[split_name]}")

    except (FileNotFoundError, ValueError, KeyError, yaml.YAMLError, json.JSONDecodeError, pd.errors.ParserError, RuntimeError) as e:
        # Removed pyBigWig.BigWigError. Catching RuntimeError explicitly for pyBigWig open error.
        # Catching general OSError for HDF5 file issues might be better if needed.
        logger.error(f"Error during data processing: {e}", exc_info=True) # Log traceback for errors
        # Ensure files are closed even on error before exiting
        # Close handles in finally block
        if fasta_handle:
            # pyfaidx doesn't have an explicit close method typically
            pass 
        if histone_handles:
            for i, handle in enumerate(histone_handles):
                 try: handle.close()
                 except Exception as e: logger.warning(f"Error closing BigWig handle {i}: {e}")
        if h5_handles: 
            for split_name, handle in h5_handles.items():
                 if handle: # Check if handle was successfully created
                     try: 
                         handle.close()
                         logger.info(f"Closed HDF5 file: {output_paths[split_name]}")
                     except Exception as e: logger.warning(f"Error closing HDF5 file {output_paths[split_name]}: {e}")
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
        if h5_handles: 
            for split_name, handle in h5_handles.items():
                 if handle: # Check if handle was successfully created
                     try: 
                         handle.close()
                         logger.info(f"Closed HDF5 file: {output_paths[split_name]}")
                     except Exception as e: logger.warning(f"Error closing HDF5 file {output_paths[split_name]}: {e}")

    logger.info("Region-based data processing complete.")

if __name__ == '__main__':
    # This allows running the script directly for testing
    parser = argparse.ArgumentParser(description='EpiBench Data Processing CLI')
    setup_process_data_parser(parser)
    args = parser.parse_args()
    # Basic logger setup if run directly (can be overridden by config)
    # setup_logger()
    process_data_main(args) 