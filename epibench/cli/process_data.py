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
# from epibench.config.config_manager import ConfigManager # No longer using ConfigManager here
from epibench.utils.logging import LoggerManager # Import the LoggerManager class
from epibench.validation.config_validator import validate_process_config, ProcessConfig # Import validator

# Get a logger for this module
logger = logging.getLogger(__name__)

def generate_region_boundary_channel(target_length: int, region_start_in_window: int, region_end_in_window: int, dtype=np.float32) -> np.ndarray:
    """Generates a binary channel marking the original BED region boundaries within the target window.

    Args:
        target_length (int): The fixed length of the output sequence window.
        region_start_in_window (int): The start index of the original BED region relative to the window start (0-based).
        region_end_in_window (int): The end index (exclusive) of the original BED region relative to the window start.
        dtype: The numpy dtype for the output array.

    Returns:
        A numpy array of shape (target_length,) with 1s marking the region and 0s elsewhere.
    """
    boundary_channel = np.zeros(target_length, dtype=dtype)
    
    # Clamp start and end to be within the window bounds [0, target_length)
    start_clamped = max(0, region_start_in_window)
    end_clamped = min(target_length, region_end_in_window)
    
    if start_clamped < end_clamped: # Ensure start is less than end after clamping
        boundary_channel[start_clamped:end_clamped] = 1
        
    return boundary_channel

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
    # Setup basic logger first to catch early errors
    logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
    logger = logging.getLogger(__name__) # Re-assign logger after basic setup

    validated_config: Optional[ProcessConfig] = None
    try:
        # Validate the configuration file using the Pydantic model
        logger.info(f"Validating configuration file: {args.config}")
        validated_config = validate_process_config(args.config)
        logger.info("Configuration validated successfully.")
        
        # Now setup logger properly using validated config
        log_settings = validated_config.logging_config
        log_level_str = log_settings.level
        log_level = getattr(logging, log_level_str, logging.INFO)
        log_file = log_settings.file # Can be None
        
        # Use the LoggerManager class to setup
        # Note: LoggerManager might need adjustment if it expected the old config format
        # For now, we pass None for config_manager and use validated values
        LoggerManager.setup_logger(config_manager=None, # Pass None or adjust LoggerManager
                                   log_level_override=log_level, 
                                   log_file_override=log_file,
                                   log_to_console=True) # Assume console logging is desired
        
        # Re-log initial messages with the proper formatter
        logger.info("Starting data processing...")
        logger.info(f"Configuration file path: {args.config}")
        logger.info(f"Output directory: {args.output_dir}")
        logger.debug(f"Validated configuration:\n{validated_config.json(indent=2)}")

    except FileNotFoundError as e:
        logger.error(f"Error: Configuration file not found: {e}", exc_info=True)
        sys.exit(1)
    except (ValueError, yaml.YAMLError, json.JSONDecodeError, KeyError) as e:
        # Catch validation errors (KeyError, ValueError from Pydantic) and parsing errors
        logger.error(f"Error loading or validating configuration file {args.config}: {e}", exc_info=True)
        sys.exit(1)
    except Exception as e:
        logger.error(f"An unexpected error occurred during initial setup: {e}", exc_info=True)
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
        # 2. Extract parameters from validated Pydantic config object
        ref_genome_path = validated_config.input_paths.reference_genome
        methylation_bed_path = validated_config.input_paths.methylation_bed
        histone_bw_paths = validated_config.input_paths.histone_bigwigs
        
        # --- New Parameters for Region-Based Processing ---
        target_seq_length = validated_config.processing_params.target_sequence_length
        methyl_col_idx = validated_config.processing_params.methylation_bed_column
        num_histone_features = len(histone_bw_paths)
        output_feature_dim = 4 + num_histone_features + 1 # 4: DNA, N: Histones, 1: Region Boundary

        # --- Splitting Parameters ---
        split_config = validated_config.split_ratios
        train_ratio = split_config.train
        val_ratio = split_config.validation
        test_ratio = 1.0 - train_ratio - val_ratio
        # Pydantic validator already checks test_ratio >= 0
        # if test_ratio < 0: # Check already done in Pydantic model
        #     raise ValueError("Train and validation ratios sum to more than 1.0")
        random_seed = validated_config.random_seed

        # Validate required config parameters (Pydantic handles presence, FilePath checks existence)
        # if not ref_genome_path or not methylation_bed_path: # Already validated by Pydantic
        #     raise ValueError("Configuration must include 'reference_genome' and 'methylation_bed' paths.")
        if num_histone_features == 0:
             warnings.warn("No 'histone_bigwigs' specified in config. Feature matrix will only contain sequence data.")
             logger.warning("No 'histone_bigwigs' specified in config. Feature matrix will only contain sequence data.")
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
        # Chunk shape reflects the dimensions of a single region's data
        chunk_shape_feat = (1, target_seq_length, output_feature_dim) # 1 region, seq_len, num_features (4 + histone + boundary)
        chunk_shape_target = (1, 1) # 1 region, 1 target value
        # Store original chr (string), start (int), end (int)
        chunk_shape_coords = (1,) # Chunking by region for coordinates too
        
        for split_name, h5_path in output_paths.items():
            try:
                 logger.info(f"Initializing HDF5 file for {split_name} split: {h5_path}")
                 # Open in 'w' mode to create/overwrite
                 handle = h5py.File(h5_path, 'w') 
                 h5_handles[split_name] = handle
                 
                 # Define chunk shapes (tune based on expected data size and access patterns)
                 chunk_shape_features = (64, target_seq_length, output_feature_dim) # Chunk across samples
                 chunk_shape_targets = (64, 1)
                 chunk_shape_coords = (64,)

                 # Create datasets with maxshape=(None, ...) to allow resizing
                 logger.info(f"  Creating dataset 'features' with shape (0, {target_seq_length}, {output_feature_dim}) and chunk shape {chunk_shape_features}")
                 handle.create_dataset('features', shape=(0, target_seq_length, output_feature_dim), 
                                        maxshape=(None, target_seq_length, output_feature_dim), 
                                        dtype=np.float32, 
                                        chunks=chunk_shape_features, compression='gzip')
                 logger.info(f"  Creating dataset 'targets' with shape (0, 1) and chunk shape {chunk_shape_targets}")
                 handle.create_dataset('targets', shape=(0, 1), maxshape=(None, 1), 
                                        dtype=np.float32, 
                                        chunks=chunk_shape_targets, compression='gzip')
                 logger.info(f"  Creating dataset 'chrom' with shape (0,) and chunk shape {chunk_shape_coords}") # Storing coords separately
                 handle.create_dataset('chrom', shape=(0,), maxshape=(None,), 
                                        dtype=h5py.string_dtype(encoding='utf-8'), 
                                        chunks=chunk_shape_coords, compression='gzip')
                 logger.info(f"  Creating dataset 'start' with shape (0,) and chunk shape {chunk_shape_coords}")
                 handle.create_dataset('start', shape=(0,), maxshape=(None,), 
                                        dtype=np.int64, 
                                        chunks=chunk_shape_coords, compression='gzip')
                 logger.info(f"  Creating dataset 'end' with shape (0,) and chunk shape {chunk_shape_coords}")
                 handle.create_dataset('end', shape=(0,), maxshape=(None,), 
                                        dtype=np.int64, 
                                        chunks=chunk_shape_coords, compression='gzip') 
                                      
                 # Add metadata (optional) - Accessing validated config fields
                 handle.attrs['reference_genome'] = str(validated_config.input_paths.reference_genome)
                 handle.attrs['methylation_bed'] = str(validated_config.input_paths.methylation_bed)
                 # Convert Path objects to strings for JSON serialization
                 handle.attrs['histone_bigwigs'] = json.dumps([str(p) for p in validated_config.input_paths.histone_bigwigs]) 
                 handle.attrs['target_sequence_length'] = validated_config.processing_params.target_sequence_length
                 handle.attrs['methylation_bed_column'] = validated_config.processing_params.methylation_bed_column
                 handle.attrs['random_seed'] = validated_config.random_seed if validated_config.random_seed is not None else 'None'
                 # Placeholder for version - How to get this now?
                 # Option 1: Use importlib.metadata
                 try:
                     import importlib.metadata
                     version = importlib.metadata.version('epibench') # Replace 'epibench' with your actual package name
                 except importlib.metadata.PackageNotFoundError:
                     version = 'unknown'
                 handle.attrs['epibench_version'] = version 
                 handle.attrs['feature_channels'] = f"4 (Sequence) + {num_histone_features} (Histones) + 1 (Region Boundary)"

                 logger.info(f"Created HDF5 file for {split_name} split: {h5_path}")
            except Exception as e:
                 logger.error(f"Failed to create or initialize HDF5 file {h5_path} for split {split_name}: {e}", exc_info=True)
                 # Clean up already opened handles before raising
                 for h in h5_handles.values(): h.close()
                 raise

        # Iterate through splits and their corresponding region indices
        for split_name, indices_for_split in split_indices.items():
            logger.info(f"Processing {len(indices_for_split)} regions for {split_name} split...")
            h5_handle = h5_handles[split_name]
            
            # Use tqdm for progress bar
            for region_idx in tqdm(indices_for_split, desc=f"Processing {split_name}", unit="region"):
                chrom, bed_start, bed_end, target_methylation = all_regions[region_idx]
                
                # --- Calculate Fetch Coordinates (Subtask 26.1) ---
                center = bed_start + (bed_end - bed_start) // 2
                fetch_start = center - target_seq_length // 2
                fetch_end = fetch_start + target_seq_length
                
                # Ensure coordinates are non-negative
                if fetch_start < 0:
                    # Adjust fetch_end proportionally if start is pushed to 0
                    fetch_end -= fetch_start # fetch_end = fetch_end + abs(fetch_start)
                    fetch_start = 0
                    logger.debug(f"Adjusted fetch start to 0 for region center {center} on {chrom}. New window: {fetch_start}-{fetch_end}")
                    
                # Check against chromosome length (handle requires pyfaidx handle)
                try:
                    chrom_len = len(fasta_handle[chrom])
                    if fetch_end > chrom_len:
                        # Adjust fetch_start proportionally if end hits boundary
                        fetch_start -= (fetch_end - chrom_len)
                        fetch_end = chrom_len
                        # Re-check non-negativity after adjustment
                        if fetch_start < 0: fetch_start = 0 
                        logger.debug(f"Adjusted fetch end to {chrom_len} for region center {center} on {chrom}. New window: {fetch_start}-{fetch_end}")
                        # If the window size is still not target_seq_length after adjustment, it needs padding
                        if (fetch_end - fetch_start) < target_seq_length and fetch_start == 0:
                             pass # Padding will handle this
                        elif (fetch_end - fetch_start) < target_seq_length:
                             logger.warning(f"Could not maintain target sequence length {target_seq_length} at end of chromosome {chrom} for region {bed_start}-{bed_end}. Effective length: {fetch_end-fetch_start}")
                             # Padding will handle this
                             
                except KeyError:
                    logger.warning(f"Chromosome {chrom} not found in reference genome {ref_genome_path}. Skipping region {chrom}:{bed_start}-{bed_end}.")
                    continue # Skip this region
                except Exception as e:
                     logger.error(f"Error getting chromosome length for {chrom}: {e}. Skipping region {chrom}:{bed_start}-{bed_end}.", exc_info=True)
                     continue # Skip this region

                # If after adjustment the effective window is still smaller than target, log warning
                effective_fetch_len = fetch_end - fetch_start
                if effective_fetch_len < target_seq_length:
                     logger.debug(f"Effective fetch window {effective_fetch_len} for region {chrom}:{bed_start}-{bed_end} is less than target {target_seq_length}. Padding will be applied.")
                elif effective_fetch_len > target_seq_length:
                     # This shouldn't happen with the logic above, but catch just in case
                     logger.warning(f"Effective fetch window {effective_fetch_len} is unexpectedly larger than target {target_seq_length} for region {chrom}:{bed_start}-{bed_end}. Truncating fetch.")
                     fetch_end = fetch_start + target_seq_length # Truncate
                
                # --- Fetch Sequence (Subtask 26.1) ---
                seq = ''
                try:
                    seq = fasta_handle.get_seq(chrom, fetch_start + 1, fetch_end).seq # pyfaidx is 1-based, inclusive
                    # Handle cases where get_seq returns less than expected due to boundaries
                    if len(seq) < target_seq_length:
                        # Need padding - calculate difference and pad with 'N'
                        padding_needed = target_seq_length - len(seq)
                        if fetch_start == 0: # Padding needed at the end
                             seq += 'N' * padding_needed
                        else: # Padding needed at the beginning (unlikely with current logic)
                             seq = 'N' * padding_needed + seq
                        logger.debug(f"Padded sequence for region {chrom}:{bed_start}-{bed_end} by {padding_needed} bases.")
                        
                except Exception as e:
                    logger.warning(f"Error fetching sequence for region {chrom}:{bed_start}-{bed_end} (coords {fetch_start+1}-{fetch_end}): {e}. Skipping region.")
                    continue # Skip region if sequence fetch fails
                
                # One-hot encode sequence: (TargetSeqLength, 4)
                seq_encoded = one_hot_encode(seq)
                
                # --- Fetch Histone Marks (Subtask 26.1) ---
                # Initialize histone signal matrix: (TargetSeqLength, NumHistoneFeatures)
                histone_signals = np.zeros((target_seq_length, num_histone_features), dtype=np.float32)
                # skip_region_histone = False # Removed skip logic for individual histone errors
                
                for i, bw_handle in enumerate(histone_handles):
                    try:
                        # Get values, fill NaNs with 0
                        # pyBigWig uses 0-based, half-open intervals [start, end)
                        vals = bw_handle.values(chrom, fetch_start, fetch_end, numpy=True) 
                        vals = np.nan_to_num(vals) # Replace NaN with 0
                        actual_signal_len = len(vals)

                        if actual_signal_len == target_seq_length:
                             histone_signals[:, i] = vals
                        elif actual_signal_len < target_seq_length:
                             # Pad histone signal similar to sequence
                             padding_needed = target_seq_length - actual_signal_len
                             # Determine if padding is needed at start or end based on fetch coords
                             if fetch_start == 0 and fetch_end < chrom_len:
                                 # Padding at the end
                                 histone_signals[:actual_signal_len, i] = vals
                             elif fetch_end == chrom_len and fetch_start > 0:
                                 # Padding at the beginning
                                 histone_signals[padding_needed:, i] = vals
                             else:
                                 # Default or ambiguous case, pad at end
                                 histone_signals[:actual_signal_len, i] = vals
                             logger.debug(f"Padded histone {i+1} for region {chrom}:{bed_start}-{bed_end} by {padding_needed}.")
                        else: # actual_signal_len > target_seq_length (can happen if BigWig has different resolution)
                             logger.warning(f"Fetched histone signal {i+1} longer ({actual_signal_len}) than target ({target_seq_length}) for {chrom}:{bed_start}-{bed_end}. Truncating.")
                             histone_signals[:, i] = vals[:target_seq_length]
                             
                    except Exception as e:
                         warnings.warn(f"Error fetching signal for histone {i+1} in region {chrom}:{bed_start}-{bed_end} (coords {fetch_start}-{fetch_end}): {e}. Setting channel to zeros.")
                         # Don't skip the whole region, just leave this channel as zeros if one BigWig fails
                         histone_signals[:, i] = 0 # Ensure channel is zeroed on error
                
                # --- Generate Boundary Channel (Subtask 26.2) ---
                # Calculate original region's position relative to the fetched window
                region_start_in_window = bed_start - fetch_start
                region_end_in_window = bed_end - fetch_start
                # Create the 11th channel: a binary mask indicating the original BED region extent
                boundary_channel = generate_region_boundary_channel(
                    target_length=target_seq_length,
                    region_start_in_window=region_start_in_window,
                    region_end_in_window=region_end_in_window
                ) # Shape: (target_seq_length,)
                # Reshape to (target_seq_length, 1) for concatenation
                boundary_channel = boundary_channel.reshape(-1, 1) 
                # --- End Boundary Channel ---
                
                # Combine features: (TargetSeqLength, 4 + NumHistone + 1)
                features_matrix = np.concatenate([seq_encoded, histone_signals, boundary_channel], axis=1)
                
                # Target methylation is already available from the BED region
                # target_methylation = target_methylation (variable already holds it)
                
                # --- Shape Validation (Subtask 26.3) ---
                expected_shape = (target_seq_length, output_feature_dim)
                if features_matrix.shape != expected_shape:
                    logger.error(f"Internal Error: Final feature matrix shape mismatch for region {chrom}:{bed_start}-{bed_end}. Expected {expected_shape}, got {features_matrix.shape}. Skipping region.")
                    continue # Skip writing this malformed region
                # --- End Shape Validation ---
                
                # Append data to HDF5 datasets for the correct split
                current_size = h5_handle['features'].shape[0]
                h5_handle['features'].resize((current_size + 1, target_seq_length, output_feature_dim))
                h5_handle['targets'].resize((current_size + 1, 1))
                # Resize coordinate datasets (now storing string, int, int separately)
                h5_handle['chrom'].resize((current_size + 1,))
                h5_handle['start'].resize((current_size + 1,))
                h5_handle['end'].resize((current_size + 1,))
                
                # Assign data to the new slots
                h5_handle['features'][current_size] = features_matrix
                h5_handle['targets'][current_size] = np.array([[target_methylation]], dtype=np.float32)
                h5_handle['chrom'][current_size] = chrom
                h5_handle['start'][current_size] = bed_start
                h5_handle['end'][current_size] = bed_end

        logger.info(f"Finished processing. Total regions processed: {current_size}. Total regions skipped: {current_size - len(split_indices['train']) - len(split_indices['validation']) - len(split_indices['test'])}.")
        logger.info(f"Processed data saved to HDF5 files in: {args.output_dir}")
        # Report counts per split
        for split_name, h5_handle in h5_handles.items():
            if h5_handle:
                 count = h5_handle['features'].shape[0]
                 logger.info(f"  - {split_name}: {count} regions saved to {h5_handle.filename}")

    except Exception as e:
        logger.error(f"An error occurred during data processing: {e}", exc_info=True)
        # Ensure files are closed even if errors occur mid-processing
        sys.exit(1) # Exit with error code
    finally:
        # 5. Close all file handles
        logger.info("Closing file handles...")
        if fasta_handle:
            try:
                fasta_handle.close()
                logger.debug("Closed FASTA file handle.")
            except Exception as e:
                 logger.warning(f"Error closing FASTA handle: {e}")
                 
        for i, handle in enumerate(histone_handles):
            if handle:
                try:
                    handle.close()
                    logger.debug(f"Closed histone BigWig handle {i+1}.")
                except Exception as e:
                     logger.warning(f"Error closing histone BigWig handle {i+1}: {e}")
                     
        for split_name, handle in h5_handles.items():
            if handle:
                try:
                    handle.close()
                    logger.info(f"Closed HDF5 file handle for {split_name} split.")
                except Exception as e:
                     logger.warning(f"Error closing HDF5 handle for {split_name}: {e}")

    logger.info("Data processing finished.")

if __name__ == '__main__':
    # This allows running the script directly for testing
    parser = argparse.ArgumentParser(description='EpiBench Data Processing CLI')
    setup_process_data_parser(parser)
    args = parser.parse_args()
    # Basic logger setup if run directly (can be overridden by config)
    # setup_logger()
    process_data_main(args) 