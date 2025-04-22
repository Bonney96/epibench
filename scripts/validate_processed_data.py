import h5py
import numpy as np
import argparse
import sys
import os
import logging

# Basic logger setup
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

def validate_hdf5_file(file_path: str, expected_channels: int = 11, num_samples_to_check: int = 5):
    """Validates the structure and content of a processed HDF5 data file.

    Args:
        file_path (str): Path to the HDF5 file (e.g., train.h5).
        expected_channels (int): The expected number of feature channels.
        num_samples_to_check (int): Number of random samples to check in detail.
    """
    logger.info(f"Validating HDF5 file: {file_path}")

    if not os.path.exists(file_path):
        logger.error(f"File not found: {file_path}")
        return False

    try:
        with h5py.File(file_path, 'r') as f:
            # Check required datasets exist
            required_datasets = ['features', 'targets', 'chrom', 'start', 'end']
            for ds_name in required_datasets:
                if ds_name not in f:
                    logger.error(f"Dataset '{ds_name}' missing in {file_path}.")
                    return False
                logger.info(f"Dataset '{ds_name}' found. Shape: {f[ds_name].shape}")

            features_shape = f['features'].shape
            targets_shape = f['targets'].shape
            num_regions = features_shape[0]

            if num_regions == 0:
                logger.warning(f"File {file_path} contains 0 regions. Skipping detailed checks.")
                return True # Technically valid, just empty

            # Check shapes
            if len(features_shape) != 3:
                logger.error(f"Features dataset has incorrect rank: {len(features_shape)} (expected 3). Shape: {features_shape}")
                return False
            
            actual_channels = features_shape[2]
            if actual_channels != expected_channels:
                 logger.error(f"Features dataset has incorrect number of channels: {actual_channels} (expected {expected_channels}). Shape: {features_shape}")
                 return False
            logger.info(f"Feature shape check passed: {features_shape}")

            if targets_shape[0] != num_regions or len(targets_shape) > 2 or (len(targets_shape) == 2 and targets_shape[1] != 1):
                 logger.error(f"Targets dataset shape mismatch. Expected ({num_regions},) or ({num_regions}, 1), got {targets_shape}")
                 return False
            logger.info(f"Target shape check passed: {targets_shape}")
            
            # Check coordinate shapes
            if f['chrom'].shape[0] != num_regions or f['start'].shape[0] != num_regions or f['end'].shape[0] != num_regions:
                 logger.error(f"Coordinate dataset length mismatch with features/targets ({num_regions} regions).")
                 return False
            logger.info("Coordinate shapes check passed.")


            # Check boundary channel (last channel, index -1)
            logger.info(f"Checking boundary channel (index {expected_channels - 1}) for {num_samples_to_check} random samples...")
            boundary_channel_index = expected_channels - 1
            
            indices_to_check = np.random.choice(num_regions, size=min(num_regions, num_samples_to_check), replace=False)
            
            for i in indices_to_check:
                sample_features = f['features'][i]
                boundary_channel = sample_features[:, boundary_channel_index]
                
                # Check if values are only 0 or 1
                unique_values = np.unique(boundary_channel)
                is_binary = np.all(np.isin(unique_values, [0, 1]))
                if not is_binary:
                     logger.error(f"Sample {i}: Boundary channel contains non-binary values: {unique_values}")
                     return False

                # Optional: Check alignment with coordinates
                # This is a rough check, assumes windowing logic is consistent
                try:
                    chrom = f['chrom'][i].decode('utf-8')
                    start = f['start'][i]
                    end = f['end'][i]
                    seq_len = features_shape[1] # e.g., 10000
                    
                    # Find where boundary channel is 1
                    boundary_indices = np.where(boundary_channel == 1)[0]
                    if len(boundary_indices) > 0:
                        boundary_start = boundary_indices[0]
                        boundary_end = boundary_indices[-1] + 1 # Make exclusive end
                        
                        # Very rough check: the length of the 1s region should match BED region length
                        bed_region_len = end - start
                        boundary_len = boundary_end - boundary_start
                        
                        # Allow some tolerance due to centering/padding/boundary effects
                        if not (0 <= boundary_start < seq_len and 0 < boundary_end <= seq_len):
                             logger.warning(f"Sample {i} ({chrom}:{start}-{end}): Boundary indices [{boundary_start}-{boundary_end}] seem out of window bounds [0-{seq_len}].")
                        
                        # Check length similarity (allowing for off-by-one/small differences)
                        if abs(boundary_len - bed_region_len) > 5: 
                             logger.warning(f"Sample {i} ({chrom}:{start}-{end}): Boundary channel length ({boundary_len}) doesn't closely match BED region length ({bed_region_len}).")
                    elif (end - start) > 0: # If BED region has length but boundary is all zeros
                         logger.warning(f"Sample {i} ({chrom}:{start}-{end}): BED region has length {end-start} but boundary channel is all zeros.")

                except Exception as e:
                    logger.warning(f"Sample {i}: Error during coordinate alignment check: {e}")

            logger.info("Boundary channel basic checks passed.")

    except Exception as e:
        logger.error(f"Error opening or reading HDF5 file {file_path}: {e}", exc_info=True)
        return False

    logger.info(f"Validation successful for {file_path}")
    return True

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Validate processed EpiBench HDF5 data files.")
    parser.add_argument("h5_files", nargs='+', help="Path(s) to the HDF5 file(s) to validate (e.g., train.h5 validation.h5 test.h5)")
    parser.add_argument("--expected-channels", type=int, default=11, help="Expected number of channels in the features dataset.")
    parser.add_argument("--num-samples", type=int, default=5, help="Number of random samples to check in detail.")

    args = parser.parse_args()

    all_valid = True
    for file_path in args.h5_files:
        if not validate_hdf5_file(file_path, args.expected_channels, args.num_samples):
            all_valid = False

    if all_valid:
        logger.info("All specified files passed validation.")
        sys.exit(0)
    else:
        logger.error("One or more files failed validation.")
        sys.exit(1) 