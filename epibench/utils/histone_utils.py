import pyBigWig
import numpy as np
import logging
from typing import List, Dict, Union, Optional
from pathlib import Path

logger = logging.getLogger(__name__)

def get_histone_data(chrom: str, 
                     start: int, 
                     end: int, 
                     histone_names: List[str],
                     bigwig_paths: List[Union[str, Path]],
                     target_length: Optional[int] = None) -> np.ndarray:
    """Fetches histone modification data from multiple BigWig files for a given region.

    Args:
        chrom: Chromosome name.
        start: Start coordinate (0-based).
        end: End coordinate (0-based, exclusive).
        histone_names: List of histone mark names corresponding to the bigwig_paths.
        bigwig_paths: List of paths to the BigWig files.
        target_length: If specified, interpolates/averages data to this length.
                       Useful if BigWig resolution differs from model input length.

    Returns:
        A NumPy array of shape (num_histones, length) containing histone signals,
        or (num_histones, target_length) if target_length is specified.
        Returns an array of zeros if no data is found or errors occur for a track.
    """
    if len(histone_names) != len(bigwig_paths):
        raise ValueError("Length of histone_names must match length of bigwig_paths.")

    num_histones = len(histone_names)
    region_len = end - start
    output_len = target_length if target_length is not None else region_len
    
    all_histone_data = np.zeros((num_histones, output_len), dtype=np.float32)

    # Prepare chromosome variants (e.g., 'chr1' vs '1')
    chrom_variants = [chrom]
    if chrom.startswith('chr'):
        chrom_variants.append(chrom[3:])
    else:
        chrom_variants.append(f'chr{chrom}')

    for i, (name, bw_path) in enumerate(zip(histone_names, bigwig_paths)):
        bw_path = Path(bw_path)
        track_data = None
        if not bw_path.exists():
            logger.warning(f"BigWig file not found for {name}: {bw_path}. Skipping track.")
            continue # Keep zeros for this track

        try:
            with pyBigWig.open(str(bw_path)) as bw:
                valid_chrom = None
                bw_chroms = bw.chroms()
                for variant in chrom_variants:
                    if variant in bw_chroms:
                        valid_chrom = variant
                        break
                
                if valid_chrom is None:
                    logger.warning(f"Chromosome '{chrom}' (or variants) not found in BigWig file for {name}: {bw_path}. Available: {list(bw_chroms.keys())[:5]}... Skipping track.")
                    continue

                # Ensure query stays within chromosome bounds
                chrom_len = bw_chroms[valid_chrom]
                query_start = max(0, start)
                query_end = min(chrom_len, end)

                if query_start >= query_end:
                    logger.warning(f"Query region {valid_chrom}:{query_start}-{query_end} is invalid or outside chromosome bounds ({chrom_len}) for {name}. Skipping track.")
                    continue
                    
                # Fetch values
                # pyBigWig returns NaN for regions with no data
                track_data = bw.values(valid_chrom, query_start, query_end, numpy=True)
                # Replace NaN with 0
                track_data = np.nan_to_num(track_data)

                # Handle cases where query region was clipped due to chromosome boundaries
                # Pad with zeros if necessary to match the original desired region_len
                actual_len = len(track_data)
                if actual_len < region_len:
                    padded_data = np.zeros(region_len, dtype=np.float32)
                    pad_start = query_start - start # Amount clipped from the beginning
                    if pad_start < 0: pad_start = 0 # Should not happen with max(0, start)
                    pad_end = pad_start + actual_len
                    if pad_end > region_len: pad_end = region_len # Should not happen? 
                    
                    padded_data[pad_start:pad_end] = track_data
                    track_data = padded_data
                    logger.debug(f"Padded {name} data from {actual_len} to {region_len} due to boundary clipping.")
                elif actual_len > region_len:
                     logger.warning(f"Fetched data length ({actual_len}) > region length ({region_len}) for {name}. Truncating.")
                     track_data = track_data[:region_len]

        except Exception as e:
            logger.error(f"Error reading BigWig file for {name} ({bw_path}) at {chrom}:{start}-{end}: {e}", exc_info=True)
            continue # Keep zeros for this track

        if track_data is not None:
             # Resize/interpolate if target_length is specified
            if target_length is not None and region_len != target_length:
                try:
                    from scipy.ndimage import zoom
                    # Use zoom for interpolation/downsampling. Order=1 is linear interpolation.
                    zoom_factor = target_length / region_len
                    resized_data = zoom(track_data, zoom_factor, order=1)
                    # Ensure correct length after potential floating point issues in zoom
                    if len(resized_data) != target_length:
                        # Fallback: simple averaging or sampling if zoom fails? Or raise error?
                        # For now, adjust length manually (crude)
                        if len(resized_data) > target_length:
                            resized_data = resized_data[:target_length]
                        else:
                            padded = np.zeros(target_length, dtype=np.float32)
                            padded[:len(resized_data)] = resized_data
                            resized_data = padded
                            
                    all_histone_data[i, :] = resized_data.astype(np.float32)
                    logger.debug(f"Resized {name} data from {region_len} to {target_length}.")
                except ImportError:
                     logger.error("scipy is required for resizing histone data to target_length. Please install scipy.")
                     # Fallback: return original resolution data? Or error?
                     # Returning original resolution for now if target_length is different
                     if region_len == output_len: # Check if target_len happened to match region_len
                         all_histone_data[i, :] = track_data.astype(np.float32)
                     else:
                         # Cannot provide target_length without scipy, return zeros for this track
                         logger.warning(f"Cannot resize {name} data to {target_length} without scipy. Returning zeros.")
                         pass # Keep zeros
                except Exception as e:
                    logger.error(f"Error resizing {name} data from {region_len} to {target_length}: {e}", exc_info=True)
                    # Keep zeros for this track on error
            elif track_data.shape[0] == output_len: # Check if length already matches output length
                all_histone_data[i, :] = track_data.astype(np.float32)
            else:
                 logger.error(f"Unexpected length mismatch for {name}. Expected {output_len}, got {track_data.shape[0]}. Returning zeros.")
                 # Keep zeros

    return all_histone_data 