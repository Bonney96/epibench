import h5py
import numpy as np
import pandas as pd
from pathlib import Path
import logging
import importlib.metadata
import json
from datetime import datetime
from typing import List, Dict, Union, Optional, Any

# Assuming InterpretConfig, FeatureExtractionParams, etc. are importable
# from ..validation.config_validator import InterpretConfig, FeatureExtractionParams
# For standalone use, define dummy classes or expect dicts:
from pydantic import BaseModel, Field # Use BaseModel for config typing hints

# Plotting libraries
import matplotlib
matplotlib.use('Agg') # Use Agg backend for non-interactive plotting
import matplotlib.pyplot as plt
import seaborn as sns

# Import BigWig utility
from ..utils.histone_utils import get_histone_data

logger = logging.getLogger(__name__)

# --- Dummy Pydantic models if config_validator cannot be imported directly ---
# Replace with actual imports if possible
class DummyFeatureExtractionParams(BaseModel):
    use_absolute_value: bool = True
    top_k: Optional[int] = None
    threshold: Optional[float] = None

class DummyInterpretConfig(BaseModel):
    interpretation: Any = None
    feature_extraction: DummyFeatureExtractionParams = Field(default_factory=DummyFeatureExtractionParams)
    training_config: Union[str, Path] = "path/to/dummy/train_config.yaml"

class DummyCliArgs(BaseModel):
    checkpoint: str = "dummy_checkpoint.pth"
    input_data: str = "dummy_input.h5"

# --- End Dummy Models ---


def save_interpretation_results(output_dir: Union[str, Path], 
                                filename_prefix: str, 
                                attributions: np.ndarray, 
                                coordinates: List[Dict[str, Any]], 
                                interpret_config: Any, # Use actual InterpretConfig if possible
                                cli_args: Any): # Use actual argparse.Namespace if possible
    """Saves interpretation results (attributions, coordinates, metadata) to an HDF5 file.

    Args:
        output_dir: Directory to save the HDF5 file.
        filename_prefix: Prefix for the output filename.
        attributions: NumPy array of attribution scores (n_samples, seq_len, features).
        coordinates: List of dictionaries, each with 'chrom', 'start', 'end'.
        interpret_config: Validated interpretation configuration object (e.g., InterpretConfig).
        cli_args: Parsed command-line arguments (e.g., argparse.Namespace).
    """
    output_path = Path(output_dir) / f"{filename_prefix}_attributions.h5"
    logger.info(f"Saving interpretation results to: {output_path}")

    try:
        # Ensure output directory exists
        output_path.parent.mkdir(parents=True, exist_ok=True)

        # Convert coordinates to NumPy arrays
        num_samples = len(coordinates)
        # Handle potential None values in coordinates list
        # Provide default values if coordinate dict is None
        default_chrom = ""
        default_pos = -1 
        
        chroms = np.array([
            c['chrom'] if c is not None else default_chrom for c in coordinates
        ], dtype=h5py.string_dtype(encoding='utf-8'))
        starts = np.array([
            c['start'] if c is not None else default_pos for c in coordinates
        ], dtype=np.int64)
        ends = np.array([
            c['end'] if c is not None else default_pos for c in coordinates
        ], dtype=np.int64)

        with h5py.File(output_path, 'w') as f:
            # Compression settings
            compression_opts = {
                'compression': 'gzip',
                'compression_opts': 4, # Balance speed and size
                'shuffle': True # Often helps float compression
            }

            # Create datasets
            logger.debug(f"Creating dataset 'attributions' with shape {attributions.shape} and dtype {attributions.dtype}")
            f.create_dataset(
                'attributions', 
                data=attributions.astype(np.float32), # Ensure float32
                chunks=(min(64, num_samples),) + attributions.shape[1:], # Chunk by sample
                **compression_opts
            )

            coord_group = f.create_group('coordinates')
            logger.debug(f"Creating dataset 'coordinates/chrom' with shape {chroms.shape} and dtype {chroms.dtype}")
            coord_group.create_dataset('chrom', data=chroms, chunks=(min(1024, num_samples),), compression='gzip')
            logger.debug(f"Creating dataset 'coordinates/start' with shape {starts.shape} and dtype {starts.dtype}")
            coord_group.create_dataset('start', data=starts, chunks=(min(1024, num_samples),), compression='gzip')
            logger.debug(f"Creating dataset 'coordinates/end' with shape {ends.shape} and dtype {ends.dtype}")
            coord_group.create_dataset('end', data=ends, chunks=(min(1024, num_samples),), compression='gzip')

            # Add metadata as attributes
            meta_group = f.create_group('metadata')
            # Safely access nested attributes from potentially dummy config
            interp_params = getattr(interpret_config, 'interpretation', None)
            ig_params = getattr(interp_params, 'integrated_gradients', None) if interp_params else None

            meta_group.attrs['attribution_method'] = getattr(interp_params, 'method', 'Unknown')
            meta_group.attrs['n_steps'] = getattr(ig_params, 'n_steps', 'Unknown')
            meta_group.attrs['baseline_type'] = getattr(ig_params, 'baseline_type', 'Unknown')
            meta_group.attrs['custom_baseline_path'] = str(getattr(ig_params, 'custom_baseline_path', 'N/A'))
            meta_group.attrs['target_output_index'] = getattr(ig_params, 'target_output_index', 'Unknown')
            meta_group.attrs['model_checkpoint'] = str(getattr(cli_args, 'checkpoint', 'Unknown'))
            meta_group.attrs['training_config'] = str(getattr(interpret_config, 'training_config', 'Unknown'))
            meta_group.attrs['input_data_file'] = str(getattr(cli_args, 'input_data', 'Unknown'))
            meta_group.attrs['creation_date'] = datetime.now().isoformat()
            try:
                meta_group.attrs['epibench_version'] = importlib.metadata.version('epibench')
            except importlib.metadata.PackageNotFoundError:
                meta_group.attrs['epibench_version'] = 'unknown'
                
            # Save full config as JSON string (optional but helpful)
            # Convert Pydantic model to dict first
            config_dict = interpret_config.model_dump(mode='json') if hasattr(interpret_config, 'model_dump') else {}
            meta_group.attrs['interpret_config_json'] = json.dumps(config_dict)

        logger.info("Successfully saved interpretation results.")

    except Exception as e:
        logger.error(f"Failed to save interpretation results to {output_path}: {e}", exc_info=True)
        # Optionally re-raise or handle error
        raise

def extract_and_save_features(output_dir: Union[str, Path],
                              filename_prefix: str,
                              attributions: np.ndarray,
                              coordinates: List[Dict[str, Any]],
                              feature_extraction_config: Any): # Use actual FeatureExtractionParams if possible
    """Extracts important features based on attribution scores and saves them to a TSV file.

    Args:
        output_dir: Directory to save the TSV file.
        filename_prefix: Prefix for the output filename.
        attributions: NumPy array of attribution scores (n_samples, seq_len, features).
                      Assumes attributions are per base pair if seq_len == features dim.
                      Needs clarification if seq_len != features.
        coordinates: List of dictionaries, each with 'chrom', 'start', 'end'.
        feature_extraction_config: Configuration object with extraction parameters 
                                     (e.g., FeatureExtractionParams).
    """
    output_path = Path(output_dir) / f"{filename_prefix}_extracted_features.tsv"
    logger.info(f"Extracting important features based on config: {feature_extraction_config}")
    
    top_k = getattr(feature_extraction_config, 'top_k', None)
    threshold = getattr(feature_extraction_config, 'threshold', None)
    use_abs = getattr(feature_extraction_config, 'use_absolute_value', True)

    if top_k is None and threshold is None:
        logger.info("No top_k or threshold specified in feature_extraction config. Skipping feature extraction.")
        return
        
    all_extracted_features = []

    try:
        num_samples, seq_len, num_channels = attributions.shape
        
        # Decision: Are attributions per base (seq_len) or per channel (num_channels)?
        # Assuming attribution is per base pair for now, summing across channels if needed.
        # If attribution is per input channel (e.g., histone marks), the logic needs adjustment.
        if num_channels > 1:
             logger.warning(f"Attribution array has {num_channels} channels. Summing across channels to get per-base score.")
             # Sum absolute attributions across channels to get a single score per base
             per_base_scores = np.sum(np.abs(attributions), axis=2) # Shape: (num_samples, seq_len)
        else:
             per_base_scores = attributions.squeeze(axis=2) # Shape: (num_samples, seq_len)
             
        if use_abs:
            per_base_scores = np.abs(per_base_scores)

        for i in range(num_samples):
            sample_scores = per_base_scores[i] # Shape: (seq_len,)
            coords = coordinates[i]
            chrom = coords['chrom']
            start_coord = coords['start'] # Base coordinate for the window
            
            indices_to_save = []
            scores_to_save = []
            ranks_to_save = []

            if top_k is not None:
                # Get indices of top k scores (highest first)
                # Ensure k is not larger than sequence length
                effective_k = min(top_k, len(sample_scores))
                top_indices = np.argsort(sample_scores)[::-1][:effective_k]
                indices_to_save = top_indices
                scores_to_save = sample_scores[top_indices]
                ranks_to_save = list(range(1, effective_k + 1))
            elif threshold is not None:
                # Get indices where score meets threshold
                threshold_indices = np.where(sample_scores >= threshold)[0]
                # Sort by score descending for consistency
                sorted_indices = np.argsort(sample_scores[threshold_indices])[::-1]
                indices_to_save = threshold_indices[sorted_indices]
                scores_to_save = sample_scores[indices_to_save]
                # No rank assigned for thresholding
                ranks_to_save = [np.nan] * len(indices_to_save) 

            # Create records for this sample
            for rank, score, feature_idx in zip(ranks_to_save, scores_to_save, indices_to_save):
                genomic_pos = start_coord + feature_idx # Calculate absolute genomic position
                all_extracted_features.append({
                    'sample_index': i,
                    'chrom': chrom,
                    'genomic_position': genomic_pos,
                    'score': score,
                    'rank': rank, 
                    'feature_index_in_window': feature_idx,
                    'window_start': start_coord, # Add window coords for context
                    'window_end': coords['end']
                })

        if not all_extracted_features:
            logger.warning("No features met the extraction criteria (top_k/threshold).")
            return

        # Create DataFrame and save
        df = pd.DataFrame(all_extracted_features)
        # Define column order
        columns = ['sample_index', 'chrom', 'genomic_position', 'score', 'rank', 'feature_index_in_window', 'window_start', 'window_end']
        # Reorder columns, handling potential missing 'rank' column if only threshold was used
        if 'rank' not in df.columns: columns.remove('rank')
        df = df[columns]
        
        df.to_csv(output_path, sep='\t', index=False, float_format='%.6g')
        logger.info(f"Saved {len(df)} extracted features to: {output_path}")

    except Exception as e:
        logger.error(f"Failed to extract and save features: {e}", exc_info=True)
        # Don't halt execution if feature extraction fails, just log error

# --- Placeholder for Plotting Function --- #
def generate_and_save_plots(output_dir: Union[str, Path],
                            filename_prefix: str,
                            attributions: np.ndarray,
                            predictions: np.ndarray,
                            actuals: np.ndarray,
                            coordinates: List[Dict[str, Any]],
                            config: Any,
                            secondary_predictions: Optional[np.ndarray] = None): # Use actual InterpretConfig type hint if possible
    """Generates and saves visualization plots for attributions.

    Creates a multi-panel plot showing:
        1. Attribution heatmap (DNA + histone channels).
        2. Ground truth histone coverage heatmap (fetched from BigWigs).
        3. Region indicator track.
    """
    logger.info("Starting plot generation...")
    
    vis_config = getattr(config, 'visualization', None)
    if not vis_config:
        logger.warning("Visualization configuration missing. Skipping plot generation.")
        return
        
    output_params = getattr(config, 'output', None)
    if not output_params or not output_params.generate_plots:
        logger.info("Plot generation disabled in output config. Skipping.")
        return

    # Ensure required visualization parameters are present
    if not getattr(vis_config, 'histone_bigwig_paths', None) or not getattr(vis_config, 'histone_names', None):
        logger.error("Missing histone_bigwig_paths or histone_names in visualization config. Cannot generate plots.")
        return
        
    num_samples = attributions.shape[0]
    seq_len = attributions.shape[1]
    # Determine number of samples to plot
    max_plots = getattr(vis_config, 'max_samples_to_plot', 20) if getattr(vis_config, 'max_samples_to_plot', 20) is not None else num_samples
    if max_plots == 0:
         logger.info("max_samples_to_plot set to 0. Skipping individual plot generation.")
         return
         
    samples_to_plot = min(num_samples, max_plots)
    plot_dpi = getattr(vis_config, 'plot_dpi', 150)
    
    logger.info(f"Generating plots for the first {samples_to_plot} samples.")

    # Define expected structure: 4 DNA + N Histone + 1 Boundary
    num_dna_channels = 4
    histone_channel_names_config = getattr(vis_config, 'histone_names', []) # Names from config
    num_histone_channels_config = len(histone_channel_names_config)
    expected_total_channels = num_dna_channels + num_histone_channels_config + 1 # Including boundary channel
    num_channels_actual = attributions.shape[2]
            
    dna_channel_names = ['A', 'C', 'G', 'T']
    
    # Determine the actual number of histone channels present in the data
    num_histone_channels_actual = max(0, num_channels_actual - num_dna_channels - 1)
    
    # Generate labels based on actual data structure
    if num_histone_channels_actual == num_histone_channels_config:
        histone_channel_names_actual = histone_channel_names_config
    else:
        logger.warning(f"Mismatch between configured histone names ({num_histone_channels_config}) and detected histone channels ({num_histone_channels_actual}). Using generic names.")
        histone_channel_names_actual = [f'Histone_{i+1}' for i in range(num_histone_channels_actual)]
        
    # Define labels for the attribution plot (excluding the boundary channel)
    y_labels_attr = dna_channel_names + histone_channel_names_actual
    num_channels_to_plot_attr = num_dna_channels + num_histone_channels_actual
    
    # Define labels for the ground truth plot (using names from config)
    y_labels_gt = histone_channel_names_config # Ground truth uses config names
    num_histone_channels_gt = num_histone_channels_config

    # Log channel handling
    logger.info(f"Attribution data has {num_channels_actual} channels. Plotting {num_channels_to_plot_attr} channels (DNA + Histone). Ground truth has {num_histone_channels_gt} channels.")

    # Determine a suitable shared color scale for attributions (excluding boundary channel)
    abs_max_attr = np.percentile(np.abs(attributions[:samples_to_plot, :, :num_channels_to_plot_attr]), 99.5)
    vmin_attr, vmax_attr = -abs_max_attr, abs_max_attr

    for i in range(samples_to_plot):
        logger.debug(f"Generating plot for sample index {i}")
        # Select only DNA and Histone attribution channels for plotting
        sample_attributions_to_plot = attributions[i, :, :num_channels_to_plot_attr] # Shape: (seq_len, num_dna+num_histone_actual)
        sample_coords = coordinates[i]
        pred_score = predictions[i]
        true_score = actuals[i]
        chrom = sample_coords['chrom']
        start = sample_coords['start']
        end = sample_coords['end']

        # --- Prepare Title ---
        title_parts = [
            f"Feature Importance - Sample {i}",
            f"Region: {chrom}:{start}-{end}",
            f"Actual: {true_score.item():.3f}",
            f"AML Pred: {pred_score.item():.3f}"
        ]
        title_color = 'black'
        
        # Comparative analysis if secondary predictions are available
        if secondary_predictions is not None and i < len(secondary_predictions):
            secondary_pred_score = secondary_predictions[i]
            diff = pred_score.item() - secondary_pred_score.item()
            title_parts.append(f"CD34 Pred: {secondary_pred_score.item():.3f} (Diff: {diff:+.3f})")
            
            # Check for DMR
            if abs(diff) > 0.3:
                title_color = 'red'
                title_parts[0] += " - DMR"

        final_title = " | ".join(title_parts)

        try:
            # Fetch ground truth histone data using the utility function
            ground_truth_histones = get_histone_data(
                chrom=chrom,
                start=start,
                end=end,
                histone_names=histone_channel_names_config, # Names from config
                bigwig_paths=vis_config.histone_bigwig_paths, # Paths from config
                target_length=seq_len # Resize to match attribution length
            )
            # Ensure ground truth data has the correct shape (num_histones_config, seq_len)
            if ground_truth_histones.shape != (num_histone_channels_gt, seq_len):
                 logger.warning(f"Unexpected shape for ground truth histone data for sample {i}: {ground_truth_histones.shape}. Expected: ({num_histone_channels_gt}, {seq_len}). Skipping ground truth plot.")
                 ground_truth_histones = None # Cannot plot if shape is wrong

            # Create figure with 3 subplots: attribution, ground truth, region
            # Adjust height ratios based on number of tracks
            height_ratios = [num_channels_to_plot_attr, num_histone_channels_gt, 1] # Give more space to heatmaps
            fig, axes = plt.subplots(3, 1, figsize=(15, 2 + 0.6 * num_channels_to_plot_attr + 0.6 * num_histone_channels_gt), 
                                     sharex=True, gridspec_kw={'height_ratios': height_ratios})
            
            fig.suptitle(final_title, fontsize=14, color=title_color)

            # --- 1. Attribution Heatmap --- #
            ax1 = axes[0]
            # Transpose for heatmap: (channels_to_plot, seq_len)
            sns.heatmap(sample_attributions_to_plot.T, ax=ax1, cmap="vlag", 
                        vmin=vmin_attr, vmax=vmax_attr, 
                        cbar_kws={'label': 'Attribution Score'}, 
                        yticklabels=y_labels_attr)
            ax1.set_title("Feature Importance (Integrated Gradients)")
            ax1.set_yticks(np.arange(len(y_labels_attr)) + 0.5) # Center ticks
            ax1.set_yticklabels(y_labels_attr, rotation=0)
            ax1.tick_params(axis='y', length=0) # Remove y-axis ticks
            ax1.set_ylabel("")

            # --- 2. Ground Truth Histone Coverage Heatmap --- #
            ax2 = axes[1]
            if ground_truth_histones is not None:
                # Use a sequential colormap for coverage
                sns.heatmap(ground_truth_histones, ax=ax2, cmap="YlOrRd", 
                            cbar_kws={'label': 'Ground Truth Signal'},
                            yticklabels=y_labels_gt) # Use names from config
                ax2.set_title("Ground Truth Histone Coverage")
                ax2.set_yticks(np.arange(len(y_labels_gt)) + 0.5) # Center ticks
                ax2.set_yticklabels(y_labels_gt, rotation=0)
                ax2.tick_params(axis='y', length=0) # Remove y-axis ticks
                ax2.set_ylabel("")
            else:
                 ax2.set_title("Ground Truth Histone Coverage (Data Unavailable)")
                 ax2.set_yticks([])

            # --- 3. Region Indicator --- #
            ax3 = axes[2]
            # Calculate relative start/end within the seq_len window
            # User confirmed 'start' and 'end' are the Region of Interest (ROI).
            # Assume the seq_len window (e.g., 10000bp) is centered around the ROI's midpoint.
            roi_mid_point = start + (end - start) // 2
            window_start_coord = roi_mid_point - seq_len // 2 # Estimated start coordinate of the 10kb window
            
            # Calculate ROI start/end relative to the window_start_coord
            relative_start = max(0, start - window_start_coord)
            relative_end = min(seq_len, end - window_start_coord) # Relative end is end_coord - window_start_coord

            region_indicator = np.zeros(seq_len)
            # Ensure coordinates are valid within the window
            if relative_start < relative_end and relative_start < seq_len and relative_end > 0:
                 # Make sure indices are integers for slicing
                 plot_start_idx = int(np.floor(relative_start))
                 plot_end_idx = int(np.ceil(relative_end))
                 region_indicator[max(0, plot_start_idx):min(seq_len, plot_end_idx)] = 1
            else:
                 logger.warning(f"Calculated relative region invalid or outside window for sample {i}: rel_start={relative_start}, rel_end={relative_end}. Plotting full bar.")
                 region_indicator[:] = 1 # Fallback to full bar if coords invalid

            # Plot the accurate region bar
            ax3.fill_between(np.arange(seq_len), 0, region_indicator, color='steelblue')
            ax3.set_yticks([])
            ax3.set_ylabel("Region", rotation=0, labelpad=20)
            ax3.set_xlabel("Sequence Position")
            
            # --- Customize x-axis ticks on the bottom plot (ax3) --- 
            # Determine a suitable interval based on sequence length
            if seq_len > 5000:
                 tick_interval = 2000
            elif seq_len > 1000:
                 tick_interval = 1000
            elif seq_len > 200:
                tick_interval = 200
            else:
                 tick_interval = 50 # Default for shorter sequences
                 
            # Generate tick positions
            major_ticks = np.arange(0, seq_len + 1, tick_interval) # Include seq_len if it's a multiple
            ax3.set_xticks(major_ticks)
            # Optionally set minor ticks if desired
            # minor_ticks = np.arange(0, seq_len, tick_interval // 2)
            # ax3.set_xticks(minor_ticks, minor=True)
            # ax3.grid(which='minor', axis='x', linestyle=':', alpha=0.7) # Example for minor grid
            
            # Rotate labels slightly if they still overlap (optional)
            # plt.setp(ax3.get_xticklabels(), rotation=30, ha="right")
            # --- End x-axis tick customization ---
            
            # Add vertical dashed lines to the heatmaps using calculated relative coords
            if relative_start < relative_end and relative_start < seq_len and relative_end > 0:
                 line_color = 'blue'
                 line_style = '--'
                 line_width = 1.0
                 # Use the precise relative_start and relative_end for lines
                 ax1.axvline(x=relative_start, color=line_color, linestyle=line_style, linewidth=line_width)
                 ax1.axvline(x=relative_end, color=line_color, linestyle=line_style, linewidth=line_width)
                 if ground_truth_histones is not None: # Only add to ax2 if it was plotted
                     ax2.axvline(x=relative_start, color=line_color, linestyle=line_style, linewidth=line_width)
                     ax2.axvline(x=relative_end, color=line_color, linestyle=line_style, linewidth=line_width)
            # Optional: Add dashed lines if TSS or other features are known (keep commented)
            # ax1.axvline(x=tss_pos, color='purple', linestyle=':', linewidth=1.5)
            # ax2.axvline(x=tss_pos, color='purple', linestyle=':', linewidth=1.5)
            # ax3.axvline(x=tss_pos, color='purple', linestyle=':', linewidth=1.5)

            plt.tight_layout(rect=[0, 0.03, 1, 0.95]) # Adjust layout to prevent title overlap

            # Save the figure
            output_file = Path(output_dir) / f"{filename_prefix}_viz_sample_{i}.png"
            fig.savefig(output_file, dpi=plot_dpi, bbox_inches='tight')
            plt.close(fig) # Close figure to free memory
            logger.debug(f"Saved plot for sample {i} to {output_file}")

        except Exception as e:
            logger.error(f"Failed to generate plot for sample index {i}: {e}", exc_info=True)
            plt.close(fig) # Ensure figure is closed even on error

    logger.info(f"Finished generating {samples_to_plot} plots.") 