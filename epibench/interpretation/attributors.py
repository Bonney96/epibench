import torch
import logging
from captum.attr import IntegratedGradients
from typing import Optional, Union
import numpy as np
import h5py
from pathlib import Path

logger = logging.getLogger(__name__)

def calculate_integrated_gradients(model: torch.nn.Module, 
                                   inputs: torch.Tensor, 
                                   baseline: torch.Tensor, 
                                   target_index: Optional[int] = None, 
                                   n_steps: int = 50) -> torch.Tensor:
    """Calculates feature attributions using the Integrated Gradients method.

    Args:
        model: The PyTorch model to interpret.
        inputs: The input tensor for which to calculate attributions.
        baseline: The baseline tensor to compare against.
        target_index: The index of the target output neuron (for classification/multi-output).
                      Set to None for single-output regression models.
        n_steps: The number of steps for the path integral approximation.

    Returns:
        A tensor containing the attribution scores for each input feature.
    """
    ig = IntegratedGradients(model)
    logger.debug(f"Calculating Integrated Gradients with n_steps={n_steps}, target={target_index}")
    try:
        attributions = ig.attribute(
            inputs=inputs,
            baselines=baseline,
            target=target_index, 
            n_steps=n_steps,
            return_convergence_delta=False # Set to True if delta is needed
        )
        logger.debug("Integrated Gradients calculation successful.")
        return attributions
    except Exception as e:
        logger.error(f"Error during Integrated Gradients calculation: {e}", exc_info=True)
        raise # Re-raise the exception

def load_custom_baseline(file_path: Union[str, Path], target_shape: tuple) -> torch.Tensor:
    """Loads custom baseline data from a .npy or .h5 file.
    
    Currently supports loading a single baseline tensor. 
    Needs expansion if multiple baselines or specific selection logic is required.
    """
    file_path = Path(file_path)
    logger.info(f"Loading custom baseline from: {file_path}")
    try:
        if file_path.suffix == '.npy':
            baseline_data = np.load(file_path)
        elif file_path.suffix == '.h5':
            with h5py.File(file_path, 'r') as f:
                # Try common dataset names, or require a specific name via config?
                if 'baseline' in f:
                    baseline_data = f['baseline'][:]
                elif 'features' in f: # Maybe baseline uses same format as input?
                     logger.warning("Loading baseline from 'features' dataset in HDF5 file.")
                     # Potentially load only the first sample or an average?
                     # For now, loading the first sample as baseline if it matches shape.
                     baseline_data = f['features'][0]
                else:
                    raise ValueError("HDF5 baseline file must contain a 'baseline' or 'features' dataset.")
        else:
            raise ValueError(f"Unsupported baseline file format: {file_path.suffix}. Use .npy or .h5")

        baseline_tensor = torch.from_numpy(baseline_data.astype(np.float32))
        
        # Basic shape validation
        if baseline_tensor.shape != target_shape:
             # Allow broadcasting if baseline is per-channel or simpler? For now, strict shape match.
             logger.warning(f"Custom baseline shape {baseline_tensor.shape} does not match target shape {target_shape}. Attempting to use first sample if possible.")
             # Example: If baseline file contains multiple samples, try taking the first one?
             # This logic needs refinement based on expected baseline format.
             if baseline_tensor.ndim > len(target_shape) and baseline_tensor.shape[1:] == target_shape[1:]:
                 baseline_tensor = baseline_tensor[0]
                 if baseline_tensor.shape != target_shape:
                      raise ValueError(f"Custom baseline shape mismatch: Expected {target_shape}, got {baseline_tensor.shape} after trying first sample.")
             else:
                  raise ValueError(f"Custom baseline shape mismatch: Expected {target_shape}, got {baseline_tensor.shape}.")

        return baseline_tensor
        
    except FileNotFoundError:
        logger.error(f"Custom baseline file not found: {file_path}")
        raise
    except Exception as e:
        logger.error(f"Error loading custom baseline from {file_path}: {e}", exc_info=True)
        raise

def generate_baseline(baseline_type: str, 
                      input_batch: torch.Tensor, 
                      custom_path: Optional[Union[str, Path]] = None,
                      seed: int = 42) -> torch.Tensor:
    """Generates a baseline tensor based on the specified type.

    Args:
        baseline_type: Type of baseline ('zero', 'random', 'custom').
        input_batch: An example batch of inputs to match shape and device.
        custom_path: Path to file if baseline_type is 'custom'.
        seed: Random seed for 'random' baseline.

    Returns:
        A baseline tensor.
    """
    logger.info(f"Generating baseline of type: {baseline_type}")
    if baseline_type == 'zero':
        return torch.zeros_like(input_batch)
    elif baseline_type == 'random':
        # Simple Gaussian noise baseline
        torch.manual_seed(seed) # Ensure reproducibility
        # Consider scaling noise based on input std deviation if needed
        return torch.randn_like(input_batch)
    elif baseline_type == 'custom':
        if custom_path is None:
            raise ValueError("`custom_baseline_path` must be provided for baseline_type 'custom'.")
        # Load the baseline - expects shape to match a single input sample
        # We assume the custom baseline represents ONE sample, and broadcast it to the batch size
        single_baseline = load_custom_baseline(custom_path, input_batch.shape[1:]) # Target shape excludes batch dim
        # Expand baseline to match batch size
        # Use expand not repeat to save memory if baseline is simple (like all zeros)
        # Assumes baseline should be the same for all items in the batch
        return single_baseline.unsqueeze(0).expand_as(input_batch).to(input_batch.device)
    else:
        raise ValueError(f"Unsupported baseline_type: '{baseline_type}'. Choose 'zero', 'random', or 'custom'.") 