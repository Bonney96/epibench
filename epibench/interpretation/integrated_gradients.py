# -*- coding: utf-8 -*-
"""Model interpretation using Integrated Gradients (Captum)."""

import logging
from typing import Any, Optional, Tuple, Union, List

import torch
import torch.nn as nn
from captum.attr import IntegratedGradients, LayerIntegratedGradients
from captum.attr._utils.visualization import visualize_image_attr # For potential visualization
import matplotlib.pyplot as plt
import numpy as np # Add numpy import

# Assuming model and data loader structure from other modules
# from ..models.base_model import BaseModel # Example import if needed
# from torch.utils.data import DataLoader

logger = logging.getLogger(__name__)

class ModelInterpreter:
    """Interprets model predictions using Integrated Gradients.

    Uses the Captum library to calculate feature attributions.
    """

    def __init__(self, model: nn.Module, device: torch.device):
        """Initializes the ModelInterpreter.

        Args:
            model: The trained PyTorch model (nn.Module) to interpret.
                 Must be in evaluation mode (model.eval()).
            device: The torch.device (e.g., 'cuda' or 'cpu') the model is on.
        """
        if not isinstance(model, nn.Module):
            raise TypeError("Model must be an instance of torch.nn.Module.")
        # Ensure model is on the correct device and in eval mode before interpretation
        self.model = model.to(device)
        self.model.eval() # Ensure model is in evaluation mode
        self.device = device
        # Initialize IntegratedGradients. Choose LayerIntegratedGradients if needed.
        self.integrated_gradients = IntegratedGradients(self.model) 
        logger.info(f"ModelInterpreter initialized for model {type(model).__name__} on {device}.")

    def calculate_attributions(self, 
                               inputs: torch.Tensor, 
                               target: Optional[int] = None, 
                               baselines: Optional[Union[torch.Tensor, Tuple[torch.Tensor, ...]]] = None,
                               n_steps: int = 50,
                               **kwargs) -> Union[torch.Tensor, Tuple[torch.Tensor, ...], None]:
        """Calculates attribution scores for given inputs using Integrated Gradients.

        Args:
            inputs: Input tensor(s) for which to calculate attributions. 
                    Should be on the same device as the model.
            target: Output index for which gradients are computed (for multi-output models).
                    If None, the gradient with respect to the maximum output is computed.
            baselines: Baseline input(s) used in integration. If None, defaults to zero tensor(s)
                       matching input shape. Can be a single tensor or a tuple if the model
                       accepts multiple inputs.
            n_steps: Number of steps for the approximation of the integral.
            **kwargs: Additional arguments passed to the captum `attribute` method.

        Returns:
            Attribution scores as a tensor or tuple of tensors (matching input structure),
            or None if an error occurs.
        """
        logger.debug(f"Calculating attributions for input shape: {inputs.shape if isinstance(inputs, torch.Tensor) else [t.shape for t in inputs]}")
        
        # Ensure inputs are on the correct device
        if isinstance(inputs, torch.Tensor):
            if inputs.device != self.device:
                inputs = inputs.to(self.device)
                logger.warning(f"Input tensor moved to device {self.device}.")
        elif isinstance(inputs, (tuple, list)):
             inputs = tuple(t.to(self.device) if t.device != self.device else t for t in inputs)
             logger.warning(f"Input tensors moved to device {self.device}.")
        else:
            logger.error(f"Unsupported input type: {type(inputs)}. Expected Tensor or tuple/list of Tensors.")
            return None

        # Default baseline to zeros if not provided
        if baselines is None:
            if isinstance(inputs, torch.Tensor):
                baselines = torch.zeros_like(inputs)
            elif isinstance(inputs, tuple):
                baselines = tuple(torch.zeros_like(inp) for inp in inputs)
            logger.debug("Using zero tensor(s) as baseline.")
        elif isinstance(baselines, torch.Tensor):
             if baselines.device != self.device:
                 baselines = baselines.to(self.device)
        elif isinstance(baselines, tuple):
             baselines = tuple(b.to(self.device) if b.device != self.device else b for b in baselines)

        try:
            attributions = self.integrated_gradients.attribute(
                inputs=inputs,
                baselines=baselines,
                target=target,
                n_steps=n_steps,
                **kwargs
            )
            logger.info(f"Attribution calculation successful.")
            return attributions
        except Exception as e:
            logger.error(f"Error calculating attributions: {e}", exc_info=True)
            return None

    # --- Visualization Methods (Subtask 12.3) ---
    def visualize_attributions(self, 
                               attributions: torch.Tensor,
                               inputs: torch.Tensor, 
                               method: str = "heat_map", 
                               sign: str = "absolute_value", 
                               outlier_perc: float = 2,
                               cmap: str = "viridis",
                               title: str = "Feature Attributions",
                               fig_size: Tuple[int, int] = (6, 6),
                               **kwargs):
        """Visualizes attribution scores using Captum's visualization utilities.

        Note: `visualize_image_attr` is primarily designed for image data (2D or 3D).
        Adaptation might be needed for 1D sequence data (e.g., reshaping,
        or using matplotlib directly for line plots).

        Args:
            attributions: Attribution scores tensor (usually requires grad).
            inputs: Input tensor used for attribution calculation.
            method: Visualization method (e.g., 'heat_map', 'original_image').
            sign: How to handle attribution signs ('positive', 'negative', 
                  'absolute_value', 'all').
            outlier_perc: Top/bottom percentage of attributions to clip.
            cmap: Matplotlib colormap name.
            title: Title for the plot.
            fig_size: Figure size tuple.
            **kwargs: Additional arguments passed to `visualize_image_attr`.

        Returns:
            A tuple containing the matplotlib figure and axes objects, or None if
            visualization fails.
        """
        logger.info(f"Attempting to visualize attributions using method: {method}, sign: {sign}")
        
        # Basic input validation
        if not isinstance(attributions, torch.Tensor) or not isinstance(inputs, torch.Tensor):
            logger.error("Attributions and inputs must be torch.Tensors.")
            return None
        if attributions.shape != inputs.shape:
             # Captum visualization might handle broadcasting in some cases, but let's warn
             logger.warning(f"Attribution shape {attributions.shape} differs from input shape {inputs.shape}. Visualization might be unexpected.")

        # Prepare data for visualization (move to CPU, detach, convert to numpy)
        # IMPORTANT: Adjust transpose/squeeze based on expected input shape for visualize_image_attr
        #            This assumes a shape like (batch, channels, ...) where we take the first batch item
        #            and potentially need to reshape/transpose for image-like format.
        try:
            attr_np = attributions.squeeze(0).cpu().detach().numpy()
            inp_np = inputs.squeeze(0).cpu().detach().numpy()
            
            # Example: If input is (1, channels, length), transpose might be needed
            # if len(attr_np.shape) == 2: # Assuming (channels, length)
            #     attr_np = np.transpose(attr_np, (1, 0)) # -> (length, channels)
            #     inp_np = np.transpose(inp_np, (1, 0))
            # Further reshaping might be needed if visualize_image_attr expects HxWxC
            
            logger.debug(f"Visualizing with attr shape: {attr_np.shape}, input shape: {inp_np.shape}")

        except Exception as e:
            logger.error(f"Error preparing data for visualization: {e}", exc_info=True)
            return None

        try:
            fig, axes = visualize_image_attr(
                attr=attr_np,
                original_image=inp_np,
                method=method,
                sign=sign,
                outlier_perc=outlier_perc,
                cmap=cmap,
                plt_fig_axis=(plt.figure(figsize=fig_size), plt.gca()),
                use_pyplot=False, # Recommended to manage figure/axes manually
                **kwargs
            )
            # Assume visualize_image_attr returns (fig, axis) or similar based on use_pyplot=False
            if fig and axes:
                 fig.suptitle(title)
                 # fig.tight_layout() # May interfere with suptitle
                 logger.info("Attribution visualization generated successfully.")
                 return fig, axes
            else:
                 logger.error("Captum visualization function did not return figure/axes.")
                 return None

        except Exception as e:
            logger.error(f"Error during Captum visualization: {e}", exc_info=True)
            # Clean up figure if created but error occurred
            try:
                 plt.close(fig) 
            except:
                 pass
            return None

    # --- High-Attribution Region Extraction (Subtask 12.4) ---
    def extract_high_attribution_regions(self, 
                                         attributions: torch.Tensor, 
                                         threshold: Optional[float] = None,
                                         top_k: Optional[int] = None,
                                         abs_val: bool = True) -> Union[np.ndarray, None]:
        """Extracts indices of features with high attribution scores.

        Allows extraction based on an absolute threshold or the top-k scores.

        Args:
            attributions: Attribution scores tensor.
            threshold: Optional threshold value. If provided, indices where the 
                       absolute attribution score exceeds this threshold are returned.
            top_k: Optional integer. If provided, indices of the top-k features with 
                   the highest absolute attribution scores are returned.
                   Overrides threshold if both are provided.
            abs_val: Whether to use the absolute value of attributions for thresholding
                     and ranking (default: True).

        Returns:
            A NumPy array containing the indices of high-attribution features, 
            or None if an error occurs.
        """
        if not isinstance(attributions, torch.Tensor):
            logger.error("Attributions must be a torch.Tensor.")
            return None
        if threshold is None and top_k is None:
            logger.error("Either threshold or top_k must be provided.")
            return None
        if threshold is not None and top_k is not None:
             logger.warning("Both threshold and top_k provided. Using top_k.")
             threshold = None # Prioritize top_k
        if top_k is not None and (not isinstance(top_k, int) or top_k <= 0):
            logger.error("top_k must be a positive integer.")
            return None
            
        try:
            attr_np = attributions.detach().cpu().numpy()
            
            # Use absolute value if requested
            scores = np.abs(attr_np) if abs_val else attr_np
            
            if top_k is not None:
                 # Flatten the array to find top_k across all dimensions, then get indices
                 flat_scores = scores.flatten()
                 # Get indices that would sort the flattened array in descending order
                 sorted_indices_flat = np.argsort(flat_scores)[::-1]
                 # Take the top_k indices
                 top_k_indices_flat = sorted_indices_flat[:top_k]
                 # Convert flat indices back to multi-dimensional indices
                 high_attr_indices = np.array(np.unravel_index(top_k_indices_flat, scores.shape)).T
                 logger.info(f"Extracted top {top_k} attribution indices.")
                 return high_attr_indices
                 
            elif threshold is not None:
                 # Find indices where score exceeds the threshold
                 high_attr_indices = np.argwhere(scores > threshold)
                 logger.info(f"Extracted {len(high_attr_indices)} indices exceeding threshold {threshold}.")
                 return high_attr_indices
                 
            else:
                # Should not happen due to initial checks, but safeguard
                return None

        except Exception as e:
            logger.error(f"Error extracting high attribution regions: {e}", exc_info=True)
            return None

# Example usage (for testing within the module)
if __name__ == '__main__':
    # This is placeholder example code - requires a dummy model and data
    logger.info("Running ModelInterpreter example...")
    
    # Create a dummy model (e.g., simple linear layer)
    class DummyModel(nn.Module):
        def __init__(self):
            super().__init__()
            # Example: expecting input shape like (batch, features) e.g., (4, 10)
            self.linear = nn.Linear(10, 1) 
        
        def forward(self, x):
            return self.linear(x)

    dummy_model = DummyModel()
    dummy_device = torch.device("cpu")
    dummy_model.to(dummy_device)
    dummy_model.eval()

    # Create dummy input data
    dummy_input = torch.randn(4, 10, device=dummy_device) # Batch of 4, 10 features

    # Initialize interpreter
    try:
        interpreter = ModelInterpreter(dummy_model, dummy_device)

        # Calculate attributions
        attributions = interpreter.calculate_attributions(dummy_input)

        if attributions is not None:
            logger.info(f"Calculated attributions shape: {attributions.shape}")
            # Add calls to visualize or extract regions here if they were implemented
        else:
             logger.error("Failed to calculate attributions in example.")

    except Exception as e:
        logger.error(f"Error in ModelInterpreter example: {e}", exc_info=True) 