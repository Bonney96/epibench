import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import List, Optional
import logging

logger = logging.getLogger(__name__)

class SeqCNNRegressor(nn.Module):
    """Multi-branch CNN model for methylation prediction from sequence and histone marks.

    Processes input data (Sequence Length, Features) through parallel CNN branches
    with different kernel sizes, concatenates the results, and passes them through
    fully connected layers for regression.
    """
    def __init__(self,
                 input_channels: int = 11, # e.g., 4 for sequence + 7 for histone marks
                 num_filters: int = 64,
                 kernel_sizes: List[int] = [3, 9, 25, 51],
                 fc_units: List[int] = [256, 128],
                 dropout_rate: float = 0.5,
                 use_batch_norm: bool = True,
                 activation: str = 'relu'):
        """Initialize the SeqCNNRegressor model.

        Args:
            input_channels: Number of input features (e.g., 11 for one-hot DNA + histone marks).
            num_filters: Number of filters for each convolutional layer in each branch.
            kernel_sizes: List of kernel sizes for the parallel CNN branches.
            fc_units: List of integers specifying the number of units in each fully connected layer.
            dropout_rate: Dropout rate for the fully connected layers.
            use_batch_norm: Whether to use Batch Normalization after conv layers.
            activation: Activation function to use ('relu' or 'gelu').
        """
        super().__init__()

        if activation.lower() == 'relu':
            self.activation_fn = nn.ReLU()
        elif activation.lower() == 'gelu':
            self.activation_fn = nn.GELU()
        else:
            raise ValueError(f"Unsupported activation function: {activation}. Choose 'relu' or 'gelu'.")
            
        self.kernel_sizes = kernel_sizes
        self.num_branches = len(kernel_sizes)
        self.input_channels = input_channels
        self.num_filters = num_filters
        self.fc_units = fc_units
        self.dropout_rate = dropout_rate
        self.use_batch_norm = use_batch_norm

        # --- Layers --- 
        self.conv_branches = nn.ModuleList() 
        self._build_conv_branches() # Build the conv branches
        
        # Placeholder - Need to calculate the flattened size after conv branches
        # This will be done *after* building branches, potentially requiring a dummy forward pass
        self._conv_output_size = self._calculate_conv_output_size() 
        
        self.fc_layers = nn.Sequential() # Will hold the final fully connected layers
        # self._build_fc_layers() # Build FC layers using _conv_output_size

    def _build_conv_branches(self):
        """Creates the parallel convolutional branches."""
        for kernel_size in self.kernel_sizes:
            # Each branch: Conv1d -> BN (opt) -> Activation -> MaxPool1d
            branch_layers = [] 
            branch_layers.append(nn.Conv1d(
                in_channels=self.input_channels,
                out_channels=self.num_filters,
                kernel_size=kernel_size,
                padding=(kernel_size - 1) // 2 # 'same' padding approximation
            ))
            
            if self.use_batch_norm:
                branch_layers.append(nn.BatchNorm1d(self.num_filters))
            
            branch_layers.append(self.activation_fn)
            
            # Add Max Pooling - kernel size can be adjusted, e.g., 2 or 3
            # The output length after pooling affects the final flattened size.
            # Using a fixed pool size for simplicity here.
            branch_layers.append(nn.MaxPool1d(kernel_size=3, stride=3))
            
            self.conv_branches.append(nn.Sequential(*branch_layers))

    def _calculate_conv_output_size(self) -> int:
        """Calculates the flattened output size after all conv branches.
           Requires a dummy forward pass through the branches.
        """
        # Use a dummy input matching expected dimensions (batch_size=1, seq_len=10000)
        # We need to determine the output size *per branch* after conv and pool
        # Note: Using seq_len=10000 might be memory-intensive for this calculation.
        # Consider using a smaller representative seq_len if needed, but 10000 is accurate.
        dummy_input_seq_len = 10000 
        dummy_input = torch.randn(1, self.input_channels, dummy_input_seq_len)
        
        total_output_features = 0
        for branch in self.conv_branches:
            with torch.no_grad(): # No need to track gradients
                branch_output = branch(dummy_input)
            total_output_features += branch_output.numel() # Flattened size: features * length
            
        if total_output_features == 0:
             raise RuntimeError("Convolutional branches produced zero output features. Check layers/pooling.")
             
        logger.info(f"Calculated total flattened features after conv branches: {total_output_features}")
        return total_output_features

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Defines the forward pass of the model.

        Args:
            x: Input tensor of shape (batch_size, sequence_length, input_channels).
               Expected sequence_length=10000, input_channels=11.

        Returns:
            Output tensor (regression prediction), shape (batch_size, 1).
        """
        # Placeholder forward pass
        # 1. Permute input: (N, SeqLen, Channels) -> (N, Channels, SeqLen)
        # 2. Pass through each conv branch
        # 3. Flatten/Pool results from branches
        # 4. Concatenate results
        # 5. Pass through FC layers
        
        # This will be implemented in Subtask 5.4
        
        # Example: Return dummy output of correct shape for now
        batch_size = x.shape[0]
        return torch.randn(batch_size, 1, device=x.device) 

    # --- Placeholder methods for building layers (to be implemented) ---
    # def _build_fc_layers(self):
    #     pass

    def init_weights(self):
        """Initializes weights for convolutional and linear layers using Xavier uniform.
        """
        logger.info("Initializing model weights using Xavier uniform initialization.")
        for m in self.modules():
            if isinstance(m, (nn.Conv1d, nn.Linear)):
                nn.init.xavier_uniform_(m.weight)
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
            # Optional: Initialize BatchNorm layers if needed
            # elif isinstance(m, nn.BatchNorm1d):
            #     nn.init.constant_(m.weight, 1)
            #     nn.init.constant_(m.bias, 0)