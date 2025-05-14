import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import List, Optional
import logging

logger = logging.getLogger(__name__)

class SeqCNNRegressor(nn.Module):
    """Multi-branch CNN regressor for sequence data, adapted to match the reference architecture.

    Processes input data through parallel CNN branches with different kernel sizes, concatenates the results,
    then applies additional convolutional blocks, global average pooling, and fully connected layers for regression.
    """
    def __init__(self,
                 input_channels: int = 11,
                 num_filters: int = 64,
                 kernel_sizes: List[int] = [3, 9, 25, 51],
                 fc_units: int = 128,
                 dropout_rate: float = 0.5,
                 use_batch_norm: bool = True,
                 activation: str = 'relu'):
        """Initialize the SeqCNNRegressor model (reference-style).

        Args:
            input_channels: Number of input features (e.g., 11 for one-hot DNA + histone marks).
            num_filters: Number of filters for each convolutional layer in each branch.
            kernel_sizes: List of kernel sizes for the parallel CNN branches.
            fc_units: Number of units in the fully connected layer.
            dropout_rate: Dropout rate for the fully connected layers.
            use_batch_norm: Whether to use Batch Normalization after conv layers.
            activation: Activation function to use ('relu' or 'gelu').
        """
        super().__init__()

        if activation.lower() == 'relu':
            self.activation_fn = nn.ReLU(inplace=True)
        elif activation.lower() == 'gelu':
            self.activation_fn = nn.GELU()
        else:
            raise ValueError(f"Unsupported activation function: {activation}. Choose 'relu' or 'gelu'.")

        self.kernel_sizes = kernel_sizes
        self.use_batch_norm = use_batch_norm
        self.input_channels = input_channels
        self.num_filters = num_filters
        self.fc_units = fc_units
        self.dropout_rate = dropout_rate

        # 1. Convolutional branches
        self.branches = nn.ModuleList()
        self.branch_norms = nn.ModuleList()
        for k in self.kernel_sizes:
            conv = nn.Conv1d(
                in_channels=self.input_channels,
                out_channels=self.num_filters,
                kernel_size=k,
                padding=k // 2
            )
            self.branches.append(conv)
            if self.use_batch_norm:
                self.branch_norms.append(nn.BatchNorm1d(self.num_filters))
            else:
                self.branch_norms.append(nn.Identity())

        total_channels = self.num_filters * len(self.kernel_sizes)

        # 2. Additional convolution block #1
        self.conv2 = nn.Conv1d(in_channels=total_channels,
                               out_channels=total_channels * 2,
                               kernel_size=3, padding=1)
        self.bn2 = nn.BatchNorm1d(total_channels * 2) if self.use_batch_norm else nn.Identity()
        self.pool = nn.MaxPool1d(2)

        # 3. Additional convolution block #2
        self.conv3 = nn.Conv1d(in_channels=total_channels * 2,
                               out_channels=total_channels * 4,
                               kernel_size=7, padding=3)
        self.bn3 = nn.BatchNorm1d(total_channels * 4) if self.use_batch_norm else nn.Identity()

        # 4. Additional convolution block #3
        self.conv4 = nn.Conv1d(in_channels=total_channels * 4,
                               out_channels=total_channels * 4,
                               kernel_size=3, padding=1)
        self.bn4 = nn.BatchNorm1d(total_channels * 4) if self.use_batch_norm else nn.Identity()

        # 5. Fully connected layers
        self.fc1 = nn.Linear(total_channels * 4, self.fc_units)
        self.dropout = nn.Dropout(self.dropout_rate)
        self.fc2 = nn.Linear(self.fc_units, 1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Defines the forward pass of the model (reference-style).

        Args:
            x: Input tensor of shape (batch_size, sequence_length, input_channels) or (batch_size, input_channels, sequence_length).

        Returns:
            Output tensor (regression prediction), shape (batch_size, 1).
        """
        # Permute input if needed: (N, L, C) -> (N, C, L)
        if x.shape[1] == self.input_channels:
            pass
        elif x.shape[2] == self.input_channels:
            x = x.permute(0, 2, 1)
        else:
            raise ValueError(f"Unexpected input shape: {x.shape}. Expected channels={self.input_channels}")

        # Branches
        branch_outputs = []
        for conv, bn in zip(self.branches, self.branch_norms):
            out = conv(x)
            out = bn(out)
            out = self.activation_fn(out)
            branch_outputs.append(out)
        x_cat = torch.cat(branch_outputs, dim=1)

        # Block 1: conv2 -> bn2 -> activation -> pool
        x = self.conv2(x_cat)
        x = self.bn2(x)
        x = self.activation_fn(x)
        x = self.pool(x)

        # Block 2: conv3 -> bn3 -> activation
        x = self.conv3(x)
        x = self.bn3(x)
        x = self.activation_fn(x)

        # Block 3: conv4 -> bn4 -> activation
        x = self.conv4(x)
        x = self.bn4(x)
        x = self.activation_fn(x)

        # Global average pooling across the sequence dimension
        x = x.mean(dim=2)  # (batch, channels)

        # Fully connected
        x = self.fc1(x)
        x = self.activation_fn(x)
        x = self.dropout(x)
        x = self.fc2(x)
        x = torch.sigmoid(x)
        return x

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