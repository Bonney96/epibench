import torch
import torch.nn as nn
import torch.nn.functional as F
from .base import BaseModel

class SimpleCNN(BaseModel):
    """
    A simple 1D Convolutional Neural Network model.

    Args:
        input_channels (int): Number of input channels (e.g., 4 for one-hot encoded DNA).
        seq_len (int): Length of the input sequence.
        num_classes (int): Number of output classes.
        num_filters (int): Number of filters in the convolutional layer. Default is 32.
        filter_size (int): Size of the convolutional filter. Default is 5.
    """
    def __init__(self, input_channels: int, seq_len: int, num_classes: int, num_filters: int = 32, filter_size: int = 5):
        super().__init__()
        self.conv1 = nn.Conv1d(in_channels=input_channels, out_channels=num_filters, kernel_size=filter_size, padding='same')
        # Calculate the size after convolution and pooling
        # With padding='same', the length remains seq_len after conv1
        # MaxPool1d with kernel_size=2 halves the length
        pooled_seq_len = seq_len // 2
        self.pool = nn.MaxPool1d(kernel_size=2)
        self.fc1 = nn.Linear(num_filters * pooled_seq_len, num_classes)

        # Store config
        self.config = {
            "class_name": self.__class__.__name__,
            "input_channels": input_channels,
            "seq_len": seq_len,
            "num_classes": num_classes,
            "num_filters": num_filters,
            "filter_size": filter_size,
        }

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass of the SimpleCNN model.

        Args:
            x (torch.Tensor): Input tensor of shape (batch_size, input_channels, seq_len).

        Returns:
            torch.Tensor: Output tensor of shape (batch_size, num_classes).
        """
        # Ensure input is (batch_size, channels, seq_len)
        if x.ndim == 2:
            # Assuming (batch_size, seq_len), add channel dim
            # This might need adjustment based on actual data format
            x = x.unsqueeze(1)
        elif x.ndim == 3 and x.shape[2] == self.config['input_channels']:
             # Input is (batch_size, seq_len, channels), permute
             x = x.permute(0, 2, 1)

        if x.shape[1] != self.config['input_channels'] or x.shape[2] != self.config['seq_len']:
             raise ValueError(f"Expected input shape (batch, {self.config['input_channels']}, {self.config['seq_len']}), but got {x.shape}")

        x = self.pool(F.relu(self.conv1(x)))
        # Flatten the output for the linear layer
        x = torch.flatten(x, 1) # Flatten all dimensions except batch
        x = self.fc1(x)
        return x

    def get_config(self) -> dict:
        """
        Returns the configuration of the SimpleCNN model.

        Returns:
            dict: A dictionary containing the model's configuration.
        """
        return self.config 