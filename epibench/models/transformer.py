import torch
import torch.nn as nn
import math
from .base import BaseModel

class PositionalEncoding(nn.Module):
    """Injects positional information into the input embeddings."""
    def __init__(self, d_model: int, dropout: float = 0.1, max_len: int = 5000):
        super().__init__()
        self.dropout = nn.Dropout(p=dropout)

        position = torch.arange(max_len).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2) * (-math.log(10000.0) / d_model))
        pe = torch.zeros(max_len, 1, d_model)
        pe[:, 0, 0::2] = torch.sin(position * div_term)
        pe[:, 0, 1::2] = torch.cos(position * div_term)
        self.register_buffer('pe', pe)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Args: x: Tensor, shape [seq_len, batch_size, embedding_dim]"""
        x = x + self.pe[:x.size(0)]
        return self.dropout(x)


class SimpleTransformer(BaseModel):
    """
    A simple Transformer model using TransformerEncoder.

    Args:
        input_channels (int): Number of input channels (embedding dimension).
        seq_len (int): Length of the input sequence.
        num_classes (int): Number of output classes.
        d_model (int): The number of expected features in the input (required).
        nhead (int): The number of heads in the multiheadattention models (required).
        num_encoder_layers (int): The number of sub-encoder-layers in the encoder (required).
        dim_feedforward (int): The dimension of the feedforward network model (default=2048).
        dropout (float): The dropout value (default=0.1).
        activation (str): The activation function of encoder/decoder intermediate layer, relu or gelu (default=relu).
        batch_first (bool): If True, then the input and output tensors are provided as (batch, seq, feature). Default: False.
    """
    def __init__(self, input_channels: int, seq_len: int, num_classes: int, d_model: int, nhead: int, num_encoder_layers: int,
                 dim_feedforward: int = 2048, dropout: float = 0.1, activation: str = 'relu', batch_first: bool = False):
        super().__init__()

        if batch_first:
            raise NotImplementedError("Batch first is not yet fully supported in this simple implementation")

        self.d_model = d_model
        self.batch_first = batch_first # Store batch_first

        # Embedding layer if input_channels != d_model (e.g., 4 channels to d_model)
        # Assuming input is (seq_len, batch_size, input_channels) if not batch_first
        # Or (batch_size, seq_len, input_channels) if batch_first
        # We need input to TransformerEncoder as (seq_len, batch_size, d_model)
        if input_channels != d_model:
             # Simple linear projection
             self.embedding = nn.Linear(input_channels, d_model)
        else:
            self.embedding = nn.Identity() # No embedding needed if channels match d_model

        self.pos_encoder = PositionalEncoding(d_model, dropout)
        encoder_layer = nn.TransformerEncoderLayer(d_model=d_model, nhead=nhead,
                                                   dim_feedforward=dim_feedforward,
                                                   dropout=dropout, activation=activation, batch_first=batch_first) # Pass batch_first here
        self.transformer_encoder = nn.TransformerEncoder(encoder_layer, num_layers=num_encoder_layers)

        # Output layer
        # The output of TransformerEncoder is (seq_len, batch, d_model)
        # We need to decide how to aggregate the sequence information for classification
        # Option 1: Use the output of the first token ([CLS] token style if applicable)
        # Option 2: Average pooling over the sequence length
        self.output_layer = nn.Linear(d_model, num_classes)

        # Store config
        self.config = {
            "class_name": self.__class__.__name__,
            "input_channels": input_channels,
            "seq_len": seq_len,
            "num_classes": num_classes,
            "d_model": d_model,
            "nhead": nhead,
            "num_encoder_layers": num_encoder_layers,
            "dim_feedforward": dim_feedforward,
            "dropout": dropout,
            "activation": activation,
            "batch_first": batch_first
        }

        self.init_weights()

    def init_weights(self) -> None:
        initrange = 0.1
        if hasattr(self, 'embedding') and not isinstance(self.embedding, nn.Identity):
            self.embedding.weight.data.uniform_(-initrange, initrange)
        self.output_layer.bias.data.zero_()
        self.output_layer.weight.data.uniform_(-initrange, initrange)

    def forward(self, src: torch.Tensor, src_mask: torch.Tensor = None) -> torch.Tensor:
        """
        Forward pass of the SimpleTransformer model.

        Args:
            src (torch.Tensor): Input tensor. Shape depends on batch_first.
                               If batch_first=False: (seq_len, batch_size, input_channels)
                               If batch_first=True: (batch_size, seq_len, input_channels) - NOT fully supported yet.
            src_mask (torch.Tensor, optional): The additive mask for the src sequence. Defaults to None.

        Returns:
            torch.Tensor: Output tensor of shape (batch_size, num_classes).
        """
        # Input shape check and potential permute if batch_first is handled later
        if self.batch_first:
            # Input: (batch, seq, channels) -> Need (seq, batch, channels) for embedding/pos encoding if needed
            # This part needs careful implementation if batch_first=True is fully supported
            pass # Placeholder for potential permutation
        else:
            # Expected input: (seq_len, batch, channels)
            if src.ndim != 3 or src.shape[0] != self.config['seq_len'] or src.shape[2] != self.config['input_channels']:
                 raise ValueError(f"Expected input shape (seq_len={self.config['seq_len']}, batch, channels={self.config['input_channels']}), but got {src.shape}")


        src = self.embedding(src) * math.sqrt(self.d_model) # Scale embedding
        src = self.pos_encoder(src)
        output = self.transformer_encoder(src, src_mask)
        # Aggregate sequence output - Using mean pooling here
        output = output.mean(dim=0) # Mean across seq_len dimension (dim=0 because batch_first=False)
        output = self.output_layer(output)
        return output # Shape: (batch_size, num_classes)

    def get_config(self) -> dict:
        """
        Returns the configuration of the SimpleTransformer model.

        Returns:
            dict: A dictionary containing the model's configuration.
        """
        return self.config 