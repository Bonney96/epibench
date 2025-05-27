import pytest
import torch
from epibench.models.seq_cnn_regressor import SeqCNNRegressor

# --- Fixtures --- 

@pytest.fixture
def default_model():
    """Returns a SeqCNNRegressor model with default parameters."""
    return SeqCNNRegressor()

@pytest.fixture
def custom_model():
    """Returns a SeqCNNRegressor model with custom parameters."""
    return SeqCNNRegressor(
        input_channels=11,
        num_filters=32,
        kernel_sizes=[5, 15],
        fc_units=[128],
        dropout_rate=0.2,
        use_batch_norm=False,
        activation='gelu'
    )

@pytest.fixture
def dummy_input():
    """Returns a dummy input tensor of the expected shape."""
    batch_size = 4
    seq_len = 10000 # Standard sequence length
    input_channels = 11 # Standard channels
    return torch.randn(batch_size, seq_len, input_channels)

# --- Test Cases --- 

def test_model_initialization_default(default_model):
    """Tests if the model initializes with default parameters without errors."""
    assert isinstance(default_model, SeqCNNRegressor)
    assert len(default_model.conv_branches) == 4 # Default kernel_sizes: [3, 9, 25, 51]
    assert default_model.use_batch_norm is True
    assert isinstance(default_model.activation_fn, torch.nn.ReLU)

def test_model_initialization_custom(custom_model):
    """Tests if the model initializes with custom parameters."""
    assert isinstance(custom_model, SeqCNNRegressor)
    assert len(custom_model.conv_branches) == 2 # Custom kernel_sizes: [5, 15]
    assert custom_model.num_filters == 32
    assert custom_model.use_batch_norm is False
    assert custom_model.fc_units == [128]
    assert isinstance(custom_model.activation_fn, torch.nn.GELU)

def test_forward_pass_shape(default_model, dummy_input):
    """Tests if the forward pass returns the correct output shape."""
    output = default_model(dummy_input)
    assert isinstance(output, torch.Tensor)
    assert output.shape == (dummy_input.shape[0], 1) # (batch_size, 1)

def test_forward_pass_shape_custom(custom_model, dummy_input):
    """Tests the forward pass shape with custom model parameters."""
    output = custom_model(dummy_input)
    assert isinstance(output, torch.Tensor)
    assert output.shape == (dummy_input.shape[0], 1)

def test_init_weights_changes_weights(default_model):
    """Tests if calling init_weights actually changes the model's weights."""
    # Store original weights (example: first conv layer of first branch)
    original_weight = default_model.conv_branches[0][0].weight.data.clone()
    
    default_model.init_weights() # Initialize weights
    
    new_weight = default_model.conv_branches[0][0].weight.data
    
    # Check that weights are not the same (initialization should change them)
    # Use torch.equal for element-wise comparison
    assert not torch.equal(original_weight, new_weight)

    # Optional: Check bias initialization (if bias exists)
    if default_model.conv_branches[0][0].bias is not None:
         original_bias = default_model.conv_branches[0][0].bias.data.clone()
         # Re-initialize to be sure (though done above)
         default_model.init_weights() 
         new_bias = default_model.conv_branches[0][0].bias.data
         # Xavier init sets bias to 0
         assert torch.all(new_bias == 0) 