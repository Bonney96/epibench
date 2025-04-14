import pytest
import torch
import os
from epibench.models.base import BaseModel # Needed for load
from epibench.models.cnn import SimpleCNN
from epibench.models.transformer import SimpleTransformer, PositionalEncoding

# Test parameters
BATCH_SIZE = 4
SEQ_LEN = 128 # Use a smaller seq len for testing to speed up
INPUT_CHANNELS = 4
NUM_CLASSES = 2

# CNN specific
CNN_NUM_FILTERS = 16
CNN_FILTER_SIZE = 3

# Transformer specific
D_MODEL = 32 # Must be divisible by nhead
NHEAD = 4
NUM_ENCODER_LAYERS = 2
DIM_FEEDFORWARD = 64

# --- Fixtures --- 

@pytest.fixture
def cnn_model():
    """Provides an instance of SimpleCNN."""
    return SimpleCNN(input_channels=INPUT_CHANNELS, seq_len=SEQ_LEN, num_classes=NUM_CLASSES,
                     num_filters=CNN_NUM_FILTERS, filter_size=CNN_FILTER_SIZE)

@pytest.fixture
def transformer_model():
    """Provides an instance of SimpleTransformer."""
    return SimpleTransformer(input_channels=INPUT_CHANNELS, seq_len=SEQ_LEN, num_classes=NUM_CLASSES,
                           d_model=D_MODEL, nhead=NHEAD, num_encoder_layers=NUM_ENCODER_LAYERS,
                           dim_feedforward=DIM_FEEDFORWARD, batch_first=False)

@pytest.fixture
def dummy_cnn_input():
    """Provides dummy input tensor for CNN (batch, channels, seq_len)."""
    return torch.randn(BATCH_SIZE, INPUT_CHANNELS, SEQ_LEN)

@pytest.fixture
def dummy_transformer_input():
    """Provides dummy input tensor for Transformer (seq_len, batch, channels)."""
    return torch.randn(SEQ_LEN, BATCH_SIZE, INPUT_CHANNELS)


# --- CNN Tests --- 

def test_simple_cnn_instantiation(cnn_model):
    """Test if SimpleCNN initializes correctly."""
    assert isinstance(cnn_model, SimpleCNN)
    assert cnn_model.config["input_channels"] == INPUT_CHANNELS
    assert cnn_model.config["seq_len"] == SEQ_LEN
    assert cnn_model.config["num_classes"] == NUM_CLASSES
    assert cnn_model.config["num_filters"] == CNN_NUM_FILTERS
    assert cnn_model.config["filter_size"] == CNN_FILTER_SIZE

def test_simple_cnn_forward_pass(cnn_model, dummy_cnn_input):
    """Test the forward pass of SimpleCNN."""
    output = cnn_model(dummy_cnn_input)
    assert output.shape == (BATCH_SIZE, NUM_CLASSES)

def test_simple_cnn_save_load(cnn_model, dummy_cnn_input, tmp_path):
    """Test saving and loading SimpleCNN."""
    file_prefix = os.path.join(tmp_path, "test_cnn_model")
    
    # Save
    cnn_model.save(file_prefix)
    assert os.path.exists(file_prefix + ".pt")
    assert os.path.exists(file_prefix + ".config.json")

    # Load
    loaded_model = BaseModel.load(file_prefix)
    assert isinstance(loaded_model, SimpleCNN)
    
    # Verify config matches
    assert loaded_model.config == cnn_model.config

    # Verify forward pass produces same output (ensure eval mode)
    cnn_model.eval()
    loaded_model.eval()
    with torch.no_grad():
        original_output = cnn_model(dummy_cnn_input)
        loaded_output = loaded_model(dummy_cnn_input)
    assert torch.allclose(original_output, loaded_output, atol=1e-6)

# --- Transformer Tests --- 

def test_simple_transformer_instantiation(transformer_model):
    """Test if SimpleTransformer initializes correctly."""
    assert isinstance(transformer_model, SimpleTransformer)
    assert transformer_model.config["input_channels"] == INPUT_CHANNELS
    assert transformer_model.config["seq_len"] == SEQ_LEN
    assert transformer_model.config["num_classes"] == NUM_CLASSES
    assert transformer_model.config["d_model"] == D_MODEL
    assert transformer_model.config["nhead"] == NHEAD
    assert transformer_model.config["num_encoder_layers"] == NUM_ENCODER_LAYERS
    assert not transformer_model.config["batch_first"] # Explicitly check batch_first
    assert isinstance(transformer_model.pos_encoder, PositionalEncoding)

def test_simple_transformer_forward_pass(transformer_model, dummy_transformer_input):
    """Test the forward pass of SimpleTransformer."""
    # Transformer expects (seq_len, batch, channels)
    output = transformer_model(dummy_transformer_input)
    assert output.shape == (BATCH_SIZE, NUM_CLASSES)

def test_simple_transformer_save_load(transformer_model, dummy_transformer_input, tmp_path):
    """Test saving and loading SimpleTransformer."""
    file_prefix = os.path.join(tmp_path, "test_transformer_model")
    
    # Save
    transformer_model.save(file_prefix)
    assert os.path.exists(file_prefix + ".pt")
    assert os.path.exists(file_prefix + ".config.json")

    # Load
    # Note: We need to explicitly import SimpleTransformer for the dynamic load to find it
    loaded_model = BaseModel.load(file_prefix)
    assert isinstance(loaded_model, SimpleTransformer)
    
    # Verify config matches
    assert loaded_model.config == transformer_model.config

    # Verify forward pass produces same output (ensure eval mode)
    transformer_model.eval()
    loaded_model.eval()
    with torch.no_grad():
        original_output = transformer_model(dummy_transformer_input)
        loaded_output = loaded_model(dummy_transformer_input)
    assert torch.allclose(original_output, loaded_output, atol=1e-6)


# Test input dimension handling in SimpleCNN
def test_simple_cnn_input_permutation(cnn_model):
    """Test if SimpleCNN correctly handles (batch, seq, channels) input."""
    batch_size = 2
    seq_len = cnn_model.config['seq_len']
    input_channels = cnn_model.config['input_channels']
    num_classes = cnn_model.config['num_classes']
    dummy_input_permuted = torch.randn(batch_size, seq_len, input_channels)
    
    try:
        output = cnn_model(dummy_input_permuted)
        assert output.shape == (batch_size, num_classes)
    except ValueError:
        pytest.fail("SimpleCNN failed to handle permuted input (batch, seq, channels).")

def test_simple_cnn_input_unsqueeze(cnn_model):
    """Test if SimpleCNN correctly handles (batch, seq_len) input."""
    # This test assumes input_channels == 1 for unsqueezing to work correctly
    # Modify the test or model logic if this assumption changes.
    if cnn_model.config['input_channels'] == 1:
        batch_size = 2
        seq_len = cnn_model.config['seq_len']
        num_classes = cnn_model.config['num_classes']
        dummy_input_2d = torch.randn(batch_size, seq_len)
        
        try:
            output = cnn_model(dummy_input_2d)
            assert output.shape == (batch_size, num_classes)
        except ValueError:
            pytest.fail("SimpleCNN failed to handle 2D input (batch, seq_len).")
    else:
        pytest.skip("Skipping 2D input test because input_channels != 1")
