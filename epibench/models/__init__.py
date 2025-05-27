import logging
from typing import Type, Dict, List
import torch.nn as nn

# Import model classes from their respective files
from .base import BaseModel # Assuming a base class exists
from .cnn import SimpleCNN
from .transformer import SimpleTransformer
from .seq_cnn_regressor import SeqCNNRegressor

logger = logging.getLogger(__name__)

# Dictionary mapping model names (strings) to model classes
_MODEL_REGISTRY: Dict[str, Type[BaseModel]] = {
    'SimpleCNN': SimpleCNN,
    'SimpleTransformer': SimpleTransformer,
    'SeqCNNRegressor': SeqCNNRegressor,
    # Add other models here as they are created
    # e.g., 'MyAwesomeModel': MyAwesomeModel,
}

def get_model(model_name: str) -> Type[BaseModel]:
    """Factory function to get a model class by its name.

    Args:
        model_name (str): The name of the model to retrieve.

    Returns:
        Type[BaseModel]: The corresponding model class.

    Raises:
        ValueError: If the model_name is not found in the registry.
    """
    model_class = _MODEL_REGISTRY.get(model_name)
    if model_class is None:
        logger.error(f"Model name '{model_name}' not found in registry. Available models: {list(_MODEL_REGISTRY.keys())}")
        raise ValueError(f"Unknown model name: {model_name}. Available models: {list(_MODEL_REGISTRY.keys())}")
    
    logger.info(f"Retrieved model class: {model_name}")
    return model_class

# Optionally, make registry available for inspection
def available_models() -> List[str]:
    """Returns a list of available model names."""
    return list(_MODEL_REGISTRY.keys())

# This allows `from epibench.models import models` and using `models.get_model`
# However, a more direct approach is often preferred:
# `from epibench.models import get_model`
# To keep consistency with the existing code in train.py, we keep this structure for now.
# If refactoring later, consider changing train.py to use the direct import.
class ModelAccessor:
    def get_model(self, model_name: str) -> Type[BaseModel]:
        return get_model(model_name)
    
    def available_models(self) -> List[str]:
        return available_models()

models = ModelAccessor()
