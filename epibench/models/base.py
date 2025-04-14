import torch
import torch.nn as nn
import json
import os
import importlib

class BaseModel(nn.Module):
    """
    Base class for all models in EpiBench.
    Provides a common interface and potential shared functionalities,
    including saving and loading.
    """
    def __init__(self):
        super().__init__()

    def forward(self, x):
        """
        Defines the forward pass of the model.
        Must be implemented by subclasses.

        Args:
            x (torch.Tensor): Input tensor.

        Raises:
            NotImplementedError: If the forward pass is not implemented.
        """
        raise NotImplementedError("Forward pass not implemented in the base model.")

    def get_config(self) -> dict:
        """
        Returns the configuration of the model.
        Should be overridden by subclasses to return specific parameters.

        Returns:
            dict: A dictionary containing the model's configuration.
        """
        # Subclasses must implement this to return their specific config
        raise NotImplementedError("get_config() must be implemented by subclasses.")

    def save(self, file_path_prefix: str):
        """
        Saves the model's state dict and configuration.

        Args:
            file_path_prefix (str): The prefix for the saved files.
                                      ".pt" will be appended for the state_dict,
                                      ".config.json" will be appended for the config.
        """
        # Ensure directory exists
        dir_name = os.path.dirname(file_path_prefix)
        if dir_name:
            os.makedirs(dir_name, exist_ok=True)

        # Save state dict
        state_dict_path = file_path_prefix + ".pt"
        torch.save(self.state_dict(), state_dict_path)
        print(f"Model state dict saved to {state_dict_path}")

        # Save config
        config = self.get_config()
        config_path = file_path_prefix + ".config.json"
        with open(config_path, 'w') as f:
            json.dump(config, f, indent=4)
        print(f"Model configuration saved to {config_path}")

    @classmethod
    def load(cls, file_path_prefix: str) -> 'BaseModel':
        """
        Loads a model from a saved state dict and configuration file.

        Args:
            file_path_prefix (str): The prefix used when saving the model.

        Returns:
            BaseModel: The loaded model instance.

        Raises:
            FileNotFoundError: If the config or state dict file doesn't exist.
            KeyError: If 'class_name' is missing in the config.
            AttributeError: If the loaded class_name is not found.
        """
        config_path = file_path_prefix + ".config.json"
        state_dict_path = file_path_prefix + ".pt"

        if not os.path.exists(config_path):
            raise FileNotFoundError(f"Configuration file not found: {config_path}")
        if not os.path.exists(state_dict_path):
            raise FileNotFoundError(f"State dictionary file not found: {state_dict_path}")

        # Load config
        with open(config_path, 'r') as f:
            config = json.load(f)

        if "class_name" not in config:
            raise KeyError("'class_name' missing in the loaded configuration.")

        class_name = config.pop("class_name")

        # Dynamically find the class
        # This assumes the model class is defined in epibench.models.* structure
        # Adjust the module path if necessary
        try:
            module_name = f"epibench.models.{class_name.lower()}" # Heuristic module name
            models_module = importlib.import_module(module_name)
            model_class = getattr(models_module, class_name)
        except (ImportError, AttributeError) as e:
             # Fallback: try finding class in the current module's scope (less robust)
             # Or provide a more sophisticated registry mechanism if needed
             print(f"Warning: Could not dynamically import {class_name} from {module_name}. Trying globals(). Error: {e}")
             # Attempting to find class in globals() where load might be called
             # This is fragile and might fail depending on execution context.
             # A better approach would be a model registry.
             if class_name in globals():
                 model_class = globals()[class_name]
             else:
                 raise AttributeError(f"Model class '{class_name}' not found. Ensure it's imported or defined.")

        if not issubclass(model_class, BaseModel):
             raise TypeError(f"Loaded class '{class_name}' is not a subclass of BaseModel.")

        # Instantiate model with loaded config
        model = model_class(**config)

        # Load state dict
        model.load_state_dict(torch.load(state_dict_path))
        model.eval() # Set model to evaluation mode by default after loading
        print(f"Model '{class_name}' loaded successfully from {file_path_prefix}")

        return model 