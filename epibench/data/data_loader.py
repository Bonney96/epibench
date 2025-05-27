import logging
import os
from typing import Dict, Tuple, Optional, Any
import torch
from torch.utils.data import DataLoader, Dataset

from . import datasets # Import the datasets module

logger = logging.getLogger(__name__)

def validate_dataloader_config(config: Dict[str, Any]) -> Dict[str, Any]:
    """Validates the data loading section of the configuration.

    Args:
        config (Dict[str, Any]): The main configuration dictionary.

    Returns:
        Dict[str, Any]: The validated and potentially normalized data configuration section.

    Raises:
        ValueError: If required keys are missing or have invalid values.
    """
    if 'data' not in config:
        raise ValueError("Configuration missing required section: 'data'")

    data_config = config['data']

    required_keys = ['train_path', 'val_path', 'test_path']
    for key in required_keys:
        if key not in data_config or not data_config[key]:
            raise ValueError(f"Data configuration missing required key: '{key}'")
        if not isinstance(data_config[key], str):
             raise ValueError(f"Data configuration key '{key}' must be a string path.")
        # Basic check if file exists, more robust checks happen in HDF5Dataset
        # if not os.path.exists(data_config[key]):
        #     logger.warning(f"Data file path specified in config does not exist: {data_config[key]}")
             # raise FileNotFoundError(f"Data file not found: {data_config[key]}") 

    # Validate optional parameters with defaults
    data_config.setdefault('batch_size', 64)
    data_config.setdefault('num_workers', 0)
    data_config.setdefault('shuffle_train', True)
    data_config.setdefault('pin_memory', torch.cuda.is_available())

    if not isinstance(data_config['batch_size'], int) or data_config['batch_size'] <= 0:
        raise ValueError("'batch_size' must be a positive integer.")
    if not isinstance(data_config['num_workers'], int) or data_config['num_workers'] < 0:
        raise ValueError("'num_workers' must be a non-negative integer.")
    if not isinstance(data_config['shuffle_train'], bool):
        raise ValueError("'shuffle_train' must be a boolean.")
    if not isinstance(data_config['pin_memory'], bool):
        raise ValueError("'pin_memory' must be a boolean.")

    logger.info("Data loader configuration validated successfully.")
    return data_config

def create_dataloaders(config: Dict[str, Any]) -> Tuple[DataLoader, DataLoader, DataLoader]:
    """Creates PyTorch DataLoaders for train, validation, and test sets.

    Args:
        config (Dict[str, Any]): The main configuration dictionary.

    Returns:
        Tuple[DataLoader, DataLoader, DataLoader]: A tuple containing the
            training, validation, and testing DataLoader instances.
    """
    logger.info("Creating DataLoaders...")
    try:
        data_config = validate_dataloader_config(config)
    except (ValueError, FileNotFoundError) as e:
        logger.error(f"Invalid data configuration: {e}", exc_info=True)
        raise

    train_path = data_config['train_path']
    val_path = data_config['val_path']
    test_path = data_config['test_path']
    batch_size = data_config['batch_size']
    num_workers = data_config['num_workers']
    shuffle_train = data_config['shuffle_train']
    pin_memory = data_config['pin_memory']

    # TODO: Add support for transforms/augmentation later
    transform = None 
    target_transform = None

    try:
        # Create Datasets
        logger.info(f"Loading training data from: {train_path}")
        train_dataset = datasets.HDF5Dataset(train_path, transform=transform, target_transform=target_transform)
        logger.info(f"Loading validation data from: {val_path}")
        val_dataset = datasets.HDF5Dataset(val_path, transform=transform, target_transform=target_transform)
        logger.info(f"Loading testing data from: {test_path}")
        test_dataset = datasets.HDF5Dataset(test_path, transform=transform, target_transform=target_transform)

        # Create DataLoaders
        logger.info(f"Creating DataLoader instances (Batch size: {batch_size}, Workers: {num_workers}, Shuffle Train: {shuffle_train}, Pin Memory: {pin_memory})")
        train_loader = DataLoader(
            dataset=train_dataset,
            batch_size=batch_size,
            shuffle=shuffle_train,
            num_workers=num_workers,
            pin_memory=pin_memory,
            drop_last=False # Keep last batch even if smaller
        )

        val_loader = DataLoader(
            dataset=val_dataset,
            batch_size=batch_size,
            shuffle=False, # No shuffling for validation
            num_workers=num_workers,
            pin_memory=pin_memory,
            drop_last=False
        )

        test_loader = DataLoader(
            dataset=test_dataset,
            batch_size=batch_size,
            shuffle=False, # No shuffling for testing
            num_workers=num_workers,
            pin_memory=pin_memory,
            drop_last=False
        )

        logger.info("DataLoaders created successfully.")
        return train_loader, val_loader, test_loader

    except Exception as e:
        logger.error(f"Failed to create DataLoaders: {e}", exc_info=True)
        # Clean up dataset file handles if necessary
        if 'train_dataset' in locals() and train_dataset: train_dataset.close()
        if 'val_dataset' in locals() and val_dataset: val_dataset.close()
        if 'test_dataset' in locals() and test_dataset: test_dataset.close()
        raise 