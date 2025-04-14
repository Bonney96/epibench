import torch
from torch.utils.data import Dataset
import h5py
import numpy as np
import os
from typing import Optional, Callable, List, Dict, Tuple, Any
import warnings
import logging

logger = logging.getLogger(__name__)

class SequenceDataset(Dataset):
    """PyTorch Dataset for loading sequence and epigenetic data.

    Assumes data is preprocessed and stored in an HDF5 file.
    The HDF5 file is expected to have datasets like 'features', 'targets',
    and potentially 'metadata' or specific keys for sample IDs/chromosomes.
    """
    def __init__(self, hdf5_path: str, transform: Optional[Callable] = None):
        """Initialize the dataset.

        Args:
            hdf5_path: Path to the HDF5 file containing the dataset.
            transform: Optional transform to be applied on a sample.
        """
        if not os.path.exists(hdf5_path):
            raise FileNotFoundError(f"HDF5 file not found at: {hdf5_path}")

        self.hdf5_path = hdf5_path
        self.transform = transform
        self._file_handle = None # Lazily open the HDF5 file
        self._num_samples = 0

        # Open the file once to get the number of samples
        with h5py.File(self.hdf5_path, 'r') as f:
            if 'features' not in f:
                 raise ValueError(f"HDF5 file {hdf5_path} missing required 'features' dataset.")
            if 'targets' not in f:
                 raise ValueError(f"HDF5 file {hdf5_path} missing required 'targets' dataset.")

            # Assuming the first dimension is the number of samples
            self._num_samples = f['features'].shape[0]
            if f['targets'].shape[0] != self._num_samples:
                 raise ValueError("Features and targets datasets have different number of samples.")

            # Optionally load other metadata if needed later
            # e.g., self.sample_ids = f['sample_ids'][:] if 'sample_ids' in f else None

    def _open_hdf5(self):
        """Opens the HDF5 file if it's not already open."""
        if self._file_handle is None:
            self._file_handle = h5py.File(self.hdf5_path, 'r')

    def __len__(self) -> int:
        """Return the total number of samples in the dataset."""
        return self._num_samples

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor]:
        """Fetch a sample (features and target) at the given index."""
        self._open_hdf5() # Ensure file is open

        # Directly access data assuming idx is valid (checked by DataLoader or subclass)
        features = self._file_handle['features'][idx]
        target = self._file_handle['targets'][idx]

        # Convert numpy arrays to PyTorch tensors
        features_tensor = torch.from_numpy(features.astype(np.float32))
        target_tensor = torch.from_numpy(target.astype(np.float32))

        # Apply transformations if any
        sample = {'features': features_tensor, 'target': target_tensor}
        if self.transform:
            sample = self.transform(sample)

        # Return as tuple (common practice for DataLoader)
        return sample['features'], sample['target']

    def close(self) -> None:
        """Close the HDF5 file handle."""
        if self._file_handle is not None:
            self._file_handle.close()
            self._file_handle = None

    def __del__(self):
        """Ensure the HDF5 file is closed when the object is deleted."""
        self.close()


class LeakageFreeSequenceDataset(SequenceDataset):
    """A Dataset wrapper that allows selecting specific chromosomes.

    This class extends SequenceDataset to enable chromosome-based splitting,
    preventing data leakage between train/validation/test sets.

    Assumes the HDF5 file contains a dataset named 'chromosomes' mapping
    each sample index to its chromosome name (e.g., as strings).
    """
    def __init__(self, hdf5_path: str, include_chromosomes: Optional[List[str]] = None, exclude_chromosomes: Optional[List[str]] = None, transform: Optional[Callable] = None):
        """Initialize the dataset, filtering by chromosomes.

        Args:
            hdf5_path: Path to the HDF5 file.
            include_chromosomes: A list of chromosome names to include. If None, all are included initially.
            exclude_chromosomes: A list of chromosome names to exclude. Applied after inclusion.
            transform: Optional transform to be applied on a sample.

        Raises:
            ValueError: If 'chromosomes' dataset is missing in HDF5 or if indices cannot be determined.
            FileNotFoundError: If the HDF5 file does not exist.
        """
        super().__init__(hdf5_path, transform) # Initialize base to get total size

        self.include_chromosomes = include_chromosomes
        self.exclude_chromosomes = exclude_chromosomes
        self.original_indices: np.ndarray = np.arange(super().__len__()) # Store mapping to original indices
        self.filtered_indices: np.ndarray = self.original_indices # Initially all indices

        # Load chromosome information and filter indices
        self._filter_by_chromosome()

        # Update the effective length of the dataset
        self._num_samples = len(self.filtered_indices)

    def _filter_by_chromosome(self):
        """Filters the dataset indices based on include/exclude chromosomes."""
        all_chromosomes = None
        try:
            with h5py.File(self.hdf5_path, 'r') as f:
                if 'chromosomes' not in f:
                    raise ValueError(f"HDF5 file {self.hdf5_path} missing required 'chromosomes' dataset for leakage-free splitting.")
                # Load chromosome data - Assuming it's an array of strings matching sample order
                all_chromosomes_bytes = f['chromosomes'][:]
                # Decode bytes to strings if necessary (HDF5 often stores strings as bytes)
                all_chromosomes = [c.decode('utf-8') if isinstance(c, bytes) else str(c) for c in all_chromosomes_bytes]

                if len(all_chromosomes) != len(self.original_indices):
                    raise ValueError("Length of 'chromosomes' dataset does not match number of samples.")

        except Exception as e:
            raise ValueError(f"Error reading 'chromosomes' dataset from {self.hdf5_path}: {e}") from e

        # Create boolean mask based on include/exclude lists
        mask = np.ones(len(self.original_indices), dtype=bool)

        if self.include_chromosomes is not None:
            include_set = set(self.include_chromosomes)
            mask &= np.array([c in include_set for c in all_chromosomes])

        if self.exclude_chromosomes is not None:
            exclude_set = set(self.exclude_chromosomes)
            mask &= np.array([c not in exclude_set for c in all_chromosomes])

        self.filtered_indices = self.original_indices[mask]

        if len(self.filtered_indices) == 0:
            warnings.warn("Chromosome filtering resulted in an empty dataset.")

    def __len__(self) -> int:
        """Return the number of samples *after* chromosome filtering."""
        return len(self.filtered_indices)

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor]:
        """Fetch a sample using the filtered index."""
        if not 0 <= idx < len(self.filtered_indices):
            raise IndexError(f"Index {idx} out of range for filtered dataset size {len(self.filtered_indices)}")

        # Map the filtered index back to the original HDF5 index
        original_idx = self.filtered_indices[idx]

        # Call the parent class's __getitem__ with the original index
        return super().__getitem__(original_idx)

class HDF5Dataset(Dataset):
    """PyTorch Dataset for loading data from HDF5 files.

    Assumes the HDF5 file contains at least 'features' and 'targets' datasets,
    and optionally 'chromosomes' and 'coordinates'.

    Args:
        h5_path (str): Path to the HDF5 file.
        transform (Optional[Callable]): Optional transform to be applied on a sample.
        target_transform (Optional[Callable]): Optional transform to be applied on a label.
    """
    def __init__(self, h5_path: str, transform: Optional[Callable] = None, target_transform: Optional[Callable] = None):
        super().__init__()
        self.h5_path = h5_path
        self.transform = transform
        self.target_transform = target_transform
        self._file_handle: Optional[h5py.File] = None
        self._length: Optional[int] = None

        # Validate file existence and basic structure immediately
        try:
            with h5py.File(self.h5_path, 'r') as f:
                if 'features' not in f:
                    raise ValueError(f"HDF5 file {h5_path} is missing required dataset 'features'.")
                if 'targets' not in f:
                    raise ValueError(f"HDF5 file {h5_path} is missing required dataset 'targets'.")
                self._length = f['features'].shape[0]
                if f['targets'].shape[0] != self._length:
                     raise ValueError(f"Mismatch in number of samples between 'features' ({self._length}) and 'targets' ({f['targets'].shape[0]}) in {h5_path}.")
        except FileNotFoundError:
            logger.error(f"HDF5 file not found: {self.h5_path}")
            raise
        except Exception as e:
            logger.error(f"Error initializing HDF5Dataset from {self.h5_path}: {e}", exc_info=True)
            raise

        logger.info(f"Initialized HDF5Dataset from {self.h5_path}. Found {self._length} samples.")

    def _open_file(self):
        """Opens the HDF5 file if it's not already open. Should be called by __getitem__."""
        if self._file_handle is None:
            try:
                self._file_handle = h5py.File(self.h5_path, 'r')
                logger.debug(f"Opened HDF5 file for reading: {self.h5_path}")
            except Exception as e:
                logger.error(f"Failed to open HDF5 file {self.h5_path} in worker process: {e}", exc_info=True)
                raise RuntimeError(f"Could not open HDF5 file: {self.h5_path}") from e

    def __len__(self) -> int:
        """Returns the number of samples in the dataset."""
        if self._length is None:
            # This should have been set in __init__, but as a fallback:
            try:
                with h5py.File(self.h5_path, 'r') as f:
                    self._length = f['features'].shape[0]
            except Exception as e:
                logger.error(f"Failed to determine length from {self.h5_path}: {e}")
                return 0 # Return 0 if length cannot be determined
        return self._length

    def __getitem__(self, idx: int) -> Tuple[Any, Any]:
        """Loads and returns a sample from the dataset at the given index.

        Opens the HDF5 file if necessary (handle worker processes).

        Args:
            idx (int): Index of the sample to retrieve.

        Returns:
            Tuple[Any, Any]: A tuple containing the features and the target.
                             The types depend on the transforms applied.
        """
        self._open_file() # Ensure file is open in the current worker process

        if self._file_handle is None:
             raise RuntimeError(f"HDF5 file handle is not open for {self.h5_path}")

        try:
            # Load data for the given index
            # Convert to PyTorch tensors
            features = torch.from_numpy(self._file_handle['features'][idx].astype(np.float32))
            target = torch.from_numpy(self._file_handle['targets'][idx].astype(np.float32))

            # Apply transforms if they exist
            if self.transform:
                features = self.transform(features)
            if self.target_transform:
                target = self.target_transform(target)

            return features, target

        except IndexError:
            logger.error(f"Index {idx} out of bounds for dataset with length {self._length} in file {self.h5_path}")
            raise
        except Exception as e:
            logger.error(f"Error reading sample {idx} from {self.h5_path}: {e}", exc_info=True)
            # You might want to return dummy data or raise a specific error
            # For now, re-raising the exception
            raise RuntimeError(f"Failed to retrieve sample {idx} from {self.h5_path}") from e

    def close(self):
        """Closes the HDF5 file handle if it's open."""
        if self._file_handle is not None:
            try:
                self._file_handle.close()
                self._file_handle = None
                logger.debug(f"Closed HDF5 file: {self.h5_path}")
            except Exception as e:
                logger.error(f"Error closing HDF5 file {self.h5_path}: {e}", exc_info=True)

    def __del__(self):
        """Ensure the file handle is closed when the object is deleted."""
        self.close()
