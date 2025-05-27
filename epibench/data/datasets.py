import torch
from torch.utils.data import Dataset
import h5py
import numpy as np
import os
from typing import Optional, Callable, List, Dict, Tuple, Any, Union
import warnings
import logging

logger = logging.getLogger(__name__)

CoordinateInfo = Dict[str, Union[str, int]]  # Type hint for coordinate information

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
    """PyTorch Dataset for loading data from HDF5 files created by EpiBench.

    Handles loading 'features' and 'targets' datasets. Optionally loads genomic
    coordinates ('chrom', 'start', 'end') if they exist in the file.

    Args:
        h5_path (str): Path to the HDF5 file.
        transform (Optional[Callable]): Optional transform applied to features.
        target_transform (Optional[Callable]): Optional transform applied to targets.
    """
    def __init__(self, h5_path: str, transform: Optional[Callable] = None, target_transform: Optional[Callable] = None):
        super().__init__()
        self.h5_path = h5_path
        self.transform = transform
        self.target_transform = target_transform
        self._file_handle: Optional[h5py.File] = None
        self._features_ds: Optional[h5py.Dataset] = None
        self._targets_ds: Optional[h5py.Dataset] = None
        self._chrom_ds: Optional[h5py.Dataset] = None
        self._start_ds: Optional[h5py.Dataset] = None
        self._end_ds: Optional[h5py.Dataset] = None
        self._length: Optional[int] = None
        self.has_coordinates: bool = False

        # Validate file existence and basic structure immediately
        try:
            with h5py.File(self.h5_path, 'r') as f:
                if 'features' not in f:
                    logger.error(f"HDF5 file {h5_path} is missing required dataset 'features'.")
                    raise ValueError(f"HDF5 file {h5_path} missing required dataset 'features'.")
                if 'targets' not in f:
                    logger.error(f"HDF5 file {h5_path} is missing required dataset 'targets'.")
                    raise ValueError(f"HDF5 file {h5_path} missing required dataset 'targets'.")
                self._length = f['features'].shape[0]
                if f['targets'].shape[0] != self._length:
                    error_msg = f"Feature count ({self._length}) and target count ({f['targets'].shape[0]}) mismatch in {h5_path}."
                    logger.error(error_msg)
                    raise ValueError(error_msg)

                # Check for coordinate datasets
                coord_keys = ['chrom', 'start', 'end']
                if all(key in f for key in coord_keys):
                    if (f['chrom'].shape[0] == self._length and 
                        f['start'].shape[0] == self._length and 
                        f['end'].shape[0] == self._length):
                        self.has_coordinates = True
                        logger.info(f"Coordinate datasets ('chrom', 'start', 'end') found in {h5_path} and match features length.")
                    else:
                        logger.warning(f"Coordinate datasets found in {h5_path}, but their lengths do not match the features dataset. Coordinates will not be loaded.")
                        self.has_coordinates = False
                else:
                    logger.info(f"Coordinate datasets ('chrom', 'start', 'end') not found or incomplete in {h5_path}. Coordinates will not be loaded.")
                    self.has_coordinates = False

        except FileNotFoundError:
            logger.error(f"HDF5 file not found: {h5_path}")
            raise
        except Exception as e:
            logger.error(f"Failed to open or validate HDF5 file {h5_path}: {e}", exc_info=True) # Log with stacktrace
            raise # Re-raise after logging

        logger.info(f"Initialized HDF5Dataset from {self.h5_path}. Found {self._length} samples.")

    def _open_file(self):
        """Opens the HDF5 file if it's not already open and assigns dataset handles."""
        if self._file_handle is None:
            try:
                self._file_handle = h5py.File(self.h5_path, 'r')
                self._features_ds = self._file_handle['features']
                self._targets_ds = self._file_handle['targets']
                if self.has_coordinates:
                    self._chrom_ds = self._file_handle['chrom']
                    self._start_ds = self._file_handle['start']
                    self._end_ds = self._file_handle['end']
                logger.debug(f"Opened HDF5 file: {self.h5_path}")
            except Exception as e:
                logger.error(f"Failed to open HDF5 file {self.h5_path} in worker process: {e}")
                # Reset handles to ensure we don't use potentially bad ones
                self._file_handle = None
                self._features_ds = None
                self._targets_ds = None
                self._chrom_ds = None
                self._start_ds = None
                self._end_ds = None
                raise # Re-raise to propagate the error

    def __len__(self) -> int:
        """Return the total number of samples in the dataset."""
        if self._length is None:
            # Should have been set in init, but as a fallback:
            self._open_file()
            self._length = self._features_ds.shape[0] if self._features_ds is not None else 0
            logger.warning("__len__ called before HDF5 length was initialized.")
        return self._length

    def __getitem__(self, idx: int) -> Tuple[Any, Any, CoordinateInfo]:
        """Fetch a sample (features, target, coordinates) at the given index.

        Opens the HDF5 file if necessary (for use with multiprocessing in DataLoader).

        Args:
            idx: The index of the sample to retrieve.

        Returns:
            A tuple containing:
                - features: Data features (typically a torch.Tensor).
                - target: Target value(s) (typically a torch.Tensor).
                - coordinates: Dictionary with 'chrom', 'start', 'end' if available,
                               otherwise an empty dictionary.
        """
        self._open_file() # Ensure file handle is open, especially for multiprocessing

        if self._features_ds is None or self._targets_ds is None:
             # This might happen if _open_file failed
             raise RuntimeError(f"HDF5 dataset handles not initialized for {self.h5_path}")

        try:
            # Get features and target
            features = self._features_ds[idx]
            target = self._targets_ds[idx]

            # Apply transforms if provided
            if self.transform:
                features = self.transform(features)
            if self.target_transform:
                target = self.target_transform(target)

            # Get coordinates if available
            coordinates = {}
            if self.has_coordinates:
                if self._chrom_ds is not None and self._start_ds is not None and self._end_ds is not None:
                    try:
                        chrom_val = self._chrom_ds[idx]
                        start_val = int(self._start_ds[idx]) # Ensure integer
                        end_val = int(self._end_ds[idx]) # Ensure integer
                        # Decode chromosome name if it's bytes
                        chrom_str = chrom_val.decode('utf-8') if isinstance(chrom_val, bytes) else str(chrom_val)
                        coordinates = {'chrom': chrom_str, 'start': start_val, 'end': end_val}
                    except IndexError:
                         logger.error(f"IndexError fetching coordinates for index {idx} in {self.h5_path}. Dataset length: {self._length}")
                         raise # Re-raise to indicate a problem
                    except Exception as e:
                         logger.error(f"Error fetching coordinates for index {idx} in {self.h5_path}: {e}", exc_info=True)
                         # Return empty dict for this item, but log the error
                         coordinates = {}
                else:
                     # This case should ideally not happen if has_coordinates is True and _open_file worked
                     logger.warning(f"Coordinate datasets were expected but handles are None for index {idx} in {self.h5_path}")
                     coordinates = {}

            # Return features, target, and coordinates
            return features, target, coordinates

        except IndexError:
            logger.error(f"Index {idx} out of range for HDF5 dataset {self.h5_path} with length {self._length}")
            raise # Re-raise the IndexError
        except Exception as e:
            # Catch potential errors during data loading or transform
            logger.error(f"Error loading item {idx} from {self.h5_path}: {e}", exc_info=True) # Log with stacktrace
            # Decide how to handle: re-raise, return None, return dummy data?
            # Re-raising is often safest to signal the problem upstream.
            raise

    def get_coordinates(self, idx: int) -> Optional[CoordinateInfo]:
        """Retrieve genomic coordinates for a specific index, if available.

        Args:
            idx: Index of the sample.

        Returns:
            Dictionary containing 'chrom', 'start', 'end' keys if coordinates
            are available, otherwise None.

        Raises:
            IndexError: If the index is out of bounds.
            RuntimeError: If HDF5 dataset handles are not initialized.
        """
        if not self.has_coordinates:
            return None

        self._open_file() # Ensure file is open

        if self._chrom_ds is None or self._start_ds is None or self._end_ds is None:
            # This indicates an issue during file opening or state corruption
            raise RuntimeError(f"Coordinate dataset handles not initialized for {self.h5_path} despite has_coordinates=True.")

        if not 0 <= idx < self.__len__():
             raise IndexError(f"Index {idx} out of range for dataset size {self.__len__()}")

        try:
            chrom_val = self._chrom_ds[idx]
            start_val = int(self._start_ds[idx])
            end_val = int(self._end_ds[idx])
            chrom_str = chrom_val.decode('utf-8') if isinstance(chrom_val, bytes) else str(chrom_val)
            return {'chrom': chrom_str, 'start': start_val, 'end': end_val}
        except Exception as e:
            logger.error(f"Error retrieving coordinates for index {idx} from {self.h5_path}: {e}", exc_info=True)
            # Depending on desired behavior, could return None or re-raise
            return None # Return None on error accessing specific coordinates

    def close(self):
        """Close the HDF5 file handle if it's open."""
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
