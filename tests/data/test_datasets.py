import pytest
import torch
from torch.utils.data import DataLoader
import h5py
import numpy as np
import os

from epibench.data.datasets import SequenceDataset, LeakageFreeSequenceDataset

# Fixture to create a dummy HDF5 file for testing
@pytest.fixture(scope='module') # Use module scope for efficiency
def dummy_hdf5_file(tmp_path_factory):
    file_path = tmp_path_factory.mktemp("data") / "dummy_dataset.h5"
    n_samples = 20
    seq_len = 100 # Smaller for testing
    n_features = 11
    with h5py.File(file_path, 'w') as f:
        f.create_dataset('features', data=np.random.rand(n_samples, seq_len, n_features).astype(np.float32))
        f.create_dataset('targets', data=np.random.rand(n_samples, 1).astype(np.float32))
        # Create chromosome dataset
        chroms = ([f'chr{i}'.encode('utf-8') for i in range(1, 6)] * (n_samples // 5))
        f.create_dataset('chromosomes', data=chroms)
    return str(file_path)

# --- Tests for SequenceDataset with DataLoader ---

def test_sequence_dataset_with_dataloader(dummy_hdf5_file):
    batch_size = 4
    dataset = SequenceDataset(hdf5_path=dummy_hdf5_file)
    assert len(dataset) == 20

    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True)

    num_batches = 0
    for features, targets in dataloader:
        assert isinstance(features, torch.Tensor)
        assert isinstance(targets, torch.Tensor)
        # Check batch size (last batch might be smaller)
        expected_batch_size = min(batch_size, len(dataset) - num_batches * batch_size)
        assert features.shape[0] == expected_batch_size
        assert targets.shape[0] == expected_batch_size
        # Check feature dimensions (N, SeqLen, Features)
        assert features.shape[1] == 100 # Matches dummy data seq_len
        assert features.shape[2] == 11  # Matches dummy data n_features
        assert targets.shape[1] == 1   # Matches dummy data target dim
        assert features.dtype == torch.float32
        assert targets.dtype == torch.float32
        num_batches += 1

    assert num_batches == np.ceil(len(dataset) / batch_size)
    dataset.close() # Important to close the handle

# --- Tests for LeakageFreeSequenceDataset with DataLoader ---

def test_leakagefree_dataset_include(dummy_hdf5_file):
    include_chroms = ['chr1', 'chr3']
    dataset = LeakageFreeSequenceDataset(hdf5_path=dummy_hdf5_file, include_chromosomes=include_chroms)
    # Expect 4 samples per chromosome * 2 chromosomes = 8 samples
    assert len(dataset) == 8
    dataset.close()

def test_leakagefree_dataset_exclude(dummy_hdf5_file):
    exclude_chroms = ['chr1', 'chr5']
    dataset = LeakageFreeSequenceDataset(hdf5_path=dummy_hdf5_file, exclude_chromosomes=exclude_chroms)
    # Expect 20 total - (4 samples * 2 excluded chroms) = 12 samples
    assert len(dataset) == 12
    dataset.close()

def test_leakagefree_dataset_include_exclude(dummy_hdf5_file):
    include_chroms = ['chr1', 'chr2', 'chr3']
    exclude_chroms = ['chr2', 'chr4'] # Exclude overrides include
    dataset = LeakageFreeSequenceDataset(
        hdf5_path=dummy_hdf5_file,
        include_chromosomes=include_chroms,
        exclude_chromosomes=exclude_chroms
    )
    # Included: chr1, chr3. Excluded: chr2, chr4. Total 20. Chroms 1, 3 should remain (4*2=8)
    assert len(dataset) == 8
    dataset.close()

def test_leakagefree_dataset_with_dataloader(dummy_hdf5_file):
    include_chroms = ['chr2', 'chr4']
    batch_size = 3
    dataset = LeakageFreeSequenceDataset(hdf5_path=dummy_hdf5_file, include_chromosomes=include_chroms)
    assert len(dataset) == 8 # 4 samples * 2 chroms

    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=False) # No shuffle for predictable batches

    num_batches = 0
    all_features = []
    for features, targets in dataloader:
        assert isinstance(features, torch.Tensor)
        assert isinstance(targets, torch.Tensor)
        expected_batch_size = min(batch_size, len(dataset) - num_batches * batch_size)
        assert features.shape[0] == expected_batch_size
        assert targets.shape[0] == expected_batch_size
        assert features.shape[1] == 100
        assert features.shape[2] == 11
        assert targets.shape[1] == 1
        all_features.append(features)
        num_batches += 1

    assert num_batches == np.ceil(len(dataset) / batch_size) # ceil(8/3) = 3
    # Check total number of samples processed
    total_samples_processed = sum(f.shape[0] for f in all_features)
    assert total_samples_processed == len(dataset)

    dataset.close()
