import pytest
import pandas as pd
import numpy as np
import os
from unittest.mock import patch, MagicMock

from epibench.data.base import (
    GenomicRegion,
    SequenceData,
    MethylationData,
    HistoneMarkData,
    BaseGenomicData
)

# --- Tests for GenomicRegion ---

def test_genomic_region_init():
    region = GenomicRegion("chr1", 100, 200, '+', 'geneA')
    assert region.chromosome == "chr1"
    assert region.start == 100
    assert region.end == 200
    assert region.strand == '+'
    assert region.name == 'geneA'

def test_genomic_region_init_defaults():
    region = GenomicRegion("chrX", 500, 550)
    assert region.chromosome == "chrX"
    assert region.start == 500
    assert region.end == 550
    assert region.strand is None
    assert region.name is None

def test_genomic_region_invalid_coords():
    with pytest.raises(ValueError, match="Invalid genomic coordinates"): # noqa
        GenomicRegion("chr1", -10, 100) # Negative start
    with pytest.raises(ValueError, match="Invalid genomic coordinates"): # noqa
        GenomicRegion("chr1", 100, 50) # Start > end

def test_genomic_region_invalid_strand():
    with pytest.raises(ValueError, match="Invalid strand"): # noqa
        GenomicRegion("chr1", 100, 200, strand='x')

def test_genomic_region_len():
    region = GenomicRegion("chr1", 100, 250)
    assert len(region) == 150

def test_genomic_region_str():
    region1 = GenomicRegion("chr1", 100, 200, '+', 'geneA')
    assert str(region1) == "chr1:100-200(+) [geneA]"
    region2 = GenomicRegion("chr2", 300, 400)
    assert str(region2) == "chr2:300-400"
    region3 = GenomicRegion("chr3", 500, 600, '-')
    assert str(region3) == "chr3:500-600(-)"
    region4 = GenomicRegion("chr4", 700, 800, name="enhancer")
    assert str(region4) == "chr4:700-800 [enhancer]"

def test_genomic_region_repr():
    region = GenomicRegion("chr1", 100, 200, '+', 'geneA')
    assert repr(region) == "GenomicRegion(chromosome='chr1', start=100, end=200, strand='+', name='geneA')"

def test_genomic_region_eq():
    region1 = GenomicRegion("chr1", 100, 200, '+', 'geneA')
    region2 = GenomicRegion("chr1", 100, 200, '+', 'geneA')
    region3 = GenomicRegion("chr1", 100, 200, '-', 'geneA') # Different strand
    region4 = GenomicRegion("chr2", 100, 200, '+', 'geneA') # Different chrom
    assert region1 == region2
    assert region1 != region3
    assert region1 != region4
    assert region1 != "chr1:100-200"

def test_genomic_region_hash():
    region1 = GenomicRegion("chr1", 100, 200, '+', 'geneA')
    region2 = GenomicRegion("chr1", 100, 200, '+', 'geneA')
    region3 = GenomicRegion("chr1", 100, 200, '-', 'geneA')
    region_set = {region1, region2, region3}
    assert len(region_set) == 2 # region1 and region2 should hash the same
    assert region1 in region_set
    assert region3 in region_set

def test_genomic_region_overlaps():
    region1 = GenomicRegion("chr1", 100, 200)
    # Overlap cases
    assert region1.overlaps(GenomicRegion("chr1", 150, 250)) # Overlap end
    assert region1.overlaps(GenomicRegion("chr1", 50, 150))  # Overlap start
    assert region1.overlaps(GenomicRegion("chr1", 120, 180)) # Contained within
    assert region1.overlaps(GenomicRegion("chr1", 50, 250))  # Contains
    assert region1.overlaps(GenomicRegion("chr1", 100, 200)) # Exact match
    # Non-overlap cases
    assert not region1.overlaps(GenomicRegion("chr1", 200, 300)) # Adjacent end
    assert not region1.overlaps(GenomicRegion("chr1", 50, 100))  # Adjacent start
    assert not region1.overlaps(GenomicRegion("chr1", 300, 400)) # Disjoint
    assert not region1.overlaps(GenomicRegion("chr2", 100, 200)) # Different chromosome

# --- Tests for BaseGenomicData (Indirectly via subclasses) ---

@pytest.fixture
def temp_file(tmp_path):
    def _temp_file(content, suffix=".txt"):
        file = tmp_path / f"temp{suffix}"
        file.write_text(content)
        return str(file)
    return _temp_file

# --- Tests for SequenceData ---

def test_sequence_data_init_empty():
    seq_data = SequenceData()
    assert not seq_data.is_loaded()
    assert seq_data.sequences == {}
    assert seq_data.metadata == {}

def test_sequence_data_init_with_data():
    initial_seqs = {"seq1": "ACGT", "seq2": "TGCA"}
    initial_meta = {"source": "manual"}
    seq_data = SequenceData(sequences=initial_seqs, metadata=initial_meta)
    assert seq_data.is_loaded()
    assert seq_data.sequences == initial_seqs
    assert seq_data.metadata == initial_meta

def test_sequence_data_metadata_methods():
    seq_data = SequenceData()
    assert seq_data.get_metadata() == {}
    seq_data.update_metadata({"key1": "value1"})
    assert seq_data.get_metadata() == {"key1": "value1"}
    assert seq_data.get_metadata("key1") == "value1"
    assert seq_data.get_metadata("key_unknown") is None
    seq_data.update_metadata({"key1": "value_new", "key2": 123})
    assert seq_data.get_metadata() == {"key1": "value_new", "key2": 123}


@patch('epibench.data.base.SeqIO') # Mock Bio.SeqIO
def test_sequence_data_load_success(mock_seqio, temp_file):
    fasta_content = ">seqA\nACGTN\n>seqB\nGGCCTA"
    fasta_path = temp_file(fasta_content, suffix=".fasta")

    # Setup mock SeqIO.parse to return mock records
    mock_record_a = MagicMock()
    mock_record_a.id = "seqA"
    mock_record_a.seq = "ACGTN"
    mock_record_b = MagicMock()
    mock_record_b.id = "seqB"
    mock_record_b.seq = "GGCCTA"
    mock_seqio.parse.return_value = [mock_record_a, mock_record_b]

    seq_data = SequenceData()
    assert not seq_data.is_loaded()
    seq_data.load_data(fasta_path)

    mock_seqio.parse.assert_called_once_with(fasta_path, "fasta")
    assert seq_data.is_loaded()
    assert seq_data.sequences == {"seqA": "ACGTN", "seqB": "GGCCTA"}
    assert seq_data.get_metadata('source_file') == fasta_path
    assert seq_data.get_metadata('format') == 'fasta'
    assert seq_data.get_metadata('count') == 2


def test_sequence_data_load_file_not_found(tmp_path):
    seq_data = SequenceData()
    fasta_path = str(tmp_path / "nonexistent.fasta")
    seq_data.load_data(fasta_path)
    assert not seq_data.is_loaded()
    assert seq_data.sequences == {}

@patch.dict('sys.modules', {'Bio': None}) # Simulate Biopython not installed
def test_sequence_data_load_biopython_missing(temp_file):
    fasta_content = ">seqA\nACGTN"
    fasta_path = temp_file(fasta_content, suffix=".fasta")
    seq_data = SequenceData()
    seq_data.load_data(fasta_path)
    assert not seq_data.is_loaded()

def test_sequence_data_load_unsupported_format(temp_file):
    content = "some data"
    file_path = temp_file(content, suffix=".txt")
    seq_data = SequenceData()
    seq_data.load_data(file_path, file_format="unsupported")
    assert not seq_data.is_loaded()

def test_sequence_data_validate_success():
    seq_data = SequenceData(sequences={"seq1": "ACGT", "seq2": "N"})
    assert seq_data.validate_data() is True

def test_sequence_data_validate_empty_sequences():
    seq_data = SequenceData(sequences={"seq1": "", "seq2": "ACGT"})
    with pytest.warns(UserWarning, match="Sequence seq1 is empty."):
        assert seq_data.validate_data() is True

def test_sequence_data_validate_invalid_chars():
    seq_data = SequenceData(sequences={"seq1": "ACGTX", "seq2": "N"})
    with pytest.warns(UserWarning, match="Sequence seq1 contains invalid characters."):
        assert seq_data.validate_data() is False

def test_sequence_data_validate_not_loaded():
    seq_data = SequenceData()
    assert seq_data.validate_data() is False

def test_sequence_data_get_data():
    initial_seqs = {"seq1": "ACGT", "seq2": "TGCA"}
    seq_data = SequenceData(sequences=initial_seqs)
    assert seq_data.get_data() == initial_seqs

def test_sequence_data_get_data_before_load():
     seq_data = SequenceData()
     with pytest.warns(UserWarning, match="Attempting to get data before loading."):
         assert seq_data.get_data() == {}

def test_sequence_data_get_sequence():
    initial_seqs = {"seq1": "ACGT", "seq2": "TGCA"}
    seq_data = SequenceData(sequences=initial_seqs)
    assert seq_data.get_sequence("seq1") == "ACGT"
    assert seq_data.get_sequence("seq_unknown") is None

def test_sequence_data_get_sequence_length():
    initial_seqs = {"seq1": "ACGT", "seq2": "TGCA"}
    seq_data = SequenceData(sequences=initial_seqs)
    assert seq_data.get_sequence_length("seq1") == 4
    assert seq_data.get_sequence_length("seq_unknown") is None

# --- Tests for MethylationData ---

def test_methylation_data_init_empty():
    meth_data = MethylationData()
    assert not meth_data.is_loaded()
    assert meth_data.methylation_levels is None
    assert meth_data.metadata == {}

def test_methylation_data_init_with_data():
    initial_df = pd.DataFrame({
        'chromosome': ['chr1', 'chr1'],
        'position': [100, 150],
        'methylation_level': [0.8, 0.2]
    })
    initial_meta = {"source": "manual"}
    meth_data = MethylationData(methylation_levels=initial_df, metadata=initial_meta)
    assert meth_data.is_loaded()
    pd.testing.assert_frame_equal(meth_data.methylation_levels, initial_df)
    assert meth_data.metadata == initial_meta

def test_methylation_data_load_bedgraph_success(temp_file):
    bedgraph_content = "chr1\t100\t101\t0.75\nchr1\t200\t201\t0.25"
    bedgraph_path = temp_file(bedgraph_content, suffix=".bedgraph")
    meth_data = MethylationData()
    meth_data.load_data(bedgraph_path, file_format='bedgraph')

    assert meth_data.is_loaded()
    df = meth_data.get_data()
    assert isinstance(df, pd.DataFrame)
    assert list(df.columns) == ['chromosome', 'position', 'methylation_level']
    assert len(df) == 2
    assert df.iloc[0]['chromosome'] == 'chr1'
    assert df.iloc[0]['position'] == 100
    assert df.iloc[0]['methylation_level'] == 0.75
    assert meth_data.get_metadata('format') == 'bedgraph'

def test_methylation_data_load_bismark_cov_success(temp_file):
    bismark_content = "chr2\t500\t501\t60.0\t3\t2\nchr2\t600\t601\t20.0\t1\t4"
    bismark_path = temp_file(bismark_content, suffix=".cov")
    meth_data = MethylationData()
    meth_data.load_data(bismark_path, file_format='bismark_cov')

    assert meth_data.is_loaded()
    df = meth_data.get_data()
    assert isinstance(df, pd.DataFrame)
    assert 'chromosome' in df.columns
    assert 'position' in df.columns
    assert 'methylation_level' in df.columns
    assert 'coverage' in df.columns
    assert len(df) == 2
    assert df.iloc[0]['chromosome'] == 'chr2'
    assert df.iloc[0]['position'] == 500
    assert df.iloc[0]['methylation_level'] == 0.60
    assert df.iloc[0]['coverage'] == 5
    assert meth_data.get_metadata('format') == 'bismark_cov'

def test_methylation_data_load_file_not_found(tmp_path):
    meth_data = MethylationData()
    meth_data.load_data(str(tmp_path / "no.bedgraph"), file_format='bedgraph')
    assert not meth_data.is_loaded()
    assert meth_data.get_data() is None

def test_methylation_data_load_unsupported_format(temp_file):
    content = "some data"
    file_path = temp_file(content)
    meth_data = MethylationData()
    meth_data.load_data(file_path, file_format="unsupported")
    assert not meth_data.is_loaded()

def test_methylation_data_validate_success():
    df = pd.DataFrame({
        'chromosome': ['chr1', 'chr1'],
        'position': [100, 150],
        'methylation_level': [0.8, 0.2],
        'coverage': [10, 20]
    })
    meth_data = MethylationData(methylation_levels=df)
    assert meth_data.validate_data() is True

def test_methylation_data_validate_missing_required_col():
    df = pd.DataFrame({
        'chromosome': ['chr1', 'chr1'],
        # 'position': [100, 150], # Missing position
        'methylation_level': [0.8, 0.2]
    })
    meth_data = MethylationData(methylation_levels=df)
    assert meth_data.validate_data() is False

def test_methylation_data_validate_wrong_type():
    df = pd.DataFrame({
        'chromosome': ['chr1', 'chr1'],
        'position': ['100', '150'], # Position as string
        'methylation_level': [0.8, 0.2]
    })
    meth_data = MethylationData(methylation_levels=df)
    assert meth_data.validate_data() is False # Position type check

def test_methylation_data_validate_invalid_meth_level():
    df = pd.DataFrame({
        'chromosome': ['chr1', 'chr1'],
        'position': [100, 150],
        'methylation_level': [1.2, -0.1] # Levels out of range
    })
    meth_data = MethylationData(methylation_levels=df)
    with pytest.warns(UserWarning, match="Methylation levels outside the expected \[0, 1\] range"): # noqa
        assert meth_data.validate_data() is True # Should still be True but warn

def test_methylation_data_validate_negative_position():
    df = pd.DataFrame({
        'chromosome': ['chr1', 'chr1'],
        'position': [-10, 150], # Negative position
        'methylation_level': [0.8, 0.2]
    })
    meth_data = MethylationData(methylation_levels=df)
    with pytest.warns(UserWarning, match="Methylation data contains negative 'position' values."):
         assert meth_data.validate_data() is True # Should still be True but warn

def test_methylation_data_validate_non_numeric_coverage():
    df = pd.DataFrame({
        'chromosome': ['chr1', 'chr1'],
        'position': [100, 150],
        'methylation_level': [0.8, 0.2],
        'coverage': ['10', '20'] # Coverage as string
    })
    meth_data = MethylationData(methylation_levels=df)
    with pytest.warns(UserWarning, match="'coverage' column exists but is not numeric."):
        assert meth_data.validate_data() is True # Should still be True but warn

def test_methylation_data_validate_not_loaded():
    meth_data = MethylationData()
    assert meth_data.validate_data() is False

def test_methylation_data_get_data():
    initial_df = pd.DataFrame({'chromosome': ['chr1'], 'position': [100], 'methylation_level': [0.5]})
    meth_data = MethylationData(methylation_levels=initial_df)
    pd.testing.assert_frame_equal(meth_data.get_data(), initial_df)

def test_methylation_data_get_data_before_load():
    meth_data = MethylationData()
    with pytest.warns(UserWarning, match="Attempting to get data before loading."):
         assert meth_data.get_data() is None

def test_methylation_data_get_methylation_at_success():
    df = pd.DataFrame({
        'chromosome': ['chr1', 'chr1', 'chr1', 'chr2'],
        'position': [50, 100, 199, 100],
        'methylation_level': [0.1, 0.8, 0.2, 0.9]
    })
    meth_data = MethylationData(methylation_levels=df)
    region = GenomicRegion("chr1", 100, 200)
    result_df = meth_data.get_methylation_at(region)

    expected_df = pd.DataFrame({
        'chromosome': ['chr1', 'chr1'],
        'position': [100, 199],
        'methylation_level': [0.8, 0.2]
    }, index=[1, 2]) # Index should match original selection
    pd.testing.assert_frame_equal(result_df, expected_df)

def test_methylation_data_get_methylation_at_no_match():
    df = pd.DataFrame({
        'chromosome': ['chr1', 'chr1'],
        'position': [50, 250],
        'methylation_level': [0.1, 0.9]
    })
    meth_data = MethylationData(methylation_levels=df)
    region = GenomicRegion("chr1", 100, 200)
    result_df = meth_data.get_methylation_at(region)
    assert result_df is not None
    assert result_df.empty

def test_methylation_data_get_methylation_at_wrong_chrom():
    df = pd.DataFrame({
        'chromosome': ['chr2', 'chr2'],
        'position': [100, 150],
        'methylation_level': [0.1, 0.9]
    })
    meth_data = MethylationData(methylation_levels=df)
    region = GenomicRegion("chr1", 100, 200)
    result_df = meth_data.get_methylation_at(region)
    assert result_df is not None
    assert result_df.empty

def test_methylation_data_get_methylation_at_not_loaded():
    meth_data = MethylationData()
    region = GenomicRegion("chr1", 100, 200)
    with pytest.warns(UserWarning, match="Cannot get methylation for chr1:100-200: Data not loaded or invalid."):
        assert meth_data.get_methylation_at(region) is None

# --- Tests for HistoneMarkData ---

# Mock pyBigWig module
@pytest.fixture(autouse=True)
def mock_pybigwig(monkeypatch):
    mock_bw = MagicMock()

    # Function to create a new mock handle each time
    def create_mock_handle(*args, **kwargs):
        handle = MagicMock()
        handle.chroms.return_value = {'chr1': 10000, 'chr2': 5000}
        handle.stats.return_value = [0.5] # Default stats value
        handle.close = MagicMock()
        return handle

    mock_bw.open = MagicMock(side_effect=create_mock_handle) # Use side_effect

    # Simulate pyBigWig module
    mock_module = MagicMock()
    mock_module.open = mock_bw.open
    monkeypatch.setitem(__import__('sys').modules, 'pyBigWig', mock_module)
    # Patch the import within the base module itself if necessary
    # This might depend on how pyBigWig is imported in base.py
    patcher = patch('epibench.data.base.pyBigWig', mock_module, create=True)
    patcher.start()
    yield mock_module # Expose the mock module if needed
    patcher.stop()


def test_histone_data_init_no_load():
    hist_data = HistoneMarkData()
    assert hist_data.file_path is None
    assert not hist_data.is_loaded()
    assert hist_data.signal_file_handle is None
    assert hist_data.metadata == {}

def test_histone_data_init_with_load(mock_pybigwig):
    test_path = "dummy.bw"
    hist_data = HistoneMarkData(file_path=test_path)
    mock_pybigwig.open.assert_called_once_with(test_path)
    assert hist_data.is_loaded()
    assert hist_data.file_path == test_path
    assert hist_data.signal_file_handle is not None
    assert hist_data.get_metadata('source_file') == test_path
    assert hist_data.get_metadata('format') == 'bigwig'
    assert 'chromosomes' in hist_data.get_metadata()

def test_histone_data_load_success(mock_pybigwig):
    hist_data = HistoneMarkData()
    test_path = "test.bw"
    hist_data.load_data(test_path)
    mock_pybigwig.open.assert_called_once_with(test_path)
    assert hist_data.is_loaded()
    assert hist_data.file_path == test_path

def test_histone_data_load_closes_previous(mock_pybigwig):
    hist_data = HistoneMarkData("first.bw")
    first_handle = hist_data.signal_file_handle
    assert first_handle is not None
    first_handle.close.assert_not_called()

    hist_data.load_data("second.bw")
    first_handle.close.assert_called_once()
    assert hist_data.is_loaded()
    assert hist_data.file_path == "second.bw"
    assert hist_data.signal_file_handle is not first_handle

def test_histone_data_load_file_not_found(mock_pybigwig):
    mock_pybigwig.open.side_effect = FileNotFoundError
    hist_data = HistoneMarkData()
    hist_data.load_data("nonexistent.bw")
    assert not hist_data.is_loaded()
    assert hist_data.signal_file_handle is None
    assert hist_data.file_path is None

def test_histone_data_load_runtime_error(mock_pybigwig):
    mock_pybigwig.open.side_effect = RuntimeError("BigWig error")
    hist_data = HistoneMarkData()
    hist_data.load_data("bad.bw")
    assert not hist_data.is_loaded()
    assert hist_data.signal_file_handle is None
    assert hist_data.file_path is None

@patch.dict('sys.modules', {'pyBigWig': None}) # Simulate pyBigWig not installed
def test_histone_data_load_pybigwig_missing():
    hist_data = HistoneMarkData()
    hist_data.load_data("any.bw")
    assert not hist_data.is_loaded()

def test_histone_data_load_unsupported_format():
    hist_data = HistoneMarkData()
    hist_data.load_data("some.file", file_format="unsupported")
    assert not hist_data.is_loaded()
    assert hist_data.file_path is None

def test_histone_data_validate_success(mock_pybigwig):
    hist_data = HistoneMarkData("valid.bw")
    assert hist_data.validate_data() is True

def test_histone_data_validate_not_loaded():
    hist_data = HistoneMarkData()
    assert hist_data.validate_data() is False

def test_histone_data_validate_handle_none():
    hist_data = HistoneMarkData()
    hist_data._data_loaded = True # Force loaded state
    hist_data.signal_file_handle = None
    assert hist_data.validate_data() is False

def test_histone_data_validate_invalid_handle(mock_pybigwig):
    hist_data = HistoneMarkData("valid.bw")
    hist_data.signal_file_handle.chroms.side_effect = Exception("Invalid handle")
    assert hist_data.validate_data() is False

def test_histone_data_get_data(mock_pybigwig):
    hist_data = HistoneMarkData("valid.bw")
    handle = hist_data.get_data()
    assert handle is not None
    assert isinstance(handle, MagicMock) # Check it's a mock object as expected

def test_histone_data_get_data_before_load():
    hist_data = HistoneMarkData()
    with pytest.warns(UserWarning, match="Attempting to get data before loading."):
         assert hist_data.get_data() is None

def test_histone_data_get_signal_success_mean(mock_pybigwig):
    hist_data = HistoneMarkData("test.bw")
    mock_handle = hist_data.signal_file_handle
    mock_handle.stats.return_value = [0.75] # Single value for mean
    region = GenomicRegion("chr1", 100, 200)

    signal = hist_data.get_signal_in_region(region, summary_type='mean')
    mock_handle.stats.assert_called_once_with("chr1", 100, 200, type='mean', nBins=1)
    assert signal == 0.75

def test_histone_data_get_signal_success_binned(mock_pybigwig):
    hist_data = HistoneMarkData("test.bw")
    mock_handle = hist_data.signal_file_handle
    # Simulate some None values returned by stats for bins
    mock_handle.stats.return_value = [0.1, 0.2, None, 0.4, None]
    region = GenomicRegion("chr1", 500, 1000)

    signal_array = hist_data.get_signal_in_region(region, summary_type='max', n_bins=5)
    mock_handle.stats.assert_called_once_with("chr1", 500, 1000, type='max', nBins=5)
    expected_array = np.array([0.1, 0.2, 0.0, 0.4, 0.0]) # Nones replaced with 0
    np.testing.assert_array_equal(signal_array, expected_array)

def test_histone_data_get_signal_chrom_not_found(mock_pybigwig):
    hist_data = HistoneMarkData("test.bw")
    region = GenomicRegion("chr3", 100, 200) # chr3 not in mock chroms
    # No warning expected, should just return 0 / zeros
    signal_mean = hist_data.get_signal_in_region(region, summary_type='mean')
    signal_binned = hist_data.get_signal_in_region(region, summary_type='mean', n_bins=10)
    assert signal_mean == 0.0
    np.testing.assert_array_equal(signal_binned, np.zeros(10))

def test_histone_data_get_signal_no_signal_in_region(mock_pybigwig):
    hist_data = HistoneMarkData("test.bw")
    mock_handle = hist_data.signal_file_handle
    mock_handle.stats.return_value = [None] # Simulate region exists but no data
    region = GenomicRegion("chr1", 100, 200)

    signal = hist_data.get_signal_in_region(region, summary_type='mean')
    assert signal == 0.0

def test_histone_data_get_signal_runtime_error(mock_pybigwig):
    hist_data = HistoneMarkData("test.bw")
    mock_handle = hist_data.signal_file_handle
    mock_handle.stats.side_effect = RuntimeError("Query error")
    region = GenomicRegion("chr1", 100, 200)
    # No warning expected, should return 0 / zeros
    signal_mean = hist_data.get_signal_in_region(region, summary_type='mean')
    signal_binned = hist_data.get_signal_in_region(region, summary_type='mean', n_bins=5)
    assert signal_mean == 0.0
    np.testing.assert_array_equal(signal_binned, np.zeros(5))

def test_histone_data_get_signal_invalid_summary_type(mock_pybigwig):
    hist_data = HistoneMarkData("test.bw")
    region = GenomicRegion("chr1", 100, 200)
    signal = hist_data.get_signal_in_region(region, summary_type='invalid_type')
    assert signal is None

def test_histone_data_get_signal_not_loaded():
    hist_data = HistoneMarkData() # Not loaded
    region = GenomicRegion("chr1", 100, 200)
    signal = hist_data.get_signal_in_region(region)
    assert signal is None

def test_histone_data_close(mock_pybigwig):
    hist_data = HistoneMarkData("test.bw")
    mock_handle = hist_data.signal_file_handle
    assert hist_data.is_loaded()
    mock_handle.close.assert_not_called()
    hist_data.close()
    mock_handle.close.assert_called_once()
    assert not hist_data.is_loaded()
    assert hist_data.signal_file_handle is None
    # Test closing again does nothing
    hist_data.close()
    mock_handle.close.assert_called_once() # Still called only once

def test_histone_data_del_closes_handle(mock_pybigwig):
    hist_data = HistoneMarkData("test.bw")
    mock_handle = hist_data.signal_file_handle
    mock_handle.close.assert_not_called()
    # Simulate garbage collection
    del hist_data
    # Check that close was called via __del__
    mock_handle.close.assert_called_once()
