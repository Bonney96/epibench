# epibench/data/base.py
from abc import ABC, abstractmethod
from typing import Any, Dict, List, Optional, Union, Set
import numpy as np
import pandas as pd
import warnings
from Bio import SeqIO

class GenomicRegion:
    """Represents a genomic region."""
    def __init__(self, chromosome: str, start: int, end: int, strand: Optional[str] = None, name: Optional[str] = None):
        """Initialize a GenomicRegion.

        Args:
            chromosome: Chromosome name (e.g., 'chr1').
            start: Start position (0-based inclusive).
            end: End position (0-based exclusive).
            strand: Optional strand ('+' or '-').
            name: Optional name for the region.

        Raises:
            ValueError: If coordinates are invalid (negative, start > end) or strand is invalid.
        """
        if start < 0 or end < 0 or start > end:
            raise ValueError(f"Invalid genomic coordinates: start={start}, end={end}. Must be non-negative and start <= end.")
        if strand is not None and strand not in ['+', '-']:
            raise ValueError(f"Invalid strand: '{strand}'. Must be '+' or '-'.")

        self.chromosome = str(chromosome)
        self.start = int(start)
        self.end = int(end)
        self.strand = strand
        self.name = name # Optional name for the region

    def __len__(self) -> int:
        """Return the length of the region (end - start)."""
        return self.end - self.start

    def __str__(self) -> str:
        """Return a string representation (e.g., chr1:1000-2000(+))."""
        s = f"{self.chromosome}:{self.start}-{self.end}"
        if self.strand:
            s += f"({self.strand})"
        if self.name:
            s += f" [{self.name}]"
        return s

    def __repr__(self) -> str:
        return f"GenomicRegion(chromosome='{self.chromosome}', start={self.start}, end={self.end}, strand='{self.strand}', name='{self.name}')"

    def __eq__(self, other):
        if not isinstance(other, GenomicRegion):
            return NotImplemented
        return (self.chromosome == other.chromosome and
                self.start == other.start and
                self.end == other.end and
                self.strand == other.strand and
                self.name == other.name)

    def __hash__(self):
        return hash((self.chromosome, self.start, self.end, self.strand, self.name))

    def overlaps(self, other: 'GenomicRegion') -> bool:
        """Check if this region overlaps with another."""
        if self.chromosome != other.chromosome:
            return False
        return max(self.start, other.start) < min(self.end, other.end)

class BaseGenomicData(ABC):
    """Abstract base class for genomic data."""
    def __init__(self, metadata: Optional[Dict[str, Any]] = None):
        self.metadata = metadata or {}
        self._data_loaded = False # Internal flag

    @abstractmethod
    def load_data(self, *args, **kwargs) -> None:
        """Abstract method to load data. Should set self._data_loaded = True on success."""
        pass

    @abstractmethod
    def validate_data(self) -> bool:
        """Abstract method to validate the loaded data. Should check self._data_loaded."""
        pass

    @abstractmethod
    def get_data(self) -> Any:
        """Abstract method to retrieve the core data structure."""
        pass

    def is_loaded(self) -> bool:
        """Check if data has been loaded."""
        return self._data_loaded

    def get_metadata(self, key: Optional[str] = None) -> Union[Dict[str, Any], Any]:
        """Retrieve metadata."""
        if key:
            return self.metadata.get(key)
        return self.metadata

    def update_metadata(self, new_metadata: Dict[str, Any]) -> None:
        """Update metadata."""
        self.metadata.update(new_metadata)

class SequenceData(BaseGenomicData):
    """Represents sequence data, typically loaded from FASTA files.
    Stores sequences as a dictionary: {sequence_id: sequence_string}.
    """
    def __init__(self, sequences: Optional[Dict[str, str]] = None, metadata: Optional[Dict[str, Any]] = None):
        """Initialize SequenceData.

        Args:
            sequences: Optional dictionary of sequence IDs to sequence strings.
                       If provided, data is considered loaded.
            metadata: Optional dictionary for metadata.
        """
        super().__init__(metadata)
        self.sequences: Dict[str, str] = sequences or {}
        if self.sequences:
            self._data_loaded = True

    def load_data(self, file_path: str, file_format: str = 'fasta') -> None:
        """Load sequence data from a file.

        Currently supports FASTA format using Biopython. Other formats can be added.

        Args:
            file_path: Path to the sequence file.
            file_format: Format of the file (default: 'fasta'). Case-insensitive.

        Raises:
            Prints error messages to console on ImportError (Biopython missing),
            FileNotFoundError, or other exceptions during parsing. Sets data
            as not loaded on failure.
        """
        self.sequences = {} # Reset sequences
        self._data_loaded = False
        if file_format.lower() == 'fasta':
            try:
                self.sequences = {record.id: str(record.seq) for record in SeqIO.parse(file_path, "fasta")}
                self.update_metadata({'source_file': file_path, 'format': 'fasta', 'count': len(self.sequences)})
                self._data_loaded = True
                print(f"Loaded {len(self.sequences)} sequences from {file_path}")
            except ImportError:
                print("Error: Biopython is required to load FASTA files. Install with 'pip install biopython'")
            except FileNotFoundError:
                 print(f"Error: FASTA file not found at {file_path}")
            except Exception as e:
                print(f"Error loading FASTA file {file_path}: {e}")
        else:
            print(f"Unsupported sequence file format: {file_format}")
            # Add support for other formats if needed

    def validate_data(self) -> bool:
        """Validate sequence data (checks for valid characters)."""
        if not self.is_loaded():
            print("Error: Sequence data not loaded.")
            return False
        if not self.sequences:
            warnings.warn("Sequence data is empty.")
            return True # Empty is valid, but maybe warn

        valid_chars = set("ATCGN") # Case-insensitive check below
        valid = True
        for seq_id, seq in self.sequences.items():
            if not seq:
                 warnings.warn(f"Sequence {seq_id} is empty.")
                 continue # Allow empty sequences?
            if not set(seq.upper()).issubset(valid_chars):
                warnings.warn(f"Sequence {seq_id} contains invalid characters.")
                valid = False # Or change behavior based on strictness
        return valid

    def get_data(self) -> Dict[str, str]:
        """Retrieve the dictionary of sequence IDs to sequences."""
        if not self.is_loaded():
            warnings.warn("Attempting to get data before loading.")
        return self.sequences

    def get_sequence(self, seq_id: str) -> Optional[str]:
        """Get a specific sequence by ID."""
        return self.sequences.get(seq_id)

    def get_sequence_length(self, seq_id: str) -> Optional[int]:
        """Get the length of a specific sequence."""
        seq = self.get_sequence(seq_id)
        return len(seq) if seq is not None else None

class MethylationData(BaseGenomicData):
    """Represents methylation data, typically loaded from tabular formats.
    Stores data in a pandas DataFrame. Expects columns like:
    'chromosome', 'position', 'methylation_level' (0-1).
    Optional columns: 'strand', 'coverage', 'count_methylated', 'count_unmethylated'.
    """
    def __init__(self, methylation_levels: Optional[pd.DataFrame] = None, metadata: Optional[Dict[str, Any]] = None):
        """Initialize MethylationData.

        Args:
            methylation_levels: Optional pandas DataFrame containing methylation data.
                                If provided, data is considered loaded.
            metadata: Optional dictionary for metadata.
        """
        super().__init__(metadata)
        self.methylation_levels: Optional[pd.DataFrame] = methylation_levels
        if self.methylation_levels is not None:
            self._data_loaded = True

    def load_data(self, file_path: str, file_format: str = 'bedgraph', required_cols: Optional[List[str]] = None, **kwargs) -> None:
        """Load methylation data from a file.

        Supports formats like bedGraph and Bismark coverage files.

        Args:
            file_path: Path to the methylation data file.
            file_format: Format of the file (default: 'bedgraph'). Case-insensitive.
                         Supported: 'bedgraph', 'bismark_cov'.
            required_cols: Optional list of columns considered essential after loading.
                           Defaults to ['chromosome', 'position', 'methylation_level'].
                           Validation will fail if these are missing.
            **kwargs: Additional keyword arguments passed directly to pandas.read_csv
                      (e.g., `sep=' ', compression='gzip'`).

        Raises:
            Prints error messages to console on FileNotFoundError or other exceptions
            during parsing. Sets data as not loaded on failure.
        """
        self.methylation_levels = None # Reset
        self._data_loaded = False
        default_required = ['chromosome', 'position', 'methylation_level']
        required_cols = required_cols or default_required

        try:
            if file_format.lower() == 'bedgraph':
                 # Simple bedGraph (chr, start, end, value) -> chr, pos, level
                 df = pd.read_csv(file_path, sep='\t', header=None, comment='#', low_memory=False,
                                  names=['chromosome', 'start', 'end', 'methylation_level'], **kwargs)
                 df['position'] = df['start'] # Assume position is start
                 self.methylation_levels = df[['chromosome', 'position', 'methylation_level']]

            elif file_format.lower() == 'bismark_cov':
                # Bismark coverage file (chr, start, end, percentage, count_methylated, count_unmethylated)
                df = pd.read_csv(file_path, sep='\t', header=None, comment='#', low_memory=False,
                                 names=['chromosome', 'start', 'end', 'percentage', 'count_methylated', 'count_unmethylated'], **kwargs)
                df['position'] = df['start'] # Bismark output is 1-based, but often treated as 0-based start position
                df['methylation_level'] = df['percentage'] / 100.0
                df['coverage'] = df['count_methylated'] + df['count_unmethylated']
                self.methylation_levels = df[['chromosome', 'position', 'methylation_level', 'coverage']] # Keep coverage if available

            # Add parsers for other relevant formats here
            else:
                print(f"Unsupported methylation file format: {file_format}")
                return

            self.update_metadata({'source_file': file_path, 'format': file_format, 'records': len(self.methylation_levels)})
            self._data_loaded = True
            print(f"Loaded {len(self.methylation_levels)} records from {file_path}")

        except FileNotFoundError:
            print(f"Error: Methylation data file not found at {file_path}")
        except Exception as e:
            print(f"Error loading methylation data from {file_path} (format: {file_format}): {e}")
            self.methylation_levels = None

    def validate_data(self) -> bool:
        """Validate methylation data (checks columns, types, ranges)."""
        if not self.is_loaded():
            print("Error: Methylation data not loaded.")
            return False
        if self.methylation_levels is None or self.methylation_levels.empty:
            warnings.warn("Methylation data is empty or failed to load.")
            return True # Technically valid if empty

        df = self.methylation_levels
        required_cols = ['chromosome', 'position', 'methylation_level']
        valid = True

        # Check required columns exist
        missing_cols = [col for col in required_cols if col not in df.columns]
        if missing_cols:
            print(f"Error: Methylation data missing required columns: {missing_cols}")
            return False # Hard fail if core columns missing

        # Check types
        if not pd.api.types.is_numeric_dtype(df['position']):
            print("Error: 'position' column must be numeric.")
            valid = False
        if not pd.api.types.is_numeric_dtype(df['methylation_level']):
            print("Error: 'methylation_level' column must be numeric.")
            valid = False
        if 'coverage' in df.columns and not pd.api.types.is_numeric_dtype(df['coverage']):
            warnings.warn("'coverage' column exists but is not numeric.")
            # Don't fail validation for optional column type issue, just warn

        # Check ranges
        if 'position' in df.columns and pd.api.types.is_numeric_dtype(df['position']) and df['position'].min() < 0:
            warnings.warn("Methylation data contains negative 'position' values.")
            # valid = False # Decide if this is critical
        if 'methylation_level' in df.columns and pd.api.types.is_numeric_dtype(df['methylation_level']):
            if not ((df['methylation_level'] >= 0) & (df['methylation_level'] <= 1)).all():
                warnings.warn("Methylation levels outside the expected [0, 1] range detected.")
                # valid = False # Decide if this is critical

        return valid

    def get_data(self) -> Optional[pd.DataFrame]:
        """Retrieve the methylation data as a pandas DataFrame."""
        if not self.is_loaded():
            warnings.warn("Attempting to get data before loading.")
        return self.methylation_levels

    def get_methylation_at(self, region: GenomicRegion) -> Optional[pd.DataFrame]:
        """Get methylation levels within a specific genomic region.

        Filters the internal DataFrame based on the provided GenomicRegion's
        chromosome, start (inclusive), and end (exclusive) coordinates.

        Args:
            region: The GenomicRegion object specifying the area of interest.

        Returns:
            A pandas DataFrame containing methylation records within the specified region,
            or None if the data is not loaded, invalid, required columns are missing,
            or an unexpected error occurs during filtering. Returns an empty DataFrame
            if the region is valid but contains no data points.
        """
        if not self.validate_data() or self.methylation_levels is None:
             warnings.warn(f"Cannot get methylation for {region}: Data not loaded or invalid.")
             return None
        try:
            mask = (
                (self.methylation_levels['chromosome'] == region.chromosome) &
                (self.methylation_levels['position'] >= region.start) &
                (self.methylation_levels['position'] < region.end) # Assume position is 0-based start
            )
            return self.methylation_levels.loc[mask] # Use .loc for potentially better performance/clarity
        except KeyError:
             print(f"Error: Required columns missing for region query ({region}).")
             return None
        except Exception as e:
            print(f"Unexpected error getting methylation for region {region}: {e}")
            return None

class HistoneMarkData(BaseGenomicData):
    """Represents histone mark data, typically signal values loaded from BigWig files.
    Stores a handle to the opened BigWig file using pyBigWig.
    """
    def __init__(self, file_path: Optional[str] = None, metadata: Optional[Dict[str, Any]] = None):
        """Initialize HistoneMarkData.

        Args:
            file_path: Optional path to the BigWig file. If provided, attempts
                       to load the file immediately.
            metadata: Optional dictionary for metadata.
        """
        super().__init__(metadata)
        self.signal_file_handle = None # pyBigWig file handle
        self.file_path = file_path
        if file_path:
            self.load_data(file_path) # Optionally load on init

    def load_data(self, file_path: str, file_format: str = 'bigwig', **kwargs) -> None:
        """Load histone mark data from a file (currently supports BigWig).

        Opens the file using pyBigWig and stores the handle.

        Args:
            file_path: Path to the BigWig file.
            file_format: Format of the file (default: 'bigwig'). Case-insensitive.
            **kwargs: Additional keyword arguments passed directly to pyBigWig.open().

        Raises:
            Prints error messages to console on ImportError (pyBigWig missing),
            FileNotFoundError, RuntimeError (often from pyBigWig), or other
            exceptions during file opening. Sets data as not loaded on failure.
        """
        self.close() # Close previous file if any
        self._data_loaded = False
        self.file_path = file_path

        if file_format.lower() == 'bigwig':
            try:
                import pyBigWig
                self.signal_file_handle = pyBigWig.open(file_path, **kwargs)
                chroms = self.signal_file_handle.chroms()
                self.update_metadata({
                    'source_file': file_path,
                    'format': 'bigwig',
                    'chromosomes': list(chroms.keys()),
                    'total_length': sum(chroms.values())
                })
                self._data_loaded = True
                print(f"Opened BigWig file: {file_path}")
            except ImportError:
                print("Error: pyBigWig is required to load BigWig files. Install with 'pip install pyBigWig'")
            except FileNotFoundError:
                 print(f"Error: BigWig file not found at {file_path}")
            except RuntimeError as e: # pyBigWig often raises RuntimeError for file issues
                 print(f"Error opening BigWig file {file_path}: {e}")
            except Exception as e:
                print(f"Unexpected error opening BigWig file {file_path}: {e}")
            finally:
                 if not self._data_loaded: # Ensure handle is None if loading failed
                     self.signal_file_handle = None
                     self.file_path = None
        else:
            print(f"Unsupported histone mark file format: {file_format}")
            self.file_path = None

    def validate_data(self) -> bool:
        """Validate histone mark data (checks if file handle is valid)."""
        if not self.is_loaded():
            print("Error: Histone mark data not loaded.")
            return False
        if self.signal_file_handle is None:
            print("Error: BigWig file handle is not available.")
            return False
        # Check if it's a valid pyBigWig object (basic check by accessing a method)
        try:
            _ = self.signal_file_handle.chroms()
            return True
        except Exception as e:
             print(f"Error: BigWig file handle appears invalid: {e}")
             return False

    def get_data(self) -> Any:
        """Retrieve the signal data object (pyBigWig file handle)."""
        if not self.is_loaded():
            warnings.warn("Attempting to get data before loading.")
        return self.signal_file_handle

    def get_signal_in_region(self, region: GenomicRegion, summary_type: str = 'mean', n_bins: Optional[int] = None) -> Optional[Union[float, np.ndarray]]:
        """Get signal values within a specific genomic region.

        Args:
            region: The GenomicRegion to query.
            summary_type: Type of summary statistic ('mean', 'min', 'max', 'std', 'coverage').
            n_bins: If provided, returns an array of stats for this many bins. Otherwise, returns a single stat for the region.

        Returns:
            A float (if n_bins is None) or numpy array (if n_bins > 0) with the signal value(s),
            or None if the region is invalid, not found, or an error occurs. Returns 0.0 / array of 0.0
            if the region is valid but contains no signal data.
        """
        valid_summary_types = {'mean', 'min', 'max', 'std', 'coverage'}
        if summary_type not in valid_summary_types:
             print(f"Error: Invalid summary_type '{summary_type}'. Must be one of {valid_summary_types}")
             return None

        if not self.validate_data():
            return None

        handle = self.get_data()
        try:
            # Check chromosome exists in the file
            if region.chromosome not in handle.chroms():
                 # warnings.warn(f"Chromosome {region.chromosome} not found in BigWig file {self.file_path}.")
                 return np.zeros(n_bins) if n_bins else 0.0 # Return zeros/zero if chrom not found

            # Fetch stats
            # Note: pyBigWig uses 0-based, half-open intervals [start, end)
            result = handle.stats(region.chromosome, region.start, region.end, type=summary_type, nBins=n_bins or 1) # nBins=1 if None

            if n_bins:
                # handle.stats returns list of values, Nones where bins have no data
                result_array = np.array(result, dtype=float)
                result_array[np.isnan(result_array)] = 0.0 # Replace None/nan with 0
                return result_array
            else:
                # handle.stats returns a list with one value, or None if region empty
                return result[0] if result is not None and result[0] is not None else 0.0

        except RuntimeError as e:
            # pyBigWig can raise RuntimeError for invalid queries (e.g., start > end after clipping)
            # This can happen if the query region is entirely outside the chromosome bounds known to pyBigWig
            # warnings.warn(f"RuntimeError querying BigWig {self.file_path} for region {region}: {e}. Returning 0.")
            return np.zeros(n_bins) if n_bins else 0.0
        except Exception as e:
            print(f"Unexpected error querying BigWig {self.file_path} for region {region}: {e}")
            return None

    def close(self) -> None:
        """Close the BigWig file handle if it's open."""
        if self.signal_file_handle:
            try:
                self.signal_file_handle.close()
                # print(f"Closed BigWig file: {self.file_path}")
            except Exception as e:
                 print(f"Error closing BigWig file {self.file_path}: {e}")
            finally:
                 self.signal_file_handle = None
                 self._data_loaded = False
                 # Keep self.file_path? Maybe reset if close is meant to fully detach. Let's keep it.

    def __del__(self):
        """Ensure file handle is closed when object is garbage collected."""
        self.close()

# Example Usage (Conceptual) - Updated
# seq_data = SequenceData()
# seq_data.load_data("sequences.fasta")
# if seq_data.validate_data():
#     sequences = seq_data.get_data()
#     print(f"Seq length: {seq_data.get_sequence_length(list(sequences.keys())[0])}")

# meth_data = MethylationData()
# meth_data.load_data("bismark_output.cov.gz", file_format='bismark_cov', compression='gzip')
# if meth_data.validate_data():
#     region = GenomicRegion("chr1", 1000, 2000, name="PromoterX")
#     print(f"Querying region: {region}")
#     meth_in_region = meth_data.get_methylation_at(region)
#     print(meth_in_region.head())

# histone_data = HistoneMarkData("H3K4me3.bw") # Load on init
# if histone_data.validate_data():
#     region = GenomicRegion("chr1", 1000, 2000)
#     signal_mean = histone_data.get_signal_in_region(region, summary_type='mean')
#     binned_signal = histone_data.get_signal_in_region(region, summary_type='mean', n_bins=10)
#     print(f"Mean signal in {region}: {signal_mean}")
#     print(f"Binned signal in {region}: {binned_signal}")
# histone_data.close() # Explicitly close 