"""
EpiBench Log Manager

Core logging infrastructure for tracking EpiBench pipeline executions
with robust file-based storage and minimal performance impact.
"""

import json
import logging
import os
import platform
import subprocess
import sys
import tempfile
import threading
import time
import uuid
from datetime import datetime
from pathlib import Path
from typing import Dict, Any, Optional, List, Union

import psutil
import numpy as np

from .log_schema import LogSchema

logger = logging.getLogger(__name__)


class NumpyJSONEncoder(json.JSONEncoder):
    """Custom JSON encoder that handles numpy types and other special cases."""
    
    def default(self, obj):
        if isinstance(obj, np.integer):
            return int(obj)
        elif isinstance(obj, np.floating):
            if np.isnan(obj) or np.isinf(obj):
                return None
            return float(obj)
        elif isinstance(obj, np.ndarray):
            return obj.tolist()
        elif isinstance(obj, np.bool_):
            return bool(obj)
        elif isinstance(obj, datetime):
            return obj.isoformat()
        elif isinstance(obj, Path):
            return str(obj)
        return super(NumpyJSONEncoder, self).default(obj)


class LogManager:
    """
    Manages EpiBench execution logs with thread-safe operations,
    atomic writes, and minimal performance impact.
    """
    
    def __init__(self, log_directory: Union[str, Path], enable_backup: bool = True):
        """
        Initialize the LogManager.
        
        Args:
            log_directory: Directory where logs will be stored
            enable_backup: Whether to create backups of existing logs
        """
        self.log_directory = Path(log_directory)
        self.log_directory.mkdir(parents=True, exist_ok=True)
        self.enable_backup = enable_backup
        self._lock = threading.Lock()
        self._current_log = None
        self._log_file_path = None
        self._buffer = []
        self._buffer_size = 10  # Flush buffer after this many updates
        
    def create_log(self, sample_id: str, base_output_directory: Union[str, Path]) -> str:
        """
        Create a new log for a pipeline execution.
        
        Args:
            sample_id: Identifier for the sample being processed
            base_output_directory: Root directory for pipeline outputs
            
        Returns:
            execution_id: Unique identifier for this execution
        """
        with self._lock:
            # Generate unique execution ID
            execution_id = str(uuid.uuid4())
            
            # Create log structure
            self._current_log = LogSchema.create_empty_log()
            
            # Fill in initial metadata
            self._current_log["execution_metadata"]["execution_id"] = execution_id
            self._current_log["execution_metadata"]["timestamp_start"] = LogSchema.format_timestamp()
            self._current_log["execution_metadata"]["epibench_version"] = self._get_epibench_version()
            self._current_log["execution_metadata"]["epibench_commit_hash"] = self._get_git_commit_hash()
            
            # Input configuration
            self._current_log["input_configuration"]["sample_id"] = sample_id
            
            # Output information
            self._current_log["output_information"]["base_output_directory"] = str(base_output_directory)
            
            # Runtime information
            self._current_log["runtime_information"]["hardware_specs"] = self._collect_hardware_info()
            self._current_log["runtime_information"]["software_environment"] = self._collect_software_info()
            self._current_log["runtime_information"]["compute_environment"] = self._detect_compute_environment()
            
            # Pipeline information
            self._current_log["pipeline_information"]["command_line_args"] = sys.argv
            
            # Set log file path
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            log_filename = f"epibench_log_{sample_id}_{timestamp}_{execution_id[:8]}.json"
            self._log_file_path = self.log_directory / log_filename
            
            # Save initial log
            self._save_log_atomic()
            
            logger.info(f"Created new log for sample {sample_id} with execution ID {execution_id}")
            return execution_id
    
    def update_log(self, updates: Dict[str, Any], immediate_save: bool = False) -> None:
        """
        Update the current log with new information.
        
        Args:
            updates: Dictionary of updates to apply to the log
            immediate_save: Whether to save immediately or buffer the update
        """
        if self._current_log is None:
            logger.warning("No active log to update")
            return
            
        with self._lock:
            # Apply updates to the log
            self._apply_updates(self._current_log, updates)
            
            if immediate_save:
                self._save_log_atomic()
            else:
                # Buffer the update for batch saving
                self._buffer.append(updates)
                if len(self._buffer) >= self._buffer_size:
                    self._save_log_atomic()
                    self._buffer.clear()
    
    def finalize_log(self, status: str = "completed", error_info: Optional[Dict[str, Any]] = None) -> None:
        """
        Finalize the current log at the end of execution.
        
        Args:
            status: Final execution status
            error_info: Error information if execution failed
        """
        if self._current_log is None:
            logger.warning("No active log to finalize")
            return
            
        with self._lock:
            # Update execution metadata
            self._current_log["execution_metadata"]["timestamp_end"] = LogSchema.format_timestamp()
            self._current_log["execution_metadata"]["execution_status"] = status
            
            # Calculate duration
            start_time = datetime.fromisoformat(self._current_log["execution_metadata"]["timestamp_start"])
            end_time = datetime.fromisoformat(self._current_log["execution_metadata"]["timestamp_end"])
            duration = (end_time - start_time).total_seconds()
            self._current_log["runtime_information"]["duration_seconds"] = duration
            
            # Add error information if provided
            if error_info:
                self._current_log["error_information"] = error_info
            
            # Save final log
            self._save_log_atomic()
            logger.info(f"Finalized log with status: {status}")
    
    def load_log(self, log_path: Union[str, Path]) -> Dict[str, Any]:
        """
        Load an existing log from disk.
        
        Args:
            log_path: Path to the log file
            
        Returns:
            Log data as dictionary
        """
        log_path = Path(log_path)
        if not log_path.exists():
            raise FileNotFoundError(f"Log file not found: {log_path}")
            
        with open(log_path, 'r') as f:
            log_data = json.load(f)
            
        # Validate the loaded log
        is_valid, error_msg = LogSchema.validate_log(log_data)
        if not is_valid:
            logger.warning(f"Loaded log failed validation: {error_msg}")
            
        return log_data
    
    def _save_log_atomic(self) -> None:
        """Save the log atomically to prevent corruption."""
        if self._current_log is None or self._log_file_path is None:
            return
            
        # Create backup if enabled and file exists
        if self.enable_backup and self._log_file_path.exists():
            self._create_backup()
        
        # Write to temporary file first
        temp_fd, temp_path = tempfile.mkstemp(
            dir=self._log_file_path.parent,
            prefix=f".{self._log_file_path.stem}_",
            suffix=".tmp"
        )
        
        try:
            # Write log data
            with os.fdopen(temp_fd, 'w') as f:
                json.dump(self._current_log, f, indent=2, cls=NumpyJSONEncoder)
            
            # Atomic rename
            os.replace(temp_path, self._log_file_path)
            
        except Exception as e:
            # Clean up temporary file on error
            if os.path.exists(temp_path):
                os.unlink(temp_path)
            logger.error(f"Failed to save log: {e}")
            raise
    
    def _create_backup(self) -> None:
        """Create a backup of the existing log file."""
        if not self._log_file_path.exists():
            return
            
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        backup_path = self._log_file_path.with_suffix(f".{timestamp}.backup")
        
        try:
            self._log_file_path.rename(backup_path)
            logger.debug(f"Created backup: {backup_path}")
        except Exception as e:
            logger.error(f"Failed to create backup: {e}")
    
    def _apply_updates(self, target: Dict[str, Any], updates: Dict[str, Any]) -> None:
        """Recursively apply updates to a dictionary."""
        for key, value in updates.items():
            if key in target and isinstance(target[key], dict) and isinstance(value, dict):
                # Recursive update for nested dictionaries
                self._apply_updates(target[key], value)
            else:
                # Direct update
                target[key] = value
    
    def _get_epibench_version(self) -> str:
        """Get the installed EpiBench version."""
        try:
            import epibench
            return getattr(epibench, '__version__', 'unknown')
        except:
            return 'unknown'
    
    def _get_git_commit_hash(self) -> Optional[str]:
        """Get the current git commit hash."""
        try:
            result = subprocess.run(
                ['git', 'rev-parse', 'HEAD'],
                capture_output=True,
                text=True,
                check=True
            )
            return result.stdout.strip()
        except:
            return None
    
    def _detect_compute_environment(self) -> str:
        """Detect the compute environment."""
        # Check for common HPC/cloud indicators
        if os.environ.get('SLURM_JOB_ID'):
            return 'compute1'  # SLURM cluster
        elif os.environ.get('PBS_JOBID'):
            return 'compute2'  # PBS cluster
        elif os.environ.get('AWS_EXECUTION_ENV'):
            return 'cloud'  # AWS
        else:
            return 'local'
    
    def _collect_hardware_info(self) -> Dict[str, Any]:
        """Collect hardware information."""
        info = {}
        
        # CPU information
        try:
            info['cpu_info'] = {
                'model': platform.processor() or 'unknown',
                'cores': psutil.cpu_count(logical=False),
                'threads': psutil.cpu_count(logical=True)
            }
        except:
            info['cpu_info'] = {'error': 'Failed to collect CPU info'}
        
        # Memory information
        try:
            mem = psutil.virtual_memory()
            info['memory_gb'] = round(mem.total / (1024**3), 2)
        except:
            info['memory_gb'] = None
        
        # GPU information
        info['gpu_info'] = self._collect_gpu_info()
        
        return info
    
    def _collect_gpu_info(self) -> Optional[List[Dict[str, Any]]]:
        """Collect GPU information if available."""
        gpus = []
        
        # Try nvidia-smi
        try:
            result = subprocess.run(
                ['nvidia-smi', '--query-gpu=name,memory.total', '--format=csv,noheader,nounits'],
                capture_output=True,
                text=True,
                check=True
            )
            for line in result.stdout.strip().split('\n'):
                if line:
                    parts = line.split(', ')
                    if len(parts) >= 2:
                        gpus.append({
                            'name': parts[0],
                            'memory_gb': round(float(parts[1]) / 1024, 2)
                        })
        except:
            pass
        
        # Try PyTorch
        if not gpus:
            try:
                import torch
                if torch.cuda.is_available():
                    for i in range(torch.cuda.device_count()):
                        props = torch.cuda.get_device_properties(i)
                        gpus.append({
                            'name': props.name,
                            'memory_gb': round(props.total_memory / (1024**3), 2)
                        })
            except:
                pass
        
        return gpus if gpus else None
    
    def _collect_software_info(self) -> Dict[str, Any]:
        """Collect software environment information."""
        info = {}
        
        # OS information
        info['os_info'] = {
            'system': platform.system(),
            'release': platform.release(),
            'version': platform.version()
        }
        
        # Python version
        info['python_version'] = platform.python_version()
        
        # CUDA version
        info['cuda_version'] = self._get_cuda_version()
        
        # Key packages
        info['key_packages'] = self._get_key_package_versions()
        
        return info
    
    def _get_cuda_version(self) -> Optional[str]:
        """Get CUDA version if available."""
        try:
            import torch
            if torch.cuda.is_available():
                return torch.version.cuda
        except:
            pass
        return None
    
    def _get_key_package_versions(self) -> Dict[str, str]:
        """Get versions of key packages."""
        packages = {}
        key_packages = [
            'numpy', 'pandas', 'torch', 'tensorflow', 
            'scikit-learn', 'scipy', 'h5py', 'yaml'
        ]
        
        for package in key_packages:
            try:
                module = __import__(package)
                version = getattr(module, '__version__', 'unknown')
                packages[package] = version
            except ImportError:
                pass
        
        return packages 