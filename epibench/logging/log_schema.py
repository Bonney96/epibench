"""
EpiBench Log Schema Definition

This module defines the JSON schema for the EpiBench logging system,
ensuring comprehensive capture of all metadata required for reproducibility.
"""

from typing import Dict, Any, List, Optional
from datetime import datetime
import json
from pathlib import Path


# JSON Schema Definition for EpiBench Logs
LOG_SCHEMA = {
    "$schema": "http://json-schema.org/draft-07/schema#",
    "title": "EpiBench Run Log",
    "description": "Complete log of an EpiBench pipeline execution for reproducibility",
    "type": "object",
    "required": [
        "log_version",
        "execution_metadata",
        "input_configuration",
        "runtime_information",
        "pipeline_information",
        "output_information"
    ],
    "properties": {
        "log_version": {
            "type": "string",
            "description": "Version of the logging schema",
            "const": "1.0.0"
        },
        "execution_metadata": {
            "type": "object",
            "description": "Metadata about the execution environment and timing",
            "required": ["execution_id", "timestamp_start", "epibench_version"],
            "properties": {
                "execution_id": {
                    "type": "string",
                    "description": "Unique identifier for this execution (UUID)"
                },
                "timestamp_start": {
                    "type": "string",
                    "format": "date-time",
                    "description": "ISO 8601 timestamp when execution started"
                },
                "timestamp_end": {
                    "type": ["string", "null"],
                    "format": "date-time",
                    "description": "ISO 8601 timestamp when execution completed"
                },
                "epibench_version": {
                    "type": "string",
                    "description": "EpiBench package version"
                },
                "epibench_commit_hash": {
                    "type": ["string", "null"],
                    "description": "Git commit hash of EpiBench code"
                },
                "execution_status": {
                    "type": "string",
                    "enum": ["running", "completed", "failed", "interrupted"],
                    "description": "Overall status of the execution"
                }
            }
        },
        "input_configuration": {
            "type": "object",
            "description": "Input files and configuration for the run",
            "required": ["sample_id"],
            "properties": {
                "sample_id": {
                    "type": "string",
                    "description": "Identifier for the sample being processed"
                },
                "regions_file_path": {
                    "type": ["string", "null"],
                    "description": "Path to the genomic regions file"
                },
                "config_files": {
                    "type": "object",
                    "description": "Paths to configuration files used",
                    "properties": {
                        "process_config": {"type": ["string", "null"]},
                        "train_config": {"type": ["string", "null"]},
                        "interpret_config": {"type": ["string", "null"]},
                        "compare_config": {"type": ["string", "null"]},
                        "samples_config": {"type": ["string", "null"]}
                    }
                }
            }
        },
        "runtime_information": {
            "type": "object",
            "description": "Runtime environment and performance information",
            "properties": {
                "duration_seconds": {
                    "type": ["number", "null"],
                    "description": "Total execution time in seconds"
                },
                "compute_environment": {
                    "type": "string",
                    "enum": ["compute1", "compute2", "local", "cloud", "other"],
                    "description": "Compute environment identifier"
                },
                "hardware_specs": {
                    "type": "object",
                    "description": "Hardware specifications",
                    "properties": {
                        "cpu_info": {
                            "type": "object",
                            "properties": {
                                "model": {"type": "string"},
                                "cores": {"type": "integer"},
                                "threads": {"type": "integer"}
                            }
                        },
                        "memory_gb": {"type": "number"},
                        "gpu_info": {
                            "type": ["array", "null"],
                            "items": {
                                "type": "object",
                                "properties": {
                                    "name": {"type": "string"},
                                    "memory_gb": {"type": "number"}
                                }
                            }
                        }
                    }
                },
                "software_environment": {
                    "type": "object",
                    "description": "Software environment details",
                    "properties": {
                        "os_info": {
                            "type": "object",
                            "properties": {
                                "system": {"type": "string"},
                                "release": {"type": "string"},
                                "version": {"type": "string"}
                            }
                        },
                        "python_version": {"type": "string"},
                        "cuda_version": {"type": ["string", "null"]},
                        "key_packages": {
                            "type": "object",
                            "description": "Versions of key Python packages",
                            "additionalProperties": {"type": "string"}
                        }
                    }
                }
            }
        },
        "pipeline_information": {
            "type": "object",
            "description": "Information about pipeline execution",
            "properties": {
                "command_line_args": {
                    "type": "array",
                    "items": {"type": "string"},
                    "description": "Command-line arguments used to start the pipeline"
                },
                "pipeline_stages": {
                    "type": "array",
                    "description": "Stages executed in the pipeline",
                    "items": {
                        "type": "object",
                        "properties": {
                            "name": {"type": "string"},
                            "status": {
                                "type": "string",
                                "enum": ["pending", "running", "completed", "failed", "skipped"]
                            },
                            "start_time": {
                                "type": ["string", "null"],
                                "format": "date-time"
                            },
                            "end_time": {
                                "type": ["string", "null"],
                                "format": "date-time"
                            },
                            "duration_seconds": {"type": ["number", "null"]},
                            "error_message": {"type": ["string", "null"]}
                        }
                    }
                },
                "checkpoint_data": {
                    "type": ["object", "null"],
                    "description": "Checkpoint information from pipeline execution"
                }
            }
        },
        "output_information": {
            "type": "object",
            "description": "Output paths and locations",
            "required": ["base_output_directory"],
            "properties": {
                "base_output_directory": {
                    "type": "string",
                    "description": "Root directory for all outputs"
                },
                "output_paths": {
                    "type": "object",
                    "properties": {
                        "processed_data": {"type": ["string", "null"]},
                        "training_output": {"type": ["string", "null"]},
                        "evaluation_output": {"type": ["string", "null"]},
                        "prediction_output": {"type": ["string", "null"]},
                        "interpretation_output": {"type": ["string", "null"]},
                        "temp_configs": {"type": ["string", "null"]}
                    }
                }
            }
        },
        "performance_metrics": {
            "type": ["object", "null"],
            "description": "Model performance metrics from evaluation",
            "properties": {
                "mse": {"type": ["number", "null"]},
                "r_squared": {"type": ["number", "null"]},
                "mae": {"type": ["number", "null"]},
                "pearson_correlation": {"type": ["number", "null"]},
                "spearman_correlation": {"type": ["number", "null"]},
                "additional_metrics": {
                    "type": "object",
                    "description": "Additional model-specific metrics",
                    "additionalProperties": true
                }
            }
        },
        "configuration_parameters": {
            "type": ["object", "null"],
            "description": "Aggregated configuration parameters from all sources",
            "properties": {
                "effective_config": {
                    "type": "object",
                    "description": "Merged configuration from all sources",
                    "additionalProperties": true
                },
                "temp_configs_content": {
                    "type": "object",
                    "description": "Content of temporary configuration files",
                    "additionalProperties": {
                        "type": "object",
                        "description": "Configuration content by filename"
                    }
                }
            }
        },
        "error_information": {
            "type": ["object", "null"],
            "description": "Error details if execution failed",
            "properties": {
                "error_type": {"type": "string"},
                "error_message": {"type": "string"},
                "error_traceback": {"type": ["string", "null"]},
                "failed_stage": {"type": ["string", "null"]}
            }
        },
        "custom_metadata": {
            "type": ["object", "null"],
            "description": "Additional custom metadata",
            "additionalProperties": true
        }
    }
}


class LogSchema:
    """Helper class for working with the EpiBench log schema."""
    
    @staticmethod
    def get_schema() -> Dict[str, Any]:
        """Return the JSON schema for EpiBench logs."""
        return LOG_SCHEMA
    
    @staticmethod
    def create_empty_log() -> Dict[str, Any]:
        """Create an empty log structure with required fields."""
        return {
            "log_version": "1.0.0",
            "execution_metadata": {
                "execution_id": None,
                "timestamp_start": None,
                "timestamp_end": None,
                "epibench_version": None,
                "epibench_commit_hash": None,
                "execution_status": "running"
            },
            "input_configuration": {
                "sample_id": None,
                "regions_file_path": None,
                "config_files": {}
            },
            "runtime_information": {
                "duration_seconds": None,
                "compute_environment": None,
                "hardware_specs": {},
                "software_environment": {}
            },
            "pipeline_information": {
                "command_line_args": [],
                "pipeline_stages": [],
                "checkpoint_data": None
            },
            "output_information": {
                "base_output_directory": None,
                "output_paths": {}
            },
            "performance_metrics": None,
            "configuration_parameters": None,
            "error_information": None,
            "custom_metadata": None
        }
    
    @staticmethod
    def validate_log(log_data: Dict[str, Any]) -> tuple[bool, Optional[str]]:
        """
        Validate a log against the schema.
        
        Returns:
            tuple: (is_valid, error_message)
        """
        try:
            import jsonschema
            jsonschema.validate(instance=log_data, schema=LOG_SCHEMA)
            return True, None
        except jsonschema.ValidationError as e:
            return False, str(e)
        except ImportError:
            # Fallback to basic validation if jsonschema not available
            required_fields = LOG_SCHEMA["required"]
            for field in required_fields:
                if field not in log_data:
                    return False, f"Missing required field: {field}"
            return True, None
    
    @staticmethod
    def format_timestamp(dt: Optional[datetime] = None) -> str:
        """Format a datetime object as ISO 8601 string."""
        if dt is None:
            dt = datetime.now()
        return dt.isoformat() 