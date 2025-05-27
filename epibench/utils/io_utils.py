# epibench/utils/io_utils.py

import os
import logging
import json
import pandas as pd
from typing import Any, Dict, Union

logger = logging.getLogger(__name__)

def ensure_dir(dir_path: str):
    """Ensures that a directory exists, creating it if necessary.

    Args:
        dir_path (str): The path to the directory.
    """
    if dir_path: # Only proceed if the path is not empty
        try:
            os.makedirs(dir_path, exist_ok=True)
            logger.debug(f"Ensured directory exists: {dir_path}")
        except OSError as e:
            logger.error(f"Error creating directory {dir_path}: {e}", exc_info=True)
            raise # Re-raise the exception as this is often critical

def load_predictions(file_path: str) -> Union[pd.DataFrame, Any]:
    """Loads predictions from a file (e.g., CSV).

    Placeholder implementation - adjust based on actual prediction format.

    Args:
        file_path (str): Path to the prediction file.

    Returns:
        Union[pd.DataFrame, Any]: Loaded predictions (e.g., a DataFrame).
                                  Returns None if loading fails.
    """
    logger.info(f"Loading predictions from: {file_path} (Placeholder Implementation)")
    try:
        # Assuming CSV format for now
        if file_path.lower().endswith('.csv'):
            return pd.read_csv(file_path)
        # Add support for other formats like .parquet, .npy, .txt if needed
        # elif file_path.lower().endswith('.parquet'):
        #     return pd.read_parquet(file_path)
        else:
            logger.warning(f"Unsupported prediction file format for loading: {file_path}. Assuming CSV.")
            # Attempt CSV read as a fallback, might fail
            return pd.read_csv(file_path) 

    except FileNotFoundError:
        logger.error(f"Prediction file not found: {file_path}")
        raise # Re-raise as it's likely a critical error
    except Exception as e:
        logger.error(f"Error loading predictions from {file_path}: {e}", exc_info=True)
        return None # Return None or re-raise depending on desired error handling

def save_results(results: Dict[str, Any], file_path: str):
    """Saves results dictionary to a file (e.g., JSON).

    Args:
        results (Dict[str, Any]): Dictionary containing the results to save.
        file_path (str): Path to the output file.
    """
    logger.info(f"Saving results to: {file_path}")
    ensure_dir(os.path.dirname(file_path))
    
    try:
        if file_path.lower().endswith('.json'):
            with open(file_path, 'w') as f:
                # Handle potential numpy types for JSON serialization
                def convert_numpy(obj):
                     if isinstance(obj, np.integer):
                         return int(obj)
                     elif isinstance(obj, np.floating):
                         return float(obj)
                     elif isinstance(obj, np.ndarray):
                         return obj.tolist()
                     # Add other type conversions if needed
                     return obj # Or raise TypeError for unsupported types
                     
                json.dump(results, f, indent=4, default=convert_numpy)
            logger.info("Results saved successfully as JSON.")
        elif file_path.lower().endswith('.csv'):
             # Requires results to be easily convertible to a DataFrame
             try:
                 df = pd.DataFrame.from_dict(results, orient='index') # Example conversion
                 df.to_csv(file_path)
                 logger.info("Results saved successfully as CSV.")
             except Exception as df_e:
                 logger.error(f"Could not convert results to DataFrame for CSV saving: {df_e}")
                 logger.warning(f"Falling back to saving results as JSON: {file_path}.json")
                 save_results(results, file_path + ".json") # Retry as JSON
        else:
            logger.warning(f"Unsupported file extension for saving results: {file_path}. Saving as JSON.")
            save_results(results, file_path + ".json") # Save as JSON by default
            
    except TypeError as te:
         logger.error(f"Type error saving results to {file_path} (possibly non-serializable type like numpy): {te}")
         logger.error("Consider implementing custom JSON serialization for numpy types if needed.")
         raise # Re-raise
    except Exception as e:
        logger.error(f"Error saving results to {file_path}: {e}", exc_info=True)
        raise # Re-raise 