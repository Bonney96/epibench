import logging
import sys
import os
from typing import Optional

from epibench.config.config_manager import ConfigManager # Import ConfigManager

class LoggerManager:
    """Manages the configuration of the logging system based on ConfigManager."""

    _is_configured = False # Class variable to track if logger is already configured

    @staticmethod
    def setup_logger(config_manager: Optional[ConfigManager] = None, 
                       default_log_level: int = logging.INFO,
                       default_log_file: Optional[str] = None,
                       default_log_format: str = '%(asctime)s - %(levelname)s - %(name)s - %(message)s',
                       force_reconfigure: bool = False):
        """Configures the root logger based on ConfigManager settings or defaults.

        Args:
            config_manager: An instance of ConfigManager to read settings from.
            default_log_level: Default logging level if not specified in config.
            default_log_file: Default log file path if not specified in config.
            default_log_format: Default log format string if not specified in config.
            force_reconfigure: If True, removes existing handlers and reconfigures.
        """
        if LoggerManager._is_configured and not force_reconfigure:
             logging.debug("Logger already configured. Skipping reconfiguration.")
             return

        # --- Determine configuration source --- 
        log_level = default_log_level
        log_file = default_log_file
        log_format_str = default_log_format

        if config_manager:
            try:
                # Use get_nested for potentially nested config keys
                level_name = config_manager.get_nested('logging.level', logging.getLevelName(default_log_level)).upper()
                log_level = getattr(logging, level_name, default_log_level)
                
                log_file = config_manager.get_nested('logging.file', default_log_file)
                log_format_str = config_manager.get_nested('logging.format', default_log_format)
                
                logging.debug(f"Loaded logging config: Level={logging.getLevelName(log_level)}, File={log_file}, Format='{log_format_str}'")
            except Exception as e:
                logging.error(f"Error reading logging configuration: {e}. Using defaults.", exc_info=True)
                # Revert to defaults if config reading fails
                log_level = default_log_level
                log_file = default_log_file
                log_format_str = default_log_format
        else:
            logging.debug("No ConfigManager provided. Using default logging settings.")

        # Get the root logger
        logger = logging.getLogger() # Get root logger
        logger.setLevel(log_level) # Set the base level for the root logger

        # Remove existing handlers if reconfiguring or first time
        if force_reconfigure or not LoggerManager._is_configured:
            for handler in logger.handlers[:]:
                try:
                    handler.close()
                except Exception:
                    pass 
                logger.removeHandler(handler)
            logging.info("Removed existing logging handlers.") # Use root logger to log this

        # Define format
        try:
            log_format = logging.Formatter(log_format_str)
        except Exception as e:
             logging.error(f"Invalid log format string '{log_format_str}': {e}. Using default format.")
             log_format = logging.Formatter(default_log_format) # Fallback to default format

        # --- Console Handler --- 
        console_handler = logging.StreamHandler(sys.stdout) # Log to stdout
        console_handler.setLevel(log_level) # Capture specified level and above
        console_handler.setFormatter(log_format)
        logger.addHandler(console_handler)

        # --- File Handler (optional) --- 
        if log_file:
            try:
                log_dir = os.path.dirname(log_file)
                if log_dir:
                    os.makedirs(log_dir, exist_ok=True)
                
                file_handler = logging.FileHandler(log_file, mode='a') # Append mode
                file_handler.setLevel(log_level) # Capture specified level and above
                file_handler.setFormatter(log_format)
                logger.addHandler(file_handler)
                log_message = f"Logging configured. Level: {logging.getLevelName(log_level)}. Log file: {log_file}"
            except Exception as e:
                logger.error(f"Failed to set up log file handler at {log_file}: {e}", exc_info=True)
                log_message = f"Logging configured. Level: {logging.getLevelName(log_level)}. Outputting to console only (file log failed)."
        else:
             log_message = f"Logging configured. Level: {logging.getLevelName(log_level)}. Outputting to console only."

        # Log the final configuration status (using the root logger itself)
        logger.info(log_message)
        LoggerManager._is_configured = True

# Example usage:
# if __name__ == '__main__':
#     # Basic configuration
#     LoggerManager.setup_logger(log_level=logging.DEBUG, log_file='app.log')
# 
#     # Get specific loggers for different modules
#     module_logger = logging.getLogger('my_module')
#     another_logger = logging.getLogger('another.module')
# 
#     module_logger.debug("This is a debug message from my_module.")
#     module_logger.info("This is an info message from my_module.")
#     another_logger.warning("This is a warning from another.module.")
#     logging.error("This is a root error message.") 
# 
#     # Try reconfiguring (will be skipped unless force_reconfigure=True)
#     LoggerManager.setup_logger(log_level=logging.INFO) 
# 
#     # Force reconfigure
#     LoggerManager.setup_logger(log_level=logging.INFO, force_reconfigure=True, log_file='app_reconfigured.log')
#     module_logger.info("Info message after reconfiguration.") 