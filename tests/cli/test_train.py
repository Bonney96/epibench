import unittest
import os
import sys
from unittest.mock import patch, MagicMock

# Add project root to sys.path to allow imports from epibench
project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..'))
if project_root not in sys.path:
    sys.path.insert(0, project_root)

# Import the main function from the script we want to test
from epibench.cli.train import main as train_main
from epibench.cli.train import setup_arg_parser

# Basic test class structure
class TestTrainCLI(unittest.TestCase):

    def setUp(self):
        """Set up for test methods."""
        # Create dummy config file for testing
        self.config_path = "test_config.yaml"
        with open(self.config_path, "w") as f:
            f.write("""
            # Minimal config for testing basic path
            model:
              name: DummyModel # Assume a dummy model exists for testing
              params:
                input_dim: 10
                output_dim: 1
            data:
              # Need data loading details here
              train_path: dummy_train.h5
              val_path: dummy_val.h5
              batch_size: 32
            training:
              epochs: 1
              learning_rate: 0.001
              optimizer: Adam
              loss_function: MSELoss
            logging:
              level: DEBUG # Use DEBUG for test output
              file: test_train.log
            """)
        # Create dummy data files if needed by data loader (mocking might be better)
        # open("dummy_train.h5", "a").close()
        # open("dummy_val.h5", "a").close()

    def tearDown(self):
        """Clean up after test methods."""
        if os.path.exists(self.config_path):
            os.remove(self.config_path)
        if os.path.exists("test_train.log"):
             os.remove("test_train.log")
        # Remove dummy data files
        # if os.path.exists("dummy_train.h5"): os.remove("dummy_train.h5")
        # if os.path.exists("dummy_val.h5"): os.remove("dummy_val.h5")

    def test_setup_arg_parser(self):
        """Test if the argument parser is set up correctly."""
        parser = setup_arg_parser()
        self.assertIsInstance(parser, argparse.ArgumentParser)
        # Test required argument
        with self.assertRaises(SystemExit):
            parser.parse_args([]) # No config provided
        # Test config argument
        args = parser.parse_args(["-c", "config.yaml"])
        self.assertEqual(args.config, "config.yaml")
        # Test hpo flag
        args = parser.parse_args(["-c", "config.yaml", "--hpo"])
        self.assertTrue(args.hpo)
        args = parser.parse_args(["-c", "config.yaml"])
        self.assertFalse(args.hpo)
        # Test log level
        args = parser.parse_args(["-c", "config.yaml", "--log-level", "DEBUG"])
        self.assertEqual(args.log_level, "DEBUG")
        args = parser.parse_args(["-c", "config.yaml"])
        self.assertEqual(args.log_level, "INFO") # Default

    @patch('epibench.cli.train.ConfigManager')
    @patch('epibench.cli.train.create_dataloaders')
    @patch('epibench.cli.train.models.get_model')
    @patch('epibench.cli.train.Trainer')
    @patch('epibench.cli.train.HPOptimizer')
    @patch('epibench.utils.logging.LoggerManager.setup_logger') # Patch logger setup
    @patch('sys.argv', ['epibench', 'train', '-c', 'test_config.yaml'])
    def test_main_standard_training_runs(self, mock_setup_logger, mock_hpo, mock_trainer, mock_get_model, mock_create_dataloaders, mock_config):
        """Test if the main function runs standard training without crashing."""
        
        # --- Mock Configuration ---
        mock_config_instance = MagicMock()
        mock_config_instance.config = {
            'model': {'name': 'DummyModel', 'params': {}},
            'data': {'batch_size': 32},
            'training': {},
            'logging': {'level': 'DEBUG'}
        }
        mock_config.return_value = mock_config_instance

        # --- Mock Data Loaders ---
        mock_train_loader = MagicMock()
        mock_val_loader = MagicMock()
        mock_create_dataloaders.return_value = (mock_train_loader, mock_val_loader)

        # --- Mock Model --- 
        MockModelClass = MagicMock()
        mock_model_instance = MagicMock()
        MockModelClass.return_value = mock_model_instance
        mock_get_model.return_value = MockModelClass

        # --- Mock Trainer --- 
        mock_trainer_instance = MagicMock()
        mock_trainer_instance.train.return_value = {"final_val_loss": 0.1} # Mock history
        mock_trainer.return_value = mock_trainer_instance

        # Run the main function
        try:
            train_main()
        except SystemExit as e:
             self.fail(f"train_main exited unexpectedly with code {e.code}")

        # --- Assertions ---
        mock_setup_logger.assert_called_once()
        mock_config.assert_called_once_with('test_config.yaml')
        mock_create_dataloaders.assert_called_once_with(mock_config_instance.config)
        mock_get_model.assert_called_once_with('DummyModel')
        MockModelClass.assert_called_once_with() # Assuming no params in this basic test
        mock_trainer.assert_called_once()
        mock_trainer_instance.train.assert_called_once_with(mock_train_loader, mock_val_loader)
        mock_hpo.assert_not_called() # HPO should not be called

    # TODO: Add test for HPO path
    # TODO: Add test for configuration loading errors
    # TODO: Add test for data loading errors
    # TODO: Add test for model creation errors
    # TODO: Add test for file outputs (logs, checkpoints) - might need tempfile module

if __name__ == '__main__':
    unittest.main() 