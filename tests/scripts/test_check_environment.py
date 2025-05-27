#!/usr/bin/env python3
import os
import sys
import unittest
from unittest.mock import patch, mock_open, MagicMock
import importlib.metadata
from packaging.version import Version
from packaging.requirements import Requirement

# Adjust path to import the script
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '../../scripts')))
import check_environment

# Disable colorama for tests
check_environment.GREEN = check_environment.RED = check_environment.YELLOW = check_environment.RESET = ""

class TestCheckEnvironment(unittest.TestCase):

    @patch('os.path.exists')
    def test_requirements_file_not_found(self, mock_exists):
        """Test case when requirements.txt does not exist."""
        mock_exists.return_value = False
        errors, suggestions = check_environment.check_python_packages()
        self.assertEqual(len(errors), 1)
        self.assertTrue("requirements.txt not found" in errors[0])
        self.assertEqual(suggestions, [])

    @patch('os.path.exists')
    @patch('builtins.open', new_callable=mock_open, read_data="requests==2.25.1\ninvalid-requirement\n")
    @patch('check_environment.Requirement') # Mock Requirement directly
    def test_requirements_parse_error(self, mock_requirement_cls, mock_file, mock_exists):
        """Test case when requirements.txt has an invalid line."""
        mock_exists.return_value = True

        # Define side effect for Requirement constructor
        def requirement_side_effect(req_string):
            if req_string == "invalid-requirement":
                raise ValueError("Simulated parsing error")
            else:
                # For valid strings, return a basic mock object
                # We don't need its attributes for this specific test
                mock_req = MagicMock(spec=Requirement)
                mock_req.name = req_string.split('==')[0] # Basic name parsing
                return mock_req

        mock_requirement_cls.side_effect = requirement_side_effect

        errors, suggestions = check_environment.check_python_packages()

        # We expect exactly one error: the parsing error
        self.assertEqual(len(errors), 1)
        self.assertTrue("Could not parse requirement 'invalid-requirement'" in errors[0])
        self.assertEqual(suggestions, []) # No suggestions if parsing failed

    @patch('os.path.exists')
    @patch('builtins.open', new_callable=mock_open, read_data="requests>=2.0.0\nmissing_pkg==1.0\nnumpy<1.20")
    @patch('importlib.metadata.version')
    def test_package_checks(self, mock_version, mock_file, mock_exists):
        """Test various package installation states."""
        mock_exists.return_value = True

        def version_side_effect(package_name):
            if package_name == 'requests':
                return '2.25.1'
            elif package_name == 'missing_pkg':
                raise importlib.metadata.PackageNotFoundError
            elif package_name == 'numpy':
                return '1.21.0' # This version is too high
            else:
                 raise importlib.metadata.PackageNotFoundError

        mock_version.side_effect = version_side_effect

        errors, suggestions = check_environment.check_python_packages()

        # Expected errors:
        # 1. missing_pkg not found
        # 2. numpy version too high
        self.assertEqual(len(errors), 2)
        self.assertTrue(any("'missing_pkg' is not installed" in e for e in errors))
        self.assertTrue(any("'numpy': Installed version 1.21.0 does not meet requirement '<1.20'" in e for e in errors))

        # Expected suggestions:
        # 1. Install missing_pkg
        # 2. Install numpy<1.20
        self.assertEqual(len(suggestions), 2)
        self.assertTrue(any('Run: pip install "missing_pkg==1.0"' in s for s in suggestions))
        self.assertTrue(any('Run: pip install "numpy<1.20"' in s for s in suggestions))

    @patch('os.path.exists')
    @patch('builtins.open', new_callable=mock_open, read_data="requests==2.25.1")
    @patch('importlib.metadata.version')
    def test_package_ok(self, mock_version, mock_file, mock_exists):
        """Test case when all packages are installed and versions match."""
        mock_exists.return_value = True
        mock_version.return_value = '2.25.1'

        errors, suggestions = check_environment.check_python_packages()
        self.assertEqual(len(errors), 0)
        self.assertEqual(len(suggestions), 0)

    # --- TODO: Add tests for check_external_tools --- #

    # --- TODO: Add tests for check_environment_variables --- #

    def test_main_success(self):
        """Test main() function calls sys.exit(0) on success."""
        with patch('check_environment.check_python_packages', return_value=([], [])) as mock_check_pkgs, \
             patch('check_environment.check_external_tools', return_value=([], [])) as mock_check_tools, \
             patch('check_environment.check_environment_variables', return_value=([], [])) as mock_check_env, \
             patch('sys.exit') as mock_exit:

            check_environment.main() # Run the function
            mock_exit.assert_called_once_with(0)

    def test_main_failure(self):
        """Test main() function calls sys.exit(1) on failure."""
        with patch('check_environment.check_python_packages', return_value=(["Package error"], ["Install package"])) as mock_check_pkgs, \
             patch('check_environment.check_external_tools', return_value=([], [])) as mock_check_tools, \
             patch('check_environment.check_environment_variables', return_value=([], [])) as mock_check_env, \
             patch('sys.exit') as mock_exit:

            check_environment.main() # Run the function
            mock_exit.assert_called_once_with(1)
            # TODO: Capture stdout/stderr to check output messages

if __name__ == '__main__':
    unittest.main() 