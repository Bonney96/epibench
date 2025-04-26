#!/usr/bin/env python3
"""
Validates the execution environment for the EpiBench pipeline.

Checks for:
- Correct Python package versions based on requirements.txt
- Presence and basic functionality of required external tools
- Necessary environment variables

Usage:
  python scripts/check_environment.py
"""

import os
import sys
import shutil
import subprocess
import importlib.metadata
from packaging.requirements import Requirement
from packaging.version import parse as parse_version

try:
    import colorama
    colorama.init()
    GREEN = colorama.Fore.GREEN
    RED = colorama.Fore.RED
    YELLOW = colorama.Fore.YELLOW
    RESET = colorama.Style.RESET_ALL
except ImportError:
    print("Warning: colorama not installed. Output will not be colored.")
    GREEN = RED = YELLOW = RESET = ""

# --- Configuration ---
# Assume requirements.txt is in the project root
REQUIREMENTS_FILE = os.path.join(os.path.dirname(__file__), '..', 'requirements.txt')

# Placeholder lists - these should be updated based on actual project needs
REQUIRED_EXTERNAL_TOOLS = {
    'git': {'version_arg': '--version'},
    # 'docker': {'version_arg': '--version'},
    # 'some_bio_tool': {'version_arg': '-v', 'min_version': '1.2.0'}
}
REQUIRED_ENV_VARS = [
    # 'API_KEY',
    # 'DATA_DIR'
]
# --- End Configuration ---

def check_python_packages():
    """Validates installed Python packages against requirements.txt."""
    print(f"{YELLOW}Checking Python packages...{RESET}")
    parsing_errors = []
    requirements = []
    install_errors = []
    install_suggestions = []

    if not os.path.exists(REQUIREMENTS_FILE):
        parsing_errors.append(f"requirements.txt not found at {REQUIREMENTS_FILE}")
        return parsing_errors, [] # Return only parsing errors and no suggestions

    try:
        with open(REQUIREMENTS_FILE, 'r') as f:
            requirements_lines = f.readlines()
    except IOError as e:
        parsing_errors.append(f"Could not read {REQUIREMENTS_FILE}: {e}")
        return parsing_errors, []

    # --- Step 1: Parse Requirements --- #
    for line in requirements_lines:
        line = line.strip()
        if line and not line.startswith('#'):
            try:
                requirements.append(Requirement(line))
            except Exception as e:
                # Collect parsing errors
                parsing_errors.append(f"Could not parse requirement '{line}': {e}")

    # If any parsing errors occurred, return immediately
    if parsing_errors:
        return parsing_errors, []

    # --- Step 2: Check Installed Versions (only if parsing succeeded) --- #
    for req in requirements:
        try:
            installed_version_str = importlib.metadata.version(req.name)
            installed_version = parse_version(installed_version_str)
            # Use contains method, allow checking against prereleases
            # The 'prereleases' argument to contains() determines if installed prereleases
            # can satisfy the requirement. It's not an attribute of 'req'.
            if not req.specifier.contains(installed_version, prereleases=True):
                install_errors.append(
                    f"Package '{req.name}': Installed version {installed_version} "
                    f"does not meet requirement '{req.specifier}'"
                )
                # Format suggestion string correctly
                install_suggestions.append(f'Run: pip install "{req}"')
            else:
                 print(f"  {GREEN}[OK]{RESET} {req.name} ({installed_version} matches {req.specifier})")

        except importlib.metadata.PackageNotFoundError:
            install_errors.append(f"Package '{req.name}' is not installed.")
            # Format suggestion string correctly
            install_suggestions.append(f'Run: pip install "{req}"')
        except Exception as e:
             # Catch other potential errors during version checking
             install_errors.append(f"Error checking package '{req.name}': {e}")

    if not install_errors:
         print(f"{GREEN}All Python packages OK.{RESET}")

    # Return installation errors and suggestions
    return install_errors, install_suggestions

def check_external_tools():
    """Validates presence and optionally versions of external tools."""
    print(f"{YELLOW}Checking external tools...{RESET}")
    errors = []
    suggestions = []

    if not REQUIRED_EXTERNAL_TOOLS:
        print(f"  {GREEN}No external tools configured for checking.{RESET}")
        return errors, suggestions

    for tool, config in REQUIRED_EXTERNAL_TOOLS.items():
        path = shutil.which(tool)
        if not path:
            errors.append(f"External tool '{tool}' not found in PATH.")
            suggestions.append(f"Install '{tool}' and ensure it's in your system PATH.")
            continue

        print(f"  {GREEN}[OK]{RESET} Found {tool} at {path}")

        # Optional: Version check
        version_arg = config.get('version_arg')
        min_version_str = config.get('min_version')

        if version_arg:
            try:
                # Run tool --version or similar command
                result = subprocess.run([path, version_arg], capture_output=True, text=True, check=True, timeout=5)
                # Basic version extraction (may need refinement per tool)
                version_output = result.stdout.strip() or result.stderr.strip()
                # Try to parse the first line containing digits
                tool_version_str = None
                for line in version_output.splitlines():
                     parts = line.split()
                     for part in parts:
                         if any(char.isdigit() for char in part):
                             # Attempt to clean up common prefixes/suffixes
                             cleaned_part = ''.join(c for c in part if c.isdigit() or c == '.')
                             try:
                                 # Check if it parses as a version
                                 parse_version(cleaned_part)
                                 tool_version_str = cleaned_part
                                 break
                             except Exception:
                                 continue
                     if tool_version_str:
                         break

                if tool_version_str:
                    print(f"     Version: {tool_version_str}")
                    if min_version_str:
                        tool_version = parse_version(tool_version_str)
                        min_version = parse_version(min_version_str)
                        if tool_version < min_version:
                            errors.append(
                                f"Tool '{tool}' version {tool_version} is older than "
                                f"required minimum {min_version}."
                            )
                            suggestions.append(f"Upgrade '{tool}' to version {min_version} or later.")
                        else:
                            print(f"     {GREEN}[OK]{RESET} Version meets minimum requirement ({min_version_str})")
                else:
                    print(f"     {YELLOW}Could not parse version from output:{RESET}\\n{version_output}")

            except FileNotFoundError:
                 errors.append(f"Could not execute '{tool}' found at {path}. Is it executable?")
            except subprocess.CalledProcessError as e:
                errors.append(f"Command '{tool} {version_arg}' failed with exit code {e.returncode}")
                suggestions.append(f"Check '{tool}' installation and configuration.")
            except subprocess.TimeoutExpired:
                errors.append(f"Command '{tool} {version_arg}' timed out.")
            except Exception as e:
                 errors.append(f"Error checking version for '{tool}': {e}")

    if not errors:
        print(f"{GREEN}All external tools OK.{RESET}")

    return errors, suggestions


def check_environment_variables():
    """Validates presence of required environment variables."""
    print(f"{YELLOW}Checking environment variables...{RESET}")
    errors = []
    suggestions = []

    if not REQUIRED_ENV_VARS:
        print(f"  {GREEN}No environment variables configured for checking.{RESET}")
        return errors, suggestions

    for var_name in REQUIRED_ENV_VARS:
        value = os.environ.get(var_name)
        if value is None:
            errors.append(f"Required environment variable '{var_name}' is not set.")
            suggestions.append(f"Set the environment variable '{var_name}'. "
                               f"Refer to project documentation for expected value/format.")
        else:
            # Optional: Add validation logic here if needed
            print(f"  {GREEN}[OK]{RESET} Environment variable '{var_name}' is set.")
            # Example validation: if var_name == 'DATA_DIR' and not os.path.isdir(value):
            #     errors.append(f"Environment variable 'DATA_DIR' points to a non-existent directory: {value}")

    if not errors:
        print(f"{GREEN}All environment variables OK.{RESET}")

    return errors, suggestions

def main():
    """Runs all environment checks and reports results."""
    print(f"{YELLOW}--- Starting Environment Validation ---{RESET}")
    all_errors = []
    all_suggestions = []

    pkg_errors, pkg_suggestions = check_python_packages()
    all_errors.extend(pkg_errors)
    all_suggestions.extend(pkg_suggestions)
    print("-" * 30)

    tool_errors, tool_suggestions = check_external_tools()
    all_errors.extend(tool_errors)
    all_suggestions.extend(tool_suggestions)
    print("-" * 30)

    env_errors, env_suggestions = check_environment_variables()
    all_errors.extend(env_errors)
    all_suggestions.extend(env_suggestions)
    print("-" * 30)

    print(f"{YELLOW}--- Validation Summary ---{RESET}")
    if not all_errors:
        print(f"{GREEN}Environment validation successful!{RESET}")
        sys.exit(0)
    else:
        print(f"{RED}Environment validation failed with {len(all_errors)} error(s):{RESET}")
        for i, error in enumerate(all_errors):
            print(f"  {RED}[Error {i+1}]{RESET} {error}")

        if all_suggestions:
            print(f"\n{YELLOW}Suggestions:{RESET}")
            unique_suggestions = sorted(list(set(all_suggestions)))
            for suggestion in unique_suggestions:
                print(f"  - {suggestion}")

        print(f"\n{RED}Please resolve the issues above before proceeding.{RESET}")
        sys.exit(1)

if __name__ == "__main__":
    main()