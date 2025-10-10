#!/usr/bin/env python3
"""
Release script for PyTradePath
"""

import os
import sys
import subprocess
import shutil
import zipfile
import argparse
from pathlib import Path

def run_command(cmd, check=True):
    """Run a command and return the result."""
    print(f"Running: {cmd}")
    try:
        result = subprocess.run(cmd, shell=True, check=check, 
                              capture_output=True, text=True)
        if result.stdout:
            print(result.stdout)
        if result.stderr:
            print(result.stderr)
        return result
    except subprocess.CalledProcessError as e:
        print(f"Command failed with exit code {e.returncode}")
        print(f"stdout: {e.stdout}")
        print(f"stderr: {e.stderr}")
        if check:
            sys.exit(1)
        return e

def clean_build_dirs():
    """Clean build directories."""
    dirs_to_clean = ['build', 'dist', '*.egg-info']
    for pattern in dirs_to_clean:
        for path in Path('.').glob(pattern):
            if path.is_dir():
                shutil.rmtree(path)
                print(f"Removed directory: {path}")
            elif path.is_file():
                path.unlink()
                print(f"Removed file: {path}")

def create_zip_package():
    """Create a zip package of the project."""
    print("Creating zip archive...")
    
    # Define files and directories to exclude
    exclude_patterns = [
        '.git', 'dist', 'build', '__pycache__', '*.pyc', 
        'logs', 'cache', 'temp', 'tmp', 'exported_data'
    ]
    
    # Create the zip file
    with zipfile.ZipFile('pytradepath-release.zip', 'w', zipfile.ZIP_DEFLATED) as zf:
        for root, dirs, files in os.walk('.'):
            # Skip excluded directories
            dirs[:] = [d for d in dirs if not any(pattern in d for pattern in exclude_patterns)]
            
            for file in files:
                file_path = os.path.join(root, file)
                # Skip excluded files
                if any(pattern in file_path for pattern in exclude_patterns):
                    continue
                    
                # Add file to zip
                zf.write(file_path)
    
    print("Zip archive created successfully!")

def build_package():
    """Build the package."""
    print("Building package...")
    
    # Clean previous builds
    clean_build_dirs()
    
    # Check if setuptools is available
    try:
        import setuptools
        # Build source distribution and wheel
        run_command("python setup.py sdist bdist_wheel")
        print("Package built successfully!")
        return True
    except ImportError:
        print("setuptools not available, creating zip archive instead...")
        create_zip_package()
        return False

def check_package():
    """Check the package."""
    print("Checking package...")
    
    # Check if twine is available
    try:
        import twine
        # Check the package with twine
        result = run_command("python -m twine check dist/*", check=False)
        if result.returncode != 0:
            print("Package check failed!")
            return False
        print("Package check passed!")
        return True
    except ImportError:
        print("twine not available, skipping package check...")
        return True

def list_dist_files():
    """List files in the dist directory."""
    print("Files in dist directory:")
    if os.path.exists('dist'):
        for file in os.listdir('dist'):
            print(f"  {file}")
    else:
        print("  No dist directory found")

def main():
    """Main function."""
    parser = argparse.ArgumentParser(description="PyTradePath Release Script")
    parser.add_argument("--package-only", action="store_true", 
                        help="Create package only without full release process")
    args = parser.parse_args()
    
    print("PyTradePath Release Script")
    print("=" * 30)
    
    # Check if we're in the right directory
    if not os.path.exists('setup.py'):
        print("Error: setup.py not found. Please run this script from the project root directory.")
        sys.exit(1)
    
    if args.package_only:
        # Just create the package
        create_zip_package()
        print("\nPackage created successfully!")
        return
    
    # Build the package
    package_built = build_package()
    
    if package_built:
        # Check the package
        if check_package():
            # List the built files
            list_dist_files()
            
            print("\nRelease preparation completed successfully!")
            print("\nTo upload to PyPI, run:")
            print("  python -m twine upload dist/*")
            print("\nTo test the package locally, run:")
            print("  pip install dist/pytradepath-*.whl")
        else:
            print("\nRelease preparation failed!")
            sys.exit(1)
    else:
        print("\nZip package created successfully!")
        print("File: pytradepath-release.zip")

if __name__ == "__main__":
    main()