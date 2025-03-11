#!/usr/bin/env python
"""
Test script to diagnose basic-pitch installation issues
"""

import os
import sys
import tempfile
import subprocess
from pathlib import Path

def run_command(cmd):
    """Run a command and return the output"""
    print(f"Running: {cmd}")
    result = subprocess.run(cmd, shell=True, capture_output=True, text=True)
    if result.returncode != 0:
        print(f"Error ({result.returncode}):")
        print(result.stderr)
    else:
        print("Success!")
        print(result.stdout)
    return result.returncode == 0

def test_packages():
    """Test importing packages to identify issues"""
    print("\n\n=== Testing imports ===")
    packages_to_test = [
        "basic_pitch",
        "basic_pitch.inference",
        "basic_pitch.midi",
        "tensorflow",
    ]
    
    for package in packages_to_test:
        cmd = f"python -c 'import {package}; print(f\"Successfully imported {package}\"'"
        run_command(cmd)

def check_installed_packages():
    """Check what packages are installed"""
    print("\n\n=== Checking installed packages ===")
    run_command("pip freeze | grep -i pitch")
    run_command("pip freeze | grep -i tensorflow")
    run_command("pip freeze | grep -i midi")

def check_basic_pitch_contents():
    """Check the contents of the basic_pitch package"""
    print("\n\n=== Checking basic_pitch package contents ===")
    try:
        import basic_pitch
        pkg_path = Path(basic_pitch.__file__).parent
        print(f"basic_pitch package location: {pkg_path}")
        
        print("\nListing files in basic_pitch package:")
        for f in pkg_path.glob("**/*"):
            if f.is_file():
                rel_path = f.relative_to(pkg_path)
                print(f"  {rel_path}")
    except ImportError as e:
        print(f"Error importing basic_pitch: {e}")

def reinstall_basic_pitch():
    """Reinstall basic_pitch"""
    print("\n\n=== Reinstalling basic_pitch ===")
    run_command("pip uninstall -y basic-pitch")
    run_command("pip install basic-pitch==0.4.0")
    run_command("pip install 'basic-pitch[tf]'")

def main():
    print(f"Python version: {sys.version}")
    check_installed_packages()
    check_basic_pitch_contents()
    test_packages()
    reinstall_basic_pitch()
    check_installed_packages()
    check_basic_pitch_contents()
    test_packages()

if __name__ == "__main__":
    main() 