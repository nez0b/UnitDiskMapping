#!/usr/bin/env python
"""
Test runner script for unit disk mapping tests.
"""
import sys
import os
import pytest

if __name__ == "__main__":
    print("Running Unit Disk Mapping tests...")
    
    # Add project root and src directories to the Python path
    project_root = os.path.dirname(os.path.abspath(__file__))
    src_dir = os.path.join(project_root, "src")
    
    if project_root not in sys.path:
        sys.path.insert(0, project_root)
    if src_dir not in sys.path:
        sys.path.insert(0, src_dir)
    
    # Print current Python path for debugging
    print(f"Python path: {sys.path}")
    
    # Run pytest with verbose output on both test directories
    test_dirs = ["tests", "tests_additional"]
    sys.exit(pytest.main(["-v"] + test_dirs))