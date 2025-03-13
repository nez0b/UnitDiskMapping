"""
Configuration file for pytest.
"""
import sys
import os

# Add the parent directory to path so we can import our package
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))