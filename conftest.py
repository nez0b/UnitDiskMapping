"""
pytest configuration file to ensure proper import paths.
"""
import sys
import os

# Add project paths to enable imports
project_root = os.path.dirname(os.path.abspath(__file__))
src_dir = os.path.join(project_root, "src")

if project_root not in sys.path:
    sys.path.insert(0, project_root)
if src_dir not in sys.path:
    sys.path.insert(0, src_dir)