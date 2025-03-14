"""
Unit Disk Mapping package.

This package provides tools for mapping optimization problems to unit disk graphs.
"""

import sys
import os

# Add the project root and src directories to the Python path
project_root = os.path.dirname(os.path.abspath(__file__))
src_dir = os.path.join(project_root, "src")
if src_dir not in sys.path:
    sys.path.insert(0, src_dir)
if project_root not in sys.path:
    sys.path.insert(0, project_root)

# Import main components from src
from src.unit_disk_mapping import *
from src.gadgets import *
from src.gadgets_ext import *
from src.weighted import *

__version__ = "0.1.0"