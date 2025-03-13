"""
Unit Disk Mapping

A Python package for mapping problems to unit disk graphs for quantum computing.
"""

# Basic types
from core import ONE_INSTANCE
from core import SimpleCell, AbstractCell
from core import Node, UnWeightedNode, WeightedNode
from core import GridGraph
from mapping import UnWeighted, Weighted
from mapping import MappingGrid, MappingResult
from mapping import map_graph, embed_graph, print_config, map_config_back

# Path decomposition
from pathdecomposition import pathwidth, MinhThiTrick, Greedy

# Utilities
from utils import unit_disk_graph, is_independent_set

# Version information
__version__ = "0.1.0"