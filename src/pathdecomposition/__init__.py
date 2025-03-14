"""
Path decomposition module for the UnitDiskMapping package.

This module provides algorithms and data structures for computing
path decompositions of graphs, which are used in the mapping process.
"""

from .pathdecomposition import (
    Layout,
    vsep_and_neighbors,
    vsep_updated,
    vsep_updated_neighbors,
    PathDecompositionMethod,
    MinhThiTrick,
    Greedy,
    pathwidth
)

__all__ = [
    'Layout',
    'vsep_and_neighbors',
    'vsep_updated',
    'vsep_updated_neighbors',
    'PathDecompositionMethod',
    'MinhThiTrick',
    'Greedy',
    'pathwidth'
]