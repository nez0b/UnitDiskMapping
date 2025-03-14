"""
Tests for the utils module functionality.
"""
import pytest
import networkx as nx
import numpy as np
from src.utils import (
    rotate90_around, reflectx_around, reflecty_around, 
    reflectdiag_around, reflectoffdiag_around,
    unit_disk_graph, is_independent_set, is_diff_by_const
)

def test_symmetry_operations():
    """Test symmetry transformation functions."""
    center = (2, 2)
    loc = (4, 3)
    
    # Test each transformation matches the expected result
    assert rotate90_around(loc, center) == (1, 4)
    assert reflectx_around(loc, center) == (4, 1)
    assert reflecty_around(loc, center) == (0, 3)
    assert reflectdiag_around(loc, center) == (1, 0)
    assert reflectoffdiag_around(loc, center) == (3, 4)

def test_unit_disk_graph():
    """Test unit disk graph creation."""
    # Test with locations that are close
    locs = [(0, 0), (1, 0), (0, 1), (5, 5)]
    g = unit_disk_graph(locs, 1.5)
    
    # Check the graph has the right structure
    assert g.number_of_nodes() == 4
    # Nodes 0, 1, 2 should form a triangle (all within distance 1.5)
    # Node 3 should be isolated
    assert g.number_of_edges() == 3
    assert (0, 1) in g.edges()
    assert (0, 2) in g.edges()
    assert (1, 2) in g.edges()
    assert list(g.neighbors(3)) == []
    
    # Test with a larger radius
    g_large = unit_disk_graph(locs, 8.0)
    # All nodes should be connected
    assert g_large.number_of_edges() == 6

def test_is_independent_set():
    """Test independent set verification."""
    # Create a simple graph
    g = nx.Graph()
    g.add_edges_from([(0, 1), (1, 2), (2, 3), (3, 0)])  # A 4-cycle
    
    # Valid independent sets
    assert is_independent_set(g, [1, 0, 1, 0])  # Nodes 0 and 2
    assert is_independent_set(g, [0, 1, 0, 1])  # Nodes 1 and 3
    
    # Invalid independent sets
    assert not is_independent_set(g, [1, 1, 0, 0])  # Nodes 0 and 1 are adjacent
    assert not is_independent_set(g, [1, 0, 0, 1])  # Nodes 0 and 3 are adjacent

def test_is_diff_by_const():
    """Test array difference by constant."""
    # Arrays that differ by a constant
    arr1 = np.array([1, 2, 3, 4])
    arr2 = np.array([3, 4, 5, 6])
    is_diff, diff = is_diff_by_const(arr1, arr2)
    assert is_diff
    assert diff == -2
    
    # Arrays that don't differ by a constant
    arr3 = np.array([1, 2, 3, 4])
    arr4 = np.array([2, 4, 6, 8])
    is_diff, diff = is_diff_by_const(arr3, arr4)
    assert not is_diff
    assert diff == 0
    
    # Test with infinity
    arr5 = np.array([1, 2, np.inf, 4])
    arr6 = np.array([3, 4, np.inf, 6])
    is_diff, diff = is_diff_by_const(arr5, arr6)
    assert is_diff
    assert diff == -2
    
    # Test with mixed infinity
    arr7 = np.array([1, 2, np.inf, 4])
    arr8 = np.array([3, 4, 5, 6])
    is_diff, diff = is_diff_by_const(arr7, arr8)
    assert not is_diff
    assert diff == 0