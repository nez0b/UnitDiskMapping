"""
Tests for the copyline module functionality.
"""
import pytest
from copyline import CopyLine, create_copylines, center_location, copyline_locations
import networkx as nx

def test_copyline_creation():
    """Test CopyLine structure creation."""
    # Create a CopyLine manually
    line = CopyLine(vertex=1, vslot=2, hslot=3, vstart=1, vstop=4, hstop=5)
    
    # Check attributes
    assert line.vertex == 1
    assert line.vslot == 2
    assert line.hslot == 3
    assert line.vstart == 1
    assert line.vstop == 4
    assert line.hstop == 5

def test_create_copylines():
    """Test creating copylines for a graph."""
    # Create a simple graph
    g = nx.path_graph(3)  # Path with 3 nodes
    
    # Create copylines with a specific vertex order
    vertex_order = [0, 1, 2]
    lines = create_copylines(g, vertex_order)
    
    # Check we have the right number of lines
    assert len(lines) == 3
    
    # Check each line has the correct vertex
    assert lines[0].vertex == 0
    assert lines[1].vertex == 1
    assert lines[2].vertex == 2
    
    # Check positions
    for i, line in enumerate(lines):
        assert line.vslot == i+1
        assert line.hslot == i+1

def test_center_location():
    """Test calculating center locations for copylines."""
    # Create a copyline
    line = CopyLine(vertex=1, vslot=2, hslot=3, vstart=1, vstop=4, hstop=5)
    
    # Calculate center location with different paddings
    padding = 2
    I, J = center_location(line, padding)
    
    # Check correct calculation
    s = 4  # spacing factor
    expected_I = s * (line.hslot - 1) + padding + 2
    expected_J = s * (line.vslot - 1) + padding + 1
    assert I == expected_I
    assert J == expected_J

def test_copyline_locations():
    """Test generating node locations for copylines."""
    # Create a copyline
    line = CopyLine(vertex=1, vslot=2, hslot=2, vstart=1, vstop=3, hstop=3)
    padding = 2
    
    # Generate locations for unweighted
    locs_unweighted = copyline_locations("UnWeightedNode", line, padding)
    
    # Check we have nodes (exact count may vary by implementation)
    assert len(locs_unweighted) > 0
    
    # Generate locations for weighted
    locs_weighted = copyline_locations("WeightedNode", line, padding)
    
    # Check node count matches between implementations
    assert len(locs_unweighted) == len(locs_weighted)
    
    # Check node locations are the same
    for i in range(len(locs_unweighted)):
        assert locs_unweighted[i].loc == locs_weighted[i].loc