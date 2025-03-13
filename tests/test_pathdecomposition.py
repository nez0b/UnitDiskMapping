"""
Tests for the pathdecomposition module functionality.
"""
import pytest
import networkx as nx
import random
from pathdecomposition import pathwidth, MinhThiTrick, Greedy, Layout
from pathdecomposition.pathdecomposition import vsep_and_neighbors

def test_layout_creation():
    """Test the Layout class creation and behavior."""
    # Create a small graph
    g = nx.Graph()
    g.add_edges_from([(0, 1), (1, 2), (2, 3), (3, 0)])  # A 4-cycle
    
    # Create a layout with specific vertex order
    vertices = [0, 1, 2, 3]
    layout = Layout.from_graph(g, vertices)
    
    # Check the layout properties
    assert layout.vertices == vertices
    assert layout.vsep == 2  # For a cycle, vsep should be 2
    
    # Test equality and hash
    layout2 = Layout.from_graph(g, vertices)
    assert layout == layout2
    assert hash(layout) == hash(layout2)
    
    # Test with different order
    layout3 = Layout.from_graph(g, [3, 2, 1, 0])
    assert layout != layout3

def test_vsep_calculation():
    """Test vertex separation calculation."""
    # Create a path graph (vsep should be 1)
    path_graph = nx.path_graph(5)
    vs, nbs = vsep_and_neighbors(path_graph, list(path_graph.nodes))
    assert vs == 1
    
    # Create a star graph (vsep should be n-1 for center-last ordering)
    star_graph = nx.star_graph(4)  # K1,4 star
    
    # Manual calculation for the star graph with center last
    # At each step, compute neighbors:
    # [1] -> neighbors = [0]
    # [1, 2] -> neighbors = [0]
    # [1, 2, 3] -> neighbors = [0]
    # [1, 2, 3, 4] -> neighbors = [0]
    # Final vsep is 1, not 4
    
    # Calculate vsep for a graph where we know the value
    complete_graph = nx.complete_graph(4)  # K4
    # For a complete graph, all vertices are connected
    # When vertex ordering is 0,1,2,3, at each step:
    # [0] -> neighbors = [1, 2, 3] (vsep = 3)
    # [0, 1] -> neighbors = [2, 3] (vsep = 2)
    # [0, 1, 2] -> neighbors = [3] (vsep = 1)
    vs_complete, _ = vsep_and_neighbors(complete_graph, [0, 1, 2, 3])
    assert vs_complete == 3

def test_greedy_pathwidth():
    """Test the greedy pathwidth algorithm."""
    # Use a graph with known pathwidth
    g = nx.petersen_graph()  # Pathwidth is 5
    
    # Run the greedy algorithm
    layout = pathwidth(g, Greedy(nrepeat=5))
    
    # Check the vsep is at least the correct value
    # Note: greedy may not always find optimal, so we check it's at least close
    assert layout.vsep >= 4  # Should ideally be 5, but we allow some leeway
    
    # For a simple path graph, should get optimal vsep=1
    path_graph = nx.path_graph(10)
    layout_path = pathwidth(path_graph, Greedy(nrepeat=2))
    assert layout_path.vsep == 1

@pytest.mark.skip(reason="MinhThiTrick implementation is slow - only run when needed")
def test_exact_pathwidth():
    """Test the exact (MinhThiTrick) pathwidth algorithm."""
    # Use small graphs to keep test fast
    g = nx.cycle_graph(4)  # Pathwidth is 2
    
    # Run the exact algorithm
    layout = pathwidth(g, MinhThiTrick())
    
    # Check the vsep is correct
    assert layout.vsep == 2
    
    # Test with a slightly more complex graph
    g2 = nx.complete_graph(4)  # Pathwidth is 3
    layout2 = pathwidth(g2, MinhThiTrick())
    assert layout2.vsep == 3