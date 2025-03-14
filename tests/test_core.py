"""
Tests for the core module functionality.
"""
import pytest
import networkx as nx
from src.core import Node, GridGraph

def test_grid_graph():
    """Test GridGraph functionality."""
    # Create a grid graph with 3 nodes
    grid = GridGraph((5, 5), [Node(2, 3), Node(2, 4), Node(5, 5)], 1.2)
    
    # Convert to NetworkX graph and check edge count (should be 1 - only between nearby nodes)
    g = grid.to_networkx()
    assert g.number_of_edges() == 1
    assert g.number_of_nodes() == 3
    
    # Check vertices match
    assert list(range(grid.num_vertices())) == list(range(g.number_of_nodes()))
    
    # Check neighbors match for node 1 (0-indexed)
    assert grid.neighbors(1) == list(g.neighbors(1))
    
    # Create a grid graph with larger radius (all nodes should be connected)
    grid_large_radius = GridGraph((5, 5), [Node(2, 3), Node(2, 4), Node(5, 5)], 4.0)
    g_large = grid_large_radius.to_networkx()
    
    # Check edge count - with larger radius, all nodes should be connected
    assert g_large.number_of_edges() == 3
    
    # Check vertices match
    assert list(range(grid_large_radius.num_vertices())) == list(range(g_large.number_of_nodes()))
    
    # Check neighbors match
    assert sorted(grid_large_radius.neighbors(1)) == sorted(list(g_large.neighbors(1)))

def test_node_functionality():
    """Test Node class functionality."""
    # Test different initialization methods
    n1 = Node(1, 2)  # Separate x, y coordinates
    n2 = Node((1, 2))  # Tuple coordinates
    n3 = Node([1, 2])  # List coordinates
    n4 = Node(1, 2, 5)  # With weight
    
    # Check they're equivalent
    assert n1.loc == (1, 2)
    assert n2.loc == (1, 2)
    assert n3.loc == (1, 2)
    assert n4.loc == (1, 2)
    assert n4.weight == 5
    
    # Test indexing and iteration
    assert n1[0] == 1
    assert n1[1] == 2
    assert list(n1) == [1, 2]
    
    # Test offset
    n5 = n1.offset((1, 2))
    assert n5.loc == (2, 4)

def test_cell_matrix():
    """Test conversion between GridGraph and cell matrix."""
    # Create a grid graph
    grid = GridGraph((3, 3), [Node(0, 0), Node(0, 2), Node(2, 0), Node(2, 2)], 1.5)
    
    # Convert to cell matrix
    matrix = grid.cell_matrix()
    
    # Check dimensions
    assert len(matrix) == 3
    assert len(matrix[0]) == 3
    
    # Check occupied cells
    assert matrix[0][1].is_empty  # Empty cell at 0,1
    assert matrix[0][0].occupied  # Occupied cell at 0,0
    assert matrix[0][2].occupied  # Occupied cell at 0,2
    assert matrix[2][0].occupied  # Occupied cell at 2,0
    assert matrix[2][2].occupied  # Occupied cell at 2,2
    
    # Convert back to GridGraph
    grid2 = GridGraph.from_cell_matrix(matrix, 1.5)
    
    # Check the nodes match
    assert len(grid.nodes) == len(grid2.nodes)
    assert set(n.loc for n in grid.nodes) == set(n.loc for n in grid2.nodes)
    
    # Check the edges match
    g1 = grid.to_networkx()
    g2 = grid2.to_networkx()
    assert g1.number_of_edges() == g2.number_of_edges()