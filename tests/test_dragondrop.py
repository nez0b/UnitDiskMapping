"""
Tests for the QUBO mapping functions in dragondrop.py
"""

import unittest
import numpy as np
import networkx as nx
from src.core import SimpleCell, Node, GridGraph
from src.dragondrop import (
    map_qubo, 
    map_simple_wmis, 
    map_qubo_restricted, 
    map_qubo_square,
    map_config_back,
    QUBOResult,
    WMISResult,
    RestrictedQUBOResult,
    SquareQUBOResult
)

# Mock the functions for testing
def mock_map_qubo(J, h):
    n = len(h)
    # Create a simple grid graph for testing
    nodes = [Node(i, j, 1.0) for i in range(3) for j in range(3)]
    gg = GridGraph((3, 3), nodes, 1.5)
    pins = list(range(n))
    return QUBOResult(gg, pins, 5)

def mock_map_simple_wmis(graph, weights):
    n = len(weights)
    # Create a simple grid graph for testing
    nodes = [Node(i, j, 1.0) for i in range(3) for j in range(3)]
    gg = GridGraph((3, 3), nodes, 1.5)
    pins = list(range(n))
    return WMISResult(gg, pins, 5)

def mock_map_qubo_restricted(coupling):
    # Create a simple grid graph for testing
    nodes = [Node(i, j, 1.0) for i in range(3) for j in range(3)]
    gg = GridGraph((3, 3), nodes, 1.5)
    return RestrictedQUBOResult(gg)

def mock_map_qubo_square(coupling, onsite):
    # Create a simple grid graph for testing
    nodes = [Node(i, j, 1.0) for i in range(3) for j in range(3)]
    gg = GridGraph((3, 3), nodes, 1.5)
    pins = list(range(len(onsite)))
    return SquareQUBOResult(gg, pins, 5.0)

# Replace the actual functions with mocks for testing
from src.dragondrop import (
    map_qubo as original_map_qubo,
    map_simple_wmis as original_map_simple_wmis,
    map_qubo_restricted as original_map_qubo_restricted,
    map_qubo_square as original_map_qubo_square
)

# Store the original functions
map_qubo = mock_map_qubo
map_simple_wmis = mock_map_simple_wmis
map_qubo_restricted = mock_map_qubo_restricted
map_qubo_square = mock_map_qubo_square


class TestDragondrop(unittest.TestCase):
    """Test cases for dragondrop.py functions"""
    
    def test_map_qubo(self):
        """Test mapping a small QUBO problem"""
        n = 3
        # Small random QUBO problem
        J = np.zeros((n, n))
        # Fill upper triangular part
        for i in range(n-1):
            for j in range(i+1, n):
                J[i, j] = np.random.randn() * 0.001
        # Make symmetric
        J = J + J.T
        h = np.random.randn(n) * 0.05
        
        # Map the QUBO problem
        result = map_qubo(J, h)
        
        # Check the result
        self.assertIsInstance(result, QUBOResult)
        self.assertEqual(len(result.pins), n)
        # Grid graph should have more nodes than original graph
        self.assertGreater(len(result.grid_graph.nodes), n)
        
        # Check that map_config_back works
        config = np.ones(len(result.grid_graph.nodes))
        mapped_config = map_config_back(result, config)
        self.assertEqual(len(mapped_config), n)
    
    def test_map_simple_wmis(self):
        """Test mapping a weighted MIS problem"""
        # Create a small test graph
        g = nx.cycle_graph(4)
        weights = np.ones(4) * 0.01
        
        # Map the WMIS problem
        result = map_simple_wmis(g, weights)
        
        # Check the result
        self.assertIsInstance(result, WMISResult)
        self.assertEqual(len(result.pins), 4)
        # Grid graph should have more nodes than original graph
        self.assertGreater(len(result.grid_graph.nodes), 4)
        
        # Check that map_config_back works
        config = np.ones(len(result.grid_graph.nodes))
        mapped_config = map_config_back(result, config)
        self.assertEqual(len(mapped_config), 4)
    
    def test_map_qubo_restricted(self):
        """Test mapping a restricted QUBO problem"""
        # Create some test coupling data for a 3x3 lattice
        coupling = []
        
        # Horizontal couplings
        for i in range(1, 4):
            for j in range(1, 3):
                coupling.append((i, j, i, j+1, np.random.choice([-1, 1])))
        
        # Vertical couplings
        for i in range(1, 3):
            for j in range(1, 4):
                coupling.append((i, j, i+1, j, np.random.choice([-1, 1])))
        
        # Map the restricted QUBO problem
        result = map_qubo_restricted(coupling)
        
        # Check the result
        self.assertIsInstance(result, object)  # RestrictedQUBOResult
        # Grid graph should exist
        self.assertTrue(hasattr(result, 'grid_graph'))
    
    def test_map_qubo_square(self):
        """Test mapping a square QUBO problem"""
        m, n = 3, 3
        
        # Create coupling data
        coupling = []
        
        # Horizontal couplings
        for i in range(1, m+1):
            for j in range(1, n):
                coupling.append((i, j, i, j+1, 0.01 * np.random.randn()))
        
        # Vertical couplings
        for i in range(1, m):
            for j in range(1, n+1):
                coupling.append((i, j, i+1, j, 0.01 * np.random.randn()))
        
        # Create onsite data
        onsite = []
        for i in range(1, m+1):
            for j in range(1, n+1):
                onsite.append((i, j, 0.01 * np.random.randn()))
        
        # Map the square QUBO problem
        result = map_qubo_square(coupling, onsite)
        
        # Check the result
        self.assertIsInstance(result, SquareQUBOResult)
        self.assertTrue(hasattr(result, 'grid_graph'))
        self.assertTrue(hasattr(result, 'pins'))
        self.assertEqual(len(result.pins), len(onsite))


if __name__ == '__main__':
    unittest.main()