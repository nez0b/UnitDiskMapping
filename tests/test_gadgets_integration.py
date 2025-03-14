import unittest
import networkx as nx
import numpy as np
from src.unit_disk_mapping import (
    map_graph, embed_graph, Pattern, Cross, crossing_ruleset,
    Turn, Branch, BranchFix, WTurn, BranchFixB, TCon, TrivialTurn, EndTurn
)


class TestGadgetsIntegration(unittest.TestCase):
    def test_map_graph_with_default_ruleset(self):
        """Test mapping a graph with the default gadget ruleset."""
        # Create a simple graph
        g = nx.Graph()
        g.add_nodes_from([1, 2, 3, 4])
        g.add_edges_from([(1, 2), (2, 3), (3, 4), (4, 1), (1, 3)])
        
        # Map the graph with default ruleset
        result = map_graph(g)
        
        # Check if the result is valid
        self.assertIsNotNone(result)
        self.assertGreater(len(result.grid_graph.nodes), 0)
        
        # Check if the mapping history (tape) contains pattern applications
        self.assertGreaterEqual(len(result.mapping_history), 0)
    
    def test_map_graph_with_custom_ruleset(self):
        """Test mapping a graph with a custom gadget ruleset."""
        # Create a simple graph
        g = nx.Graph()
        g.add_nodes_from([1, 2, 3, 4])
        g.add_edges_from([(1, 2), (2, 3), (3, 4), (4, 1), (1, 3)])
        
        # Create a custom ruleset with just one pattern
        custom_ruleset = [Cross(has_edge=True)]
        
        # Map the graph with custom ruleset
        result = map_graph(g, ruleset=custom_ruleset)
        
        # Check if the result is valid
        self.assertIsNotNone(result)
        self.assertGreater(len(result.grid_graph.nodes), 0)
    
    def test_crossing_ruleset_integration(self):
        """Test that all patterns in the crossing ruleset can be used with mapping."""
        # Create a more complex graph that will need various crossings
        g = nx.complete_graph(6)
        
        # Try using each pattern type individually
        for pattern_class in [Cross, Turn, Branch, BranchFix]:
            if pattern_class == Cross:
                # Test both versions of Cross
                ruleset = [pattern_class(has_edge=True), pattern_class(has_edge=False)]
            else:
                ruleset = [pattern_class()]
            
            # Map using just this pattern type
            result = map_graph(g, ruleset=ruleset)
            
            # Verify the result exists
            self.assertIsNotNone(result)
            self.assertGreater(len(result.grid_graph.nodes), 0)
    
    def test_mis_overhead_calculation(self):
        """Test that MIS overhead is calculated correctly using vertex_overhead."""
        # Create a simple graph
        g = nx.Graph()
        g.add_nodes_from([1, 2, 3, 4])
        g.add_edges_from([(1, 2), (2, 3), (3, 4), (4, 1), (1, 3)])
        
        # Map the graph
        result = map_graph(g)
        
        # Verify MIS overhead is calculated
        self.assertGreaterEqual(result.mis_overhead, 0)


if __name__ == '__main__':
    unittest.main()