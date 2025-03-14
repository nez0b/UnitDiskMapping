import unittest
import networkx as nx
import numpy as np
from src.gadgets import (
    Pattern,
    Cross,
    Turn,
    Branch,
    BranchFix,
    WTurn,
    BranchFixB,
    TCon,
    TrivialTurn,
    EndTurn,
    RotatedGadget,
    ReflectedGadget,
    rotated_and_reflected,
    crossing_ruleset
)


class TestGadgets(unittest.TestCase):
    def test_cross_pattern(self):
        # Test Cross with edge
        cross = Cross(has_edge=True)
        self.assertEqual(cross.size(), (3, 3))
        self.assertEqual(cross.cross_location(), (1, 1))
        self.assertTrue(cross.is_connected())
        
        # Test source graph
        locs, graph, pins = cross.source_graph()
        self.assertEqual(len(locs), 6)
        self.assertEqual(nx.number_of_edges(graph), 5)
        self.assertEqual(len(pins), 4)
        
        # Test mapped graph
        locs, graph, pins = cross.mapped_graph()
        self.assertEqual(len(locs), 5)
        self.assertEqual(len(pins), 4)
        
        # Test source and mapped matrices
        source_matrix = cross.source_matrix()
        mapped_matrix = cross.mapped_matrix()
        self.assertEqual(len(source_matrix), 3)
        self.assertEqual(len(mapped_matrix), 3)
    
    def test_turn_pattern(self):
        turn = Turn()
        self.assertEqual(turn.size(), (4, 4))
        self.assertEqual(turn.cross_location(), (2, 1))
        self.assertFalse(turn.is_connected())
        
        # Test source graph
        locs, graph, pins = turn.source_graph()
        self.assertEqual(len(locs), 5)
        self.assertEqual(nx.number_of_edges(graph), 4)
        self.assertEqual(len(pins), 2)
        
        # Test mapped graph
        locs, graph, pins = turn.mapped_graph()
        self.assertEqual(len(locs), 3)
        self.assertEqual(len(pins), 2)
    
    def test_rotated_gadget(self):
        # Create a pattern and its rotation
        cross = Cross(has_edge=True)
        rotated = RotatedGadget(cross, 1)  # Rotate 90 degrees
        
        # Test size changes after rotation (3,3) -> (3,3)
        self.assertEqual(rotated.size(), (3, 3))
        
        # Test source graph after rotation
        locs_orig, _, _ = cross.source_graph()
        locs_rot, _, _ = rotated.source_graph()
        
        # Number of nodes should be the same
        self.assertEqual(len(locs_orig), len(locs_rot))
        
        # But the coordinates should be different (rotated)
        orig_coords = set(loc.loc for loc in locs_orig)
        rot_coords = set(loc.loc for loc in locs_rot)
        self.assertNotEqual(orig_coords, rot_coords)
    
    def test_reflected_gadget(self):
        # Create a pattern and its reflection
        cross = Cross(has_edge=True)
        reflected = ReflectedGadget(cross, "x")  # Reflect across x-axis
        
        # Size should remain the same
        self.assertEqual(reflected.size(), cross.size())
        
        # Test source graph after reflection
        locs_orig, _, _ = cross.source_graph()
        locs_ref, _, _ = reflected.source_graph()
        
        # Number of nodes should be the same
        self.assertEqual(len(locs_orig), len(locs_ref))
        
        # But the coordinates should be different (reflected)
        orig_coords = set(loc.loc for loc in locs_orig)
        ref_coords = set(loc.loc for loc in locs_ref)
        self.assertNotEqual(orig_coords, ref_coords)
    
    def test_pattern_matching(self):
        # Create a pattern
        cross = Cross(has_edge=True)
        
        # Create a matrix that should match the pattern
        matrix = cross.source_matrix()
        
        # Test that the pattern matches its own matrix
        self.assertTrue(cross.match(matrix, 0, 0))
        
        # Modify the matrix to not match
        matrix[0][0] = matrix[0][0].__class__(occupied=not matrix[0][0].occupied)
        self.assertFalse(cross.match(matrix, 0, 0))
    
    def test_pattern_application(self):
        # Create a pattern
        cross = Cross(has_edge=True)
        
        # Get source and mapped matrices
        source = cross.source_matrix()
        mapped = cross.mapped_matrix()
        
        # Apply the gadget to the source matrix
        cross.apply_gadget(source, 0, 0)
        
        # Check if the result matches the mapped matrix
        for i in range(len(mapped)):
            for j in range(len(mapped[i])):
                self.assertEqual(source[i][j].occupied, mapped[i][j].occupied)
    
    def test_crossing_ruleset(self):
        # Check that the crossing ruleset has the expected patterns
        self.assertEqual(len(crossing_ruleset), 10)
        
        # Check we have each pattern type in the ruleset
        pattern_types = set(type(p) for p in crossing_ruleset)
        self.assertEqual(len(pattern_types), 9)  # Cross appears as two different instances
        
        # Ensure all patterns have the necessary methods
        for pattern in crossing_ruleset:
            self.assertTrue(hasattr(pattern, 'size'))
            self.assertTrue(hasattr(pattern, 'cross_location'))
            self.assertTrue(hasattr(pattern, 'source_graph'))
            self.assertTrue(hasattr(pattern, 'mapped_graph'))
    
    def test_rotated_gadget_creation(self):
        # Just test that we can create a rotated gadget
        cross = Cross(has_edge=True)
        rotated = RotatedGadget(cross, 1)
        
        # Check that the rotated gadget is a Pattern instance
        self.assertIsInstance(rotated, Pattern)
        
        # Check that source_graph works
        locs, graph, pins = rotated.source_graph()
        # The number of nodes should match the original pattern
        orig_locs, _, _ = cross.source_graph()
        self.assertEqual(len(locs), len(orig_locs))


if __name__ == '__main__':
    unittest.main()