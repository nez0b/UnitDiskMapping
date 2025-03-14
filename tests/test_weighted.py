import networkx as nx
import numpy as np
import pytest
import sys
import os

# Ensure src directory is in path
project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
src_dir = os.path.join(project_root, "src")
if src_dir not in sys.path:
    sys.path.insert(0, src_dir)
if project_root not in sys.path:
    sys.path.insert(0, project_root)

from src.unit_disk_mapping import (
    ONE_INSTANCE,
    Node, WeightedNode, SimpleCell, GridGraph,
    Weighted, UnWeighted,
    Cross, Turn, 
    source_centers, mapped_centers,
    move_centers, trace_centers,
    map_weights, map_configs_back,
    crossing_ruleset_weighted,
    map_graph
)
from src.weighted import simple_gadget_rule, WeightedGadget, weighted


def test_weighted_gadget_creation():
    """Test creation of weighted gadgets."""
    # Create a weighted cross gadget
    cross = Cross()
    weighted_cross = WeightedGadget(cross)
    
    # Check that it inherits the base properties
    assert weighted_cross.size() == cross.size()
    assert weighted_cross.cross_location() == cross.cross_location()
    assert weighted_cross.is_connected() == cross.is_connected()


def test_weighted_helper_factory():
    """Test the weighted helper factory function."""
    cross = Cross()
    weighted_cross = weighted(cross)
    
    # Verify it creates a WeightedGadget
    assert isinstance(weighted_cross, WeightedGadget)
    
    # Test with custom weights
    source_weights = [1, 2, 3, 4, 5, 6, 7, 8, 9]
    mapped_weights = [2, 3, 4, 5, 6, 7, 8, 9, 10]
    
    weighted_cross_custom = weighted(cross, source_weights, mapped_weights)
    
    # Check that the weights are assigned
    src_locs, _, _ = weighted_cross_custom.source_graph()
    map_locs, _, _ = weighted_cross_custom.mapped_graph()
    
    # Verify weights have been applied to nodes
    for i, loc in enumerate(src_locs):
        if i < len(source_weights):
            assert loc.weight == source_weights[i]
    
    for i, loc in enumerate(map_locs):
        if i < len(mapped_weights):
            assert loc.weight == mapped_weights[i]


def test_center_functions():
    """Test center location functions."""
    cross = Cross()
    
    # Test source centers
    src_centers = source_centers(cross)
    assert len(src_centers) > 0
    assert all(isinstance(center, tuple) and len(center) == 2 for center in src_centers)
    
    # Test mapped centers
    map_centers = mapped_centers(cross)
    assert len(map_centers) > 0
    assert all(isinstance(center, tuple) and len(center) == 2 for center in map_centers)
    
    # Test move centers
    moved = move_centers(src_centers, 1, 2)
    assert len(moved) == len(src_centers)
    for (x1, y1), (x2, y2) in zip(moved, src_centers):
        assert x1 == x2 + 1
        assert y1 == y2 + 2
    
    # Test trace centers
    traced = trace_centers(cross, 5, 5)
    assert len(traced) == len(map_centers)


def test_weighted_ruleset():
    """Test the weighted crossing ruleset."""
    # Check that we have a weighted ruleset
    assert len(crossing_ruleset_weighted) > 0
    
    # Check that all entries are weighted gadgets by checking class name
    # This avoids issues with multiple imports of the same class
    for pattern in crossing_ruleset_weighted:
        assert pattern.__class__.__name__ == "WeightedGadget"


def test_simple_gadget_rule():
    """Test the simple gadget rule creator."""
    cross = Cross()
    weighted_cross = simple_gadget_rule(cross, 2, 3)
    
    # Verify it's a WeightedGadget
    assert isinstance(weighted_cross, WeightedGadget)
    
    # Check weights in source and mapped graphs
    src_locs, _, _ = weighted_cross.source_graph()
    map_locs, _, _ = weighted_cross.mapped_graph()
    
    # All source nodes should have weight 2
    for loc in src_locs:
        assert loc.weight == 2
    
    # All mapped nodes should have weight 3
    for loc in map_locs:
        assert loc.weight == 3


def test_weighted_graph_mapping():
    """Test mapping a weighted graph."""
    # Create a simple graph
    G = nx.path_graph(3)
    
    # Map with weighted mode
    result = map_graph(G, mode=Weighted(), ruleset=crossing_ruleset_weighted)
    
    # Check that the result has nodes with weights
    assert len(result.grid_graph.nodes) > 0
    
    # In weighted mode, weights are either integers or ONE instances
    for node in result.grid_graph.nodes:
        # Check that the weight is either a numeric value or ONE_INSTANCE
        # ONE_INSTANCE is technically allowed as the weight can be converted to 1 when needed
        assert isinstance(node.weight, (int, float)) or node.weight == ONE_INSTANCE


def test_map_configs_back():
    """Test mapping configurations back from weighted graphs."""
    # Create a simple graph
    G = nx.path_graph(3)
    
    # Map with weighted mode
    result = map_graph(G, mode=Weighted(), ruleset=crossing_ruleset_weighted)
    
    # Create a mock configuration
    config = np.zeros((result.grid_graph.size[0], result.grid_graph.size[1]))
    
    # Set some values
    for i, node in enumerate(result.grid_graph.nodes):
        if i % 2 == 0:  # Set every other node
            x, y = node.loc
            config[x, y] = 1
    
    # Map back
    original_config = map_configs_back(result, config)
    
    # Check that we got a result with the right size
    assert len(original_config) == max(line.vertex for line in result.lines) + 1


if __name__ == "__main__":
    pytest.main(["-xvs", __file__])