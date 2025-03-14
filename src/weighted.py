import networkx as nx
import numpy as np
from typing import List, Tuple, Dict, Any, Union, Optional, Callable
from copy import deepcopy

from src.core import SimpleCell, Node, GridGraph, ONE_INSTANCE
from src.utils import unit_disk_graph
from src.gadgets import Pattern, crossing_ruleset
from src.mapping import Weighted, apply_crossing_gadgets, WeightedMCell, convert_mcell_to_simplecell, convert_simplecell_to_mcell


class WeightedGadget(Pattern):
    """
    Wrapper for transforming a gadget pattern to handle weights.
    
    This class decorates an existing gadget pattern to support weight mapping
    from source to target graphs.
    """
    
    def __init__(self, gadget, source_weights=None, mapped_weights=None):
        """
        Initialize a weighted gadget.
        
        Args:
            gadget: The base gadget pattern to wrap
            source_weights: Optional weights for the source graph
            mapped_weights: Optional weights for the mapped graph
        """
        super().__init__()
        self.gadget = gadget
        self.source_weights = source_weights
        self.mapped_weights = mapped_weights
        
    def size(self):
        """Return the size of the pattern."""
        return self.gadget.size()
    
    def cross_location(self):
        """Return the location of the cross."""
        return self.gadget.cross_location()
    
    def is_connected(self):
        """Return whether the pattern has connected cells."""
        return self.gadget.is_connected()
    
    def connected_nodes(self):
        """Return the list of connected nodes."""
        return self.gadget.connected_nodes()
    
    def source_graph(self):
        """Return the source graph with weights."""
        locs, graph, pins = self.gadget.source_graph()
        
        # Apply weights if provided
        if self.source_weights is not None:
            weighted_locs = []
            for i, loc in enumerate(locs):
                if i < len(self.source_weights):
                    weighted_locs.append(Node(loc.loc, weight=self.source_weights[i]))
                else:
                    weighted_locs.append(loc)
            return weighted_locs, graph, pins
        
        return locs, graph, pins
    
    def mapped_graph(self):
        """Return the mapped graph with weights."""
        locs, graph, pins = self.gadget.mapped_graph()
        
        # Apply weights if provided
        if self.mapped_weights is not None:
            weighted_locs = []
            for i, loc in enumerate(locs):
                if i < len(self.mapped_weights):
                    weighted_locs.append(Node(loc.loc, weight=self.mapped_weights[i]))
                else:
                    weighted_locs.append(loc)
            return weighted_locs, graph, pins
        
        return locs, graph, pins
    
    def vertex_overhead(self):
        """Return the vertex overhead accounting for weights."""
        return self.gadget.vertex_overhead()
    
    def match(self, matrix, i, j):
        """Match the pattern in a matrix."""
        return self.gadget.match(matrix, i, j)
    
    def apply_gadget(self, matrix, i, j):
        """Apply the gadget to a matrix."""
        return self.gadget.apply_gadget(matrix, i, j)


def weighted(gadget, source_weights=None, mapped_weights=None):
    """
    Transform a gadget to support weights.
    
    Args:
        gadget: The gadget to transform
        source_weights: Optional weights for the source graph
        mapped_weights: Optional weights for the mapped graph
        
    Returns:
        A WeightedGadget that wraps the input gadget
    """
    return WeightedGadget(gadget, source_weights, mapped_weights)


def source_centers(gadget):
    """
    Get the centers of the source graph of a gadget.
    
    Args:
        gadget: The gadget to analyze
        
    Returns:
        List of center node coordinates
    """
    locs, _, pins = gadget.source_graph()
    return [locs[p].loc for p in pins]


def mapped_centers(gadget):
    """
    Get the centers of the mapped graph of a gadget.
    
    Args:
        gadget: The gadget to analyze
        
    Returns:
        List of center node coordinates
    """
    locs, _, pins = gadget.mapped_graph()
    return [locs[p].loc for p in pins]


def move_centers(centers, dx, dy):
    """
    Move all center points by a given offset.
    
    Args:
        centers: List of center coordinates
        dx: X offset
        dy: Y offset
        
    Returns:
        New list of center coordinates
    """
    return [(x + dx, y + dy) for x, y in centers]


def trace_centers(gadget, x, y):
    """
    Trace the centers of a gadget from a specific position.
    
    Args:
        gadget: The gadget to analyze
        x: X position
        y: Y position
        
    Returns:
        Mapped center coordinates
    """
    cx, cy = gadget.cross_location()
    return move_centers(mapped_centers(gadget), x - cx, y - cy)


def map_weights(grid, weighted_grid, ruleset=None):
    """
    Map weights from one grid to another using gadget patterns.
    
    Args:
        grid: The original grid
        weighted_grid: The weighted grid to modify
        ruleset: Optional custom ruleset
        
    Returns:
        Updated weighted grid with mapped weights
    """
    # Get the appropriate ruleset
    patterns = ruleset if ruleset is not None else crossing_ruleset_weighted
    
    # Simple matrix conversion for matching
    simple_content = convert_mcell_to_simplecell(grid.content)
    weighted_content = convert_mcell_to_simplecell(weighted_grid.content)
    
    # Apply patterns
    for j in range(1, len(grid.lines)+1):
        for i in range(1, len(grid.lines)+1):
            for pattern in patterns:
                # Calculate pattern application point
                from src.mapping import crossat
                cx, cy = crossat(grid, i, j)
                pl = pattern.cross_location() if callable(pattern.cross_location) else pattern.cross_location
                x, y = cx - pl[0], cy - pl[1]
                
                # Check if pattern matches
                if pattern.match(simple_content, x, y):
                    # Get the centers of the source pattern
                    centers = trace_centers(pattern, x, y)
                    
                    # Map the weights
                    src_locs, _, _ = pattern.source_graph()
                    map_locs, _, _ = pattern.mapped_graph()
                    
                    # For each center in the mapped graph
                    for center_x, center_y in centers:
                        # Find the closest node in the pattern
                        min_dist = float('inf')
                        closest_node = None
                        
                        for node in map_locs:
                            nx, ny = node.loc
                            # Adjust for pattern position
                            nx, ny = nx + x - pl[0], ny + y - pl[1]
                            dist = (nx - center_x)**2 + (ny - center_y)**2
                            
                            if dist < min_dist:
                                min_dist = dist
                                closest_node = node
                        
                        # Update weight in the weighted grid if node exists
                        if closest_node and 0 <= center_x < len(weighted_grid.content) and 0 <= center_y < len(weighted_grid.content[0]):
                            if not weighted_content[center_x][center_y].is_empty:
                                weighted_content[center_x][center_y] = SimpleCell(
                                    occupied=True,
                                    weight=closest_node.weight
                                )
    
    # Convert back to MCell format
    convert_simplecell_to_mcell(weighted_content, weighted_grid.content)
    return weighted_grid


def map_configs_back(mapping_result, config):
    """
    Map configurations back from weighted grids.
    
    Args:
        mapping_result: The mapping result
        config: The configuration matrix
        
    Returns:
        Configuration for the original graph
    """
    # Simplified version based on mapping.map_config_back
    max_vertex = max(line.vertex for line in mapping_result.lines)
    result = [0] * (max_vertex + 1)
    
    # For each vertex line in the original mapping
    for line in mapping_result.lines:
        vertex = line.vertex
        
        # Find boundary nodes with the right weight
        for node in mapping_result.grid_graph.nodes:
            i, j = node.loc
            # Check if this is a boundary node
            if i == 0 or j == 0 or i == mapping_result.grid_graph.size[0]-1 or j == mapping_result.grid_graph.size[1]-1:
                # Take weighted values into account
                if not isinstance(node.weight, int) or node.weight == 1:
                    if 0 <= i < config.shape[0] and 0 <= j < config.shape[1]:
                        if config[i, j] != 0:
                            result[vertex] = 1
                            break
    
    return result


# Simple weighted crossing rules
def simple_gadget_rule(gadget, source_weight=1, mapped_weight=1):
    """Create a simple weighted gadget rule."""
    source_weights = [source_weight] * len(gadget.source_graph()[0])
    mapped_weights = [mapped_weight] * len(gadget.mapped_graph()[0])
    return weighted(gadget, source_weights, mapped_weights)


# Define the weighted crossing ruleset
from src.gadgets import Cross, Turn, Branch, BranchFix, WTurn, BranchFixB, TCon, TrivialTurn, EndTurn

crossing_ruleset_weighted = [
    simple_gadget_rule(Cross(has_edge=True), 1, 2),
    simple_gadget_rule(Cross(has_edge=False), 1, 2),
    simple_gadget_rule(Turn(), 1, 2),
    simple_gadget_rule(Branch(), 1, 2),
    simple_gadget_rule(BranchFix(), 1, 2),
    simple_gadget_rule(WTurn(), 1, 2),
    simple_gadget_rule(BranchFixB(), 1, 2),
    simple_gadget_rule(TCon(), 1, 2),
    simple_gadget_rule(TrivialTurn(), 1, 2),
    simple_gadget_rule(EndTurn(), 1, 2)
]