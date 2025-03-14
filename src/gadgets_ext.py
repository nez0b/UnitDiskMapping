import networkx as nx
import numpy as np
from typing import List, Tuple, Dict, Any, Union, Optional, Callable
from copy import deepcopy
from functools import lru_cache
import math

from .core import SimpleCell, Node, GridGraph, ONE_INSTANCE
from .utils import unit_disk_graph
from .gadgets import Pattern, Cross, Turn, Branch, BranchFix, WTurn
from .gadgets import BranchFixB, TCon, TrivialTurn, EndTurn, crossing_ruleset


class ComplexGadget(Pattern):
    """Extended base class for more complex gadgets with boundary configuration support."""
    
    def __init__(self):
        super().__init__()
    
    def source_boundary_config(self):
        """
        Get the boundary configuration of the source graph.
        
        Returns:
            A dictionary mapping boundary indices to values
        """
        src_locs, _, pins = self.source_graph()
        return {i: 1 for i in range(len(pins))}
    
    def mapped_boundary_config(self):
        """
        Get the boundary configuration of the mapped graph.
        
        Returns:
            A dictionary mapping boundary indices to values
        """
        map_locs, _, pins = self.mapped_graph()
        return {i: 1 for i in range(len(pins))}
    
    def source_entry_to_configs(self, entry):
        """
        Convert a source entry to configurations.
        
        Args:
            entry: Entry configuration
            
        Returns:
            List of configurations
        """
        return [entry]
    
    def mapped_entry_to_compact(self, entry):
        """
        Convert a mapped entry to compact representation.
        
        Args:
            entry: Entry configuration
            
        Returns:
            Compact representation
        """
        return entry
    
    def mis_overhead(self):
        """
        Calculate the MIS overhead of the pattern.
        
        Returns:
            MIS overhead value
        """
        return self.vertex_overhead()


class EnhancedRotatedGadget(ComplexGadget):
    """
    An enhanced version of RotatedGadget with improved transformation handling.
    """
    
    def __init__(self, gadget, n):
        super().__init__()
        self.gadget = gadget
        self.n = n % 4  # Normalize to 0-3
        self._cached_values = {}
    
    def size(self):
        """Return the size of the pattern after rotation."""
        m, n = self.gadget.size()
        return (n, m) if self.n % 2 == 1 else (m, n)
    
    @lru_cache(maxsize=8)
    def _get_offset(self):
        """Get the offset needed to keep the pattern within bounds."""
        m, n = self.gadget.size()
        center = self.gadget.cross_location()
        
        # Calculate transformed corners
        min_x, min_y = float('inf'), float('inf')
        corners = [(0, 0), (0, n-1), (m-1, 0), (m-1, n-1)]
        
        for corner in corners:
            x, y = corner
            loc = (x, y)
            for _ in range(self.n):
                loc = self._rotate90(loc, center)
            min_x = min(min_x, loc[0])
            min_y = min(min_y, loc[1])
        
        return (1 - min_x, 1 - min_y)
    
    @staticmethod
    def _rotate90(loc, center):
        """Rotate a location 90 degrees around a center point."""
        x, y = loc
        cx, cy = center
        return (cx + (cy - y), cy + (x - cx))
    
    def cross_location(self):
        """Return the location of the cross after rotation."""
        if 'cross_location' in self._cached_values:
            return self._cached_values['cross_location']
        
        # Get the original cross location
        original_center = self.gadget.cross_location()
        
        # Apply rotation around the original center
        center = original_center
        for _ in range(self.n):
            center = self._rotate90(center, original_center)
            
        # Apply offset to keep in bounds
        offset = self._get_offset()
        result = (center[0] + offset[0], center[1] + offset[1])
        
        self._cached_values['cross_location'] = result
        return result
    
    def is_connected(self):
        """Return whether the pattern has connected cells."""
        return self.gadget.is_connected()
    
    def connected_nodes(self):
        """Return the list of connected nodes."""
        return self.gadget.connected_nodes()
    
    def _apply_transform(self, node, center):
        """Apply the rotation transformation to a node."""
        x, y = node.loc
        loc = (x, y)
        for _ in range(self.n):
            loc = self._rotate90(loc, center)
        offset = self._get_offset()
        return Node((loc[0] + offset[0], loc[1] + offset[1]), weight=node.weight)
    
    def source_graph(self):
        """Return the source graph after rotation."""
        if 'source_graph' in self._cached_values:
            return self._cached_values['source_graph']
        
        locs, graph, pins = self.gadget.source_graph()
        center = self.gadget.cross_location()
        transformed_locs = [self._apply_transform(loc, center) for loc in locs]
        result = (transformed_locs, graph, pins)
        
        self._cached_values['source_graph'] = result
        return result
    
    def mapped_graph(self):
        """Return the mapped graph after rotation."""
        if 'mapped_graph' in self._cached_values:
            return self._cached_values['mapped_graph']
        
        locs, graph, pins = self.gadget.mapped_graph()
        center = self.gadget.cross_location()
        transformed_locs = [self._apply_transform(loc, center) for loc in locs]
        transformed_graph = unit_disk_graph([loc.loc for loc in transformed_locs], 1.5)
        result = (transformed_locs, transformed_graph, pins)
        
        self._cached_values['mapped_graph'] = result
        return result
    
    def source_boundary_config(self):
        """Get the boundary configuration of the source graph."""
        # Rotate the boundary configurations
        base_config = self.gadget.source_boundary_config() if hasattr(self.gadget, 'source_boundary_config') else super().source_boundary_config()
        # In the simple case, rotation doesn't change boundary configs
        return base_config
    
    def mapped_boundary_config(self):
        """Get the boundary configuration of the mapped graph."""
        # Rotate the boundary configurations
        base_config = self.gadget.mapped_boundary_config() if hasattr(self.gadget, 'mapped_boundary_config') else super().mapped_boundary_config()
        # In the simple case, rotation doesn't change boundary configs
        return base_config
    
    def source_matrix(self):
        """Override to create a properly rotated source matrix."""
        if 'source_matrix' in self._cached_values:
            return self._cached_values['source_matrix']
            
        # Get the transformed nodes and build the matrix directly
        transformed_locs, _, _ = self.source_graph()
        m, n = self.size()  # This already handles the size correctly for rotated patterns
        
        # Create empty matrix
        matrix = [[SimpleCell.create_empty() for _ in range(n)] for _ in range(m)]
        
        # Add nodes to the matrix
        for node in transformed_locs:
            i, j = node.loc
            if 0 <= i < m and 0 <= j < n:  # Ensure we're within bounds
                matrix[i][j] = SimpleCell(occupied=True, weight=node.weight)
                
        self._cached_values['source_matrix'] = matrix
        return matrix

    def source_entry_to_configs(self, entry):
        """Convert a source entry to configurations."""
        if hasattr(self.gadget, 'source_entry_to_configs'):
            # Apply the rotation to each config
            base_configs = self.gadget.source_entry_to_configs(entry)
            # In the simple case, return as is
            return base_configs
        return super().source_entry_to_configs(entry)
    
    def mapped_matrix(self):
        """Override to create a properly rotated mapped matrix."""
        if 'mapped_matrix' in self._cached_values:
            return self._cached_values['mapped_matrix']
            
        # Get the transformed nodes and build the matrix directly
        transformed_locs, _, _ = self.mapped_graph()
        m, n = self.size()  # This already handles the size correctly for rotated patterns
        
        # Create empty matrix
        matrix = [[SimpleCell.create_empty() for _ in range(n)] for _ in range(m)]
        
        # Add nodes to the matrix
        for node in transformed_locs:
            i, j = node.loc
            if 0 <= i < m and 0 <= j < n:  # Ensure we're within bounds
                matrix[i][j] = SimpleCell(occupied=True, weight=node.weight)
                
        self._cached_values['mapped_matrix'] = matrix
        return matrix
        
    def mapped_entry_to_compact(self, entry):
        """Convert a mapped entry to compact representation."""
        if hasattr(self.gadget, 'mapped_entry_to_compact'):
            # Apply the rotation to the compact representation
            return self.gadget.mapped_entry_to_compact(entry)
        return super().mapped_entry_to_compact(entry)
    
    def mis_overhead(self):
        """Calculate the MIS overhead of the pattern."""
        # MIS overhead doesn't change with rotation
        if hasattr(self.gadget, 'mis_overhead'):
            return self.gadget.mis_overhead()
        return super().mis_overhead()


class EnhancedReflectedGadget(ComplexGadget):
    """
    An enhanced version of ReflectedGadget with improved transformation handling.
    """
    
    def __init__(self, gadget, mirror):
        super().__init__()
        self.gadget = gadget
        self.mirror = mirror  # "x", "y", "diag", or "offdiag"
        self._cached_values = {}
    
    def size(self):
        """Return the size of the pattern after reflection."""
        m, n = self.gadget.size()
        return (n, m) if self.mirror in ["diag", "offdiag"] else (m, n)
    
    @lru_cache(maxsize=8)
    def _get_offset(self):
        """Get the offset needed to keep the pattern within bounds."""
        m, n = self.gadget.size()
        center = self.gadget.cross_location()
        
        # Calculate transformed corners
        min_x, min_y = float('inf'), float('inf')
        corners = [(0, 0), (0, n-1), (m-1, 0), (m-1, n-1)]
        
        for corner in corners:
            x, y = corner
            if self.mirror == "x":
                loc = self._reflectx((x, y), center)
            elif self.mirror == "y":
                loc = self._reflecty((x, y), center)
            elif self.mirror == "diag":
                loc = self._reflectdiag((x, y), center)
            elif self.mirror == "offdiag":
                loc = self._reflectoffdiag((x, y), center)
            else:
                raise ValueError(f"Invalid mirror direction: {self.mirror}")
                
            min_x = min(min_x, loc[0])
            min_y = min(min_y, loc[1])
        
        return (1 - min_x, 1 - min_y)
    
    @staticmethod
    def _reflectx(loc, center):
        """Reflect a location across the x-axis through the center point."""
        x, y = loc
        cx, cy = center
        return (x, cy - (y - cy))
    
    @staticmethod
    def _reflecty(loc, center):
        """Reflect a location across the y-axis through the center point."""
        x, y = loc
        cx, cy = center
        return (cx - (x - cx), y)
    
    @staticmethod
    def _reflectdiag(loc, center):
        """Reflect a location across the main diagonal through the center point."""
        x, y = loc
        cx, cy = center
        dx, dy = x - cx, y - cy
        return (cx + dy, cy + dx)
    
    @staticmethod
    def _reflectoffdiag(loc, center):
        """Reflect a location across the off-diagonal through the center point."""
        x, y = loc
        cx, cy = center
        dx, dy = x - cx, y - cy
        return (cx - dy, cy - dx)
    
    def cross_location(self):
        """Return the location of the cross after reflection."""
        if 'cross_location' in self._cached_values:
            return self._cached_values['cross_location']
        
        # Get the original cross location
        original_center = self.gadget.cross_location()
        cx, cy = original_center
        
        # Apply reflection around the original center
        if self.mirror == "x":
            loc = self._reflectx(original_center, original_center)
        elif self.mirror == "y":
            loc = self._reflecty(original_center, original_center)
        elif self.mirror == "diag":
            loc = self._reflectdiag(original_center, original_center)
        elif self.mirror == "offdiag":
            loc = self._reflectoffdiag(original_center, original_center)
        else:
            loc = original_center
        
        # Apply offset to keep in bounds
        offset = self._get_offset()
        result = (loc[0] + offset[0], loc[1] + offset[1])
        
        self._cached_values['cross_location'] = result
        return result
    
    def is_connected(self):
        """Return whether the pattern has connected cells."""
        return self.gadget.is_connected()
    
    def connected_nodes(self):
        """Return the list of connected nodes."""
        return self.gadget.connected_nodes()
    
    def _apply_transform(self, node, center):
        """Apply the reflection transformation to a node."""
        x, y = node.loc
        
        if self.mirror == "x":
            loc = self._reflectx((x, y), center)
        elif self.mirror == "y":
            loc = self._reflecty((x, y), center)
        elif self.mirror == "diag":
            loc = self._reflectdiag((x, y), center)
        elif self.mirror == "offdiag":
            loc = self._reflectoffdiag((x, y), center)
        else:
            raise ValueError(f"Invalid mirror direction: {self.mirror}")
        
        offset = self._get_offset()
        return Node((loc[0] + offset[0], loc[1] + offset[1]), weight=node.weight)
    
    def source_graph(self):
        """Return the source graph after reflection."""
        if 'source_graph' in self._cached_values:
            return self._cached_values['source_graph']
        
        locs, graph, pins = self.gadget.source_graph()
        center = self.gadget.cross_location()
        transformed_locs = [self._apply_transform(loc, center) for loc in locs]
        result = (transformed_locs, graph, pins)
        
        self._cached_values['source_graph'] = result
        return result
    
    def mapped_graph(self):
        """Return the mapped graph after reflection."""
        if 'mapped_graph' in self._cached_values:
            return self._cached_values['mapped_graph']
        
        locs, graph, pins = self.gadget.mapped_graph()
        center = self.gadget.cross_location()
        transformed_locs = [self._apply_transform(loc, center) for loc in locs]
        transformed_graph = unit_disk_graph([loc.loc for loc in transformed_locs], 1.5)
        result = (transformed_locs, transformed_graph, pins)
        
        self._cached_values['mapped_graph'] = result
        return result
    
    def source_boundary_config(self):
        """Get the boundary configuration of the source graph."""
        # Reflect the boundary configurations
        base_config = self.gadget.source_boundary_config() if hasattr(self.gadget, 'source_boundary_config') else super().source_boundary_config()
        # In the simple case, reflection doesn't change boundary configs
        return base_config
    
    def mapped_boundary_config(self):
        """Get the boundary configuration of the mapped graph."""
        # Reflect the boundary configurations
        base_config = self.gadget.mapped_boundary_config() if hasattr(self.gadget, 'mapped_boundary_config') else super().mapped_boundary_config()
        # In the simple case, reflection doesn't change boundary configs
        return base_config
    
    def source_matrix(self):
        """Override to create a properly reflected source matrix."""
        if 'source_matrix' in self._cached_values:
            return self._cached_values['source_matrix']
            
        # Get the transformed nodes and build the matrix directly
        transformed_locs, _, _ = self.source_graph()
        m, n = self.size()  # This already handles the size correctly for reflected patterns
        
        # Create empty matrix
        matrix = [[SimpleCell.create_empty() for _ in range(n)] for _ in range(m)]
        
        # Add nodes to the matrix
        for node in transformed_locs:
            i, j = node.loc
            if 0 <= i < m and 0 <= j < n:  # Ensure we're within bounds
                matrix[i][j] = SimpleCell(occupied=True, weight=node.weight)
                
        self._cached_values['source_matrix'] = matrix
        return matrix

    def mapped_matrix(self):
        """Override to create a properly reflected mapped matrix."""
        if 'mapped_matrix' in self._cached_values:
            return self._cached_values['mapped_matrix']
            
        # Get the transformed nodes and build the matrix directly
        transformed_locs, _, _ = self.mapped_graph()
        m, n = self.size()  # This already handles the size correctly for reflected patterns
        
        # Create empty matrix
        matrix = [[SimpleCell.create_empty() for _ in range(n)] for _ in range(m)]
        
        # Add nodes to the matrix
        for node in transformed_locs:
            i, j = node.loc
            if 0 <= i < m and 0 <= j < n:  # Ensure we're within bounds
                matrix[i][j] = SimpleCell(occupied=True, weight=node.weight)
                
        self._cached_values['mapped_matrix'] = matrix
        return matrix
    
    def source_entry_to_configs(self, entry):
        """Convert a source entry to configurations."""
        if hasattr(self.gadget, 'source_entry_to_configs'):
            # Apply the reflection to each config
            base_configs = self.gadget.source_entry_to_configs(entry)
            # In the simple case, return as is
            return base_configs
        return super().source_entry_to_configs(entry)
    
    def mapped_entry_to_compact(self, entry):
        """Convert a mapped entry to compact representation."""
        if hasattr(self.gadget, 'mapped_entry_to_compact'):
            # Apply the reflection to the compact representation
            return self.gadget.mapped_entry_to_compact(entry)
        return super().mapped_entry_to_compact(entry)
    
    def mis_overhead(self):
        """Calculate the MIS overhead of the pattern."""
        # MIS overhead doesn't change with reflection
        if hasattr(self.gadget, 'mis_overhead'):
            return self.gadget.mis_overhead()
        return super().mis_overhead()


# Advanced gadget transformations

def enhanced_rotated(gadget, n):
    """
    Create an enhanced rotated gadget.
    
    Args:
        gadget: The base gadget
        n: Number of 90-degree rotations
        
    Returns:
        Enhanced rotated gadget
    """
    return EnhancedRotatedGadget(gadget, n)


def enhanced_reflected(gadget, mirror):
    """
    Create an enhanced reflected gadget.
    
    Args:
        gadget: The base gadget
        mirror: Reflection direction ("x", "y", "diag", "offdiag")
        
    Returns:
        Enhanced reflected gadget
    """
    return EnhancedReflectedGadget(gadget, mirror)


def enhanced_rotated_and_reflected(pattern):
    """
    Generate all unique rotated and reflected variants of a pattern with enhanced support.
    
    Args:
        pattern: The base pattern
        
    Returns:
        List of all unique transformed patterns
    """
    patterns = [pattern]  # Start with the original pattern
    
    # For simple comparison, let's use pattern's size and cross_location
    # as a simple heuristic for uniqueness instead of full matrix comparison
    def pattern_signature(p):
        """Get a signature for a pattern based on size and cross location."""
        try:
            size = p.size()
            cross_loc = p.cross_location()
            return (size, cross_loc)
        except Exception:
            return None
            
    # Get signatures for checking uniqueness
    signatures = [pattern_signature(pattern)]
    
    # Add rotated variants with robust error handling
    for i in range(1, 4):
        try:
            # Create rotated variant
            rotated = enhanced_rotated(pattern, i)
            
            # Get its signature for uniqueness check
            rotated_sig = pattern_signature(rotated)
            if rotated_sig is None:
                continue  # Skip if we can't get a signature
            
            # Check if this is a new unique pattern
            if rotated_sig not in signatures:
                patterns.append(rotated)
                signatures.append(rotated_sig)
                print(f"Added rotated variant {i * 90}°")
        except Exception as e:
            print(f"Error creating rotated variant {i}: {e}")
    
    # Add reflected variants with robust error handling
    for mirror in ["x", "y", "diag", "offdiag"]:
        try:
            # Create reflected variant
            reflected = enhanced_reflected(pattern, mirror)
            
            # Get its signature for uniqueness check
            reflected_sig = pattern_signature(reflected)
            if reflected_sig is None:
                continue  # Skip if we can't get a signature
            
            # Check if this is a new unique pattern
            if reflected_sig not in signatures:
                patterns.append(reflected)
                signatures.append(reflected_sig)
                print(f"Added reflected variant {mirror}")
        except Exception as e:
            print(f"Error creating reflected variant {mirror}: {e}")
    
    return patterns


# Enhanced complex gadget patterns

class StarPattern(ComplexGadget):
    """Pattern for star-like configurations."""
    
    def __init__(self):
        super().__init__()
    
    def size(self):
        """Return the size of the pattern."""
        return (5, 5)
    
    def cross_location(self):
        """Return the location of the cross."""
        return (2, 2)
    
    def source_graph(self):
        """Return the source graph (locs, graph, pins)."""
        # ⋅ ⋅ ● ⋅ ⋅
        # ⋅ ⋅ ● ⋅ ⋅
        # ● ● ● ● ●
        # ⋅ ⋅ ● ⋅ ⋅
        # ⋅ ⋅ ● ⋅ ⋅
        locs = [
            Node((0, 2)), Node((1, 2)), Node((2, 0)), Node((2, 1)), 
            Node((2, 2)), Node((2, 3)), Node((2, 4)), Node((3, 2)), 
            Node((4, 2))
        ]
        edges = [(1, 2), (2, 5), (3, 5), (5, 6), (5, 7), (5, 8), (5, 9)]
        g = nx.Graph()
        for i, (src, dst) in enumerate(edges):
            g.add_edge(src - 1, dst - 1)  # Convert from 1-indexed to 0-indexed
        pins = [0, 2, 6, 8]  # convert from 1-indexed to 0-indexed
        return locs, g, pins
    
    def mapped_graph(self):
        """Return the mapped graph (locs, graph, pins)."""
        # ⋅ ⋅ ● ⋅ ⋅
        # ⋅ ● ⋅ ● ⋅
        # ● ⋅ ● ⋅ ●
        # ⋅ ● ⋅ ● ⋅
        # ⋅ ⋅ ● ⋅ ⋅
        locs = [
            Node((0, 2)), Node((1, 1)), Node((1, 3)), 
            Node((2, 0)), Node((2, 2)), Node((2, 4)), 
            Node((3, 1)), Node((3, 3)), Node((4, 2))
        ]
        pins = [0, 3, 5, 8]  # convert from 1-indexed to 0-indexed
        return locs, unit_disk_graph([loc.loc for loc in locs], 1.5), pins


class SpiralPattern(ComplexGadget):
    """Pattern for spiral configurations."""
    
    def __init__(self):
        super().__init__()
    
    def size(self):
        """Return the size of the pattern."""
        return (5, 5)
    
    def cross_location(self):
        """Return the location of the cross."""
        return (2, 2)
    
    def source_graph(self):
        """Return the source graph (locs, graph, pins)."""
        # ⋅ ⋅ ● ⋅ ⋅
        # ⋅ ⋅ ● ⋅ ⋅
        # ● ● ● ● ●
        # ⋅ ⋅ ⋅ ⋅ ●
        # ⋅ ⋅ ⋅ ⋅ ●
        locs = [
            Node((0, 2)), Node((1, 2)), Node((2, 0)), Node((2, 1)), 
            Node((2, 2)), Node((2, 3)), Node((2, 4)), Node((3, 4)), 
            Node((4, 4))
        ]
        edges = [(1, 2), (2, 5), (3, 5), (5, 6), (5, 7), (7, 8), (8, 9)]
        g = nx.Graph()
        for i, (src, dst) in enumerate(edges):
            g.add_edge(src - 1, dst - 1)  # Convert from 1-indexed to 0-indexed
        pins = [0, 2, 6, 8]  # convert from 1-indexed to 0-indexed
        return locs, g, pins
    
    def mapped_graph(self):
        """Return the mapped graph (locs, graph, pins)."""
        # ⋅ ⋅ ● ⋅ ⋅
        # ⋅ ● ⋅ ⋅ ⋅
        # ● ⋅ ● ● ●
        # ⋅ ⋅ ⋅ ⋅ ●
        # ⋅ ⋅ ⋅ ⋅ ●
        locs = [
            Node((0, 2)), Node((1, 1)), 
            Node((2, 0)), Node((2, 2)), Node((2, 3)), Node((2, 4)), 
            Node((3, 4)), Node((4, 4))
        ]
        pins = [0, 2, 5, 7]  # convert from 1-indexed to 0-indexed
        return locs, unit_disk_graph([loc.loc for loc in locs], 1.5), pins


# Additional complex patterns that can be further enhanced
class DiagonalCross(ComplexGadget):
    """Pattern for diagonal crossings."""
    
    def __init__(self):
        super().__init__()
    
    def size(self):
        """Return the size of the pattern."""
        return (5, 5)
    
    def cross_location(self):
        """Return the location of the cross."""
        return (2, 2)
    
    def source_graph(self):
        """Return the source graph (locs, graph, pins)."""
        # ● ⋅ ⋅ ⋅ ⋅
        # ⋅ ● ⋅ ⋅ ⋅
        # ⋅ ⋅ ● ⋅ ⋅
        # ⋅ ⋅ ⋅ ● ⋅
        # ⋅ ⋅ ⋅ ⋅ ●
        locs = [
            Node((0, 0)), Node((1, 1)), Node((2, 2)), Node((3, 3)), Node((4, 4))
        ]
        edges = [(1, 2), (2, 3), (3, 4), (4, 5)]
        g = nx.Graph()
        for i, (src, dst) in enumerate(edges):
            g.add_edge(src - 1, dst - 1)  # Convert from 1-indexed to 0-indexed
        pins = [0, 4]  # convert from 1-indexed to 0-indexed
        return locs, g, pins
    
    def mapped_graph(self):
        """Return the mapped graph (locs, graph, pins)."""
        # ● ⋅ ⋅ ⋅ ⋅
        # ⋅ ⋅ ⋅ ⋅ ⋅
        # ⋅ ⋅ ● ⋅ ⋅
        # ⋅ ⋅ ⋅ ⋅ ⋅
        # ⋅ ⋅ ⋅ ⋅ ●
        locs = [
            Node((0, 0)), Node((2, 2)), Node((4, 4))
        ]
        pins = [0, 2]  # convert from 1-indexed to 0-indexed
        return locs, unit_disk_graph([loc.loc for loc in locs], 1.5), pins


class DoubleCross(ComplexGadget):
    """Pattern for double crossings."""
    
    def __init__(self):
        super().__init__()
    
    def size(self):
        """Return the size of the pattern."""
        return (7, 7)
    
    def cross_location(self):
        """Return the location of the cross."""
        return (3, 3)
    
    def source_graph(self):
        """Return the source graph (locs, graph, pins)."""
        # ⋅ ⋅ ⋅ ● ⋅ ⋅ ⋅
        # ⋅ ⋅ ⋅ ● ⋅ ⋅ ⋅
        # ⋅ ⋅ ⋅ ● ⋅ ⋅ ⋅
        # ● ● ● ● ● ● ●
        # ⋅ ⋅ ⋅ ● ⋅ ⋅ ⋅
        # ⋅ ⋅ ⋅ ● ⋅ ⋅ ⋅
        # ⋅ ⋅ ⋅ ● ⋅ ⋅ ⋅
        locs = [
            Node((0, 3)), Node((1, 3)), Node((2, 3)), 
            Node((3, 0)), Node((3, 1)), Node((3, 2)), Node((3, 3)), 
            Node((3, 4)), Node((3, 5)), Node((3, 6)), 
            Node((4, 3)), Node((5, 3)), Node((6, 3))
        ]
        edges = [
            (1, 2), (2, 3), (3, 7), (4, 5), (5, 6), (6, 7), 
            (7, 8), (8, 9), (9, 10), (7, 11), (11, 12), (12, 13)
        ]
        g = nx.Graph()
        for i, (src, dst) in enumerate(edges):
            g.add_edge(src - 1, dst - 1)  # Convert from 1-indexed to 0-indexed
        pins = [0, 3, 9, 12]  # convert from 1-indexed to 0-indexed
        return locs, g, pins
    
    def mapped_graph(self):
        """Return the mapped graph (locs, graph, pins)."""
        # ⋅ ⋅ ⋅ ● ⋅ ⋅ ⋅
        # ⋅ ⋅ ⋅ ⋅ ⋅ ⋅ ⋅
        # ⋅ ⋅ ● ⋅ ● ⋅ ⋅
        # ● ⋅ ⋅ ⋅ ⋅ ⋅ ●
        # ⋅ ⋅ ● ⋅ ● ⋅ ⋅
        # ⋅ ⋅ ⋅ ⋅ ⋅ ⋅ ⋅
        # ⋅ ⋅ ⋅ ● ⋅ ⋅ ⋅
        locs = [
            Node((0, 3)), Node((2, 2)), Node((2, 4)), 
            Node((3, 0)), Node((3, 6)), 
            Node((4, 2)), Node((4, 4)), Node((6, 3))
        ]
        pins = [0, 3, 4, 7]  # convert from 1-indexed to 0-indexed
        return locs, unit_disk_graph([loc.loc for loc in locs], 1.5), pins


# Advanced ruleset with complex patterns
enhanced_crossing_ruleset = [
    # Original patterns
    Cross(has_edge=True),
    Cross(has_edge=False),
    Turn(),
    Branch(),
    BranchFix(),
    WTurn(),
    BranchFixB(),
    TCon(),
    TrivialTurn(),
    EndTurn(),
    
    # New complex patterns
    StarPattern(),
    SpiralPattern(),
    DiagonalCross(),
    DoubleCross()
]


# Generate all variations of complex patterns
def generate_enhanced_ruleset():
    """
    Generate an enhanced ruleset with all variations of complex patterns.
    
    Returns:
        List of all pattern variations
    """
    result = []
    
    # Add basic patterns with variations
    for pattern in enhanced_crossing_ruleset:
        # Add the original pattern
        result.append(pattern)
        
        # Add rotated and reflected variations for complex patterns
        if isinstance(pattern, (StarPattern, SpiralPattern, DiagonalCross, DoubleCross)):
            variations = enhanced_rotated_and_reflected(pattern)
            # Skip the first one as it's the original
            result.extend(variations[1:])
    
    return result


# Complete enhanced ruleset with all variations
complete_enhanced_crossing_ruleset = generate_enhanced_ruleset()