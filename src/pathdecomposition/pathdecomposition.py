"""
Path decomposition algorithms for graphs.

This module provides data structures and algorithms for computing
path decompositions of graphs, which are useful for various graph
algorithms and optimization problems.
"""

import networkx as nx
from typing import List, Tuple, Set, Dict, Any, Union, Optional
from dataclasses import dataclass
from abc import ABC

@dataclass
class Layout:
    """
    Represents a path decomposition layout of a graph.
    
    Attributes:
        vertices: List of vertices in the path decomposition order
        vsep: Vertex separation number
        neighbors: Neighbors of the vertices in the path
        disconnected: Vertices not in or adjacent to the path
    """
    vertices: List[int]
    vsep: int
    neighbors: List[int]
    disconnected: List[int]
    
    def __hash__(self):
        return hash(tuple(self.vertices))
    
    def __eq__(self, other):
        if not isinstance(other, Layout):
            return False
        return self.vsep == other.vsep and self.vertices == other.vertices
    
    @classmethod
    def from_graph(cls, g: nx.Graph, vertices: List[int]):
        """Create a Layout from a graph and vertex order."""
        vs, nbs = vsep_and_neighbors(g, vertices)
        all_vertices = set(g.nodes())
        disconnected = list(all_vertices - set(nbs) - set(vertices))
        return cls(vertices, vs, nbs, disconnected)
    
    def vsep_last(self):
        """Get the vertex separation at the last step."""
        return len(self.neighbors)
    
    def update(self, g: nx.Graph, v: int):
        """Update the layout by adding vertex v."""
        vertices = self.vertices + [v]
        vs_new, neighbors_new, disconnected = vsep_updated_neighbors(g, self, v)
        vs_new = max(self.vsep, vs_new)
        return Layout(vertices, vs_new, neighbors_new, disconnected)

def vsep_and_neighbors(g: nx.Graph, vertices: List[int]) -> Tuple[int, List[int]]:
    """
    Calculate vertex separation and neighbors.
    
    Args:
        g: Input graph
        vertices: Ordered list of vertices
        
    Returns:
        (vertex_separation, neighbors)
    """
    vs = 0
    nbs = []
    
    # For each prefix of the vertex order
    for i in range(len(vertices)):
        # Get the vertices up to i
        S = vertices[:i+1]
        S_set = set(S)
        
        # Find neighbors of S that are not in S
        current_nbs = []
        for v in g.nodes():
            if v not in S_set:
                # Check if v has a neighbor in S
                if any(u in g.neighbors(v) for u in S):
                    current_nbs.append(v)
        
        # Update max vertex separation
        vsi = len(current_nbs)
        if vsi > vs:
            vs = vsi
        
        # At the last step, save the neighbors
        if i == len(vertices) - 1:
            nbs = current_nbs
    
    return vs, nbs

def vsep_updated(g: nx.Graph, layout: Layout, v: int) -> int:
    """
    Calculate the vertex separation if v is added to the layout.
    
    Args:
        g: Input graph
        layout: Current layout
        v: Vertex to add
        
    Returns:
        Updated vertex separation
    """
    vs = layout.vsep_last()
    
    # If v is already a neighbor, it reduces vsep by 1
    if v in layout.neighbors:
        vs -= 1
    
    # For each neighbor of v not already in vertices or neighbors
    for w in g.neighbors(v):
        if w not in layout.vertices and w not in layout.neighbors:
            vs += 1
    
    # Vertex separation is the maximum of the current max and the new value
    return max(vs, layout.vsep)

def vsep_updated_neighbors(g: nx.Graph, layout: Layout, v: int) -> Tuple[int, List[int], List[int]]:
    """
    Calculate the vertex separation, neighbors, and disconnected vertices if v is added.
    
    Args:
        g: Input graph
        layout: Current layout
        v: Vertex to add
        
    Returns:
        (updated_vsep, updated_neighbors, updated_disconnected)
    """
    vs = layout.vsep_last()
    nbs = layout.neighbors.copy()
    disc = layout.disconnected.copy()
    
    # If v is already a neighbor, remove it and reduce vsep
    if v in nbs:
        nbs.remove(v)
        vs -= 1
    else:
        disc.remove(v)
    
    # Add new neighbors
    for w in g.neighbors(v):
        if w not in layout.vertices and w not in nbs:
            vs += 1
            nbs.append(w)
            if w in disc:
                disc.remove(w)
    
    vs = max(vs, layout.vsep)
    return vs, nbs, disc

class PathDecompositionMethod(ABC):
    """Base class for path decomposition methods."""
    pass

class MinhThiTrick(PathDecompositionMethod):
    """
    A path decomposition method based on the Branching method.
    
    In memory of Minh-Thi Nguyen, one of the main developers of this method.
    She left us in a truck accident at her 24 years old.
    - https://www.cbsnews.com/boston/news/cyclist-killed-minh-thi-nguyen-cambridge-bike-safety/
    """
    pass

class Greedy(PathDecompositionMethod):
    """A path decomposition method based on the Greedy method."""
    def __init__(self, nrepeat=10):
        self.nrepeat = nrepeat

def pathwidth(g: nx.Graph, method) -> Layout:
    """
    Compute the optimal path decomposition of graph g.
    
    Args:
        g: Input graph
        method: The path decomposition method (MinhThiTrick or Greedy)
        
    Returns:
        A Layout object representing the path decomposition
    """
    if isinstance(method, MinhThiTrick):
        from .branching import branch_and_bound
        return branch_and_bound(g)
    elif isinstance(method, Greedy):
        from .greedy import greedy_decompose
        
        results = []
        for _ in range(method.nrepeat):
            results.append(greedy_decompose(g))
        
        # Return the layout with minimum vertex separation
        return min(results, key=lambda layout: layout.vsep)
    else:
        raise ValueError(f"Unknown path decomposition method: {method}")