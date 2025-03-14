"""
Branch and bound algorithm for computing optimal path decompositions.

This module implements the branch and bound algorithm for computing
the optimal path decomposition of a graph, following the approach in:

Coudert, D., Mazauric, D., & Nisse, N. (2014).
Experimental evaluation of a branch and bound algorithm for computing pathwidth.
https://doi.org/10.1007/978-3-319-07959-2_5
"""

import networkx as nx
from typing import Dict, List, Optional, Set, Tuple
from .pathdecomposition import Layout, vsep_updated
from .greedy import greedy_exact

def branch_and_bound(g: nx.Graph) -> Layout:
    """
    Compute the optimal path decomposition using branch and bound.
    
    Args:
        g: Input graph
        
    Returns:
        A Layout object representing the optimal path decomposition
    """
    # Start with empty layout
    empty_layout = Layout.from_graph(g, [])
    
    # Use all vertices for the initial best layout
    vertices = list(g.nodes())
    best_layout = Layout.from_graph(g, vertices)
    
    # Memoization dictionary for visited layouts
    visited_layouts = {}
    
    # Call the recursive helper function
    return branch_and_bound_helper(g, empty_layout, best_layout, visited_layouts)

def branch_and_bound_helper(g: nx.Graph, p: Layout, l: Layout, 
                          visited: Dict[Layout, bool]) -> Layout:
    """
    Recursive helper function for branch and bound.
    
    This implements the branch_and_bound! function from the Julia version.
    
    Args:
        g: Input graph
        p: Current prefix layout
        l: Best layout found so far
        visited: Dictionary of visited layouts
        
    Returns:
        The best layout found
    """
    # Get all vertices
    vertices = list(g.nodes())
    
    # Check if the current layout is promising
    if p.vsep < l.vsep and p not in visited:
        # Apply greedy exact rules to extend the layout
        p2 = greedy_exact(g, p)
        
        # Calculate vertex separation
        vsep_p2 = p2.vsep
        
        # Check if we have a complete layout with better separation
        p2_vertices = set(p2.vertices)
        if set(vertices) == p2_vertices and vsep_p2 < l.vsep:
            return p2
        else:
            # Remember the current best vertex separation
            current = l.vsep
            
            # Combine neighbors and disconnected vertices
            remaining = p2.neighbors + p2.disconnected
            
            # Sort vertices by increasing updated vertex separation
            vsep_order = sorted(range(len(remaining)), 
                               key=lambda i: vsep_updated(g, p2, remaining[i]))
            
            # Try each vertex in order
            for i in vsep_order:
                v = remaining[i]
                
                # Check if adding v is promising
                if vsep_updated(g, p2, v) < l.vsep:
                    # Recursively try adding v
                    l3 = branch_and_bound_helper(g, p2.update(g, v), l, visited)
                    
                    # Update best layout if improved
                    if l3.vsep < l.vsep:
                        l = l3
            
            # Update visited layouts dictionary
            visited[p] = not (l.vsep < current and p.vsep == l.vsep)
    
    return l