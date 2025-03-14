"""
Greedy path decomposition algorithm for graphs.

This module implements greedy algorithms for computing
path decompositions of graphs, following the approach in the Julia package.
"""

import networkx as nx
import random
from typing import List, Set, Dict, Any, Tuple
from .pathdecomposition import Layout, vsep_updated, vsep_updated_neighbors

def greedy_exact(g: nx.Graph, p: Layout) -> Layout:
    """
    Greedily extend a partial layout using exact rules.
    
    This algorithm implements the greedy_exact function from the Julia version,
    which applies deterministic rules to add vertices that can be safely added.
    
    Args:
        g: Input graph
        p: Current partial layout
        
    Returns:
        Updated Layout object
    """
    keep_going = True
    
    while keep_going:
        keep_going = False
        
        # Try vertices in disconnected and neighbors lists
        for vertex_list in [p.disconnected, p.neighbors]:
            for v in vertex_list.copy():  # Using copy to avoid modification during iteration
                # Check if all neighbors of v are in vertices or neighbors
                all_neighbors_covered = True
                
                for nb in g.neighbors(v):
                    if nb not in p.vertices and nb not in p.neighbors:
                        all_neighbors_covered = False
                        break
                
                # If all neighbors are covered, add v to the layout
                if all_neighbors_covered:
                    p = p.update(g, v)
                    keep_going = True
                    break  # Need to break since we've modified the list
            
            if keep_going:
                break
        
        # If we didn't add any vertex in the previous step
        if not keep_going:
            # Check for vertices with exactly one uncovered neighbor
            for v in p.neighbors.copy():
                uncovered_neighbors = [nb for nb in g.neighbors(v)
                                     if nb not in p.vertices and nb not in p.neighbors]
                
                if len(uncovered_neighbors) == 1:
                    p = p.update(g, v)
                    keep_going = True
                    break
    
    return p

def greedy_step(g: nx.Graph, p: Layout, vertex_list: List[int]) -> Layout:
    """
    Take a single greedy step by choosing the best vertex from a list.
    
    Args:
        g: Input graph
        p: Current layout
        vertex_list: List of vertices to consider
        
    Returns:
        Updated Layout with one more vertex
    """
    # Calculate the layout after adding each vertex
    layouts = [p.update(g, v) for v in vertex_list]
    
    # Get the vertex separation for each layout
    costs = [layout.vsep for layout in layouts]
    
    # Find the minimum cost
    best_cost = min(costs)
    
    # Select a random layout with the minimum cost (to break ties)
    best_indices = [i for i, cost in enumerate(costs) if cost == best_cost]
    chosen_index = random.choice(best_indices)
    
    return layouts[chosen_index]

def greedy_decompose(g: nx.Graph) -> Layout:
    """
    Compute a path decomposition using the greedy algorithm.
    
    This follows the Julia implementation of greedy_decompose.
    
    Args:
        g: Input graph
        
    Returns:
        A Layout object representing the path decomposition
    """
    # Start with an empty layout
    vertices = []
    neighbors = []
    disconnected = list(g.nodes())
    p = Layout(vertices, 0, neighbors, disconnected)
    
    while True:
        # Apply exact rules to extend the layout
        p = greedy_exact(g, p)
        
        # If there are neighbors, choose the best one
        if p.neighbors:
            p = greedy_step(g, p, p.neighbors)
        # Otherwise, if there are disconnected vertices, choose the best one
        elif p.disconnected:
            p = greedy_step(g, p, p.disconnected)
        # If no more vertices to add, we're done
        else:
            break
    
    return p