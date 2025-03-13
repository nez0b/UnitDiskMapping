import networkx as nx
import random
from typing import List, Set, Dict
from .pathdecomposition import Layout, vsep_updated

def greedy_decompose(g: nx.Graph) -> Layout:
    """
    Greedily decompose a graph into a path decomposition.
    
    Args:
        g: Input graph
        
    Returns:
        A Layout object representing the path decomposition
    """
    # Initialize with empty layout
    vertices = []
    neighbors = []
    disconnected = list(g.nodes())
    layout = Layout(vertices, 0, neighbors, disconnected)
    
    # Remaining vertices to process
    remaining = set(g.nodes())
    
    # Process all vertices
    while remaining:
        # Get the scores for all remaining vertices
        scores = {}
        for v in remaining:
            # Calculate the impact on vertex separation
            new_vsep = vsep_updated(g, layout, v)
            
            # Calculate the score (lower is better)
            # Weight by degree to prioritize high-degree vertices
            scores[v] = (new_vsep, -g.degree(v))
        
        # Select the vertex with the lowest score
        # Break ties by random selection of equally good vertices
        min_score = min(scores.values())
        candidates = [v for v in scores if scores[v] == min_score]
        v = random.choice(candidates)
        
        # Update the layout
        layout = layout.update(g, v)
        
        # Remove v from remaining
        remaining.remove(v)
    
    return layout