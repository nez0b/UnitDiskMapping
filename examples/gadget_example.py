"""
Example demonstrating the use of gadgets in unit disk mapping.

This script shows how to use gadgets to create and transform
crossing patterns for unit disk graph mapping.
"""

import networkx as nx
import matplotlib.pyplot as plt
from matplotlib.patches import Circle
import numpy as np

from src.unit_disk_mapping import (
    Cross, Turn, Branch, BranchFix, WTurn, BranchFixB, TCon, TrivialTurn, EndTurn,
    RotatedGadget, ReflectedGadget, crossing_ruleset
)


def plot_pattern(ax, pattern, title):
    """Plot a pattern on a given matplotlib axis."""
    source_matrix = pattern.source_matrix()
    m, n = len(source_matrix), len(source_matrix[0])
    
    # Plot grid
    for i in range(m+1):
        ax.axhline(i, color='gray', lw=0.5)
    for j in range(n+1):
        ax.axvline(j, color='gray', lw=0.5)
    
    # Plot source graph
    locs, graph, pins = pattern.source_graph()
    
    # Plot nodes
    for i, loc in enumerate(locs):
        x, y = loc.loc
        color = 'red' if i in pins else 'blue'
        ax.add_patch(Circle((y+0.5, m-x-0.5), 0.3, color=color))
    
    # Plot edges
    for edge in graph.edges():
        src, dst = edge
        loc1 = locs[src].loc
        loc2 = locs[dst].loc
        x1, y1 = loc1
        x2, y2 = loc2
        ax.plot([y1+0.5, y2+0.5], [m-x1-0.5, m-x2-0.5], 'k-')
    
    # Plot mapping direction
    ax.set_title(title)
    ax.set_aspect('equal')
    ax.set_xlim(-0.5, n+0.5)
    ax.set_ylim(-0.5, m+0.5)
    ax.invert_yaxis()


def plot_pattern_pair(pattern, title=None):
    """Plot a pattern and its mapped graph side by side."""
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(10, 5))
    
    if title is None:
        title = pattern.__class__.__name__
    
    # Plot source pattern
    plot_pattern(ax1, pattern, f"{title} - Source")
    
    # Plot mapped pattern
    mapped_matrix = pattern.mapped_matrix()
    m, n = len(mapped_matrix), len(mapped_matrix[0])
    
    # Plot grid
    for i in range(m+1):
        ax2.axhline(i, color='gray', lw=0.5)
    for j in range(n+1):
        ax2.axvline(j, color='gray', lw=0.5)
    
    # Plot mapped graph
    locs, graph, pins = pattern.mapped_graph()
    
    # Plot nodes
    for i, loc in enumerate(locs):
        x, y = loc.loc
        color = 'red' if i in pins else 'blue'
        ax2.add_patch(Circle((y+0.5, m-x-0.5), 0.3, color=color))
    
    # Plot edges
    for edge in graph.edges():
        u, v = edge
        loc1 = locs[u].loc
        loc2 = locs[v].loc
        x1, y1 = loc1
        x2, y2 = loc2
        ax2.plot([y1+0.5, y2+0.5], [m-x1-0.5, m-x2-0.5], 'k-')
    
    ax2.set_title(f"{title} - Mapped")
    ax2.set_aspect('equal')
    ax2.set_xlim(-0.5, n+0.5)
    ax2.set_ylim(-0.5, m+0.5)
    ax2.invert_yaxis()
    
    plt.tight_layout()
    return fig


def plot_crossing_ruleset():
    """Plot all patterns in the crossing ruleset."""
    for i, pattern in enumerate(crossing_ruleset):
        pattern_type = pattern.__class__.__name__
        if pattern_type == "Cross":
            has_edge = pattern.is_connected()
            pattern_type += f" (has_edge={has_edge})"
        fig = plot_pattern_pair(pattern, pattern_type)
        fig.savefig(f"pattern_{i+1}_{pattern_type.lower()}.png")


if __name__ == "__main__":
    # Plot all patterns in the crossing ruleset
    plot_crossing_ruleset()
    print("Pattern visualizations saved as PNG files.")