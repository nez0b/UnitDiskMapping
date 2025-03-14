#!/usr/bin/env python
"""
Script to visualize crossing points in the copyline lattice.
"""

import networkx as nx
import matplotlib.pyplot as plt
import numpy as np
from src.unit_disk_mapping import map_graph
from src.mapping import crossat, embed_graph

def main():
    """Visualize the crossing points of copylines."""
    # Create a simple graph with 5 vertices
    g = nx.Graph()
    g.add_edges_from([
        (0, 1), (0, 3), (0, 4), (2, 1), (2, 3), (2, 4)
    ])
    
    # Create the mapping grid
    ug = embed_graph(g, vertex_order=[0, 1, 2, 3, 4])
    
    # Create a figure
    plt.figure(figsize=(10, 8))
    
    # Define colors for copylines
    colors = ['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728', '#9467bd']
    
    # Plot each copyline
    vertices = [line.vertex for line in ug.lines]
    s = 2  # spacing factor
    padding = ug.padding
    
    # Plot copylines
    for i, line in enumerate(ug.lines):
        # Define simplified T-shape for visualization
        vslot_x = (line.vslot - 1) * s + padding + 1
        vstart_y = padding
        vstop_y = (len(ug.lines) - 1) * s + padding
        
        hslot_y = (line.hslot - 1) * s + padding + 2
        hstop_x = (len(ug.lines) - 1) * s + padding
        
        # Plot vertical line
        plt.plot([vslot_x, vslot_x], [vstart_y, vstop_y], color=colors[i], linewidth=2, alpha=0.7)
        
        # Plot horizontal line
        plt.plot([vslot_x, hstop_x], [hslot_y, hslot_y], color=colors[i], linewidth=2, alpha=0.7)
        
        # Add vertex label at center
        plt.text(vslot_x - 0.5, hslot_y - 0.5, str(line.vertex), fontsize=12, fontweight='bold')
    
    # Plot crossing points
    for v in range(5):
        for w in range(v+1, 5):
            cross_i, cross_j = crossat(ug, v, w)
            has_edge = g.has_edge(v, w)
            color = 'green' if has_edge else 'red'
            marker = 'o' if has_edge else 'x'
            plt.scatter(cross_j, cross_i, color=color, marker=marker, s=100, zorder=3)
            plt.text(cross_j, cross_i - 0.3, f'({v},{w})', fontsize=8, ha='center', va='bottom')
    
    # Set grid and labels
    plt.grid(True, linestyle='--', alpha=0.4)
    height = (len(ug.lines) - 1) * s + padding * 2
    width = (len(ug.lines) - 1) * s + padding * 2
    plt.xlim(0, width)
    plt.ylim(height, 0)  # Reverse y axis
    plt.xlabel('Column (j)')
    plt.ylabel('Row (i)')
    plt.title('Copylines and Crossing Points')
    
    # Save figure
    plt.tight_layout()
    plt.savefig('crossing_points.png', dpi=300)
    print('Created visualization: crossing_points.png')
    
    # Print crossing information
    print('Copyline information:')
    for i, line in enumerate(ug.lines):
        print(f'Copyline {i}: vertex={line.vertex}, vslot={line.vslot}, hslot={line.hslot}')
    
    print('\nCrossing points:')
    for v in range(5):
        for w in range(v+1, 5):
            cross_point = crossat(ug, v, w)
            has_edge = g.has_edge(v, w)
            print(f'Vertices {v} and {w} cross at {cross_point}, has edge: {has_edge}')

if __name__ == "__main__":
    main()