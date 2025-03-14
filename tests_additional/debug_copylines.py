#!/usr/bin/env python
"""
Debug script to understand copyline generation.
"""

import networkx as nx
import numpy as np
import matplotlib.pyplot as plt
from src.unit_disk_mapping import map_graph
from src.copyline import copyline_locations, CopyLine, center_location

def debug_copyline():
    """Create a single copyline and visualize its nodes."""
    # Create a simple graph with 5 vertices
    g = nx.Graph()
    g.add_edges_from([
        (0, 1), (0, 3), (0, 4), (2, 1), (2, 3), (2, 4)
    ])
    
    # Map the graph
    result = map_graph(g, vertex_order=[0, 1, 2, 3, 4])
    
    # Print copyline information
    print("Copyline Information:")
    for i, line in enumerate(result.lines):
        print(f"Copyline {i}: vertex={line.vertex}, vslot={line.vslot}, hslot={line.hslot}, vstart={line.vstart}, vstop={line.vstop}, hstop={line.hstop}")
    
    # Calculate nodes for each copyline manually
    padding = 2  # Default padding
    node_type = "UnWeightedNode"  # For unweighted mapping
    
    # Create a figure to visualize
    plt.figure(figsize=(12, 10))
    
    # Define different colors for each copyline
    colors = ['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728', '#9467bd']
    
    # Store all nodes to determine plot bounds
    all_nodes = []
    
    # For each copyline, get nodes and plot
    for i, line in enumerate(result.lines):
        # Get nodes for this copyline
        nodes = copyline_locations(node_type, line, padding=padding)
        
        # Extract locations
        locations = [node.loc for node in nodes]
        all_nodes.extend(locations)
        
        # Plot nodes
        x_coords = [loc[1] for loc in locations]  # Note: mapping.py uses (i,j) where i=row (y), j=column (x)
        y_coords = [loc[0] for loc in locations]
        
        # Plot nodes with the copyline's color
        plt.scatter(x_coords, y_coords, color=colors[i], label=f'Vertex {line.vertex}', s=100, alpha=0.7)
        
        # Plot connecting lines to show the T-shape
        center_i, center_j = center_location(line, padding)
        
        # Find locations on vertical segment
        vertical_nodes = [(i, j) for i, j in locations if abs(j - center_j) <= 1]
        if vertical_nodes:
            v_sorted = sorted(vertical_nodes, key=lambda x: x[0])
            plt.plot([v_sorted[0][1], v_sorted[-1][1]], [v_sorted[0][0], v_sorted[-1][0]], 
                     color=colors[i], linestyle='-', linewidth=2, alpha=0.5)
        
        # Find locations on horizontal segment
        horizontal_nodes = [(i, j) for i, j in locations if abs(i - center_i) <= 1]
        if horizontal_nodes:
            h_sorted = sorted(horizontal_nodes, key=lambda x: x[1])
            plt.plot([h_sorted[0][1], h_sorted[-1][1]], [h_sorted[0][0], h_sorted[-1][0]], 
                     color=colors[i], linestyle='-', linewidth=2, alpha=0.5)
    
    # Set axis limits
    if all_nodes:
        min_x = min(node[1] for node in all_nodes) - 1
        max_x = max(node[1] for node in all_nodes) + 1
        min_y = min(node[0] for node in all_nodes) - 1
        max_y = max(node[0] for node in all_nodes) + 1
        plt.xlim(min_x, max_x)
        plt.ylim(max_y, min_y)  # Reverse y-axis to match matrix coordinates
    
    # Add grid lines
    plt.grid(True, linestyle='--', alpha=0.6)
    
    # Add labels
    plt.title("Visualization of T-shaped Copylines", fontsize=16)
    plt.xlabel("X Coordinate (Column)", fontsize=14)
    plt.ylabel("Y Coordinate (Row)", fontsize=14)
    plt.legend()
    
    # Save the figure
    plt.tight_layout()
    plt.savefig("copylines_debug.png", dpi=300)
    print("Visualization saved to 'copylines_debug.png'")
    
    # Also print information about the nodes
    print("\nNodes for each copyline:")
    for i, line in enumerate(result.lines):
        nodes = copyline_locations(node_type, line, padding=padding)
        print(f"\nCopyline {i} (vertex {line.vertex}):")
        for j, node in enumerate(nodes):
            print(f"  Node {j}: location={node.loc}")

if __name__ == "__main__":
    debug_copyline()