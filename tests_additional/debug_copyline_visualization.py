#!/usr/bin/env python
"""
Debug script to visualize the grid nodes and copylines.
"""

import networkx as nx
import matplotlib.pyplot as plt
import numpy as np
from src.unit_disk_mapping import map_graph
from src.copyline import center_location

def main():
    """Create a simple visualization of all grid nodes with copylines labeled."""
    # Create a simple 5-vertex graph
    g = nx.Graph()
    g.add_edges_from([
        (0, 1), (0, 3), (0, 4), (2, 1), (2, 3), (2, 4)
    ])
    
    # Map the graph
    result = map_graph(g, vertex_order=[0, 1, 2, 3, 4])
    
    # Create a figure to visualize the entire grid
    plt.figure(figsize=(12, 10))
    
    # Define different colors for each vertex in the original graph
    colors = ['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728', '#9467bd']
    
    # Draw all nodes in gray first 
    all_nodes = result.grid_graph.nodes
    positions = {i: node.loc[::-1] for i, node in enumerate(all_nodes)}  # Swap (i,j) to (x,y)
    nx.draw_networkx_nodes(result.grid_graph.to_networkx(), positions, 
                          node_color='lightgray', node_size=100, alpha=0.5)
    
    # For each copyline, identify and color its nodes 
    for i, line in enumerate(result.lines):
        vertex = line.vertex
        
        # Calculate center location
        padding = result.padding
        center_i, center_j = center_location(line, padding)
        
        # For each node, check if it belongs to this copyline
        copyline_nodes = []
        for j, node in enumerate(all_nodes):
            x, y = node.loc
            
            # Create a more lenient matching heuristic for visualization
            s = 4  # spacing factor
            
            # Vertical segment: centered on vslot, from vstart to vstop
            vslot_x = (line.vslot - 1) * s + padding + 1
            vstart_y = (line.vstart - 1) * s + padding
            vstop_y = (line.vstop - 1) * s + padding
            
            # Horizontal segment: centered on hslot, from vslot to hstop
            hslot_y = (line.hslot - 1) * s + padding + 2
            hstop_x = (line.hstop - 1) * s + padding
            
            # On vertical segment
            on_vertical = (abs(x - vslot_x) <= 1 and vstart_y <= y <= vstop_y)
            
            # On horizontal segment
            on_horizontal = (abs(y - hslot_y) <= 1 and vslot_x <= x <= hstop_x)
            
            # Near intersection point
            near_intersection = (abs(x - vslot_x) <= 2 and abs(y - hslot_y) <= 2)
            
            if on_vertical or on_horizontal or near_intersection:
                copyline_nodes.append(j)
        
        # Draw nodes for this copyline
        if copyline_nodes:
            subgraph = result.grid_graph.to_networkx().subgraph(copyline_nodes)
            nx.draw_networkx_nodes(subgraph, positions, 
                                 node_color=colors[i], 
                                 node_size=200, 
                                 alpha=0.8, 
                                 label=f'Vertex {vertex}')
    
    # Add grid lines
    plt.grid(True, linestyle='--', alpha=0.4)
    
    # Add legend
    plt.legend()
    
    # Add axis labels and title
    plt.xlabel('Column (j)', fontsize=14)
    plt.ylabel('Row (i)', fontsize=14)
    plt.title('Grid with Copylines', fontsize=16)
    
    # Save the visualization
    plt.tight_layout()
    plt.savefig('grid_with_copylines.png', dpi=300)
    print("Visualization saved to 'grid_with_copylines.png'")

if __name__ == "__main__":
    main()