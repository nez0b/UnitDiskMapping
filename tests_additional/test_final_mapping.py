#!/usr/bin/env python
"""
Script to test the complete mapping process with fixed gadget application.
"""

import networkx as nx
import matplotlib.pyplot as plt
import numpy as np
from src.unit_disk_mapping import map_graph

def main():
    """Test the complete mapping process."""
    # Create a simple graph with 5 vertices
    g = nx.Graph()
    g.add_edges_from([
        (0, 1), (0, 3), (0, 4), (2, 1), (2, 3), (2, 4)
    ])
    
    # Map the graph with our fixed implementation
    result = map_graph(g, vertex_order=[0, 1, 2, 3, 4])
    
    # Print mapping history
    print('Original graph edges:')
    for e in g.edges():
        print(f'Edge {e}')
    
    print(f'\nOriginal graph has {len(g.edges())} edges')
    
    print('\nMapping history (applied gadgets):')
    for i, entry in enumerate(result.mapping_history):
        if len(entry) >= 6:
            pattern, x, y, v, w, has_edge = entry[:6]
            pattern_type = pattern.__class__.__name__
            pattern_has_edge = pattern.is_connected()
            correct = pattern_has_edge == has_edge
            print(f'Gadget {i+1}: {pattern_type} (has_edge={pattern_has_edge}) at ({x}, {y}) for vertices {v},{w} (has_edge={has_edge}, correct={correct})')
        else:
            pattern, x, y = entry[:3]
            pattern_type = pattern.__class__.__name__
            pattern_has_edge = pattern.is_connected()
            print(f'Gadget {i+1}: {pattern_type} (has_edge={pattern_has_edge}) at ({x}, {y})')
    
    # Count gadgets with correct edge status
    correct_gadgets = sum(1 for entry in result.mapping_history 
                         if len(entry) >= 6 and entry[0].is_connected() == entry[5])
    
    # Count gadgets with and without edges
    edge_gadgets = sum(1 for entry in result.mapping_history if len(entry) >= 6 and entry[5])
    no_edge_gadgets = sum(1 for entry in result.mapping_history if len(entry) >= 6 and not entry[5])
    
    print(f'\nTotal applied gadgets: {len(result.mapping_history)}')
    print(f'Gadgets with edges: {edge_gadgets}')
    print(f'Gadgets without edges: {no_edge_gadgets}')
    print(f'Gadgets with correct edge status: {correct_gadgets}')
    
    # Test that the grid graph has the right structure
    grid_graph = result.grid_graph
    print(f'\nFinal unit disk graph has {len(grid_graph.nodes)} nodes')
    
    # Visualize the final unit disk graph
    plt.figure(figsize=(12, 10))
    
    # Draw the nodes
    for node in grid_graph.nodes:
        i, j = node.loc
        plt.scatter(j, i, color='blue', s=50)
    
    # Draw the edges
    for i, node1 in enumerate(grid_graph.nodes):
        for j, node2 in enumerate(grid_graph.nodes[i+1:], i+1):
            i1, j1 = node1.loc
            i2, j2 = node2.loc
            # Check if within unit disk distance
            if ((i1-i2)**2 + (j1-j2)**2)**0.5 <= grid_graph.radius:
                plt.plot([j1, j2], [i1, i2], 'k-', alpha=0.2)
    
    # Add grid
    height, width = grid_graph.size
    for i in range(height+1):
        plt.axhline(i-0.5, color='gray', lw=0.5, alpha=0.3)
    for j in range(width+1):
        plt.axvline(j-0.5, color='gray', lw=0.5, alpha=0.3)
    
    # Set axis properties
    plt.title('Final Unit Disk Graph with Gadgets Applied')
    plt.xlabel('Column (j)')
    plt.ylabel('Row (i)')
    plt.xlim(-0.5, width-0.5)
    plt.ylim(height-0.5, -0.5)  # Invert y-axis
    
    plt.savefig('final_fixed_grid.png', dpi=300)
    print('Created visualization: final_fixed_grid.png')

if __name__ == "__main__":
    main()