#!/usr/bin/env python
"""
Visualize the raw grid content to debug the copylines.
"""

import networkx as nx
import matplotlib.pyplot as plt
import numpy as np
from unit_disk_mapping import map_graph
from core import GridGraph
import matplotlib.colors as mcolors

def main():
    """Create and visualize the raw grid content."""
    # Create a simple 5-vertex graph
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
    
    # Create a grid representation
    grid_shape = result.grid_graph.size
    grid = np.zeros(grid_shape)
    
    # Set the positions of all nodes
    for node in result.grid_graph.nodes:
        i, j = node.loc
        grid[i, j] = 1
    
    # Visualize the raw grid
    plt.figure(figsize=(10, 8))
    plt.imshow(grid, cmap='viridis', interpolation='nearest')
    plt.colorbar(label='Node presence')
    plt.title('Raw Grid Content')
    plt.xlabel('Column (j)')
    plt.ylabel('Row (i)')
    plt.grid(alpha=0.3)
    plt.savefig('raw_grid.png', dpi=300)
    print("Raw grid visualization saved to 'raw_grid.png'")
    
    # Now create a visualization with a unique color for each copyline
    # This time manually matching nodes to copylines using their exact positions
    grid_colors = np.zeros((grid_shape[0], grid_shape[1], 3))
    
    # Define colors for each vertex
    color_map = {
        0: [0.12, 0.47, 0.71],  # blue
        1: [1.0, 0.5, 0.05],    # orange
        2: [0.17, 0.63, 0.17],  # green
        3: [0.84, 0.15, 0.16],  # red
        4: [0.58, 0.4, 0.74]    # purple
    }
    
    # First, fill with a light gray background
    for i in range(grid_shape[0]):
        for j in range(grid_shape[1]):
            grid_colors[i, j] = [0.9, 0.9, 0.9]  # light gray
    
    # Extract all node locations
    node_locs = {node.loc: i for i, node in enumerate(result.grid_graph.nodes)}
    
    # Directly extract information on copyline locations
    from copyline import copyline_locations
    
    # Assign color based on copyline membership
    for i, line in enumerate(result.lines):
        # Get the vertex this copyline represents
        vertex = line.vertex
        color = color_map[vertex]
        
        # Get all locations for this copyline
        node_type = "UnWeightedNode"  # For unweighted mapping
        locs = copyline_locations(node_type, line, padding=result.padding)
        
        # Color each location in the grid
        for node in locs:
            i, j = node.loc
            if 0 <= i < grid_shape[0] and 0 <= j < grid_shape[1]:
                grid_colors[i, j] = color
    
    # Visualize the colored grid
    plt.figure(figsize=(12, 10))
    plt.imshow(grid_colors, interpolation='nearest')
    plt.title('Grid with Colored Copylines')
    plt.xlabel('Column (j)')
    plt.ylabel('Row (i)')
    
    # Add a legend
    import matplotlib.patches as mpatches
    legend_patches = []
    for vertex, color in color_map.items():
        patch = mpatches.Patch(color=color, label=f'Vertex {vertex}')
        legend_patches.append(patch)
    plt.legend(handles=legend_patches, loc='upper right')
    
    plt.grid(alpha=0.3)
    plt.savefig('colored_grid.png', dpi=300)
    print("Colored grid visualization saved to 'colored_grid.png'")

if __name__ == "__main__":
    main()