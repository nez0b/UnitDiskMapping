#!/usr/bin/env python
"""
Improved example of the Unit Disk Mapping package with a simple 5-vertex graph.

This example demonstrates:
1. Creating a custom 5-vertex graph
2. Mapping it to a unit disk graph
3. Visualizing both the original and mapped graphs with proper T-shaped copyline identification
"""

import networkx as nx
import matplotlib.pyplot as plt
import numpy as np
from src.unit_disk_mapping import map_graph, Greedy, print_config, map_config_back
from src.copyline import center_location

def main():
    """Create, map and visualize a 5-vertex graph."""
    print("Creating a custom 5-vertex graph...")
    
    # Create a 5-vertex graph (cycle with one chord)
    g = nx.Graph()
    g.add_edges_from([
        (0, 1), (0, 3), (0, 4), (2, 1), (2, 3),  # Cycle
        (2, 4)  # Chord
    ])
    
    print(f"Graph has {g.number_of_nodes()} vertices and {g.number_of_edges()} edges")
    
    # Map the graph using a specific vertex order for reproducibility
    print("Mapping graph to unit disk graph...")
    result = map_graph(g, vertex_order=[0, 1, 2, 3, 4])
    
    # Print some information about the result
    print(f"Mapped grid size: {result.grid_graph.size}")
    print(f"Number of vertices in mapped graph: {result.grid_graph.num_vertices()}")
    print(f"MIS overhead: {result.mis_overhead}")
    
    # Print copyline information
    print("\nCopyline Information:")
    for i, line in enumerate(result.lines):
        print(f"Copyline {i}: vertex={line.vertex}, vslot={line.vslot}, hslot={line.hslot}, vstart={line.vstart}, vstop={line.vstop}, hstop={line.hstop}")
    
    # Create a figure for visualization
    plt.figure(figsize=(15, 7))
    
    # Subplot 1: Original graph
    plt.subplot(1, 2, 1)
    pos_original = nx.spring_layout(g, seed=42)  # Position nodes using spring layout
    
    # Draw the graph
    nx.draw_networkx_nodes(g, pos_original, node_color='skyblue', node_size=700)
    nx.draw_networkx_edges(g, pos_original, width=2)
    nx.draw_networkx_labels(g, pos_original, font_size=16, font_weight='bold')
    
    plt.title("Original 5-Vertex Graph", fontsize=16)
    plt.axis('off')
    
    # Subplot 2: Mapped unit disk graph
    plt.subplot(1, 2, 2)
    
    # Get the mapped graph as a networkx graph
    mapped_graph = result.grid_graph.to_networkx()
    
    # Use the grid coordinates for node positions
    positions = {i: node.loc for i, node in enumerate(result.grid_graph.nodes)}
    
    # Group nodes by copyline - improved implementation
    copyline_nodes = {}
    
    # Define different colors for each vertex in the original graph
    # Using a colorblind-friendly colormap
    colors = ['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728', '#9467bd', '#8c564b', '#e377c2', '#7f7f7f', '#bcbd22', '#17becf']
    
    # Process each copyline and find which nodes belong to it
    for line in result.lines:
        vertex = line.vertex
        copyline_nodes[vertex] = []
        
        # Get center of this copyline
        padding = result.padding
        center_i, center_j = center_location(line, padding)
        
        # Check each node to see if it belongs to this copyline
        for i, node in enumerate(result.grid_graph.nodes):
            x, y = node.loc
            
            # Test if on vertical segment of the T
            on_vertical = (abs(x - line.vslot * 4 + (padding - 1)) <= 1 and 
                            ((line.vstart - 1) * 4 + padding <= y <= (line.vstop + 1) * 4 + padding))
            
            # Test if on horizontal segment of the T
            on_horizontal = (abs(y - line.hslot * 4 + (padding - 1)) <= 1 and 
                             ((line.vslot - 1) * 4 + padding <= x <= (line.hstop + 1) * 4 + padding))
            
            # Test if near center intersection of the T
            near_center = (abs(x - center_i) <= 2 and abs(y - center_j) <= 2)
            
            if on_vertical or on_horizontal or near_center:
                copyline_nodes[vertex].append(i)
    
    # Color nodes by copyline
    node_colors = []
    for i in range(len(result.grid_graph.nodes)):
        # Find which copyline this node belongs to
        found = False
        for vertex, nodes in copyline_nodes.items():
            if i in nodes:
                # Use the vertex index to pick a color
                color_idx = vertex % len(colors)
                node_colors.append(colors[color_idx])
                found = True
                break
        
        # If node doesn't belong to any copyline
        if not found:
            node_colors.append('lightgray')
    
    # Draw the graph with colored nodes
    nx.draw_networkx_nodes(mapped_graph, positions, node_color=node_colors, node_size=300)
    nx.draw_networkx_edges(mapped_graph, positions, width=1.5, alpha=0.7)
    
    # Add a legend to identify copylines
    import matplotlib.patches as mpatches
    legend_patches = []
    for vertex in sorted(copyline_nodes.keys()):
        color_idx = vertex % len(colors)
        patch = mpatches.Patch(color=colors[color_idx], label=f'Vertex {vertex}')
        legend_patches.append(patch)
    
    # Add the legend to the plot
    plt.legend(handles=legend_patches, loc='upper right', fontsize=10)
    
    # Use small node labels to avoid clutter
    if len(positions) < 20:  # Only label if not too many nodes
        nx.draw_networkx_labels(mapped_graph, positions, font_size=8)
    
    plt.title("Mapped Unit Disk Graph (colored by copyline)", fontsize=16)
    plt.axis('off')
    
    # Save the figure
    plt.tight_layout()
    plt.savefig("improved_5vertex_mapping.png", dpi=300)
    print("Visualization saved to 'improved_5vertex_mapping.png'")
    
    # Create and visualize a sample configuration (independent set)
    create_sample_configuration(result)

def create_sample_configuration(result):
    """Create and visualize a sample maximum independent set configuration."""
    print("\nCreating a sample independent set configuration...")
    
    # Create an empty configuration
    grid_size = result.grid_graph.size
    config = np.zeros(grid_size, dtype=int)
    
    # Find a valid independent set manually
    # In a real application, this would be solved using an MIS solver
    independent_set_nodes = []
    
    # For demonstration, select alternating nodes
    for i, node in enumerate(result.grid_graph.nodes):
        if i % 2 == 0:
            independent_set_nodes.append(i)
            config[node.loc] = 1
    
    # Print the configuration
    print("\nSample configuration on the grid:")
    config_str = print_config(result, config)
    print(config_str)
    
    # Visualize the configuration
    plt.figure(figsize=(8, 8))
    
    # Draw the graph
    mapped_graph = result.grid_graph.to_networkx()
    positions = {i: node.loc for i, node in enumerate(result.grid_graph.nodes)}
    
    # Group nodes by copyline
    copyline_nodes = {}
    colors = ['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728', '#9467bd', '#8c564b', '#e377c2', '#7f7f7f', '#bcbd22', '#17becf']
    
    # Process each copyline and find which nodes belong to it
    for line in result.lines:
        vertex = line.vertex
        copyline_nodes[vertex] = []
        
        # Get center of this copyline
        padding = result.padding
        center_i, center_j = center_location(line, padding)
        
        # Check each node to see if it belongs to this copyline
        for i, node in enumerate(result.grid_graph.nodes):
            x, y = node.loc
            
            # Test if on vertical segment of the T
            on_vertical = (abs(x - line.vslot * 4 + (padding - 1)) <= 1 and 
                            ((line.vstart - 1) * 4 + padding <= y <= (line.vstop + 1) * 4 + padding))
            
            # Test if on horizontal segment of the T
            on_horizontal = (abs(y - line.hslot * 4 + (padding - 1)) <= 1 and 
                             ((line.vslot - 1) * 4 + padding <= x <= (line.hstop + 1) * 4 + padding))
            
            # Test if near center intersection of the T
            near_center = (abs(x - center_i) <= 2 and abs(y - center_j) <= 2)
            
            if on_vertical or on_horizontal or near_center:
                copyline_nodes[vertex].append(i)
    
    # Color nodes by copyline AND independent set status
    node_colors = []
    node_sizes = []   # Use larger sizes for MIS nodes
    
    for i, node in enumerate(result.grid_graph.nodes):
        # Determine if node is in the independent set
        in_mis = config[node.loc] == 1
        
        # Find which copyline this node belongs to
        found = False
        for vertex, nodes in copyline_nodes.items():
            if i in nodes:
                # Use the vertex index to pick a color
                color_idx = vertex % len(colors)
                base_color = colors[color_idx]
                
                # Modify appearance based on MIS membership
                if in_mis:
                    # Make MIS nodes brighter and larger
                    node_colors.append(base_color)
                    node_sizes.append(400)  # Larger size
                else:
                    # Make non-MIS nodes more transparent
                    import matplotlib.colors as mcolors
                    rgba = mcolors.to_rgba(base_color)
                    faded_color = (rgba[0], rgba[1], rgba[2], 0.5)  # Half opacity
                    node_colors.append(faded_color)
                    node_sizes.append(300)  # Normal size
                
                found = True
                break
        
        # If node doesn't belong to any copyline
        if not found:
            node_colors.append('lightgray')
            node_sizes.append(300)
    
    # Draw the graph with colored nodes
    nx.draw_networkx_nodes(mapped_graph, positions, node_color=node_colors, 
                          node_size=node_sizes)
    nx.draw_networkx_edges(mapped_graph, positions, width=1.5, alpha=0.7)
    
    # Add a legend for both copylines and MIS status
    import matplotlib.patches as mpatches
    from matplotlib.lines import Line2D
    
    # Legend for copylines
    copyline_patches = []
    for vertex in sorted(copyline_nodes.keys()):
        color_idx = vertex % len(colors)
        patch = mpatches.Patch(color=colors[color_idx], label=f'Vertex {vertex}')
        copyline_patches.append(patch)
    
    # Legend for MIS status
    mis_legend = [
        Line2D([0], [0], marker='o', color='w', markerfacecolor='black', 
               markersize=12, label='In MIS (larger)'),
        Line2D([0], [0], marker='o', color='w', markerfacecolor='black', alpha=0.5,
               markersize=9, label='Not in MIS (faded)')
    ]
    
    # Combine legends
    all_legend_elements = copyline_patches + mis_legend
    plt.legend(handles=all_legend_elements, loc='upper right', fontsize=9)
    
    # Use small node labels to avoid clutter
    if len(positions) < 20:
        nx.draw_networkx_labels(mapped_graph, positions, font_size=8)
    
    plt.title("Unit Disk Graph with Independent Set\n(colored by copyline)", fontsize=16)
    plt.axis('off')
    
    # Save the figure
    plt.tight_layout()
    plt.savefig("improved_5vertex_independent_set.png", dpi=300)
    print("Configuration visualization saved to 'improved_5vertex_independent_set.png'")
    
    # Try to map the configuration back to the original graph
    try:
        original_config = map_config_back(result, config)
        print("\nMapped back to original graph:")
        print(f"Vertices in the original independent set: {[i for i, v in enumerate(original_config) if v == 1]}")
    except Exception as e:
        print(f"Error mapping configuration back: {e}")

if __name__ == "__main__":
    main()