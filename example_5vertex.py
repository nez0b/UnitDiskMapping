#!/usr/bin/env python
"""
Example usage of the Unit Disk Mapping package with a simple 5-vertex graph.

This example demonstrates:
1. Creating a custom 5-vertex graph
2. Mapping it to a unit disk graph
3. Visualizing both the original and mapped graphs
"""

import networkx as nx
import matplotlib.pyplot as plt
import numpy as np
from unit_disk_mapping import map_graph, Greedy, print_config, map_config_back

def main():
    """Create, map and visualize a 5-vertex graph."""
    print("Creating a custom 5-vertex graph...")
    
    # Create a 5-vertex graph (cycle with one chord)
    g = nx.Graph()
    g.add_edges_from([
        (0, 1), (1, 2), (2, 3), (3, 4), (4, 0),  # Cycle
        (0, 2)  # Chord
    ])
    
    print(f"Graph has {g.number_of_nodes()} vertices and {g.number_of_edges()} edges")
    
    # Map the graph using the greedy method (faster than MinhThiTrick for example purposes)
    print("Mapping graph to unit disk graph...")
    result = map_graph(g, vertex_order=Greedy(nrepeat=5))
    
    # Print some information about the result
    print(f"Mapped grid size: {result.grid_graph.size}")
    print(f"Number of vertices in mapped graph: {result.grid_graph.num_vertices()}")
    print(f"MIS overhead: {result.mis_overhead}")
    
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
    
    # Draw the graph
    nx.draw_networkx_nodes(mapped_graph, positions, node_color='lightgreen', node_size=300)
    nx.draw_networkx_edges(mapped_graph, positions, width=1.5, alpha=0.7)
    
    # Use small node labels to avoid clutter
    if len(positions) < 20:  # Only label if not too many nodes
        nx.draw_networkx_labels(mapped_graph, positions, font_size=8)
    
    plt.title("Mapped Unit Disk Graph", fontsize=16)
    plt.axis('off')
    
    # Save the figure
    plt.tight_layout()
    plt.savefig("5vertex_mapping_example.png", dpi=300)
    print("Visualization saved to '5vertex_mapping_example.png'")
    
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
    
    # Color nodes based on configuration (in MIS or not)
    node_colors = []
    for i, node in enumerate(result.grid_graph.nodes):
        if config[node.loc] == 1:
            node_colors.append('orange')  # In the independent set
        else:
            node_colors.append('lightblue')  # Not in the independent set
    
    # Draw the graph with colored nodes
    nx.draw_networkx_nodes(mapped_graph, positions, node_color=node_colors, node_size=300)
    nx.draw_networkx_edges(mapped_graph, positions, width=1.5, alpha=0.7)
    
    # Use small node labels to avoid clutter
    if len(positions) < 20:  # Only label if not too many nodes
        nx.draw_networkx_labels(mapped_graph, positions, font_size=8)
    
    plt.title("Unit Disk Graph with Independent Set Configuration", fontsize=16)
    plt.axis('off')
    
    # Save the figure
    plt.tight_layout()
    plt.savefig("5vertex_independent_set.png", dpi=300)
    print("Configuration visualization saved to '5vertex_independent_set.png'")
    
    # Try to map the configuration back to the original graph
    try:
        original_config = map_config_back(result, config)
        print("\nMapped back to original graph:")
        print(f"Vertices in the original independent set: {[i for i, v in enumerate(original_config) if v == 1]}")
    except Exception as e:
        print(f"Error mapping configuration back: {e}")

if __name__ == "__main__":
    main()