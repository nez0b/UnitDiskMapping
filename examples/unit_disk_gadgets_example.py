"""
Example showing how to use gadgets with unit disk mapping.

This example demonstrates how to use custom gadget patterns when mapping
a graph to a unit disk graph. It also shows how to examine the applied
gadgets and the resulting grid graph.
"""

import networkx as nx
import matplotlib.pyplot as plt
import numpy as np

from src.unit_disk_mapping import (
    map_graph, embed_graph, Pattern, 
    Cross, Turn, Branch, BranchFix, WTurn, BranchFixB, TCon, TrivialTurn, EndTurn,
    rotated_and_reflected
)


def plot_graph_and_mapping(original_graph, mapping_result):
    """Plot the original graph and its unit disk mapping side by side."""
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 6))
    
    # Plot original graph
    pos = nx.spring_layout(original_graph, seed=42)
    nx.draw(original_graph, pos, ax=ax1, with_labels=True, node_color='lightblue')
    ax1.set_title("Original Graph")
    
    # Plot unit disk graph
    udg = mapping_result.grid_graph.to_networkx()
    node_positions = {i: node.loc for i, node in enumerate(mapping_result.grid_graph.nodes)}
    nx.draw(udg, node_positions, ax=ax2, node_color='lightgreen')
    ax2.set_title("Unit Disk Graph Mapping")
    
    plt.tight_layout()
    plt.savefig("unit_disk_mapping.png")
    plt.close()


def example_with_default_ruleset():
    """Example using the default crossing ruleset."""
    # Create a simple graph with crossings
    g = nx.Graph()
    g.add_nodes_from(range(1, 6))
    g.add_edges_from([(1, 2), (2, 3), (3, 4), (4, 5), (5, 1), (1, 3), (1, 4)])
    
    # Map the graph with default ruleset
    result = map_graph(g)
    
    # Print information about applied gadgets
    print("Default ruleset mapping:")
    print(f"Number of nodes in original graph: {g.number_of_nodes()}")
    print(f"Number of edges in original graph: {g.number_of_edges()}")
    print(f"Number of nodes in unit disk graph: {len(result.grid_graph.nodes)}")
    print(f"Total MIS overhead: {result.mis_overhead}")
    print(f"Number of gadgets applied: {len(result.mapping_history)}")
    
    # Visualize the result
    plot_graph_and_mapping(g, result)
    
    # Print the types of gadgets that were applied
    gadget_types = {}
    for pattern, _, _ in result.mapping_history:
        pattern_type = pattern.__class__.__name__
        if pattern_type not in gadget_types:
            gadget_types[pattern_type] = 0
        gadget_types[pattern_type] += 1
    
    print("\nApplied gadget types:")
    for gadget_type, count in gadget_types.items():
        print(f"  {gadget_type}: {count}")


def example_with_custom_ruleset():
    """Example using a custom crossing ruleset."""
    # Create a simple graph with crossings
    g = nx.Graph()
    g.add_nodes_from(range(1, 6))
    g.add_edges_from([(1, 2), (2, 3), (3, 4), (4, 5), (5, 1), (1, 3), (1, 4)])
    
    # Create a custom ruleset with specific patterns
    custom_ruleset = [
        Cross(has_edge=True),
        Turn(),
        Branch()
    ]
    
    # Map the graph with custom ruleset
    result = map_graph(g, ruleset=custom_ruleset)
    
    # Print information about applied gadgets
    print("\nCustom ruleset mapping:")
    print(f"Number of nodes in original graph: {g.number_of_nodes()}")
    print(f"Number of edges in original graph: {g.number_of_edges()}")
    print(f"Number of nodes in unit disk graph: {len(result.grid_graph.nodes)}")
    print(f"Total MIS overhead: {result.mis_overhead}")
    print(f"Number of gadgets applied: {len(result.mapping_history)}")
    
    # Print the types of gadgets that were applied
    gadget_types = {}
    for pattern, _, _ in result.mapping_history:
        pattern_type = pattern.__class__.__name__
        if pattern_type not in gadget_types:
            gadget_types[pattern_type] = 0
        gadget_types[pattern_type] += 1
    
    print("\nApplied gadget types:")
    for gadget_type, count in gadget_types.items():
        print(f"  {gadget_type}: {count}")


if __name__ == "__main__":
    # Run examples
    example_with_default_ruleset()
    example_with_custom_ruleset()