"""
Example demonstrating the weighted unit disk mapping functionality.

This example demonstrates how to use the weighted unit disk mapping functionality
to map a weighted graph to a unit disk graph.
"""

import numpy as np
import networkx as nx
import matplotlib.pyplot as plt
from matplotlib.patches import Circle

from src.unit_disk_mapping import (
    map_graph, 
    Weighted, 
    MinhThiTrick, 
    WeightedNode,
    map_weights,
    map_configs_back,
    crossing_ruleset_weighted
)


def draw_weighted_graph(ax, graph, pos=None, node_color='skyblue', 
                        node_size=500, node_weights=None, title=None):
    """Draw a weighted graph with weight labels."""
    if pos is None:
        pos = nx.spring_layout(graph)
    
    # Draw nodes and edges
    nx.draw_networkx_edges(graph, pos, ax=ax)
    nx.draw_networkx_nodes(graph, pos, node_color=node_color, node_size=node_size, ax=ax)
    
    # Draw node labels
    if node_weights:
        labels = {i: f"{i}:{w}" for i, w in enumerate(node_weights)}
    else:
        labels = {i: f"{i}" for i in graph.nodes()}
    
    nx.draw_networkx_labels(graph, pos, labels=labels, font_size=10, ax=ax)
    
    if title:
        ax.set_title(title)
    ax.axis('off')


def draw_unit_disk_graph(ax, mapping_result, title=None):
    """Draw a unit disk graph with node positions in a grid."""
    # Extract nodes and coordinates
    nodes = mapping_result.grid_graph.nodes
    coords = [node.loc for node in nodes]
    weights = [node.weight for node in nodes]
    
    # Create graph with proper positions
    G = nx.Graph()
    pos = {}
    
    for i, (coord, weight) in enumerate(zip(coords, weights)):
        G.add_node(i)
        # Flip coordinates for visualization (y-axis goes up in matplotlib)
        pos[i] = (coord[1], -coord[0])
    
    # Add edges based on unit disk distance
    for i in range(len(nodes)):
        for j in range(i+1, len(nodes)):
            # Calculate Euclidean distance
            dist = ((nodes[i][0] - nodes[j][0])**2 + (nodes[i][1] - nodes[j][1])**2)**0.5
            if dist <= mapping_result.grid_graph.radius:
                G.add_edge(i, j)
    
    # Draw nodes with different colors based on weight
    weight_values = set(weights)
    colors = plt.cm.tab10(np.linspace(0, 1, len(weight_values)))
    weight_to_color = {w: colors[i] for i, w in enumerate(weight_values)}
    
    node_colors = [weight_to_color[weights[i]] for i in range(len(nodes))]
    
    # Draw graph
    nx.draw(G, pos, node_color=node_colors, node_size=300, with_labels=False, ax=ax)
    
    # Draw weight labels
    labels = {i: str(weights[i]) for i in range(len(nodes))}
    nx.draw_networkx_labels(G, pos, labels=labels, font_size=8, ax=ax)
    
    # Draw radius circles for a few nodes to illustrate connectivity
    radius = mapping_result.grid_graph.radius
    for i in range(min(5, len(nodes))):
        circle = Circle(pos[i], radius, alpha=0.1, fc='gray', ec='gray')
        ax.add_patch(circle)
    
    if title:
        ax.set_title(title)
    ax.axis('off')


def create_weighted_star_graph(n_branches=4, weights=None):
    """Create a star graph with optional weights."""
    G = nx.star_graph(n_branches)
    
    if weights is None:
        # Default weights: center has weight 3, arms have weight 1
        weights = [3] + [1] * n_branches
    
    return G, weights


def create_weighted_path_graph(n=5, weights=None):
    """Create a path graph with optional weights."""
    G = nx.path_graph(n)
    
    if weights is None:
        # Default alternating weights
        weights = [2 if i % 2 == 0 else 1 for i in range(n)]
    
    return G, weights


def create_weighted_cycle_graph(n=5, weights=None):
    """Create a cycle graph with optional weights."""
    G = nx.cycle_graph(n)
    
    if weights is None:
        # Default increasing weights
        weights = [i % 3 + 1 for i in range(n)]
    
    return G, weights


def main():
    """Main function demonstrating weighted unit disk mapping."""
    # Create example graphs
    star_graph, star_weights = create_weighted_star_graph(5)
    path_graph, path_weights = create_weighted_path_graph(6)
    cycle_graph, cycle_weights = create_weighted_cycle_graph(6)
    
    # Set up the figure
    fig, axes = plt.subplots(3, 2, figsize=(12, 15))
    
    # Draw original graphs
    draw_weighted_graph(axes[0, 0], star_graph, node_weights=star_weights, 
                        title="Weighted Star Graph")
    draw_weighted_graph(axes[1, 0], path_graph, node_weights=path_weights, 
                        title="Weighted Path Graph")
    draw_weighted_graph(axes[2, 0], cycle_graph, node_weights=cycle_weights, 
                        title="Weighted Cycle Graph")
    
    # Map graphs to unit disk graphs
    print("Mapping star graph to unit disk graph...")
    star_result = map_graph(star_graph, mode=Weighted(), 
                           ruleset=crossing_ruleset_weighted)
    
    print("Mapping path graph to unit disk graph...")
    path_result = map_graph(path_graph, mode=Weighted(),
                           ruleset=crossing_ruleset_weighted)
    
    print("Mapping cycle graph to unit disk graph...")
    cycle_result = map_graph(cycle_graph, mode=Weighted(),
                            ruleset=crossing_ruleset_weighted)
    
    # Draw mapped unit disk graphs
    draw_unit_disk_graph(axes[0, 1], star_result, 
                        title="Star Graph Mapped to Unit Disk Graph")
    draw_unit_disk_graph(axes[1, 1], path_result, 
                        title="Path Graph Mapped to Unit Disk Graph")
    draw_unit_disk_graph(axes[2, 1], cycle_result, 
                        title="Cycle Graph Mapped to Unit Disk Graph")
    
    # Adjust layout
    plt.tight_layout()
    plt.savefig("weighted_mapping_example.png", dpi=300, bbox_inches='tight')
    plt.show()
    
    # Print some statistics
    print("\nMapping Statistics:")
    print(f"Star Graph: {len(star_graph.nodes)} nodes → {len(star_result.grid_graph.nodes)} nodes")
    print(f"Path Graph: {len(path_graph.nodes)} nodes → {len(path_result.grid_graph.nodes)} nodes")
    print(f"Cycle Graph: {len(cycle_graph.nodes)} nodes → {len(cycle_result.grid_graph.nodes)} nodes")


if __name__ == "__main__":
    main()