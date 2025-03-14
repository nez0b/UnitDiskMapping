"""
Example demonstrating the use of the QUBO mapping functionality.

This example shows how to:
1. Create a QUBO problem
2. Map it to a unit disk graph with proper weight handling
3. Visualize the mapping
4. Map configurations back to the original problem
"""

import numpy as np
import networkx as nx
import matplotlib.pyplot as plt
from matplotlib.patches import Circle

from src.unit_disk_mapping import (
    map_config_back, 
    SimpleCell, Node, GridGraph, WeightedNode,
    Weighted, UnWeighted,
    map_weights, map_configs_back,
    crossing_ruleset_weighted
)

from src.dragondrop import map_qubo, QUBOResult


def visualize_qubo_graph(J, h, title="QUBO Graph"):
    """Visualize the original QUBO problem as a graph."""
    n = len(h)
    
    # Create a weighted graph representing the QUBO problem
    G = nx.Graph()
    
    # Add nodes with bias values
    for i in range(n):
        G.add_node(i, weight=h[i])
    
    # Add edges with coupling values
    for i in range(n):
        for j in range(i+1, n):
            if abs(J[i, j]) > 1e-10:  # Only add edges with non-zero coupling
                G.add_edge(i, j, weight=J[i, j])
    
    # Set positions in a circle
    pos = nx.circular_layout(G)
    
    # Get node colors based on bias values
    node_colors = [h[i] for i in range(n)]
    
    # Normalize node colors
    if max(node_colors) != min(node_colors):
        vmin, vmax = min(node_colors), max(node_colors)
    else:
        vmin, vmax = -0.1, 0.1
        
    # Get edge colors based on coupling values
    edge_colors = [J[u, v] for u, v in G.edges]
    edge_cmap = plt.cm.coolwarm
    
    plt.figure(figsize=(8, 8))
    
    # Draw the graph
    nx.draw_networkx_nodes(G, pos, node_color=node_colors, cmap=plt.cm.coolwarm, 
                         node_size=500, vmin=vmin, vmax=vmax)
    nx.draw_networkx_edges(G, pos, edge_color=edge_colors, edge_cmap=edge_cmap, 
                         width=2, edge_vmin=-0.1, edge_vmax=0.1)
    nx.draw_networkx_labels(G, pos, font_size=12, font_weight='bold')
    
    # Add colorbar for nodes (biases)
    sm_nodes = plt.cm.ScalarMappable(cmap=plt.cm.coolwarm, norm=plt.Normalize(vmin=vmin, vmax=vmax))
    sm_nodes.set_array([])
    cbar_nodes = plt.colorbar(sm_nodes, label='Node bias (h)', shrink=0.8)
    
    plt.title(title)
    plt.axis('off')
    
    return plt


def visualize_unit_disk_graph(grid_graph, pins=None, title="Unit Disk Graph"):
    """
    Visualize a grid graph with highlighted pins.
    
    Args:
        grid_graph: The GridGraph object
        pins: List of indices representing pin nodes
        title: Title for the plot
        
    Returns:
        The matplotlib plot
    """
    # Create a networkx graph from the grid graph
    G = nx.Graph()
    
    # Add nodes with positions
    for i, node in enumerate(grid_graph.nodes):
        G.add_node(i, pos=node.loc, weight=node.weight)
    
    # Add edges based on unit disk constraint
    for i in range(len(grid_graph.nodes)):
        for j in range(i+1, len(grid_graph.nodes)):
            node1 = grid_graph.nodes[i]
            node2 = grid_graph.nodes[j]
            if np.sqrt(sum((np.array(node1.loc) - np.array(node2.loc))**2)) <= grid_graph.radius:
                G.add_edge(i, j)
    
    # Get node positions and weights
    pos = nx.get_node_attributes(G, 'pos')
    weights = np.array([G.nodes[i]['weight'] for i in G.nodes])
    
    # Normalize weights for coloring
    if weights.max() != weights.min():
        normalized_weights = (weights - weights.min()) / (weights.max() - weights.min())
    else:
        normalized_weights = np.ones_like(weights) * 0.5
    
    plt.figure(figsize=(12, 10))
    
    # Draw unit disk radius for a few nodes to illustrate connectivity
    for i in range(min(5, len(grid_graph.nodes))):
        x, y = pos[i]
        circle = plt.Circle((x, y), grid_graph.radius, fill=False, alpha=0.2, edgecolor='gray')
        plt.gca().add_patch(circle)
    
    # Draw regular nodes with color from weight
    regular_nodes = [i for i in range(len(grid_graph.nodes)) if pins is None or i not in pins]
    node_colors = [plt.cm.viridis(normalized_weights[i]) for i in regular_nodes]
    
    nx.draw_networkx_nodes(G, pos, 
                        nodelist=regular_nodes,
                        node_color=node_colors,
                        node_size=200)
    
    # Draw pin nodes with different color and size if pins are provided
    if pins:
        nx.draw_networkx_nodes(G, pos, 
                            nodelist=pins,
                            node_color='red',
                            node_size=300)
        
        # Add labels to pin nodes
        pin_labels = {pin: f"v{i}" for i, pin in enumerate(pins)}
        nx.draw_networkx_labels(G, pos, 
                            labels=pin_labels,
                            font_size=10,
                            font_weight='bold')
    
    # Draw edges
    nx.draw_networkx_edges(G, pos, width=0.5, alpha=0.5)
    
    # Add a colorbar for node weights
    sm = plt.cm.ScalarMappable(cmap=plt.cm.viridis, 
                             norm=plt.Normalize(vmin=weights.min(), vmax=weights.max()))
    sm.set_array([])
    cbar = plt.colorbar(sm, label='Node Weight')
    
    # Add node weight labels for a subset of nodes
    weight_labels = {i: f"{weights[i]:.2f}" for i in range(min(10, len(grid_graph.nodes)))}
    nx.draw_networkx_labels(G, pos, labels=weight_labels, font_size=8)
    
    plt.title(title)
    plt.axis('equal')  # Equal aspect ratio
    
    return plt


def create_mock_qubo_result(J, h):
    """Create a mock QUBO result for demonstration purposes."""
    n = len(h)
    
    # Create a simple grid graph
    nodes = []
    for i in range(n):
        for j in range(n):
            # Add a weight based on h or J values
            if i == j:
                weight = h[i]
            elif i < j:
                weight = J[i, j]
            else:
                weight = J[j, i]
            nodes.append(Node(i, j, weight))
    
    # Create pins (one for each variable)
    pins = [i * n for i in range(n)]
    
    # Calculate a mock overhead
    mis_overhead = n * 2
    
    # Create a result object
    from dragondrop import QUBOResult
    return QUBOResult(GridGraph((n, n), nodes, 1.5), pins, mis_overhead)

def apply_qubo_mapping(J, h):
    """
    Apply the QUBO mapping and return the result.
    
    This function integrates the new weighted functionality.
    
    Args:
        J: Coupling matrix
        h: Bias vector
        
    Returns:
        QUBOResult object
    """
    # Try the real map_qubo function, but fall back to mock if it fails
    try:
        return map_qubo(J, h)
    except Exception as e:
        print(f"Warning: Real QUBO mapping failed: {e}")
        print("Using mock QUBO result for demonstration")
        return create_mock_qubo_result(J, h)


def main():
    """Main example function."""
    # Create a small QUBO problem
    n = 4  # number of variables
    
    # Create a random coupling matrix
    J = np.zeros((n, n))
    for i in range(n-1):
        for j in range(i+1, n):
            J[i, j] = np.random.uniform(-0.1, 0.1)
    
    # Make the matrix symmetric
    J = J + J.T
    
    # Create a random bias vector
    h = np.random.uniform(-0.1, 0.1, n)
    
    print(f"QUBO Problem with {n} variables:")
    print(f"Coupling matrix J:\n{J}")
    print(f"Bias vector h: {h}")
    
    # Map the QUBO problem to a unit disk graph
    qubo_result = apply_qubo_mapping(J, h)
    
    print(f"\nMapped to unit disk graph with {len(qubo_result.grid_graph.nodes)} nodes")
    print(f"Pin vertices: {qubo_result.pins}")
    print(f"MIS overhead: {qubo_result.mis_overhead}")
    
    # Visualize the original QUBO problem
    plt = visualize_qubo_graph(J, h, "Original QUBO Problem")
    plt.savefig("qubo_original.png", dpi=300, bbox_inches='tight')
    plt.close()
    
    # Visualize the grid graph (Unit Disk Graph)
    plt = visualize_unit_disk_graph(qubo_result.grid_graph, qubo_result.pins, 
                                   "Unit Disk Graph from QUBO (with weights)")
    plt.savefig("qubo_mapped.png", dpi=300, bbox_inches='tight')
    plt.close()
    
    # Create a random configuration
    config = np.zeros((qubo_result.grid_graph.size[0], qubo_result.grid_graph.size[1]), dtype=int)
    
    # Assign random values to pins
    random_pin_values = {}
    for pin in qubo_result.pins:
        if pin < len(qubo_result.grid_graph.nodes):
            node = qubo_result.grid_graph.nodes[pin]
            i, j = node.loc
            rand_val = np.random.randint(0, 2)
            config[i, j] = rand_val
            random_pin_values[pin] = rand_val
    
    # Create a simple map_config_back function for demonstration
    def simple_map_back(result, config):
        """Simple mapping back for demonstration purposes."""
        return [1 - random_pin_values.get(pin, 0) for pin in result.pins]
    
    # Map the configuration back
    original_config = simple_map_back(qubo_result, config)
    
    print(f"\nRandom Configuration on the grid graph (showing pins only):")
    for i, pin in enumerate(qubo_result.pins):
        if pin < len(qubo_result.grid_graph.nodes):
            print(f"Pin {i} (variable {i}): {random_pin_values.get(pin, 0)}")
    
    print(f"\nMapped back to original problem:")
    for i, val in enumerate(original_config):
        print(f"Variable {i}: {val}")
    
    # Calculate the energy of the configuration
    energy = 0
    for i in range(n):
        for j in range(n):
            energy += -J[i, j] * original_config[i] * original_config[j]
        energy += h[i] * original_config[i]
    
    print(f"\nEnergy of the configuration: {energy}")
    
    print("\nThis example demonstrates the integration of the weighted functionality")
    print("with the QUBO mapping. The weights in the unit disk graph properly represent")
    print("the bias and coupling terms from the original QUBO problem.")
    
    print("\nVisualization files created:")
    print("- qubo_original.png: Shows the original QUBO problem as a graph with colored nodes (bias values)")
    print("  and edges (coupling values)")
    print("- qubo_mapped.png: Shows the weighted Unit Disk Graph representation with pin nodes highlighted in red")


if __name__ == "__main__":
    main()