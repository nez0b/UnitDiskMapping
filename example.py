"""
Example usage of the Unit Disk Mapping package.
"""

import networkx as nx
import matplotlib.pyplot as plt
from unit_disk_mapping import map_graph, MinhThiTrick, Greedy, print_config
import numpy as np

def petersen_graph_example():
    """Example using the Petersen graph."""
    print("Creating Petersen graph...")
    graph = nx.petersen_graph()
    
    print("Mapping graph using MinhThiTrick...")
    unweighted_res = map_graph(graph, vertex_order=MinhThiTrick())
    
    print(f"Grid graph size: {unweighted_res.grid_graph.size}")
    print(f"MIS overhead: {unweighted_res.mis_overhead}")
    
    # Visualize original graph
    plt.figure(figsize=(6, 6))
    nx.draw_networkx(graph, with_labels=True, node_color="lightblue")
    plt.title("Original Petersen Graph")
    plt.savefig("petersen_original.png")
    
    # Visualize mapped graph
    mapped_graph = unweighted_res.grid_graph.to_networkx()
    positions = {i: unweighted_res.grid_graph.nodes[i].loc for i in range(len(unweighted_res.grid_graph.nodes))}
    
    plt.figure(figsize=(10, 10))
    nx.draw_networkx(mapped_graph, pos=positions, node_color="lightgreen", with_labels=False)
    plt.title("Mapped Unit Disk Graph")
    plt.savefig("petersen_mapped.png")
    
    # Create a sample configuration (all ones)
    grid_size = unweighted_res.grid_graph.size
    config = np.zeros(grid_size, dtype=int)
    
    # Set alternating nodes to 1 for demonstration
    for i, node in enumerate(unweighted_res.grid_graph.nodes):
        if i % 2 == 0:
            config[node.loc] = 1
    
    # Print configuration
    print("\nSample configuration:")
    print(print_config(unweighted_res, config))

def small_graph_example():
    """Example using a small custom graph."""
    # Create a small graph
    G = nx.Graph()
    G.add_edges_from([(0, 1), (0, 2), (1, 3), (2, 3), (2, 4), (3, 5), (4, 5)])
    
    print("Mapping graph using Greedy method...")
    result = map_graph(G, vertex_order=Greedy(nrepeat=5))
    
    print(f"Grid graph size: {result.grid_graph.size}")
    print(f"MIS overhead: {result.mis_overhead}")
    
    # Visualize original graph
    plt.figure(figsize=(6, 6))
    nx.draw_networkx(G, with_labels=True, node_color="lightblue")
    plt.title("Original Small Graph")
    plt.savefig("small_original.png")
    
    # Visualize mapped graph
    mapped_graph = result.grid_graph.to_networkx()
    positions = {i: result.grid_graph.nodes[i].loc for i in range(len(result.grid_graph.nodes))}
    
    plt.figure(figsize=(10, 10))
    nx.draw_networkx(mapped_graph, pos=positions, node_color="lightgreen", with_labels=False)
    plt.title("Mapped Unit Disk Graph")
    plt.savefig("small_mapped.png")

if __name__ == "__main__":
    print("Running Unit Disk Mapping examples...")
    
    petersen_graph_example()
    print("\n" + "="*50 + "\n")
    small_graph_example()
    
    print("\nExamples completed. Check the generated image files for visualizations.")