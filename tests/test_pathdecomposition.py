"""
Test script for path decomposition algorithms.

This script tests the path decomposition algorithms on a few simple graphs
to verify that they work correctly.
"""

import networkx as nx
import matplotlib.pyplot as plt
from src.pathdecomposition import Layout, MinhThiTrick, Greedy, pathwidth

def test_path_decomposition():
    """Test path decomposition algorithms on various graphs."""
    # Create some test graphs
    graphs = {
        "path": nx.path_graph(5),
        "cycle": nx.cycle_graph(5),
        "complete": nx.complete_graph(5),
        "petersen": nx.petersen_graph(),
        "grid": nx.grid_2d_graph(3, 3)
    }
    
    # Convert grid graph node labels to integers
    if "grid" in graphs:
        grid = graphs["grid"]
        mapping = {node: i for i, node in enumerate(grid.nodes())}
        graphs["grid"] = nx.relabel_nodes(grid, mapping)
    
    # Test both path decomposition methods
    methods = {
        "greedy": Greedy(nrepeat=5),
        "minh_thi": MinhThiTrick()
    }
    
    # For each graph and method, compute the path decomposition
    results = {}
    for graph_name, graph in graphs.items():
        results[graph_name] = {}
        
        print(f"Testing {graph_name} graph...")
        for method_name, method in methods.items():
            # Skip MinhThiTrick for larger graphs (it's slow)
            if method_name == "minh_thi" and (
                graph_name == "petersen" or 
                graph_name == "grid" or
                graph_name == "complete"
            ):
                continue
                
            print(f"  Computing with {method_name}...")
            try:
                decomp = pathwidth(graph, method)
                results[graph_name][method_name] = {
                    "decomp": decomp,
                    "success": True
                }
                print(f"    Vertex order: {decomp.vertices}")
                print(f"    Vertex separation: {decomp.vsep}")
            except Exception as e:
                results[graph_name][method_name] = {
                    "decomp": None,
                    "success": False,
                    "error": str(e)
                }
                print(f"    Failed: {e}")
    
    # Visualize some results
    for graph_name in ["path", "cycle"]:
        for method_name in methods:
            if graph_name in results and method_name in results[graph_name]:
                result = results[graph_name][method_name]
                if result["success"]:
                    visualize_path_decomposition(
                        graphs[graph_name], 
                        result["decomp"],
                        f"{graph_name}_{method_name}"
                    )
    
    return results

def visualize_path_decomposition(graph, decomp, name):
    """Visualize a path decomposition."""
    # Create figure with two subplots
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))
    
    # Draw the original graph
    pos = nx.spring_layout(graph, seed=42)
    nx.draw_networkx(graph, pos=pos, ax=ax1, node_color="lightblue", with_labels=True)
    ax1.set_title(f"Original Graph")
    ax1.axis("off")
    
    # Draw the graph with vertex order labels
    labels = {node: str(i+1) for i, node in enumerate(decomp.vertices)}
    nx.draw_networkx(graph, pos=pos, ax=ax2, node_color="lightgreen", with_labels=True)
    nx.draw_networkx_labels(graph, pos=pos, labels=labels, ax=ax2, font_color='red')
    ax2.set_title(f"Path Decomposition (vsep = {decomp.vsep})")
    ax2.axis("off")
    
    plt.tight_layout()
    plt.savefig(f"{name}_decomp.png")
    plt.close()

if __name__ == "__main__":
    print("Testing path decomposition algorithms...")
    results = test_path_decomposition()
    print("Tests completed.")