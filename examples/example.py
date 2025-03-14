"""
Example usage of the Unit Disk Mapping package.
"""

import networkx as nx
import matplotlib.pyplot as plt
import numpy as np
import sys
import os

# Add the project root to the Python path if needed
project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if project_root not in sys.path:
    sys.path.insert(0, project_root)

from src.unit_disk_mapping import map_graph, MinhThiTrick, Greedy, print_config

def visualize_grid_graph_with_copylines(result, title="Mapped Unit Disk Graph", save_path=None):
    """
    Visualize a grid graph with copylines labeled with vertex indices.
    
    Args:
        result: Mapping result object
        title: Title for the plot
        save_path: Path to save the figure
        
    Returns:
        The matplotlib figure
    """
    # Create a networkx graph from the grid graph
    mapped_graph = result.grid_graph.to_networkx()
    positions = {i: result.grid_graph.nodes[i].loc for i in range(len(result.grid_graph.nodes))}
    
    fig, ax = plt.subplots(figsize=(12, 12))
    
    # Draw the graph
    nx.draw_networkx(mapped_graph, pos=positions, node_color="lightgreen", 
                     with_labels=False, ax=ax)
    
    # Group nodes by copylines
    copyline_nodes = {}
    
    # In a real mapping result, we have access to lines attribute
    if hasattr(result, 'lines'):
        for line in result.lines:
            vertex = line.vertex
            
            # Find nodes that belong to this copyline
            line_nodes = []
            for i, node in enumerate(result.grid_graph.nodes):
                x, y = node.loc
                
                # Simple heuristic to identify nodes belonging to this copyline
                # In a real implementation, this would use more sophisticated matching
                hslot = getattr(line, 'hslot', -1)
                vstart = getattr(line, 'vstart', -1)
                vstop = getattr(line, 'vstop', -1)
                
                # Very basic matching - nodes with similar x coordinate
                # and y coordinate within the vstart-vstop range
                if ((x == hslot or abs(x - hslot) < 2) and 
                    (vstart <= y <= vstop or abs(y - vstart) < 2 or abs(y - vstop) < 2)):
                    line_nodes.append(i)
            
            copyline_nodes[vertex] = line_nodes
            
            # If we find nodes for this copyline, label it
            if line_nodes:
                # Find average position for the label
                avg_x = sum(positions[i][0] for i in line_nodes) / len(line_nodes)
                min_y = min(positions[i][1] for i in line_nodes) - 1
                
                # Add a label with the vertex number
                ax.annotate(f"Vertex {vertex}", 
                           xy=(avg_x, min_y + 0.5), 
                           xytext=(avg_x, min_y),
                           fontsize=10, 
                           ha='center',
                           bbox=dict(boxstyle="round,pad=0.3", fc="yellow", alpha=0.7),
                           arrowprops=dict(arrowstyle="->", connectionstyle="arc3,rad=.2"))
    
    ax.set_title(title)
    
    # Save the figure if a path is provided
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
    
    return fig

def petersen_graph_example():
    """Example using the Petersen graph."""
    print("Creating Petersen graph...")
    graph = nx.petersen_graph()
    
    print("Mapping graph using MinhThiTrick...")
    unweighted_res = map_graph(graph, vertex_order=MinhThiTrick())
    
    print(f"Grid graph size: {unweighted_res.grid_graph.size}")
    print(f"MIS overhead: {unweighted_res.mis_overhead}")
    
    # Create img directory if it doesn't exist
    img_dir = os.path.join(project_root, "img")
    if not os.path.exists(img_dir):
        os.makedirs(img_dir)
    
    # Visualize original graph
    plt.figure(figsize=(6, 6))
    nx.draw_networkx(graph, with_labels=True, node_color="lightblue")
    plt.title("Original Petersen Graph")
    plt.savefig(os.path.join(img_dir, "petersen_original.png"))
    
    # Visualize mapped graph with copylines labeled
    fig = visualize_grid_graph_with_copylines(
        unweighted_res, 
        title="Mapped Petersen Graph with Labeled Copylines",
        save_path=os.path.join(img_dir, "petersen_mapped_with_copylines.png")
    )
    
    # Also save a simple version without labels for comparison
    mapped_graph = unweighted_res.grid_graph.to_networkx()
    positions = {i: unweighted_res.grid_graph.nodes[i].loc for i in range(len(unweighted_res.grid_graph.nodes))}
    
    plt.figure(figsize=(10, 10))
    nx.draw_networkx(mapped_graph, pos=positions, node_color="lightgreen", with_labels=False)
    plt.title("Mapped Unit Disk Graph")
    plt.savefig(os.path.join(img_dir, "petersen_mapped.png"))
    
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
    # Create img directory if it doesn't exist
    img_dir = os.path.join(project_root, "img")
    if not os.path.exists(img_dir):
        os.makedirs(img_dir)
        
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
    plt.savefig(os.path.join(img_dir, "small_original.png"))
    
    # Visualize mapped graph with copylines labeled
    fig = visualize_grid_graph_with_copylines(
        result, 
        title="Mapped Small Graph with Labeled Copylines",
        save_path=os.path.join(img_dir, "small_mapped_with_copylines.png")
    )
    
    # Also save a simple version without labels for comparison
    mapped_graph = result.grid_graph.to_networkx()
    positions = {i: result.grid_graph.nodes[i].loc for i in range(len(result.grid_graph.nodes))}
    
    plt.figure(figsize=(10, 10))
    nx.draw_networkx(mapped_graph, pos=positions, node_color="lightgreen", with_labels=False)
    plt.title("Mapped Unit Disk Graph")
    plt.savefig(os.path.join(img_dir, "small_mapped.png"))

if __name__ == "__main__":
    print("Running Unit Disk Mapping examples...")
    
    petersen_graph_example()
    print("\n" + "="*50 + "\n")
    small_graph_example()
    
    print("\nExamples completed. Check the generated image files for visualizations.")