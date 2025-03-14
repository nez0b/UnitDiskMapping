"""
Example demonstrating the enhanced complex gadget transformations.

This example demonstrates how to use the enhanced complex gadget patterns
for mapping complex graph structures to unit disk graphs.
"""

import numpy as np
import networkx as nx
import matplotlib.pyplot as plt
from matplotlib.patches import Circle
import matplotlib.gridspec as gridspec

from src.unit_disk_mapping import (
    map_graph, 
    UnWeighted, 
    MinhThiTrick, 
    Node,
    cross_location,
    ComplexGadget,
    StarPattern,
    SpiralPattern,
    DiagonalCross,
    DoubleCross,
    enhanced_crossing_ruleset,
    complete_enhanced_crossing_ruleset
)


def draw_pattern_source_mapped(ax_source, ax_mapped, pattern, title=None):
    """Draw source and mapped graphs for a pattern."""
    # Get source and mapped data
    source_locs, source_graph, source_pins = pattern.source_graph()
    mapped_locs, mapped_graph, mapped_pins = pattern.mapped_graph()
    
    # Create a 2D grid representation
    m, n = pattern.size()
    source_grid = np.zeros((m, n))
    mapped_grid = np.zeros((m, n))
    
    # Fill in occupied cells
    for loc in source_locs:
        i, j = loc.loc
        source_grid[i, j] = 1
    
    for loc in mapped_locs:
        i, j = loc.loc
        mapped_grid[i, j] = 1
    
    # Draw grids
    ax_source.imshow(source_grid, cmap='Blues', vmin=0, vmax=1)
    ax_mapped.imshow(mapped_grid, cmap='Blues', vmin=0, vmax=1)
    
    # Add grid lines
    for i in range(m+1):
        ax_source.axhline(i-0.5, color='gray', lw=0.5)
        ax_mapped.axhline(i-0.5, color='gray', lw=0.5)
    for j in range(n+1):
        ax_source.axvline(j-0.5, color='gray', lw=0.5)
        ax_mapped.axvline(j-0.5, color='gray', lw=0.5)
    
    # Mark cross location
    cross_x, cross_y = pattern.cross_location()
    ax_source.plot(cross_y, cross_x, 'rx', markersize=10)
    ax_mapped.plot(cross_y, cross_x, 'rx', markersize=10)
    
    # Mark pins
    for pin_idx in source_pins:
        i, j = source_locs[pin_idx].loc
        ax_source.plot(j, i, 'go', markersize=8, alpha=0.7)
    
    for pin_idx in mapped_pins:
        i, j = mapped_locs[pin_idx].loc
        ax_mapped.plot(j, i, 'go', markersize=8, alpha=0.7)
    
    # Add node labels
    for i, loc in enumerate(source_locs):
        x, y = loc.loc
        ax_source.text(y, x, str(i), ha='center', va='center', color='black', fontsize=8)
    
    for i, loc in enumerate(mapped_locs):
        x, y = loc.loc
        ax_mapped.text(y, x, str(i), ha='center', va='center', color='black', fontsize=8)
    
    # Set titles
    ax_source.set_title("Source Pattern")
    ax_mapped.set_title("Mapped Pattern")
    
    if title:
        ax_source.set_ylabel(title)
    
    # Set axes
    ax_source.set_xticks(range(n))
    ax_source.set_yticks(range(m))
    ax_mapped.set_xticks(range(n))
    ax_mapped.set_yticks(range(m))


def draw_cross_graph_mapping(graph, mapped_result, pattern_name, fig_path=None):
    """Draw a graph and its unit disk mapping with complex pattern visualization."""
    fig = plt.figure(figsize=(18, 10))
    gs = gridspec.GridSpec(2, 3, width_ratios=[1, 2, 2], height_ratios=[1, 1])
    
    # Draw original graph
    ax1 = fig.add_subplot(gs[0, 0])
    pos = nx.spring_layout(graph, seed=42)
    nx.draw(graph, pos, with_labels=True, node_color='skyblue', 
            node_size=500, font_size=10, ax=ax1)
    ax1.set_title(f"Original Graph: {graph.order()} nodes, {graph.size()} edges")
    
    # Draw mapped unit disk graph
    ax2 = fig.add_subplot(gs[0, 1:])
    grid_graph = mapped_result.grid_graph
    G_mapped = nx.Graph()
    pos_mapped = {}
    
    # Add nodes
    for i, node in enumerate(grid_graph.nodes):
        G_mapped.add_node(i)
        # Flip y for visualization
        pos_mapped[i] = (node.loc[1], -node.loc[0])
    
    # Add edges based on unit disk distance
    for i in range(len(grid_graph.nodes)):
        for j in range(i+1, len(grid_graph.nodes)):
            dist = ((grid_graph.nodes[i].loc[0] - grid_graph.nodes[j].loc[0])**2 + 
                    (grid_graph.nodes[i].loc[1] - grid_graph.nodes[j].loc[1])**2)**0.5
            if dist <= grid_graph.radius:
                G_mapped.add_edge(i, j)
    
    nx.draw(G_mapped, pos_mapped, node_color='lightgreen', node_size=100, 
            width=0.5, with_labels=False, ax=ax2)
    
    # Draw radio circles for a few nodes
    for i in range(min(5, len(G_mapped.nodes()))):
        circle = Circle(pos_mapped[i], grid_graph.radius, alpha=0.1, fc='gray', ec='gray')
        ax2.add_patch(circle)
    
    ax2.set_title(f"Mapped Unit Disk Graph: {G_mapped.order()} nodes, {G_mapped.size()} edges")
    
    # Draw pattern before and after mapping
    pattern = None
    for p in enhanced_crossing_ruleset:
        if p.__class__.__name__ == pattern_name:
            pattern = p
            break
    
    if pattern:
        ax3 = fig.add_subplot(gs[1, 0])
        ax4 = fig.add_subplot(gs[1, 1])
        draw_pattern_source_mapped(ax3, ax4, pattern, title=pattern_name)
        
        # Draw statistics
        ax5 = fig.add_subplot(gs[1, 2])
        ax5.axis('off')
        statistics = [
            f"Pattern: {pattern_name}",
            f"Grid size: {pattern.size()}",
            f"Cross location: {pattern.cross_location()}",
            f"Source vertices: {len(pattern.source_graph()[0])}",
            f"Mapped vertices: {len(pattern.mapped_graph()[0])}",
            f"Vertex overhead: {pattern.vertex_overhead()}",
            f"MIS overhead: {pattern.vertex_overhead()}",
            f"\nGraph mapping statistics:",
            f"Original nodes: {graph.order()}",
            f"Mapped nodes: {G_mapped.order()}",
            f"Node expansion ratio: {G_mapped.order() / graph.order():.2f}",
            f"MIS overhead from mapping: {mapped_result.mis_overhead}"
        ]
        
        ax5.text(0.05, 0.95, "\n".join(statistics), 
                 va='top', ha='left', fontsize=12)
    
    plt.tight_layout()
    
    if fig_path:
        plt.savefig(fig_path, dpi=300, bbox_inches='tight')
    
    plt.show()


def main():
    """Main function demonstrating enhanced complex gadget transformations."""
    # Create example graphs
    triangle = nx.complete_graph(3)
    diamond = nx.Graph()
    diamond.add_edges_from([(0, 1), (0, 2), (1, 3), (2, 3)])
    butterfly = nx.Graph()
    butterfly.add_edges_from([(0, 1), (0, 2), (1, 2), (2, 3), (2, 4), (3, 4)])
    
    # Map graphs with different patterns
    print("Mapping triangle graph with StarPattern...")
    triangle_mapped = map_graph(
        triangle, 
        mode=UnWeighted(), 
        ruleset=[StarPattern()]
    )
    
    print("Mapping diamond graph with SpiralPattern...")
    diamond_mapped = map_graph(
        diamond, 
        mode=UnWeighted(), 
        ruleset=[SpiralPattern()]
    )
    
    print("Mapping butterfly graph with DoubleCross...")
    butterfly_mapped = map_graph(
        butterfly, 
        mode=UnWeighted(), 
        ruleset=[DoubleCross()]
    )
    
    # Draw results
    draw_cross_graph_mapping(
        triangle, triangle_mapped, "StarPattern", 
        fig_path="triangle_star_pattern.png"
    )
    
    draw_cross_graph_mapping(
        diamond, diamond_mapped, "SpiralPattern", 
        fig_path="diamond_spiral_pattern.png"
    )
    
    draw_cross_graph_mapping(
        butterfly, butterfly_mapped, "DoubleCross", 
        fig_path="butterfly_double_cross.png"
    )
    
    # Print statistics summary
    print("\nMapping Statistics Summary:")
    print(f"Triangle: {triangle.order()} nodes → {len(triangle_mapped.grid_graph.nodes)} nodes")
    print(f"Diamond: {diamond.order()} nodes → {len(diamond_mapped.grid_graph.nodes)} nodes")
    print(f"Butterfly: {butterfly.order()} nodes → {len(butterfly_mapped.grid_graph.nodes)} nodes")


if __name__ == "__main__":
    main()