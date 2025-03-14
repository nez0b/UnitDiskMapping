#!/usr/bin/env python
"""
Script to compare the original and fixed implementations.
"""

import networkx as nx
import matplotlib.pyplot as plt
import numpy as np
from src.unit_disk_mapping import map_graph
from src.mapping import embed_graph, crossat, apply_crossing_gadgets, UnWeighted

def main():
    """Compare original and fixed implementations."""
    # Create a simple graph with 5 vertices
    g = nx.Graph()
    g.add_edges_from([
        (0, 1), (0, 3), (0, 4), (2, 1), (2, 3), (2, 4)
    ])
    
    print('Original graph edges:')
    for e in g.edges():
        print(f'Edge {e}')
    
    # Fixed Implementation Test
    print('\n=== FIXED IMPLEMENTATION ===')
    # Map the graph with the fixed implementation
    result = map_graph(g, vertex_order=[0, 1, 2, 3, 4])
    
    # Count gadgets by edge status
    edge_gadgets = sum(1 for entry in result.mapping_history if len(entry) >= 6 and entry[5])
    no_edge_gadgets = sum(1 for entry in result.mapping_history if len(entry) >= 6 and not entry[5])
    correct_gadgets = sum(1 for entry in result.mapping_history 
                        if len(entry) >= 6 and entry[0].is_connected() == entry[5])
    
    print(f'Total applied gadgets: {len(result.mapping_history)}')
    print(f'Gadgets with edges: {edge_gadgets}')
    print(f'Gadgets without edges: {no_edge_gadgets}')
    print(f'Gadgets with correct edge status: {correct_gadgets}')
    
    # Original vs Expected
    edges_count = len(g.edges())
    vertices = list(range(5))
    total_crossings = sum(1 for v in vertices for w in vertices if v < w)
    non_edges_count = total_crossings - edges_count
    
    print(f'\nExpected gadgets:')
    print(f'Total crossings: {total_crossings}')
    print(f'Crossings with edges: {edges_count}')
    print(f'Crossings without edges: {non_edges_count}')
    
    print(f'\nFinal unit disk graph has {len(result.grid_graph.nodes)} nodes')
    
    # Visualize the final fixed grid
    plt.figure(figsize=(12, 10))
    
    # Draw the nodes
    grid_graph = result.grid_graph
    for node in grid_graph.nodes:
        i, j = node.loc
        plt.scatter(j, i, color='blue', s=50)
    
    # Draw the edges
    for i, node1 in enumerate(grid_graph.nodes):
        for j, node2 in enumerate(grid_graph.nodes[i+1:], i+1):
            i1, j1 = node1.loc
            i2, j2 = node2.loc
            # Check if within unit disk distance
            if ((i1-i2)**2 + (j1-j2)**2)**0.5 <= grid_graph.radius:
                plt.plot([j1, j2], [i1, i2], 'k-', alpha=0.2)
    
    # Add grid
    height, width = grid_graph.size
    for i in range(height+1):
        plt.axhline(i-0.5, color='gray', lw=0.5, alpha=0.3)
    for j in range(width+1):
        plt.axvline(j-0.5, color='gray', lw=0.5, alpha=0.3)
    
    # Set axis properties
    plt.title('Fixed Implementation: Final Unit Disk Graph')
    plt.xlabel('Column (j)')
    plt.ylabel('Row (i)')
    plt.xlim(-0.5, width-0.5)
    plt.ylim(height-0.5, -0.5)  # Invert y-axis
    
    plt.savefig('fixed_implementation.png', dpi=300)
    print('Created visualization: fixed_implementation.png')
    
    # Test applying gadgets with direct approach
    print('\n=== DIRECT APPROACH ===')
    ug = embed_graph(g, vertex_order=[0, 1, 2, 3, 4])
    
    # Get all vertex pairs and their crossing points
    vertices = [line.vertex for line in ug.lines]
    crossings = []
    for i, v in enumerate(vertices):
        for j, w in enumerate(vertices[i+1:], i+1):
            has_edge = g.has_edge(v, w)
            cx, cy = crossat(ug, v, w)
            crossings.append((v, w, cx, cy, has_edge))
    
    print(f'Total crossings identified: {len(crossings)}')
    print(f'Crossings with edges: {sum(1 for _, _, _, _, has_edge in crossings if has_edge)}')
    print(f'Crossings without edges: {sum(1 for _, _, _, _, has_edge in crossings if not has_edge)}')
    
    # Apply crossing gadgets directly with the original graph
    from gadgets import Cross
    mapping_history = []
    for v, w, cx, cy, has_edge in crossings:
        # Choose the appropriate pattern
        pattern = Cross(has_edge=has_edge)
        
        # Get coordinates for where gadget would be applied
        pl = pattern.cross_location()
        x, y = cx - pl[0], cy - pl[1]
        
        # Check bounds
        if x < 0 or y < 0 or x + pattern.size()[0] > len(ug.content) or y + pattern.size()[1] > len(ug.content[0]):
            print(f"Warning: Pattern for vertices {v},{w} at ({x},{y}) is out of bounds - skipping")
            continue
            
        # Record the application
        mapping_history.append((pattern, x, y, v, w, has_edge))
    
    print(f'Total gadgets that could be applied: {len(mapping_history)}')
    
    # Create a visualization showing the direct gadget placement
    plt.figure(figsize=(12, 10))
    
    # Draw the crossing lattice with copylines
    colors = ['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728', '#9467bd']
    s = 2  # spacing factor
    padding = ug.padding
    
    # Draw the copylines
    for i, line in enumerate(ug.lines):
        # Vertical line
        vslot = line.vslot
        vslot_x = (vslot - 1) * s + padding + 1
        vstart_y = padding
        vstop_y = (len(vertices) - 1) * s + padding
        
        # Horizontal line
        hslot = line.hslot
        hslot_y = (hslot - 1) * s + padding + 2
        hstop_x = (len(vertices) - 1) * s + padding
        
        plt.plot([vslot_x, vslot_x], [vstart_y, vstop_y], color=colors[i], linewidth=2, alpha=0.5)
        plt.plot([vslot_x, hstop_x], [hslot_y, hslot_y], color=colors[i], linewidth=2, alpha=0.5)
        
        # Add vertex label
        plt.text(vslot_x - 0.5, hslot_y - 0.5, str(line.vertex), fontsize=12, fontweight='bold')
    
    # Mark crossing points and gadgets
    for v, w, cx, cy, has_edge in crossings:
        # Mark the crossing point
        marker = 'o' if has_edge else 'x'
        color = 'green' if has_edge else 'red'
        plt.scatter(cy, cx, color=color, marker=marker, s=100, zorder=3)
        
        # Add label
        plt.text(cy, cx - 0.3, f'({v},{w})', fontsize=8, ha='center', va='bottom')
    
    # Mark the gadget boxes for applicaple gadgets
    for pattern, x, y, v, w, has_edge in mapping_history:
        m, n = pattern.size()
        color = 'green' if has_edge else 'red'
        
        # Draw a rectangle around where the gadget would be applied
        rect = plt.Rectangle((y, x), n, m, edgecolor=color, facecolor='none', linestyle='--', alpha=0.7)
        plt.gca().add_patch(rect)
    
    # Add grid
    height = (len(vertices) - 1) * s + padding * 2
    width = (len(vertices) - 1) * s + padding * 2
    for i in range(height+1):
        plt.axhline(i-0.5, color='gray', lw=0.5, alpha=0.3)
    for j in range(width+1):
        plt.axvline(j-0.5, color='gray', lw=0.5, alpha=0.3)
    
    # Set axis properties
    plt.title('Direct Approach: Crossing Lattice with Applicable Gadgets')
    plt.xlabel('Column (j)')
    plt.ylabel('Row (i)')
    plt.xlim(-0.5, width-0.5)
    plt.ylim(height-0.5, -0.5)  # Invert y-axis
    
    plt.savefig('direct_approach.png', dpi=300)
    print('Created visualization: direct_approach.png')

if __name__ == "__main__":
    main()