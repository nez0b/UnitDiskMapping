#!/usr/bin/env python
"""
Script to visualize gadget placement at crossing points.
"""

import networkx as nx
import matplotlib.pyplot as plt
import numpy as np
from src.unit_disk_mapping import map_graph
from src.mapping import embed_graph, crossat
from src.gadgets import Cross, crossing_ruleset
from src.core import SimpleCell, Node

def main():
    """Visualize gadget placement at crossing points."""
    # Create a simple graph with 5 vertices
    g = nx.Graph()
    g.add_edges_from([
        (0, 1), (0, 3), (0, 4), (2, 1), (2, 3), (2, 4)
    ])
    
    # Step 1: Create the mapping grid
    ug = embed_graph(g, vertex_order=[0, 1, 2, 3, 4])
    
    # Step 2: Get all vertex pairs and their crossing points
    vertices = [line.vertex for line in ug.lines]
    crossings = []
    for i, v in enumerate(vertices):
        for j, w in enumerate(vertices[i+1:], i+1):
            has_edge = g.has_edge(v, w)
            cx, cy = crossat(ug, v, w)
            crossings.append((v, w, cx, cy, has_edge))
    
    # Step 3: Create a visualization of where gadgets would be placed
    mapping_history = []
    for v, w, cx, cy, has_edge in crossings:
        # Choose the appropriate pattern
        pattern = Cross(has_edge=has_edge)
        
        # Get coordinates for where gadget would be applied
        pl = pattern.cross_location()
        x, y = cx - pl[0], cy - pl[1]
        
        # Record the application
        mapping_history.append((pattern, x, y, v, w, has_edge))
        print(f'Applied {pattern.__class__.__name__} (has_edge={has_edge}) at ({x}, {y}) for vertices {v},{w}')
    
    # Step 4: Count the gadgets
    print(f'\nTotal crossings: {len(crossings)}')
    print(f'Crossings with edges: {sum(1 for _, _, _, _, has_edge in crossings if has_edge)}')
    print(f'Crossings without edges: {sum(1 for _, _, _, _, has_edge in crossings if not has_edge)}')
    print(f'Applied gadgets: {len(mapping_history)}')
    
    # Create a visualization showing the gadget placement
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
        
        # Show the pattern box for the gadget
        pattern = Cross(has_edge=has_edge)
        pl = pattern.cross_location()
        x, y = cx - pl[0], cy - pl[1]
        m, n = pattern.size()
        
        # Draw a rectangle around where the gadget would be applied
        rect = plt.Rectangle((y, x), n, m, edgecolor=color, facecolor='none', linestyle='--', alpha=0.7)
        plt.gca().add_patch(rect)
        
        # Add label for the pattern type
        plt.text(y + n/2, x + m/2, f'Cross\nhas_edge={has_edge}', 
                 fontsize=6, ha='center', va='center', color=color)
    
    # Add grid
    height = (len(vertices) - 1) * s + padding * 2
    width = (len(vertices) - 1) * s + padding * 2
    for i in range(height+1):
        plt.axhline(i-0.5, color='gray', lw=0.5, alpha=0.3)
    for j in range(width+1):
        plt.axvline(j-0.5, color='gray', lw=0.5, alpha=0.3)
    
    # Set axis properties
    plt.title('Crossing Lattice with Gadget Placement')
    plt.xlabel('Column (j)')
    plt.ylabel('Row (i)')
    plt.xlim(-0.5, width-0.5)
    plt.ylim(height-0.5, -0.5)  # Invert y-axis
    
    plt.savefig('gadget_placement.png', dpi=300)
    print('Created visualization: gadget_placement.png')

if __name__ == "__main__":
    main()