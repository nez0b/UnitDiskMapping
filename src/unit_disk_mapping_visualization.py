#!/usr/bin/env python
"""
Comprehensive visualization of the Unit Disk Mapping process for a 5-vertex graph.
This script shows each step of the mapping process with detailed labeling:
1. Original graph structure
2. Copylines creation and visualization
3. Crossing points identification with edge status
4. Detailed gadget application at crossings
5. Final unit disk graph with node and edge visualization
"""

import networkx as nx
import matplotlib.pyplot as plt
import numpy as np
from unit_disk_mapping import map_graph
from mapping import embed_graph, crossat, apply_crossing_gadgets, UnWeighted
from copyline import center_location, CopyLine

def plot_original_graph(g):
    """Plot the original graph structure with labeled vertices and edges."""
    plt.figure(figsize=(10, 8))
    
    # Use a circular layout for pentagonal structure
    pos = nx.circular_layout(g)
    
    # Draw nodes
    nx.draw_networkx_nodes(g, pos, node_size=700, node_color='skyblue')
    
    # Draw edges with different colors based on type (cycle vs diagonal)
    cycle_edges = [(0, 1), (1, 2), (2, 3), (3, 4), (4, 0)]
    diagonal_edges = [(0, 2)]
    
    nx.draw_networkx_edges(g, pos, edgelist=cycle_edges, width=3, edge_color='blue')
    nx.draw_networkx_edges(g, pos, edgelist=diagonal_edges, width=3, edge_color='red')
    
    # Draw labels
    nx.draw_networkx_labels(g, pos, font_size=16, font_weight='bold')
    
    # Add edge labels (for diagonal)
    edge_labels = {(0, 2): "Diagonal"}
    nx.draw_networkx_edge_labels(g, pos, edge_labels=edge_labels, font_size=14)
    
    plt.title("Original 5-Vertex Graph (House Graph)", fontsize=18)
    plt.axis('off')
    plt.tight_layout()
    plt.savefig('udm_viz_original.png', dpi=300)
    print("Created visualization: udm_viz_original.png")
    plt.close()

def visualize_copylines(ug):
    """Visualize the T-shaped copylines structure with detailed labeling."""
    plt.figure(figsize=(14, 12))
    
    # Grid size
    height, width = len(ug.content), len(ug.content[0])
    
    # Create colormap for different copylines
    colors = ['blue', 'green', 'red', 'purple', 'orange']
    
    # Draw each copyline with a unique color
    for i, line in enumerate(ug.lines):
        color = colors[i % len(colors)]
        
        # Get center location
        I, J = center_location(line, ug.padding)
        
        # Calculate span points based on spacing factor
        s = 2  # spacing factor from copyline.py
        
        # Vertical span
        vstart = I + s * (line.vstart - line.hslot) + 1
        vstop = I + s * (line.vstop - line.hslot) - 1
        
        # Horizontal span
        hstop = J + s * (line.hstop - line.vslot) - 1
        
        # Draw vertical line with markers
        plt.plot([J, J], [vstart, vstop], '-', color=color, linewidth=3, alpha=0.7)
        for y in range(int(vstart), int(vstop)+1):
            plt.scatter(J, y, color=color, s=100, zorder=10)
        
        # Draw horizontal line with markers
        plt.plot([J, hstop], [I, I], '-', color=color, linewidth=3, alpha=0.7)
        for x in range(int(J), int(hstop)+1):
            plt.scatter(x, I, color=color, s=100, zorder=10)
        
        # Mark and label the center of the copyline
        plt.scatter(J, I, color=color, s=200, edgecolor='black', zorder=20)
        plt.text(J, I-0.7, f"v{line.vertex}", fontsize=14, ha='center', fontweight='bold',
                 bbox=dict(facecolor='white', alpha=0.7))
        
        # Add copyline information
        plt.text(width*0.75, i+1, f"Copyline {line.vertex}: vslot={line.vslot}, hslot={line.hslot}, " +
                f"vstart={line.vstart}, vstop={line.vstop}, hstop={line.hstop}",
                fontsize=10, color=color, bbox=dict(facecolor='white', alpha=0.7))
    
    # Add grid
    for i in range(height+1):
        plt.axhline(i-0.5, color='gray', lw=0.5, alpha=0.3)
    for j in range(width+1):
        plt.axvline(j-0.5, color='gray', lw=0.5, alpha=0.3)
    
    # Set axis properties
    plt.title('T-shaped Copylines Structure', fontsize=18)
    plt.xlabel('Column (j)', fontsize=14)
    plt.ylabel('Row (i)', fontsize=14)
    plt.xlim(-0.5, width-0.5)
    plt.ylim(height-0.5, -0.5)  # Invert y-axis
    
    # Add legend
    legend_elements = []
    for i, color in enumerate(colors[:len(ug.lines)]):
        legend_elements.append(
            plt.Line2D([0], [0], marker='o', color=color, lw=3, 
                      markersize=10, label=f'Copyline v{ug.lines[i].vertex}')
        )
    plt.legend(handles=legend_elements, loc='upper left', fontsize=12)
    
    plt.savefig('udm_viz_copylines.png', dpi=300)
    print("Created visualization: udm_viz_copylines.png")
    plt.close()

def visualize_crossing_points(ug, g):
    """Visualize the crossing points with detailed edge status information."""
    plt.figure(figsize=(14, 12))
    
    # Grid size
    height, width = len(ug.content), len(ug.content[0])
    
    # Create colormap for different copylines
    colors = ['blue', 'green', 'red', 'purple', 'orange']
    
    # Draw each copyline with a unique color but with reduced opacity
    for i, line in enumerate(ug.lines):
        color = colors[i % len(colors)]
        
        # Get center location
        I, J = center_location(line, ug.padding)
        
        # Calculate span points based on spacing factor
        s = 2  # spacing factor from copyline.py
        
        # Vertical span
        vstart = I + s * (line.vstart - line.hslot) + 1
        vstop = I + s * (line.vstop - line.hslot) - 1
        
        # Horizontal span
        hstop = J + s * (line.hstop - line.vslot) - 1
        
        # Draw vertical line with markers
        plt.plot([J, J], [vstart, vstop], '-', color=color, linewidth=2, alpha=0.3)
        for y in range(int(vstart), int(vstop)+1):
            plt.scatter(J, y, color=color, s=50, alpha=0.3)
        
        # Draw horizontal line with markers
        plt.plot([J, hstop], [I, I], '-', color=color, linewidth=2, alpha=0.3)
        for x in range(int(J), int(hstop)+1):
            plt.scatter(x, I, color=color, s=50, alpha=0.3)
        
        # Mark the center of the copyline
        plt.scatter(J, I, color=color, s=150, edgecolor='black', alpha=0.5)
        plt.text(J, I-0.7, f"v{line.vertex}", fontsize=12, ha='center', fontweight='bold')
    
    # Check all pairs of vertices for crossings and display with detailed information
    vertices = [line.vertex for line in ug.lines]
    crossing_info = []
    
    for i, v in enumerate(vertices):
        for j, w in enumerate(vertices[i+1:], i+1):
            has_edge = g.has_edge(v, w)
            cx, cy = crossat(ug, v, w)
            
            # Store crossing information
            crossing_info.append((v, w, has_edge, cx, cy))
            
            # Draw crossing point with prominent marker
            color = 'limegreen' if has_edge else 'red'
            marker = 'D' if has_edge else 'X'  # diamond for edge, X for no edge
            size = 200
            plt.scatter(cy, cx, color=color, marker=marker, s=size, edgecolor='black', linewidth=1.5, zorder=30)
            
            # Add label with detailed information
            edge_status = "EDGE EXISTS" if has_edge else "NO EDGE"
            plt.text(cy+0.5, cx, f"v{v}-v{w}\n{edge_status}", fontsize=10, ha='left', va='center',
                    bbox=dict(facecolor='white', alpha=0.8))
    
    # Add table of crossing points
    table_text = "Crossing Points:\n"
    for idx, (v, w, has_edge, cx, cy) in enumerate(crossing_info):
        status = "Edge" if has_edge else "No Edge"
        table_text += f"{idx+1}. v{v}-v{w}: ({cx},{cy}), {status}\n"
    
    plt.text(0.02, 0.02, table_text, fontsize=10, 
            transform=plt.gca().transAxes, verticalalignment='bottom',
            bbox=dict(facecolor='lightyellow', alpha=0.8))
    
    # Add grid
    for i in range(height+1):
        plt.axhline(i-0.5, color='gray', lw=0.5, alpha=0.3)
    for j in range(width+1):
        plt.axvline(j-0.5, color='gray', lw=0.5, alpha=0.3)
    
    # Set axis properties and add legend
    plt.title('Crossing Points with Edge Status', fontsize=18)
    plt.xlabel('Column (j)', fontsize=14)
    plt.ylabel('Row (i)', fontsize=14)
    plt.xlim(-0.5, width-0.5)
    plt.ylim(height-0.5, -0.5)  # Invert y-axis
    
    # Add legend
    legend_elements = [
        plt.Line2D([0], [0], marker='D', color='w', markerfacecolor='limegreen', 
                  markersize=15, label='Crossing WITH Edge'),
        plt.Line2D([0], [0], marker='X', color='w', markerfacecolor='red', 
                  markersize=15, label='Crossing WITHOUT Edge')
    ]
    plt.legend(handles=legend_elements, loc='upper right', fontsize=12)
    
    plt.savefig('udm_viz_crossings.png', dpi=300)
    print("Created visualization: udm_viz_crossings.png")
    plt.close()
    
    return crossing_info

def visualize_gadget_application(ug, g, crossing_info):
    """Visualize the grid after gadget application with detailed pattern information."""
    # Apply crossing gadgets
    grid_with_gadgets, tape = apply_crossing_gadgets(UnWeighted(), ug, original_graph=g)
    
    plt.figure(figsize=(16, 14))
    
    # Grid size
    grid = grid_with_gadgets.content
    height, width = len(grid), len(grid[0])
    
    # Draw the grid cells with different markers for cell types
    cell_coords = []
    for i in range(height):
        for j in range(width):
            cell = grid[i][j]
            if not cell.is_empty:
                if hasattr(cell, 'connected') and cell.connected:
                    color = 'green'
                    marker = 's'  # square for connected cells
                    label = 'Connected'
                elif hasattr(cell, 'doubled') and cell.doubled:
                    color = 'purple'
                    marker = '^'  # triangle for doubled cells
                    label = 'Doubled'
                else:
                    color = 'blue'
                    marker = 'o'  # circle for normal cells
                    label = 'Normal'
                plt.scatter(j, i, color=color, marker=marker, s=60, alpha=0.7)
                cell_coords.append((i, j, color, marker, label))
    
    # Create mapping for crossing points to tape entries for gadget application
    gadget_info = []
    for entry in tape:
        if len(entry) >= 6:
            pattern, x, y, v, w, has_edge = entry[:6]
            pattern_type = pattern.__class__.__name__
            pattern_height, pattern_width = pattern.size()
            
            # Find matching crossing point
            matching_crossing = None
            for cp in crossing_info:
                if cp[0] == v and cp[1] == w:
                    matching_crossing = cp
                    break
            
            if matching_crossing:
                # Draw a rectangle around the gadget area
                rect = plt.Rectangle((y-0.5, x-0.5), pattern_width, pattern_height, 
                                    fill=False, edgecolor='red', linewidth=2, zorder=20)
                plt.gca().add_patch(rect)
                
                # Add connecting line from gadget to crossing point
                _, _, _, cx, cy = matching_crossing
                mid_x = (x + pattern_height/2)
                mid_y = (y + pattern_width/2)
                plt.plot([cy, mid_y], [cx, mid_x], 'r--', linewidth=1.5, alpha=0.6)
                
                # Add detailed label
                edge_label = "WITH Edge" if has_edge else "WITHOUT Edge"
                plt.text(y+pattern_width/2, x-1.5, 
                        f"{pattern_type} Gadget\nVertices v{v}-v{w}\n{edge_label}", 
                        fontsize=9, ha='center', bbox=dict(facecolor='white', alpha=0.8))
                
                # Store gadget information
                gadget_info.append((v, w, pattern_type, has_edge, x, y, pattern_height, pattern_width))
    
    # Add grid
    for i in range(height+1):
        plt.axhline(i-0.5, color='gray', lw=0.5, alpha=0.3)
    for j in range(width+1):
        plt.axvline(j-0.5, color='gray', lw=0.5, alpha=0.3)
    
    # Set axis properties
    plt.title('Grid After Gadget Application', fontsize=18)
    plt.xlabel('Column (j)', fontsize=14)
    plt.ylabel('Row (i)', fontsize=14)
    plt.xlim(-0.5, width-0.5)
    plt.ylim(height-0.5, -0.5)  # Invert y-axis
    
    # Add legend for cell types
    legend_elements = [
        plt.Line2D([0], [0], marker='o', color='w', markerfacecolor='blue', markersize=10, label='Normal Cell'),
        plt.Line2D([0], [0], marker='s', color='w', markerfacecolor='green', markersize=10, label='Connected Cell'),
        plt.Line2D([0], [0], marker='^', color='w', markerfacecolor='purple', markersize=10, label='Doubled Cell'),
        plt.Rectangle((0, 0), 1, 1, fill=False, edgecolor='red', label='Applied Gadget')
    ]
    plt.legend(handles=legend_elements, loc='upper right', fontsize=12)
    
    # Add gadget information table
    if gadget_info:
        table_text = "Applied Gadgets:\n"
        for idx, (v, w, pattern_type, has_edge, x, y, h, w) in enumerate(gadget_info):
            edge_status = "WITH Edge" if has_edge else "WITHOUT Edge"
            table_text += f"{idx+1}. v{v}-v{w}: {pattern_type} {edge_status}, at ({x},{y})\n"
        
        plt.text(0.02, 0.02, table_text, fontsize=10, 
                transform=plt.gca().transAxes, verticalalignment='bottom',
                bbox=dict(facecolor='lightyellow', alpha=0.8))
    
    plt.savefig('udm_viz_gadgets.png', dpi=300)
    print("Created visualization: udm_viz_gadgets.png")
    plt.close()
    
    return grid_with_gadgets, gadget_info

def visualize_final_unit_disk_graph(grid_with_gadgets):
    """Visualize the final unit disk graph with node and edge details."""
    plt.figure(figsize=(16, 14))
    
    # Convert to a grid graph
    grid_graph = grid_with_gadgets.to_grid_graph()
    
    # Get grid dimensions
    grid = grid_with_gadgets.content
    height, width = len(grid), len(grid[0])
    
    # Add faint grid
    for i in range(height+1):
        plt.axhline(i-0.5, color='gray', lw=0.5, alpha=0.1)
    for j in range(width+1):
        plt.axvline(j-0.5, color='gray', lw=0.5, alpha=0.1)
    
    # Draw the edges first (unit disk connections)
    edge_count = 0
    for i, node1 in enumerate(grid_graph.nodes):
        for j, node2 in enumerate(grid_graph.nodes[i+1:], i+1):
            x1, y1 = node1.loc
            x2, y2 = node2.loc
            # Check if within unit disk distance
            dist = ((x1-x2)**2 + (y1-y2)**2)**0.5
            if dist <= grid_graph.radius:
                edge_count += 1
                plt.plot([y1, y2], [x1, x2], 'k-', alpha=0.15, linewidth=1)
    
    # Draw the nodes with consistent sizes
    for i, node in enumerate(grid_graph.nodes):
        x, y = node.loc
        plt.scatter(y, x, color='blue', s=100, alpha=0.7, edgecolor='black')
        
        # Add small node index labels to a subset of nodes
        if i % 10 == 0:  # Only label every 10th node to avoid clutter
            plt.text(y, x-0.4, f"{i}", fontsize=8, ha='center')
    
    # Set axis properties
    plt.title(f'Final Unit Disk Graph ({len(grid_graph.nodes)} nodes, ~{edge_count} edges)', fontsize=18)
    plt.xlabel('Column (j)', fontsize=14)
    plt.ylabel('Row (i)', fontsize=14)
    plt.xlim(-0.5, width-0.5)
    plt.ylim(height-0.5, -0.5)  # Invert y-axis
    
    # Add grid graph information
    info_text = f"Unit Disk Graph Properties:\n" \
                f"- Number of nodes: {len(grid_graph.nodes)}\n" \
                f"- Number of edges: ~{edge_count}\n" \
                f"- Unit disk radius: {grid_graph.radius}\n" \
                f"- Grid dimensions: {height}×{width}"
    
    plt.text(0.02, 0.02, info_text, fontsize=12, 
            transform=plt.gca().transAxes, verticalalignment='bottom',
            bbox=dict(facecolor='lightyellow', alpha=0.8))
    
    plt.savefig('udm_viz_final_udg.png', dpi=300)
    print("Created visualization: udm_viz_final_udg.png")
    plt.close()

def create_combined_visualization():
    """Create a combined visualization showing all stages."""
    plt.figure(figsize=(24, 20))
    
    # Load the individual images
    img_original = plt.imread('udm_viz_original.png')
    img_copylines = plt.imread('udm_viz_copylines.png')
    img_crossings = plt.imread('udm_viz_crossings.png')
    img_gadgets = plt.imread('udm_viz_gadgets.png')
    img_final = plt.imread('udm_viz_final_udg.png')
    
    # Create a layout of subplots
    plt.subplot(3, 2, 1)
    plt.imshow(img_original)
    plt.title('1. Original Graph', fontsize=16)
    plt.axis('off')
    
    plt.subplot(3, 2, 2)
    plt.imshow(img_copylines)
    plt.title('2. T-shaped Copylines Structure', fontsize=16)
    plt.axis('off')
    
    plt.subplot(3, 2, 3)
    plt.imshow(img_crossings)
    plt.title('3. Crossing Points with Edge Status', fontsize=16)
    plt.axis('off')
    
    plt.subplot(3, 2, 4)
    plt.imshow(img_gadgets)
    plt.title('4. Gadget Application', fontsize=16)
    plt.axis('off')
    
    plt.subplot(3, 2, 5)
    plt.imshow(img_final)
    plt.title('5. Final Unit Disk Graph', fontsize=16)
    plt.axis('off')
    
    # Add a detailed description in the last panel
    plt.subplot(3, 2, 6)
    plt.text(0.5, 0.5, 
             "Unit Disk Mapping Process:\n\n"
             "1. Original Graph: A 'house' graph with 5 vertices and 6 edges (pentagon with diagonal)\n\n"
             "2. Copylines: Each vertex is represented by a T-shaped copyline where the center\n"
             "   point corresponds to the vertex position in the vertex ordering\n\n"
             "3. Crossing Points: Identify where copylines cross and check if there's an edge\n"
             "   between the corresponding vertices in the original graph\n\n"
             "4. Gadget Application: Apply appropriate gadgets at crossing points\n"
             "   - WITH-EDGE gadget: When an edge exists between vertices\n"
             "   - WITHOUT-EDGE gadget: When no edge exists between vertices\n\n"
             "5. Final Unit Disk Graph: The resulting graph preserves the structure of the\n"
             "   original problem but has the unit disk property: vertices are connected\n"
             "   if and only if they are within a unit distance of each other\n\n"
             "This mapping preserves the Maximum Independent Set (MIS) problem, allowing\n"
             "optimization problems to be solved on unit disk graphs.",
             ha='center', va='center', fontsize=14,
             bbox=dict(facecolor='lightyellow', alpha=0.5))
    plt.axis('off')
    
    plt.tight_layout()
    plt.savefig('udm_viz_combined.png', dpi=300)
    print("Created combined visualization: udm_viz_combined.png")
    plt.close()

def main():
    """Main function to run the visualization."""
    print("Creating 5-vertex 'house' graph visualization...")
    
    # Create the house graph (pentagon with one diagonal)
    g = nx.Graph()
    # Add cycle edges (pentagon)
    g.add_edges_from([(0, 1), (1, 2), (2, 3), (3, 4), (4, 0)])
    # Add a diagonal edge
    g.add_edge(0, 2)
    
    # Plot the original graph
    plot_original_graph(g)
    
    # Create mapping grid with specified vertex order
    ug = embed_graph(g, vertex_order=[0, 1, 2, 3, 4])
    
    # Visualize the copylines
    visualize_copylines(ug)
    
    # Visualize crossing points
    crossing_info = visualize_crossing_points(ug, g)
    
    # Visualize gadget application
    grid_with_gadgets, gadget_info = visualize_gadget_application(ug, g, crossing_info)
    
    # Visualize final unit disk graph
    visualize_final_unit_disk_graph(grid_with_gadgets)
    
    # Create combined visualization
    create_combined_visualization()
    
    print("\nAll visualizations complete!")
    print("Check the udm_viz_*.png files for each step of the Unit Disk Mapping process.")
    print("The combined view is in udm_viz_combined.png")

if __name__ == "__main__":
    main()