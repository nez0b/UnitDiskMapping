#!/usr/bin/env python
"""
Unit Disk Mapping Visualization Tool

This script generates comprehensive visualizations of the Unit Disk Mapping process,
taking any graph as input. It produces detailed visualizations of:
1. Original graph structure
2. Copylines creation and visualization
3. Crossing points with edge status
4. Gadget application at crossings
5. Final unit disk graph
6. Combined view of all steps

Usage:
  python visualize_udm.py [options]

Options:
  --graph-type TYPE    Type of graph to use: cycle, path, complete, house, 
                       random, petersen, grid, custom [default: house]
  --size N            Size parameter for the graph (vertices for path/cycle/complete,
                       grid size for grid) [default: 5]
  --custom-edges LIST  Comma-separated list of edges for custom graph (e.g., "0,1,1,2,2,0")
  --vertex-order LIST  Comma-separated list of vertices for ordering (default: ascending)
  --output-dir DIR     Directory to save visualizations [default: udm_viz]
  --prefix PREFIX      Prefix for output filenames [default: udm]
  --dpi DPI            Resolution of output images [default: 300]
  --padding N          Grid padding to avoid out-of-bounds issues [default: automatic]
  --help               Show this help message and exit
"""

import networkx as nx
import matplotlib.pyplot as plt
import numpy as np
import os
import sys
import argparse
from unit_disk_mapping import map_graph
from mapping import crossat, apply_crossing_gadgets, UnWeighted
from copyline import center_location, CopyLine


def parse_arguments():
    """Parse command-line arguments."""
    parser = argparse.ArgumentParser(description='Unit Disk Mapping Visualization Tool')
    
    parser.add_argument('--graph-type', type=str, default='house',
                        choices=['cycle', 'path', 'complete', 'house', 'random', 'petersen', 'grid', 'custom'],
                        help='Type of graph to use')
    
    parser.add_argument('--size', type=int, default=5,
                        help='Size parameter for the graph (vertices for path/cycle/complete, grid size for grid)')
    
    parser.add_argument('--custom-edges', type=str, default='',
                        help='Comma-separated list of edges for custom graph (e.g., "0,1,1,2,2,0")')
    
    parser.add_argument('--vertex-order', type=str, default='',
                        help='Comma-separated list of vertices for ordering (default: ascending)')
    
    parser.add_argument('--output-dir', type=str, default='udm_viz',
                        help='Directory to save visualizations')
    
    parser.add_argument('--prefix', type=str, default='udm',
                        help='Prefix for output filenames')
    
    parser.add_argument('--dpi', type=int, default=300,
                        help='Resolution of output images')
                        
    parser.add_argument('--padding', type=int, default=0,
                        help='Grid padding (0 for automatic)')
    
    return parser.parse_args()


def create_graph(graph_type, size, custom_edges=''):
    """
    Create a graph based on the specified type and size.
    
    Args:
        graph_type: Type of graph (cycle, path, complete, house, random, petersen, grid, custom)
        size: Size parameter for the graph
        custom_edges: Comma-separated list of edges for custom graph
        
    Returns:
        A NetworkX graph
    """
    g = nx.Graph()
    
    if graph_type == 'cycle':
        g = nx.cycle_graph(size)
    elif graph_type == 'path':
        g = nx.path_graph(size)
    elif graph_type == 'complete':
        g = nx.complete_graph(size)
    elif graph_type == 'house':
        # Create a 5-vertex house graph (pentagon with one diagonal)
        g = nx.Graph()
        g.add_edges_from([(0, 1), (1, 2), (2, 3), (3, 4), (4, 0)])
        g.add_edge(0, 2)  # Diagonal
    elif graph_type == 'random':
        # Create a random graph with probability 0.5 for edges
        g = nx.gnp_random_graph(size, 0.5, seed=42)
    elif graph_type == 'petersen':
        g = nx.petersen_graph()
    elif graph_type == 'grid':
        # Create a size x size grid graph
        g = nx.grid_2d_graph(size, size)
        # Relabel nodes to be integers
        mapping = {(i, j): i*size + j for i in range(size) for j in range(size)}
        g = nx.relabel_nodes(g, mapping)
    elif graph_type == 'custom' and custom_edges:
        # Parse custom edges
        edge_list = custom_edges.split(',')
        if len(edge_list) % 2 != 0:
            raise ValueError("Custom edge list must have an even number of values")
        
        edges = [(int(edge_list[i]), int(edge_list[i+1])) for i in range(0, len(edge_list), 2)]
        g.add_edges_from(edges)
    else:
        raise ValueError(f"Invalid graph type: {graph_type}")
    
    return g


def get_graph_name(graph_type, size, custom_edges=''):
    """Get a descriptive name for the graph."""
    if graph_type == 'cycle':
        return f"Cycle Graph C{size}"
    elif graph_type == 'path':
        return f"Path Graph P{size}"
    elif graph_type == 'complete':
        return f"Complete Graph K{size}"
    elif graph_type == 'house':
        return "House Graph (Pentagon with Diagonal)"
    elif graph_type == 'random':
        return f"Random Graph G({size}, 0.5)"
    elif graph_type == 'petersen':
        return "Petersen Graph"
    elif graph_type == 'grid':
        return f"{size}×{size} Grid Graph"
    elif graph_type == 'custom':
        return "Custom Graph"
    else:
        return "Graph"


def plot_original_graph(g, graph_name, output_path, dpi=300):
    """Plot the original graph structure with labeled vertices and edges."""
    plt.figure(figsize=(10, 8))
    
    # Choose an appropriate layout based on graph type
    if graph_name.startswith("Cycle") or graph_name.startswith("House"):
        pos = nx.circular_layout(g)
    elif graph_name.startswith("Path"):
        pos = nx.kamada_kawai_layout(g)
    elif graph_name.startswith("Grid"):
        # For grid graphs, use the grid positions
        n = int(np.sqrt(g.number_of_nodes()))
        pos = {i: (i % n, i // n) for i in g.nodes()}
    else:
        pos = nx.spring_layout(g, seed=42)  # Consistent layout
    
    # Draw nodes
    nx.draw_networkx_nodes(g, pos, node_size=700, node_color='skyblue')
    
    # Draw edges
    nx.draw_networkx_edges(g, pos, width=2)
    
    # Draw labels
    nx.draw_networkx_labels(g, pos, font_size=16, font_weight='bold')
    
    plt.title(f"Original {graph_name}", fontsize=18)
    plt.axis('off')
    plt.tight_layout()
    plt.savefig(output_path, dpi=dpi)
    print(f"Created visualization: {output_path}")
    plt.close()


def visualize_copylines(ug, output_path, dpi=300):
    """Visualize the T-shaped copylines structure with detailed labeling."""
    plt.figure(figsize=(14, 12))
    
    # Grid size
    height, width = len(ug.content), len(ug.content[0])
    
    # Create colormap for different copylines
    colors = ['blue', 'green', 'red', 'purple', 'orange', 'brown', 'pink', 'cyan', 'magenta', 'gold']
    
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
        plt.scatter(J, I, color=color, s=200, zorder=20)
        plt.text(J, I-0.7, f"v{line.vertex}", fontsize=14, ha='center', fontweight='bold',
                 bbox=dict(facecolor='white', alpha=0.7))
        
        # Add copyline information (limit the number for large graphs)
        if i < 10 or len(ug.lines) <= 20:
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
    
    # Add legend (limit for large graphs)
    legend_elements = []
    for i, color in enumerate(colors[:min(len(ug.lines), 10)]):
        legend_elements.append(
            plt.Line2D([0], [0], marker='o', color=color, lw=3, 
                      markersize=10, label=f'Copyline v{ug.lines[i].vertex}')
        )
    plt.legend(handles=legend_elements, loc='upper left', fontsize=12)
    
    plt.savefig(output_path, dpi=dpi)
    print(f"Created visualization: {output_path}")
    plt.close()


def visualize_crossing_points(ug, g, output_path, dpi=300):
    """Visualize the crossing points with detailed edge status information."""
    plt.figure(figsize=(14, 12))
    
    # Grid size
    height, width = len(ug.content), len(ug.content[0])
    
    # Create colormap for different copylines
    colors = ['blue', 'green', 'red', 'purple', 'orange', 'brown', 'pink', 'cyan', 'magenta', 'gold']
    
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
        plt.scatter(J, I, color=color, s=150, alpha=0.5)
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
            plt.scatter(cy, cx, color=color, marker=marker, s=size, linewidth=1.5, zorder=30)
            
            # Add label with detailed information (limit for large graphs)
            if len(crossing_info) <= 30:
                edge_status = "EDGE" if has_edge else "NO EDGE"
                plt.text(cy+0.5, cx, f"v{v}-v{w}\n{edge_status}", fontsize=10, ha='left', va='center',
                        bbox=dict(facecolor='white', alpha=0.8))
    
    # Add table of crossing points (limit for large graphs)
    if len(crossing_info) <= 30:
        table_text = "Crossing Points:\n"
        for idx, (v, w, has_edge, cx, cy) in enumerate(crossing_info):
            status = "Edge" if has_edge else "No Edge"
            table_text += f"{idx+1}. v{v}-v{w}: ({cx},{cy}), {status}\n"
        
        plt.text(0.02, 0.02, table_text, fontsize=10, 
                transform=plt.gca().transAxes, verticalalignment='bottom',
                bbox=dict(facecolor='lightyellow', alpha=0.8))
    else:
        table_text = f"Crossings: {len(crossing_info)} points\n" + \
                     f"With Edge: {sum(1 for _, _, has_edge, _, _ in crossing_info if has_edge)}\n" + \
                     f"Without Edge: {sum(1 for _, _, has_edge, _, _ in crossing_info if not has_edge)}"
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
    
    plt.savefig(output_path, dpi=dpi)
    print(f"Created visualization: {output_path}")
    plt.close()
    
    return crossing_info


def visualize_gadget_application(ug, g, crossing_info, output_path, dpi=300):
    """Visualize the grid after gadget application with detailed pattern information."""
    # Apply crossing gadgets with error handling for out-of-bounds patterns
    try:
        grid_with_gadgets, tape = apply_crossing_gadgets(UnWeighted(), ug, original_graph=g)
    except Exception as e:
        print(f"\nWarning: Error during gadget application: {e}")
        print("This may be due to insufficient padding or other grid size issues.")
        print("Try increasing padding with --padding option.\n")
        # Return a copy of the original grid as fallback
        grid_with_gadgets = ug.copy()
        tape = []
    
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
    max_gadgets_to_show = 20  # Limit for clarity
    
    for k, entry in enumerate(tape):
        if len(entry) >= 6 and k < max_gadgets_to_show:
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
                
                # Add detailed label (only for a limited number)
                if k < 10:
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
    
    # Add gadget information table (limited for large graphs)
    if len(tape) <= 20:
        table_text = "Applied Gadgets:\n"
        for idx, (v, w, pattern_type, has_edge, x, y, h, w) in enumerate(gadget_info):
            edge_status = "WITH Edge" if has_edge else "WITHOUT Edge"
            table_text += f"{idx+1}. v{v}-v{w}: {pattern_type} {edge_status}, at ({x},{y})\n"
        
        plt.text(0.02, 0.02, table_text, fontsize=10, 
                transform=plt.gca().transAxes, verticalalignment='bottom',
                bbox=dict(facecolor='lightyellow', alpha=0.8))
    else:
        with_edge = sum(1 for _, _, _, has_edge, _, _, _, _ in gadget_info if has_edge)
        without_edge = len(gadget_info) - with_edge
        
        table_text = f"Applied Gadgets: {len(tape)} total\n" + \
                     f"With Edge: {with_edge}\n" + \
                     f"Without Edge: {without_edge}\n" + \
                     f"Showing {min(len(tape), max_gadgets_to_show)} of {len(tape)} gadgets"
                     
        plt.text(0.02, 0.02, table_text, fontsize=10, 
                transform=plt.gca().transAxes, verticalalignment='bottom',
                bbox=dict(facecolor='lightyellow', alpha=0.8))
    
    plt.savefig(output_path, dpi=dpi)
    print(f"Created visualization: {output_path}")
    plt.close()
    
    return grid_with_gadgets, gadget_info


def visualize_final_unit_disk_graph(grid_with_gadgets, output_path, dpi=300):
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
        plt.scatter(y, x, color='blue', s=100, alpha=0.7)
        
        # Add small node index labels to a subset of nodes (limit for large graphs)
        if i % max(1, len(grid_graph.nodes) // 20) == 0:
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
    
    plt.savefig(output_path, dpi=dpi)
    print(f"Created visualization: {output_path}")
    plt.close()


def create_combined_visualization(image_paths, graph_name, output_path, dpi=300):
    """Create a combined visualization showing all stages."""
    plt.figure(figsize=(24, 20))
    
    # Load the individual images
    img_original = plt.imread(image_paths['original'])
    img_copylines = plt.imread(image_paths['copylines'])
    img_crossings = plt.imread(image_paths['crossings'])
    img_gadgets = plt.imread(image_paths['gadgets'])
    img_final = plt.imread(image_paths['final'])
    
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
             f"Unit Disk Mapping Process for {graph_name}:\n\n"
             "1. Original Graph: The input graph structure\n\n"
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
    plt.savefig(output_path, dpi=dpi)
    print(f"Created combined visualization: {output_path}")
    plt.close()


def main():
    """Main function to run the visualization."""
    # Parse command-line arguments
    args = parse_arguments()
    
    # Create output directory if it doesn't exist
    if not os.path.exists(args.output_dir):
        os.makedirs(args.output_dir)
    
    # Create the graph
    try:
        g = create_graph(args.graph_type, args.size, args.custom_edges)
        graph_name = get_graph_name(args.graph_type, args.size, args.custom_edges)
    except Exception as e:
        print(f"Error creating graph: {e}")
        sys.exit(1)
    
    print(f"Creating {graph_name} visualization...")
    print(f"Graph has {g.number_of_nodes()} vertices and {g.number_of_edges()} edges")
    
    # Parse vertex order if provided
    vertex_order = None
    if args.vertex_order:
        try:
            vertex_order = [int(v) for v in args.vertex_order.split(',')]
            if not all(v in g.nodes() for v in vertex_order):
                print("Warning: Some vertices in the provided order don't exist in the graph.")
                vertex_order = None
        except Exception as e:
            print(f"Warning: Could not parse vertex order: {e}")
            vertex_order = None
    
    # Set up file paths
    image_paths = {
        'original': os.path.join(args.output_dir, f"{args.prefix}_original.png"),
        'copylines': os.path.join(args.output_dir, f"{args.prefix}_copylines.png"),
        'crossings': os.path.join(args.output_dir, f"{args.prefix}_crossings.png"),
        'gadgets': os.path.join(args.output_dir, f"{args.prefix}_gadgets.png"),
        'final': os.path.join(args.output_dir, f"{args.prefix}_final_udg.png"),
        'combined': os.path.join(args.output_dir, f"{args.prefix}_combined.png")
    }
    
    # Plot the original graph
    plot_original_graph(g, graph_name, image_paths['original'], args.dpi)
    
    # Create mapping grid with increased padding to avoid out-of-bounds issues
    # Default padding in embed_graph is 2, we'll increase it based on graph size
    if args.padding > 0:
        # Use user-specified padding
        padding = args.padding
    else:
        # Calculate automatic padding based on graph properties
        # More vertices and edges generally need more padding
        vertex_factor = g.number_of_nodes() // 2
        edge_factor = g.number_of_edges() // 3
        padding = max(4, vertex_factor + edge_factor)
    
    print(f"Using padding: {padding}")
    
    # Since embed_graph doesn't accept padding directly, we need to modify how we create the grid
    # First get the vertex order from embed_graph's logic
    if vertex_order is None:
        from mapping import MinhThiTrick, pathwidth, Layout
        vertex_order_method = MinhThiTrick()
        # Compute path decomposition
        layout = pathwidth(g, vertex_order_method)
        # Reverse the vertex order for embedding
        vertices = list(reversed(layout.vertices))
    else:
        # Create a layout from the vertex order
        from mapping import Layout
        layout = Layout.from_graph(g, list(reversed(vertex_order)))
        vertices = list(reversed(layout.vertices))
    
    # Now call ugrid directly with our padding
    from mapping import ugrid, UnWeighted
    ug = ugrid(UnWeighted(), g, vertices, padding=padding, nrow=None)
    
    # Visualize the copylines
    visualize_copylines(ug, image_paths['copylines'], args.dpi)
    
    # Visualize crossing points
    crossing_info = visualize_crossing_points(ug, g, image_paths['crossings'], args.dpi)
    
    # Visualize gadget application
    grid_with_gadgets, gadget_info = visualize_gadget_application(ug, g, crossing_info, image_paths['gadgets'], args.dpi)
    
    # Visualize final unit disk graph
    visualize_final_unit_disk_graph(grid_with_gadgets, image_paths['final'], args.dpi)
    
    # Create combined visualization
    create_combined_visualization(image_paths, graph_name, image_paths['combined'], args.dpi)
    
    print("\nAll visualizations complete!")
    print(f"Check the {args.output_dir}/{args.prefix}_*.png files for each step of the Unit Disk Mapping process.")
    print(f"The combined view is in {image_paths['combined']}")


if __name__ == "__main__":
    main()