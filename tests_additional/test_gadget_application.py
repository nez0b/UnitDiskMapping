#!/usr/bin/env python
"""
Script to test gadget application in the grid.
"""

import networkx as nx
import matplotlib.pyplot as plt
import numpy as np
from src.unit_disk_mapping import map_graph
from src.mapping import embed_graph, crossat, apply_crossing_gadgets, UnWeighted

def main():
    """Test and visualize gadget application."""
    # Create a simple graph with 5 vertices
    g = nx.Graph()
    g.add_edges_from([
        (0, 1), (0, 3), (0, 4), (2, 1), (2, 3), (2, 4)
    ])
    
    # Create mapping grid directly using embed_graph
    ug = embed_graph(g, vertex_order=[0, 1, 2, 3, 4])
    
    # Check the original graph structure vs the edges detected in the grid
    vertices = [line.vertex for line in ug.lines]
    
    # Get all vertex pairs and whether they have an edge
    print('Edges from original graph:')
    for i, v in enumerate(vertices):
        for j, w in enumerate(vertices[i+1:], i+1):
            has_edge = g.has_edge(v, w)
            print(f'Vertices {v} and {w} - has edge: {has_edge}')
    
    # Check edge information from the grid
    print('\nChecking connected cells in the grid:')
    connected_cells = []
    for i in range(len(ug.content)):
        for j in range(len(ug.content[0])):
            cell = ug.content[i][j]
            if hasattr(cell, 'connected') and cell.connected:
                connected_cells.append((i, j))
                print(f'Connected cell at ({i}, {j})')
    
    # Check if connected cells correspond to crossing points
    print('\nComparing connected cells to crossing points:')
    for i, v in enumerate(vertices):
        for j, w in enumerate(vertices[i+1:], i+1):
            has_edge = g.has_edge(v, w)
            cx, cy = crossat(ug, v, w)
            
            # Check if any connected cell is near the crossing point
            found_conn = False
            for conn_i, conn_j in connected_cells:
                if abs(conn_i - cx) <= 1 and abs(conn_j - cy) <= 1:
                    found_conn = True
                    break
            
            print(f'Vertices {v} and {w} - cross at ({cx}, {cy}), has edge: {has_edge}, connected cell found: {found_conn}')
    
    # Apply crossing gadgets with the original graph
    grid_with_gadgets, tape = apply_crossing_gadgets(UnWeighted(), ug, original_graph=g)
    
    # Print the applied gadgets
    print('\nApplied gadgets:')
    for i, entry in enumerate(tape):
        if len(entry) >= 6:
            pattern, x, y, v, w, has_edge = entry[:6]
            pattern_type = pattern.__class__.__name__
            pattern_has_edge = pattern.is_connected()
            correct = pattern_has_edge == has_edge
            print(f'Gadget {i+1}: {pattern_type} (has_edge={pattern_has_edge}) at ({x}, {y}) for vertices {v},{w} (has_edge={has_edge}, correct={correct})')
        else:
            pattern, x, y = entry[:3]
            pattern_type = pattern.__class__.__name__
            pattern_has_edge = pattern.is_connected()
            print(f'Gadget {i+1}: {pattern_type} (has_edge={pattern_has_edge}) at ({x}, {y})')
    
    # Visualize the grid with gadgets applied
    plt.figure(figsize=(12, 10))
    
    # Draw the grid cells
    grid = grid_with_gadgets.content
    height, width = len(grid), len(grid[0])
    
    for i in range(height):
        for j in range(width):
            cell = grid[i][j]
            if not cell.is_empty:
                if hasattr(cell, 'connected') and cell.connected:
                    color = 'green'
                elif hasattr(cell, 'doubled') and cell.doubled:
                    color = 'purple'
                else:
                    color = 'blue'
                plt.scatter(j, i, color=color, s=50)
    
    # Check crossings in the original graph
    print('\nCrossings in original graph:')
    for v in range(5):
        for w in range(v+1, 5):
            cross_point = crossat(ug, v, w)
            has_edge = g.has_edge(v, w)
            print(f'Vertices {v} and {w} cross at {cross_point}, has edge: {has_edge}')
            
            # Mark the crossing on the visualization
            color = 'green' if has_edge else 'red'
            plt.scatter(cross_point[1], cross_point[0], color=color, marker='x', s=100, alpha=0.5)
            plt.text(cross_point[1], cross_point[0]-0.3, f'({v},{w})', fontsize=8)
    
    # Add grid
    for i in range(height+1):
        plt.axhline(i-0.5, color='gray', lw=0.5, alpha=0.3)
    for j in range(width+1):
        plt.axvline(j-0.5, color='gray', lw=0.5, alpha=0.3)
    
    # Set axis properties
    plt.title('Grid with Gadgets Applied (blue=normal, green=connected, purple=doubled)')
    plt.xlabel('Column (j)')
    plt.ylabel('Row (i)')
    plt.xlim(-0.5, width-0.5)
    plt.ylim(height-0.5, -0.5)  # Invert y-axis
    
    plt.savefig('grid_with_gadgets.png', dpi=300)
    print('Created visualization: grid_with_gadgets.png')
    
    # Convert to a grid graph and visualize
    grid_graph = grid_with_gadgets.to_grid_graph()
    plt.figure(figsize=(12, 10))
    
    # Draw the nodes and edges
    nodes = grid_graph.nodes
    for node in nodes:
        i, j = node.loc
        plt.scatter(j, i, color='blue', s=50)
    
    # Draw the edges (unit disk connections)
    for i, node1 in enumerate(nodes):
        for j, node2 in enumerate(nodes[i+1:], i+1):
            i1, j1 = node1.loc
            i2, j2 = node2.loc
            # Check if within unit disk distance
            if ((i1-i2)**2 + (j1-j2)**2)**0.5 <= grid_graph.radius:
                plt.plot([j1, j2], [i1, i2], 'k-', alpha=0.2)
    
    # Add grid
    for i in range(height+1):
        plt.axhline(i-0.5, color='gray', lw=0.5, alpha=0.3)
    for j in range(width+1):
        plt.axvline(j-0.5, color='gray', lw=0.5, alpha=0.3)
    
    # Set axis properties
    plt.title('Final Unit Disk Graph (with gadgets applied)')
    plt.xlabel('Column (j)')
    plt.ylabel('Row (i)')
    plt.xlim(-0.5, width-0.5)
    plt.ylim(height-0.5, -0.5)  # Invert y-axis
    
    plt.savefig('final_unit_disk_graph.png', dpi=300)
    print('Created visualization: final_unit_disk_graph.png')
    
    # Count the number of expected gadgets (crossings with and without edges)
    edges_count = sum(1 for e in g.edges())
    total_crossings = sum(1 for v in range(5) for w in range(v+1, 5))
    non_edges_count = total_crossings - edges_count
    
    print(f'\nExpected gadgets:')
    print(f'Total crossings: {total_crossings}')
    print(f'Crossings with edges: {edges_count}')
    print(f'Crossings without edges: {non_edges_count}')
    print(f'Actual applied gadgets: {len(tape)}')

if __name__ == "__main__":
    main()