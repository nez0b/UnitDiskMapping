from dataclasses import dataclass
from typing import List, Tuple, Dict, Any, Union, Optional, Callable
from core import Node, ONE_INSTANCE

@dataclass
class CopyLine:
    """
    Represents a T-shaped path in the grid for each vertex in the original graph.
    
    Attributes:
        vertex: The vertex ID from the original graph
        vslot: Vertical slot position (column position)
        hslot: Horizontal slot position (row position)
        vstart: Starting point of the vertical segment
        vstop: Ending point of the vertical segment
        hstop: Ending point of the horizontal segment (there is no hstart)
    """
    vertex: int
    vslot: int
    hslot: int
    vstart: int
    vstop: int
    hstop: int

def create_copylines(g, vertex_order: List[int]) -> List[CopyLine]:
    """
    Create copy lines for a graph with a given vertex order.
    
    Args:
        g: The input graph
        vertex_order: The order of vertices
        
    Returns:
        A list of CopyLine objects
    """
    n = len(vertex_order)
    copylines = []
    
    # For each vertex in the order
    for i, v in enumerate(vertex_order):
        # Create T-shaped paths where:
        # - hslot is the position in the vertex_order (row)
        # - vslot is the position in the vertex_order (column)
        # - vstart/vstop define the vertical span
        # - hstop defines the horizontal span
        copyline = CopyLine(
            vertex=v,
            vslot=i+1,  # 1-based indexing
            hslot=i+1,  # 1-based indexing
            vstart=1,   # Start from the top
            vstop=n,    # Go to the bottom
            hstop=n     # Go to the right
        )
        copylines.append(copyline)
    
    return copylines

def center_location(tc: CopyLine, padding: int) -> Tuple[int, int]:
    """
    Calculate the center location of a copy line on the grid.
    
    Args:
        tc: The copy line
        padding: Grid padding amount
        
    Returns:
        (I, J) coordinates of the center
    """
    s = 4  # spacing factor
    I = s * (tc.hslot - 1) + padding + 2
    J = s * (tc.vslot - 1) + padding + 1
    return I, J

def node_from_type(node_type, i, j, w):
    """Create a node of the specified type with position (i,j) and weight w."""
    # For UnWeightedNode (with ONE weight)
    if node_type == "UnWeightedNode":
        return Node(i, j)
    # For WeightedNode (with numeric weight)
    else:
        return Node(i, j, w)

def copyline_locations(node_type: str, tc: CopyLine, padding: int) -> List[Node]:
    """
    Get all locations for a copy line.
    
    Args:
        node_type: "WeightedNode" or "UnWeightedNode"
        tc: The copy line
        padding: Grid padding amount
        
    Returns:
        List of Node objects representing the copy line
    """
    s = 4  # spacing factor
    nline = 0
    I, J = center_location(tc, padding=padding)
    locations = []
    
    # Grow up
    start = I + s * (tc.vstart - tc.hslot) + 1
    if tc.vstart < tc.hslot:
        nline += 1
    
    for i in range(I, start-1, -1):  # Even number of nodes up
        weight = 1 + (1 if i != start else 0)  # Half weight on last node
        locations.append(node_from_type(node_type, i, J, weight))
    
    # Grow down
    stop = I + s * (tc.vstop - tc.hslot) - 1
    if tc.vstop > tc.hslot:
        nline += 1
    
    for i in range(I, stop+1):  # Even number of nodes down
        if i == I:
            locations.append(node_from_type(node_type, i+1, J+1, 2))
        else:
            weight = 1 + (1 if i != stop else 0)
            locations.append(node_from_type(node_type, i, J, weight))
    
    # Grow right
    stop = J + s * (tc.hstop - tc.vslot) - 1
    if tc.hstop > tc.vslot:
        nline += 1
    
    for j in range(J+2, stop+1):  # Even number of nodes right
        weight = 1 + (1 if j != stop else 0)  # Half weight on last node
        locations.append(node_from_type(node_type, I, j, weight))
    
    # Center node
    locations.append(node_from_type(node_type, I, J+1, nline))
    
    return locations