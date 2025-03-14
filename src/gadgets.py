import networkx as nx
import numpy as np
from typing import List, Tuple, Dict, Any, Union, Optional, Callable
from .core import SimpleCell, Node, GridGraph, ONE_INSTANCE
from .utils import unit_disk_graph


def simplegraph(edges):
    """Create a simple graph from a list of edges."""
    g = nx.Graph()
    for i, (src, dst) in enumerate(edges):
        g.add_edge(src - 1, dst - 1)  # Convert from 1-indexed to 0-indexed
    return g


def safe_get(matrix, i, j):
    """Get a value from a matrix with boundary checking."""
    m, n = len(matrix), len(matrix[0])
    if i < 0 or i >= m or j < 0 or j >= n:
        return SimpleCell.create_empty()
    return matrix[i][j]


def safe_set(matrix, i, j, val):
    """Set a value in a matrix with boundary checking."""
    m, n = len(matrix), len(matrix[0])
    if i < 0 or i >= m or j < 0 or j >= n:
        assert val.is_empty
    else:
        matrix[i][j] = val
    return val


def locs2matrix(m, n, locs):
    """Convert a list of node locations to a matrix."""
    matrix = [[SimpleCell.create_empty() for _ in range(n)] for _ in range(m)]
    for loc in locs:
        i, j = loc.loc
        matrix[i][j] = SimpleCell(occupied=True, weight=loc.weight)
    return matrix


def add_cell(matrix, loc):
    """Add a cell to a matrix at the specified location."""
    i, j = loc.loc
    matrix[i][j] = SimpleCell(occupied=True, weight=loc.weight)


def connect_cell(matrix, i, j):
    """Mark a cell as connected."""
    # In the Python implementation, we're just setting the cell as occupied
    # since we don't have a specific visual marker for connection
    matrix[i][j] = SimpleCell(occupied=True)


# Transformation utilities
def rotate90(loc, center):
    """Rotate a location 90 degrees around a center point."""
    x, y = loc
    cx, cy = center
    return (cx + (cy - y), cy + (x - cx))


def reflectx(loc, center):
    """Reflect a location across the x-axis through the center point."""
    x, y = loc
    cx, cy = center
    return (x, cy - (y - cy))


def reflecty(loc, center):
    """Reflect a location across the y-axis through the center point."""
    x, y = loc
    cx, cy = center
    return (cx - (x - cx), y)


def reflectdiag(loc, center):
    """Reflect a location across the main diagonal through the center point."""
    x, y = loc
    cx, cy = center
    dx, dy = x - cx, y - cy
    return (cx + dy, cy + dx)


def reflectoffdiag(loc, center):
    """Reflect a location across the off-diagonal through the center point."""
    x, y = loc
    cx, cy = center
    dx, dy = x - cx, y - cy
    return (cx - dy, cy - dx)


class Pattern:
    """Base class for all pattern types."""
    
    def __init__(self):
        pass
    
    def size(self):
        """Return the size of the pattern."""
        raise NotImplementedError
    
    def cross_location(self):
        """Return the location of the cross."""
        raise NotImplementedError
    
    def is_connected(self):
        """Return whether the pattern has connected cells."""
        return False
    
    def connected_nodes(self):
        """Return the list of connected nodes."""
        return []
    
    def source_graph(self):
        """Return the source graph (locs, graph, pins)."""
        raise NotImplementedError
    
    def mapped_graph(self):
        """Return the mapped graph (locs, graph, pins)."""
        raise NotImplementedError
    
    def source_matrix(self):
        """Convert the source graph to a matrix."""
        m, n = self.size()
        locs, _, _ = self.source_graph()
        a = locs2matrix(m, n, locs)
        if self.is_connected():
            for i in self.connected_nodes():
                connect_cell(a, locs[i].loc[0], locs[i].loc[1])
        return a
    
    def mapped_matrix(self):
        """Convert the mapped graph to a matrix."""
        m, n = self.size()
        locs, _, _ = self.mapped_graph()
        return locs2matrix(m, n, locs)
    
    def match(self, matrix, i, j):
        """Check if the pattern matches at the given location."""
        a = self.source_matrix()
        m, n = len(a), len(a[0])
        for x in range(m):
            for y in range(n):
                a_cell = a[x][y]
                m_cell = safe_get(matrix, i+x, j+y)
                # Only check if the cell is occupied, not the value
                if a_cell.occupied != m_cell.occupied:
                    return False
        return True
    
    def unmatch(self, matrix, i, j):
        """Check if the unmapped pattern matches at the given location."""
        a = self.mapped_matrix()
        m, n = len(a), len(a[0])
        for x in range(m):
            for y in range(n):
                a_cell = a[x][y]
                m_cell = safe_get(matrix, i+x, j+y)
                # Only check if the cell is occupied, not the value
                if a_cell.occupied != m_cell.occupied:
                    return False
        return True
    
    def apply_gadget(self, matrix, i, j):
        """Apply the gadget at the given location."""
        a = self.mapped_matrix()
        m, n = len(a), len(a[0])
        for x in range(m):
            for y in range(n):
                safe_set(matrix, i+x, j+y, a[x][y])
        return matrix
    
    def unapply_gadget(self, matrix, i, j):
        """Unapply the gadget at the given location."""
        a = self.source_matrix()
        m, n = len(a), len(a[0])
        for x in range(m):
            for y in range(n):
                safe_set(matrix, i+x, j+y, a[x][y])
        return matrix
    
    def vertex_overhead(self):
        """Return the number of extra vertices in the mapped graph."""
        src_locs, _, _ = self.source_graph()
        map_locs, _, _ = self.mapped_graph()
        return len(map_locs) - len(src_locs)


class Cross(Pattern):
    """Pattern for crossings."""
    
    def __init__(self, has_edge=False):
        super().__init__()
        self._has_edge = has_edge
    
    def size(self):
        """Return the size of the pattern."""
        return (3, 3) if self._has_edge else (4, 5)
    
    def cross_location(self):
        """Return the location of the cross."""
        return (1, 1) if self._has_edge else (1, 2)
    
    def is_connected(self):
        """Return whether the pattern has connected cells."""
        return self._has_edge
    
    def connected_nodes(self):
        """Return the list of connected nodes."""
        return [0, 5] if self._has_edge else []
    
    def source_graph(self):
        """Return the source graph (locs, graph, pins)."""
        if self._has_edge:
            # ⋅ ● ⋅
            # ◆ ◉ ●
            # ⋅ ◆ ⋅
            locs = [Node((1, 0)), Node((1, 1)), Node((1, 2)), Node((0, 1)), Node((1, 1)), Node((2, 1))]
            g = simplegraph([(1, 2), (2, 3), (4, 5), (5, 6), (1, 6)])
            pins = [0, 3, 5, 2]  # convert from 1-indexed to 0-indexed
            return locs, g, pins
        else:
            # ⋅ ⋅ ● ⋅ ⋅
            # ● ● ◉ ● ●
            # ⋅ ⋅ ● ⋅ ⋅
            # ⋅ ⋅ ● ⋅ ⋅
            locs = [Node((1, 0)), Node((1, 1)), Node((1, 2)), Node((1, 3)), Node((1, 4)), 
                   Node((0, 2)), Node((1, 2)), Node((2, 2)), Node((3, 2))]
            g = simplegraph([(1, 2), (2, 3), (3, 4), (4, 5), (6, 7), (7, 8), (8, 9)])
            pins = [0, 5, 8, 4]  # convert from 1-indexed to 0-indexed
            return locs, g, pins
    
    def mapped_graph(self):
        """Return the mapped graph (locs, graph, pins)."""
        if self._has_edge:
            # ⋅ ● ⋅
            # ● ● ●
            # ⋅ ● ⋅
            locs = [Node((1, 0)), Node((1, 1)), Node((1, 2)), Node((0, 1)), Node((2, 1))]
            pins = [0, 3, 4, 2]  # convert from 1-indexed to 0-indexed
            return locs, unit_disk_graph([loc.loc for loc in locs], 1.5), pins
        else:
            # ⋅ ⋅ ● ⋅ ⋅
            # ● ● ● ● ●
            # ⋅ ● ● ● ⋅
            # ⋅ ⋅ ● ⋅ ⋅
            locs = [Node((1, 0)), Node((1, 1)), Node((1, 2)), Node((1, 3)), Node((1, 4)), 
                   Node((0, 2)), Node((2, 2)), Node((3, 2)), Node((2, 1)), Node((2, 3))]
            pins = [0, 5, 7, 4]  # convert from 1-indexed to 0-indexed
            return locs, unit_disk_graph([loc.loc for loc in locs], 1.5), pins


class Turn(Pattern):
    """Pattern for turns."""
    
    def __init__(self):
        super().__init__()
    
    def size(self):
        """Return the size of the pattern."""
        return (4, 4)
    
    def cross_location(self):
        """Return the location of the cross."""
        return (2, 1)
    
    def source_graph(self):
        """Return the source graph (locs, graph, pins)."""
        # ⋅ ● ⋅ ⋅
        # ⋅ ● ⋅ ⋅
        # ⋅ ● ● ●
        # ⋅ ⋅ ⋅ ⋅
        locs = [Node((0, 1)), Node((1, 1)), Node((2, 1)), Node((2, 2)), Node((2, 3))]
        g = simplegraph([(1, 2), (2, 3), (3, 4), (4, 5)])
        pins = [0, 4]  # convert from 1-indexed to 0-indexed
        return locs, g, pins
    
    def mapped_graph(self):
        """Return the mapped graph (locs, graph, pins)."""
        # ⋅ ● ⋅ ⋅
        # ⋅ ⋅ ● ⋅
        # ⋅ ⋅ ⋅ ●
        # ⋅ ⋅ ⋅ ⋅
        locs = [Node((0, 1)), Node((1, 2)), Node((2, 3))]
        pins = [0, 2]  # convert from 1-indexed to 0-indexed
        return locs, unit_disk_graph([loc.loc for loc in locs], 1.5), pins


class Branch(Pattern):
    """Pattern for branches."""
    
    def __init__(self):
        super().__init__()
    
    def size(self):
        """Return the size of the pattern."""
        return (5, 4)
    
    def cross_location(self):
        """Return the location of the cross."""
        return (2, 1)
    
    def source_graph(self):
        """Return the source graph (locs, graph, pins)."""
        # ⋅ ● ⋅ ⋅
        # ⋅ ● ⋅ ⋅
        # ⋅ ● ● ●
        # ⋅ ● ● ⋅
        # ⋅ ● ⋅ ⋅
        locs = [Node((0, 1)), Node((1, 1)), Node((2, 1)), Node((2, 2)), Node((2, 3)), 
                Node((3, 2)), Node((3, 1)), Node((4, 1))]
        g = simplegraph([(1, 2), (2, 3), (3, 4), (4, 5), (4, 6), (6, 7), (7, 8)])
        pins = [0, 4, 7]  # convert from 1-indexed to 0-indexed
        return locs, g, pins
    
    def mapped_graph(self):
        """Return the mapped graph (locs, graph, pins)."""
        # ⋅ ● ⋅ ⋅
        # ⋅ ⋅ ● ⋅
        # ⋅ ● ⋅ ●
        # ⋅ ⋅ ● ⋅
        # ⋅ ● ⋅ ⋅
        locs = [Node((0, 1)), Node((1, 2)), Node((2, 1)), Node((2, 3)), Node((3, 2)), Node((4, 1))]
        pins = [0, 3, 5]  # convert from 1-indexed to 0-indexed
        return locs, unit_disk_graph([loc.loc for loc in locs], 1.5), pins


class BranchFix(Pattern):
    """Pattern for branch fixes."""
    
    def __init__(self):
        super().__init__()
    
    def size(self):
        """Return the size of the pattern."""
        return (4, 4)
    
    def cross_location(self):
        """Return the location of the cross."""
        return (1, 1)
    
    def source_graph(self):
        """Return the source graph (locs, graph, pins)."""
        # ⋅ ● ⋅ ⋅
        # ⋅ ● ● ⋅
        # ⋅ ● ● ⋅
        # ⋅ ● ⋅ ⋅
        locs = [Node((0, 1)), Node((1, 1)), Node((1, 2)), Node((2, 2)), Node((2, 1)), Node((3, 1))]
        g = simplegraph([(1, 2), (2, 3), (3, 4), (4, 5), (5, 6)])
        pins = [0, 5]  # convert from 1-indexed to 0-indexed
        return locs, g, pins
    
    def mapped_graph(self):
        """Return the mapped graph (locs, graph, pins)."""
        # ⋅ ● ⋅ ⋅
        # ⋅ ● ⋅ ⋅
        # ⋅ ● ⋅ ⋅
        # ⋅ ● ⋅ ⋅
        locs = [Node((0, 1)), Node((1, 1)), Node((2, 1)), Node((3, 1))]
        pins = [0, 3]  # convert from 1-indexed to 0-indexed
        return locs, unit_disk_graph([loc.loc for loc in locs], 1.5), pins


class WTurn(Pattern):
    """Pattern for W-turns."""
    
    def __init__(self):
        super().__init__()
    
    def size(self):
        """Return the size of the pattern."""
        return (4, 4)
    
    def cross_location(self):
        """Return the location of the cross."""
        return (1, 1)
    
    def source_graph(self):
        """Return the source graph (locs, graph, pins)."""
        # ⋅ ⋅ ⋅ ⋅
        # ⋅ ⋅ ● ●
        # ⋅ ● ● ⋅
        # ⋅ ● ⋅ ⋅
        locs = [Node((1, 2)), Node((1, 3)), Node((2, 1)), Node((2, 2)), Node((3, 1))]
        g = simplegraph([(1, 2), (1, 4), (3, 4), (3, 5)])
        pins = [1, 4]  # convert from 1-indexed to 0-indexed
        return locs, g, pins
    
    def mapped_graph(self):
        """Return the mapped graph (locs, graph, pins)."""
        # ⋅ ⋅ ⋅ ⋅
        # ⋅ ⋅ ⋅ ●
        # ⋅ ⋅ ● ⋅
        # ⋅ ● ⋅ ⋅
        locs = [Node((1, 3)), Node((2, 2)), Node((3, 1))]
        pins = [0, 2]  # convert from 1-indexed to 0-indexed
        return locs, unit_disk_graph([loc.loc for loc in locs], 1.5), pins


class BranchFixB(Pattern):
    """Pattern for branch fix B-type."""
    
    def __init__(self):
        super().__init__()
    
    def size(self):
        """Return the size of the pattern."""
        return (4, 4)
    
    def cross_location(self):
        """Return the location of the cross."""
        return (1, 1)
    
    def source_graph(self):
        """Return the source graph (locs, graph, pins)."""
        # ⋅ ⋅ ⋅ ⋅
        # ⋅ ⋅ ● ⋅
        # ⋅ ● ● ⋅
        # ⋅ ● ⋅ ⋅
        locs = [Node((1, 2)), Node((2, 1)), Node((2, 2)), Node((3, 1))]
        g = simplegraph([(1, 3), (2, 3), (2, 4)])
        pins = [0, 3]  # convert from 1-indexed to 0-indexed
        return locs, g, pins
    
    def mapped_graph(self):
        """Return the mapped graph (locs, graph, pins)."""
        # ⋅ ⋅ ⋅ ⋅
        # ⋅ ⋅ ⋅ ⋅
        # ⋅ ● ⋅ ⋅
        # ⋅ ● ⋅ ⋅
        locs = [Node((2, 1)), Node((3, 1))]
        pins = [0, 1]  # convert from 1-indexed to 0-indexed
        return locs, unit_disk_graph([loc.loc for loc in locs], 1.5), pins


class TCon(Pattern):
    """Pattern for T-connections."""
    
    def __init__(self):
        super().__init__()
    
    def size(self):
        """Return the size of the pattern."""
        return (3, 4)
    
    def cross_location(self):
        """Return the location of the cross."""
        return (1, 1)
    
    def is_connected(self):
        """Return whether the pattern has connected cells."""
        return True
    
    def connected_nodes(self):
        """Return the list of connected nodes."""
        return [0, 1]
    
    def source_graph(self):
        """Return the source graph (locs, graph, pins)."""
        # ⋅ ◆ ⋅ ⋅
        # ◆ ● ⋅ ⋅
        # ⋅ ● ⋅ ⋅
        locs = [Node((0, 1)), Node((1, 0)), Node((1, 1)), Node((2, 1))]
        g = simplegraph([(1, 2), (1, 3), (3, 4)])
        pins = [0, 1, 3]  # convert from 1-indexed to 0-indexed
        return locs, g, pins
    
    def mapped_graph(self):
        """Return the mapped graph (locs, graph, pins)."""
        # ⋅ ● ⋅ ⋅
        # ● ⋅ ● ⋅
        # ⋅ ● ⋅ ⋅
        locs = [Node((0, 1)), Node((1, 0)), Node((1, 2)), Node((2, 1))]
        pins = [0, 1, 3]  # convert from 1-indexed to 0-indexed
        return locs, unit_disk_graph([loc.loc for loc in locs], 1.5), pins


class TrivialTurn(Pattern):
    """Pattern for trivial turns."""
    
    def __init__(self):
        super().__init__()
    
    def size(self):
        """Return the size of the pattern."""
        return (2, 2)
    
    def cross_location(self):
        """Return the location of the cross."""
        return (1, 1)
    
    def is_connected(self):
        """Return whether the pattern has connected cells."""
        return True
    
    def connected_nodes(self):
        """Return the list of connected nodes."""
        return [0, 1]
    
    def source_graph(self):
        """Return the source graph (locs, graph, pins)."""
        # ⋅ ◆
        # ◆ ⋅
        locs = [Node((0, 1)), Node((1, 0))]
        g = simplegraph([(1, 2)])
        pins = [0, 1]  # convert from 1-indexed to 0-indexed
        return locs, g, pins
    
    def mapped_graph(self):
        """Return the mapped graph (locs, graph, pins)."""
        # ⋅ ●
        # ● ⋅
        locs = [Node((0, 1)), Node((1, 0))]
        pins = [0, 1]  # convert from 1-indexed to 0-indexed
        return locs, unit_disk_graph([loc.loc for loc in locs], 1.5), pins


class EndTurn(Pattern):
    """Pattern for end turns."""
    
    def __init__(self):
        super().__init__()
    
    def size(self):
        """Return the size of the pattern."""
        return (3, 4)
    
    def cross_location(self):
        """Return the location of the cross."""
        return (1, 1)
    
    def source_graph(self):
        """Return the source graph (locs, graph, pins)."""
        # ⋅ ● ⋅ ⋅
        # ⋅ ● ● ⋅
        # ⋅ ⋅ ⋅ ⋅
        locs = [Node((0, 1)), Node((1, 1)), Node((1, 2))]
        g = simplegraph([(1, 2), (2, 3)])
        pins = [0]  # convert from 1-indexed to 0-indexed
        return locs, g, pins
    
    def mapped_graph(self):
        """Return the mapped graph (locs, graph, pins)."""
        # ⋅ ● ⋅ ⋅
        # ⋅ ⋅ ⋅ ⋅
        # ⋅ ⋅ ⋅ ⋅
        locs = [Node((0, 1))]
        pins = [0]  # convert from 1-indexed to 0-indexed
        return locs, unit_disk_graph([loc.loc for loc in locs], 1.5), pins


class RotatedGadget(Pattern):
    """A gadget rotated n times by 90 degrees."""
    
    def __init__(self, gadget, n):
        super().__init__()
        self.gadget = gadget
        self.n = n % 4  # Normalize to 0-3
        self._cached_offset = None
    
    def size(self):
        """Return the size of the pattern after rotation."""
        m, n = self.gadget.size()
        return (n, m) if self.n % 2 == 1 else (m, n)
    
    def cross_location(self):
        """Return the location of the cross after rotation."""
        cx, cy = self.gadget.cross_location()
        center = (cx, cy)
        for _ in range(self.n):
            center = rotate90(center, self.gadget.cross_location())
        offset = self._get_offset()
        return (center[0] + offset[0], center[1] + offset[1])
    
    def is_connected(self):
        """Return whether the pattern has connected cells."""
        return self.gadget.is_connected()
    
    def connected_nodes(self):
        """Return the list of connected nodes."""
        return self.gadget.connected_nodes()
    
    def _get_offset(self):
        """Get the offset needed to keep the pattern within bounds."""
        if self._cached_offset is not None:
            return self._cached_offset
            
        m, n = self.gadget.size()
        center = self.gadget.cross_location()
        
        # Calculate transformed corners
        min_x, min_y = float('inf'), float('inf')
        corners = [(0, 0), (0, n-1), (m-1, 0), (m-1, n-1)]
        
        for corner in corners:
            x, y = corner
            loc = (x, y)
            for _ in range(self.n):
                loc = rotate90(loc, center)
            min_x = min(min_x, loc[0])
            min_y = min(min_y, loc[1])
        
        self._cached_offset = (1 - min_x, 1 - min_y)
        return self._cached_offset
    
    def _apply_transform(self, node, center):
        """Apply the rotation transformation to a node."""
        x, y = node.loc
        loc = (x, y)
        for _ in range(self.n):
            loc = rotate90(loc, center)
        offset = self._get_offset()
        return Node((loc[0] + offset[0], loc[1] + offset[1]), weight=node.weight)
    
    def source_graph(self):
        """Return the source graph after rotation."""
        locs, graph, pins = self.gadget.source_graph()
        center = self.gadget.cross_location()
        transformed_locs = [self._apply_transform(loc, center) for loc in locs]
        return transformed_locs, graph, pins
    
    def mapped_graph(self):
        """Return the mapped graph after rotation."""
        locs, graph, pins = self.gadget.mapped_graph()
        center = self.gadget.cross_location()
        transformed_locs = [self._apply_transform(loc, center) for loc in locs]
        return transformed_locs, unit_disk_graph([loc.loc for loc in transformed_locs], 1.5), pins


class ReflectedGadget(Pattern):
    """A gadget reflected across an axis."""
    
    def __init__(self, gadget, mirror):
        super().__init__()
        self.gadget = gadget
        self.mirror = mirror  # "x", "y", "diag", or "offdiag"
        self._cached_offset = None
    
    def size(self):
        """Return the size of the pattern after reflection."""
        m, n = self.gadget.size()
        return (n, m) if self.mirror in ["diag", "offdiag"] else (m, n)
    
    def cross_location(self):
        """Return the location of the cross after reflection."""
        cx, cy = self.gadget.cross_location()
        loc = (cx, cy)
        
        if self.mirror == "x":
            loc = reflectx(loc, (cx, cy))
        elif self.mirror == "y":
            loc = reflecty(loc, (cx, cy))
        elif self.mirror == "diag":
            loc = reflectdiag(loc, (cx, cy))
        elif self.mirror == "offdiag":
            loc = reflectoffdiag(loc, (cx, cy))
        
        offset = self._get_offset()
        return (loc[0] + offset[0], loc[1] + offset[1])
    
    def is_connected(self):
        """Return whether the pattern has connected cells."""
        return self.gadget.is_connected()
    
    def connected_nodes(self):
        """Return the list of connected nodes."""
        return self.gadget.connected_nodes()
    
    def _get_offset(self):
        """Get the offset needed to keep the pattern within bounds."""
        if self._cached_offset is not None:
            return self._cached_offset
            
        m, n = self.gadget.size()
        center = self.gadget.cross_location()
        
        # Calculate transformed corners
        min_x, min_y = float('inf'), float('inf')
        corners = [(0, 0), (0, n-1), (m-1, 0), (m-1, n-1)]
        
        for corner in corners:
            x, y = corner
            if self.mirror == "x":
                loc = reflectx((x, y), center)
            elif self.mirror == "y":
                loc = reflecty((x, y), center)
            elif self.mirror == "diag":
                loc = reflectdiag((x, y), center)
            elif self.mirror == "offdiag":
                loc = reflectoffdiag((x, y), center)
            else:
                raise ValueError(f"Invalid mirror direction: {self.mirror}")
                
            min_x = min(min_x, loc[0])
            min_y = min(min_y, loc[1])
        
        self._cached_offset = (1 - min_x, 1 - min_y)
        return self._cached_offset
    
    def _apply_transform(self, node, center):
        """Apply the reflection transformation to a node."""
        x, y = node.loc
        
        if self.mirror == "x":
            loc = reflectx((x, y), center)
        elif self.mirror == "y":
            loc = reflecty((x, y), center)
        elif self.mirror == "diag":
            loc = reflectdiag((x, y), center)
        elif self.mirror == "offdiag":
            loc = reflectoffdiag((x, y), center)
        else:
            raise ValueError(f"Invalid mirror direction: {self.mirror}")
        
        offset = self._get_offset()
        return Node((loc[0] + offset[0], loc[1] + offset[1]), weight=node.weight)
    
    def source_graph(self):
        """Return the source graph after reflection."""
        locs, graph, pins = self.gadget.source_graph()
        center = self.gadget.cross_location()
        transformed_locs = [self._apply_transform(loc, center) for loc in locs]
        return transformed_locs, graph, pins
    
    def mapped_graph(self):
        """Return the mapped graph after reflection."""
        locs, graph, pins = self.gadget.mapped_graph()
        center = self.gadget.cross_location()
        transformed_locs = [self._apply_transform(loc, center) for loc in locs]
        return transformed_locs, unit_disk_graph([loc.loc for loc in transformed_locs], 1.5), pins


def rotated_and_reflected(pattern):
    """Generate all unique rotated and reflected variants of a pattern."""
    patterns = [pattern]
    
    # Helper function to compare matrices
    def matrix_equal(m1, m2):
        if len(m1) != len(m2) or len(m1[0]) != len(m2[0]):
            return False
        for i in range(len(m1)):
            for j in range(len(m1[0])):
                if m1[i][j].occupied != m2[i][j].occupied:
                    return False
        return True
    
    # Store source matrices for comparison
    source_matrices = [pattern.source_matrix()]
    
    # Add rotated variants
    for i in range(1, 4):
        try:
            rotated = RotatedGadget(pattern, i)
            rotated_matrix = rotated.source_matrix()
            
            # Check if this is a new unique matrix
            is_unique = True
            for existing_matrix in source_matrices:
                if len(rotated_matrix) == len(existing_matrix) and len(rotated_matrix[0]) == len(existing_matrix[0]):
                    if matrix_equal(rotated_matrix, existing_matrix):
                        is_unique = False
                        break
            
            if is_unique:
                patterns.append(rotated)
                source_matrices.append(rotated_matrix)
        except Exception as e:
            print(f"Error creating rotated variant {i}: {e}")
    
    # Add reflected variants
    for mirror in ["x", "y", "diag", "offdiag"]:
        try:
            reflected = ReflectedGadget(pattern, mirror)
            reflected_matrix = reflected.source_matrix()
            
            # Check if this is a new unique matrix
            is_unique = True
            for existing_matrix in source_matrices:
                if len(reflected_matrix) == len(existing_matrix) and len(reflected_matrix[0]) == len(existing_matrix[0]):
                    if matrix_equal(reflected_matrix, existing_matrix):
                        is_unique = False
                        break
            
            if is_unique:
                patterns.append(reflected)
                source_matrices.append(reflected_matrix)
        except Exception as e:
            print(f"Error creating reflected variant {mirror}: {e}")
    
    return patterns


# Define the crossing ruleset
crossing_ruleset = [
    Cross(has_edge=True),
    Cross(has_edge=False),
    Turn(),
    Branch(),
    BranchFix(),
    WTurn(),
    BranchFixB(),
    TCon(),
    TrivialTurn(),
    EndTurn()
]