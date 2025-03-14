# dragondrop.py - QUBO Mapping Module

This module provides functionality for mapping QUBO and related problems to weighted maximum independent set (WMIS) problems on unit disk graphs, following the algorithm described in the UnitDiskMapping paper.

## Main Functions

### `map_qubo(J, h)`

Maps a QUBO problem to a weighted MIS problem on a defected King's graph, where a QUBO problem is defined by the Hamiltonian:

```
E(z) = -∑(i<j) J_ij z_i z_j + ∑_i h_i z_i
```

**Parameters:**
- `J` (numpy.ndarray): Coupling matrix (must be symmetric)
- `h` (numpy.ndarray): Vector of onsite terms

**Returns:**
- `QUBOResult` object containing the grid graph, pin list, and MIS overhead

### `map_simple_wmis(graph, weights)`

Maps a weighted MIS problem to a weighted MIS problem on a defected King's graph.

**Parameters:**
- `graph` (networkx.Graph): The graph to map
- `weights` (numpy.ndarray): Vector of vertex weights

**Returns:**
- `WMISResult` object containing the grid graph, pin list, and MIS overhead

### `map_qubo_restricted(coupling)`

Maps a nearest-neighbor restricted QUBO problem to a weighted MIS problem on a grid graph, where the QUBO problem can be specified by a list of `(i, j, i', j', J)` tuples.

**Parameters:**
- `coupling` (list): List of tuples `(i, j, i', j', J)` defining the couplings

**Returns:**
- `RestrictedQUBOResult` object containing the grid graph

### `map_qubo_square(coupling, onsite)`

Maps a QUBO problem on a square lattice to a weighted MIS problem on a grid graph.

**Parameters:**
- `coupling` (list): List of tuples `(i, j, i', j', J)` defining the couplings
- `onsite` (list): List of tuples `(i, j, h)` defining the onsite terms

**Returns:**
- `SquareQUBOResult` object containing the grid graph, pin list, and MIS overhead

### `map_config_back(result, cfg)`

Maps a configuration back from the unit disk graph to the original graph.

**Parameters:**
- `result` (QUBOResult or similar): Result object from one of the mapping functions
- `cfg` (list or numpy.ndarray): Configuration on the unit disk graph

**Returns:**
- Configuration for the original problem

## Helper Functions

The module also contains several helper functions used by the main mapping functions:

- `glue(grid, DI, DJ)`: Glues multiple blocks into a whole
- `cell_matrix(gg)`: Converts a GridGraph to a matrix of SimpleCell objects
- `crossing_lattice(g, ordered_vertices)`: Creates a crossing lattice from a graph
- `create_copylines(g, ordered_vertices)`: Creates copy lines using path decomposition
- `remove_order(g, ordered_vertices)`: Calculates the removal order for vertices
- `render_grid(cl)`: Renders a grid from a crossing lattice
- `post_process_grid(grid, h0, h1)`: Processes the grid to add weights for 0 and 1 states
- `gadget_qubo_restricted(J)`: Creates a gadget for restricted QUBO problems
- `pad(m, top, bottom, left, right)`: Pads a matrix with empty cells
- `gadget_qubo_square()`: Creates a gadget for square QUBO problems

## Result Classes

- `QUBOResult`: Result of mapping a QUBO problem to a unit disk graph
- `WMISResult`: Result of mapping a weighted MIS problem to a unit disk graph
- `RestrictedQUBOResult`: Result of mapping a restricted QUBO problem to a unit disk graph
- `SquareQUBOResult`: Result of mapping a square lattice QUBO problem to a unit disk graph

## Example Usage

```python
import numpy as np
import networkx as nx
from unit_disk_mapping import map_qubo, map_config_back

# Create a small QUBO problem
n = 4
J = np.zeros((n, n))
for i in range(n-1):
    for j in range(i+1, n):
        J[i, j] = np.random.uniform(-0.1, 0.1)
J = J + J.T  # Make symmetric
h = np.random.uniform(-0.1, 0.1, n)

# Map to unit disk graph
result = map_qubo(J, h)

# Get information about the mapping
print(f"Mapped to unit disk graph with {len(result.grid_graph.nodes)} nodes")
print(f"Pin vertices: {result.pins}")
print(f"MIS overhead: {result.mis_overhead}")

# Create a configuration (normally found by solving the MIS problem)
config = np.random.randint(0, 2, len(result.grid_graph.nodes))

# Map the configuration back
original_config = map_config_back(result, config)
```

See `qubo_example.py` for a more complete example with visualization.