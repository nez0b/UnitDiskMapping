# Weighted Unit Disk Mapping

This module adds weighted graph support to the UnitDiskMapping package, enabling the mapping of weighted graphs to unit disk graphs with properly transferred node weights.

## Overview

The weighted functionality extends the base unit disk mapping capabilities with:

1. Support for weighted nodes in graphs
2. Weight-preserving gadget transformations
3. Weighted crossing patterns
4. Weight mapping utilities
5. Configuration mapping for weighted graphs

## Classes

### `WeightedGadget`

A decorator class that wraps existing gadget patterns to handle weights:

```python
WeightedGadget(gadget, source_weights=None, mapped_weights=None)
```

- `gadget`: Base gadget pattern to wrap
- `source_weights`: Optional weights for the source graph
- `mapped_weights`: Optional weights for the mapped graph

### Key Functions

- `weighted(gadget, source_weights=None, mapped_weights=None)`: Transform a gadget to support weights
- `source_centers(gadget)`: Get center coordinates of the source graph
- `mapped_centers(gadget)`: Get center coordinates of the mapped graph
- `move_centers(centers, dx, dy)`: Move all center points by an offset
- `trace_centers(gadget, x, y)`: Trace centers from a specific position
- `map_weights(grid, weighted_grid, ruleset=None)`: Map weights between grids
- `map_configs_back(mapping_result, config)`: Map configurations back from weighted grids

### Weighted Ruleset

The module includes a weighted version of the standard crossing ruleset:

```python
crossing_ruleset_weighted = [
    simple_gadget_rule(Cross(has_edge=True), 1, 2),
    simple_gadget_rule(Cross(has_edge=False), 1, 2),
    simple_gadget_rule(Turn(), 1, 2),
    # ... more weighted patterns
]
```

## Usage

### Basic Weighted Mapping

```python
import networkx as nx
from unit_disk_mapping import map_graph, Weighted, crossing_ruleset_weighted

# Create a weighted graph
G = nx.Graph()
G.add_edges_from([(0, 1), (1, 2), (2, 3), (3, 0)])

# Map to a unit disk graph with weights
result = map_graph(G, mode=Weighted(), ruleset=crossing_ruleset_weighted)

# Access the mapped grid graph with weights
grid_graph = result.grid_graph
for node in grid_graph.nodes:
    print(f"Node at {node.loc} has weight {node.weight}")
```

### Custom Weighted Gadgets

```python
from unit_disk_mapping import Cross, weighted

# Create a custom weighted gadget
my_weighted_cross = weighted(
    Cross(has_edge=True),
    source_weights=[2, 1, 1, 3, 2, 1],  # Weights for source nodes
    mapped_weights=[2, 1, 1, 3, 2]      # Weights for mapped nodes
)

# Use in custom ruleset
custom_ruleset = [my_weighted_cross, ...]
result = map_graph(G, mode=Weighted(), ruleset=custom_ruleset)
```

## Example

See `weighted_example.py` for a complete example demonstrating:

- Creation of weighted graphs
- Mapping to weighted unit disk graphs
- Visualization of weighted mappings
- Weight transfer through gadget patterns

## Integration with QUBO Problems

The weighted functionality is particularly useful for QUBO (Quadratic Unconstrained Binary Optimization) problems, where nodes have different weights or penalties. These weights can be preserved during the mapping process, ensuring that the mapped unit disk graph maintains the original problem semantics.