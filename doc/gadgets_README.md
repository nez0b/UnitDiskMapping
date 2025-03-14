# Gadgets Module for Unit Disk Mapping

This module implements the gadgets used in unit disk graph mapping, which are patterns of nodes and edges that represent various crossing and connection configurations. These gadgets are essential for transforming arbitrary graphs into unit disk graphs.

## Overview

Gadgets are patterns used to replace crossings in the embedded graph. The following gadget types are implemented:

1. **Cross**: Pattern for crossings with or without an edge
2. **Turn**: Pattern for turning a path
3. **Branch**: Pattern for branching a path into two paths
4. **BranchFix**: Pattern for fixing a branch
5. **WTurn**: Pattern for W-shaped turns
6. **BranchFixB**: Pattern for branch fixes of type B
7. **TCon**: Pattern for T-connections
8. **TrivialTurn**: Pattern for trivial turns
9. **EndTurn**: Pattern for end turns

Additionally, there are transformation patterns:

- **RotatedGadget**: A gadget rotated by multiples of 90 degrees
- **ReflectedGadget**: A gadget reflected across an axis

## Usage with Unit Disk Mapping

You can use the gadgets with the main mapping functionality to customize how graphs are mapped to unit disk graphs:

```python
import networkx as nx
from unit_disk_mapping import map_graph, Cross, Turn, Branch

# Create a graph
g = nx.Graph()
g.add_nodes_from(range(1, 6))
g.add_edges_from([(1, 2), (2, 3), (3, 4), (4, 5), (5, 1), (1, 3)])

# Create a custom ruleset with specific patterns
custom_ruleset = [
    Cross(has_edge=True),
    Turn(),
    Branch()
]

# Map the graph with custom ruleset
result = map_graph(g, ruleset=custom_ruleset)

# Print information about applied gadgets
print(f"Number of nodes in unit disk graph: {len(result.grid_graph.nodes)}")
print(f"Total MIS overhead: {result.mis_overhead}")
print(f"Number of gadgets applied: {len(result.mapping_history)}")
```

## Direct Usage of Gadgets

Gadgets can also be used directly for pattern matching and replacement:

```python
from unit_disk_mapping import Cross, Turn, Branch, crossing_ruleset

# Create a Cross pattern with an edge
cross = Cross(has_edge=True)

# Get the source and mapped graph representations
locs, graph, pins = cross.source_graph()
mapped_locs, mapped_graph, mapped_pins = cross.mapped_graph()

# Check if a pattern matches at a specific location in a grid
matrix = create_grid_matrix()
if cross.match(matrix, i, j):
    # Apply the gadget to replace the crossing at (i, j)
    cross.apply_gadget(matrix, i, j)

# Use the crossing ruleset
for pattern in crossing_ruleset:
    # Do something with each pattern...
    pass
```

## Visualization

You can visualize gadgets using the provided example scripts:

1. `gadget_example.py` - Shows the source and mapped configurations of each gadget
2. `unit_disk_gadgets_example.py` - Demonstrates using custom rulesets with the mapping process

## Implementation Details

Each gadget implements the following key methods:

- `source_graph()`: Returns the original graph representation (locations, graph, pins)
- `mapped_graph()`: Returns the unit disk graph representation (locations, graph, pins)
- `size()`: Returns the dimensions of the gadget
- `cross_location()`: Returns the central position of the gadget
- `match(matrix, i, j)`: Checks if the pattern matches at the given location
- `apply_gadget(matrix, i, j)`: Applies the gadget at the given location
- `vertex_overhead()`: Calculates the overhead cost of using this gadget

## Integration with Mapping

The gadgets are integrated with the mapping process in the following ways:

1. You can use the default `crossing_ruleset` in `map_graph()` for standard behavior
2. You can provide a custom ruleset to `map_graph()` for specialized mapping
3. The MIS overhead calculation uses the `vertex_overhead()` method of each applied gadget
4. The mapping history records which gadgets were applied and their locations

## Performance Considerations

- Each gadget has different vertex overhead costs
- Using simpler gadgets may result in smaller unit disk graphs but might not be able to handle all crossings
- Custom rulesets allow fine-tuning the tradeoff between graph size and pattern coverage