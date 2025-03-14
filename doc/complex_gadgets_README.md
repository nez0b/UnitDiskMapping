# Enhanced Complex Gadget Transformations

This module extends the UnitDiskMapping package with advanced gadget patterns and transformation capabilities for mapping complex graph structures to unit disk graphs.

## Overview

The enhanced complex gadget module provides:

1. Advanced gadget base class with boundary configuration support
2. Improved rotation and reflection transformations
3. Additional complex gadget patterns
4. Comprehensive ruleset generation tools
5. Performance optimizations with caching

## Classes

### Base Class

#### `ComplexGadget`

Extended base class for complex gadgets with improved functionality:

```python
class ComplexGadget(Pattern):
    def source_boundary_config(self)
    def mapped_boundary_config()
    def source_entry_to_configs(entry)
    def mapped_entry_to_compact(entry)
    def mis_overhead()
```

### Transformation Classes

#### `EnhancedRotatedGadget`

Improved version of `RotatedGadget` with caching and boundary handling:

```python
EnhancedRotatedGadget(gadget, n)
```

- `gadget`: Base gadget to rotate
- `n`: Number of 90-degree rotations (0-3)

#### `EnhancedReflectedGadget`

Improved version of `ReflectedGadget` with caching and boundary handling:

```python
EnhancedReflectedGadget(gadget, mirror)
```

- `gadget`: Base gadget to reflect
- `mirror`: Direction of reflection ("x", "y", "diag", "offdiag")

### Complex Pattern Classes

#### `StarPattern`

A star-shaped pattern with connections in four directions.

#### `SpiralPattern`

A spiral-shaped pattern with curved connections.

#### `DiagonalCross`

A pattern for handling diagonal crossings in graphs.

#### `DoubleCross`

A pattern for handling complex double crossing points.

### Key Functions

- `enhanced_rotated(gadget, n)`: Create an enhanced rotated gadget
- `enhanced_reflected(gadget, mirror)`: Create an enhanced reflected gadget
- `enhanced_rotated_and_reflected(pattern)`: Generate all unique transformations
- `generate_enhanced_ruleset()`: Generate comprehensive ruleset with all variations

### Rulesets

The module includes two predefined rulesets:

- `enhanced_crossing_ruleset`: Basic set of original and complex patterns
- `complete_enhanced_crossing_ruleset`: Comprehensive set including all variations

## Usage

### Using Complex Patterns

```python
import networkx as nx
from unit_disk_mapping import map_graph, UnWeighted, StarPattern

# Create a graph
G = nx.Graph()
G.add_edges_from([(0, 1), (1, 2), (2, 0)])

# Map using a specific complex pattern
result = map_graph(G, mode=UnWeighted(), ruleset=[StarPattern()])
```

### Using Enhanced Transformations

```python
from unit_disk_mapping import (
    Cross, enhanced_rotated, enhanced_reflected, 
    enhanced_rotated_and_reflected
)

# Create rotated variant of a pattern
rotated_cross = enhanced_rotated(Cross(), 1)  # 90 degree rotation

# Create reflected variant
reflected_cross = enhanced_reflected(Cross(), "diag")  # Diagonal reflection

# Generate all unique variants
all_cross_variants = enhanced_rotated_and_reflected(Cross())

# Use in custom ruleset
custom_ruleset = [rotated_cross, reflected_cross]
result = map_graph(G, mode=UnWeighted(), ruleset=custom_ruleset)
```

### Using Complete Ruleset

```python
from unit_disk_mapping import map_graph, UnWeighted, complete_enhanced_crossing_ruleset

# Map using the complete enhanced ruleset
result = map_graph(G, mode=UnWeighted(), ruleset=complete_enhanced_crossing_ruleset)
```

## Example

See `complex_gadgets_example.py` for a complete example demonstrating:

- Using complex gadget patterns for different graph structures
- Visualizing source and mapped patterns
- Comparing mapping statistics with different patterns

## Benefits

The enhanced complex gadget transformations provide several benefits:

1. More efficient mapping for complex graph structures
2. Reduced vertex overhead in the resulting unit disk graphs
3. Better handling of special graph topologies
4. Improved performance through caching
5. Comprehensive transformation options for optimizing mappings