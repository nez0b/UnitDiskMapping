# Unit Disk Mapping (Python)

A Python implementation of the Unit Disk Mapping algorithm for mapping optimization problems to unit disk graphs for quantum computing.

## Overview

This package implements algorithms for mapping various optimization problems (like Maximum Independent Set, QUBO, and integer factorization) to unit disk grid graphs, which can be naturally encoded in neutral-atom quantum computers.

The original implementation is in Julia and can be found at [UnitDiskMapping.jl](https://github.com/QuEraComputing/UnitDiskMapping.jl).

## Installation

This package requires:
- Python 3.7+
- NetworkX
- NumPy
- Matplotlib (for visualization)

```bash
# Not yet available on PyPI
pip install -e .
```

## Usage

Here's a simple example of mapping a graph to a unit disk graph:

```python
import networkx as nx
from unit_disk_mapping import map_graph, MinhThiTrick

# Create a graph
graph = nx.petersen_graph()

# Map the graph to a unit disk grid graph
result = map_graph(graph, vertex_order=MinhThiTrick())

# Access the mapped graph
grid_graph = result.grid_graph

# Convert to a NetworkX graph
nx_graph = grid_graph.to_networkx()
```

## Example Scripts

The package includes example scripts to demonstrate usage:

1. `example.py` - Maps and visualizes both the Petersen graph and a small custom graph
2. `example_5vertex.py` - Creates a simple 5-vertex graph, maps it to a unit disk graph, and demonstrates finding an independent set:

```bash
python example_5vertex.py
```

This example produces visualizations showing both the original graph and its unit disk mapping:

1. `5vertex_mapping_example.png` - Shows the original graph and its unit disk mapping
2. `5vertex_independent_set.png` - Visualizes an independent set configuration on the mapped graph

## Package Structure

The Python translation has the following structure:

- `core.py`: Defines the fundamental data structures (Node, Cell, GridGraph)
- `utils.py`: Utility functions for transformations and graph operations
- `copyline.py`: Implements CopyLine structure for graph embedding
- `pathdecomposition/`: Path decomposition algorithms
  - `pathdecomposition.py`: Core path decomposition functionality
  - `greedy.py`: Greedy path decomposition algorithm
  - `branching.py`: MinhThiTrick exact algorithm
- `mapping.py`: Main implementation of mapping algorithms
- `unit_disk_mapping.py`: Main module that exports all functionality

## Testing

To run tests:

```bash
python run_tests.py
```

## Features

- Map arbitrary graphs to unit disk graphs
- Support for both weighted and unweighted problems
- Path decomposition algorithms for optimizing the mapping
- Utilities for working with and visualizing unit disk graphs

## Implementation Notes

This Python implementation is a translation of the original Julia code. The core functionality is implemented, but some advanced features (like pattern matching and gadget application) are simplified or omitted.

## Citation

If you use this software in your research, please cite the original paper:

```
@article{unitdiskmapping,
    title={Quantum Optimization with Arbitrary Connectivity Using Rydberg Atom Arrays},
    author={Nguyen, Minh-Thi and Das, Shouvanik and Weidinger, Lorenz and Staudacher, Stefanie and Katz, Or and Häner, Thomas and Donvil, Brecht and Tannu, Swamit S. and Cong, Iris},
    journal={PRX Quantum},
    volume={4},
    number={1},
    pages={010316},
    year={2023},
    publisher={APS}
}
```
