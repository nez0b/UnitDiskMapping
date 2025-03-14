# Unit Disk Mapping Documentation

This directory contains documentation for the Unit Disk Mapping Python package.

## Overview Documents

- [Implementation Summary](implementation_summary.md) - High-level description of implementation details
- [Source Code Structure](src_README.md) - Overview of the source code directory structure

## Component Documentation

- [Gadgets](gadgets_README.md) - Documentation for basic gadgets implementation
- [Complex Gadgets](complex_gadgets_README.md) - Documentation for advanced gadget implementations
- [Dragondrop](dragondrop_README.md) - Documentation for the Dragondrop algorithm
- [Weighted Mapping](weighted_README.md) - Documentation for weighted mapping functionality

## Usage

For usage examples, refer to the Python scripts in the `examples/` directory.

## API Reference

The main API is exposed through the `unit_disk_mapping` module, which provides:

- Graph mapping functions (`map_graph`, `embed_graph`)
- Path decomposition algorithms (`MinhThiTrick`, `Greedy`)
- Gadget patterns for graph crossings
- Utilities for working with unit disk graphs