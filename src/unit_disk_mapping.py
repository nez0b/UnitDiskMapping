"""
Unit Disk Mapping

A Python package for mapping problems to unit disk graphs for quantum computing.
"""

# Basic types
from .core import ONE_INSTANCE
from .core import SimpleCell, AbstractCell
from .core import Node, UnWeightedNode, WeightedNode
from .core import GridGraph
from .mapping import UnWeighted, Weighted
from .mapping import MappingGrid, MappingResult
from .mapping import map_graph, embed_graph, print_config, map_config_back

# Path decomposition
from .pathdecomposition import pathwidth, MinhThiTrick, Greedy

# Utilities
from .utils import unit_disk_graph, is_independent_set

# QUBO mapping
from .dragondrop import (
    map_qubo,
    map_simple_wmis,
    map_qubo_restricted,
    map_qubo_square,
    QUBOResult,
    WMISResult,
    RestrictedQUBOResult,
    SquareQUBOResult
)

# Gadgets
from .gadgets import (
    Pattern,
    Cross,
    Turn,
    Branch,
    BranchFix,
    WTurn,
    BranchFixB,
    TCon,
    TrivialTurn,
    EndTurn,
    RotatedGadget,
    ReflectedGadget,
    rotated_and_reflected,
    crossing_ruleset
)

# Advanced Gadgets
from .gadgets_ext import (
    ComplexGadget,
    EnhancedRotatedGadget,
    EnhancedReflectedGadget,
    enhanced_rotated,
    enhanced_reflected,
    enhanced_rotated_and_reflected,
    StarPattern,
    SpiralPattern,
    DiagonalCross,
    DoubleCross,
    enhanced_crossing_ruleset,
    complete_enhanced_crossing_ruleset
)

# Weighted Module
from .weighted import (
    WeightedGadget,
    weighted,
    source_centers,
    mapped_centers,
    move_centers,
    trace_centers,
    map_weights,
    map_configs_back,
    simple_gadget_rule,
    crossing_ruleset_weighted
)

# Version information
__version__ = "0.1.0"

__all__ = [
    # Core types
    "ONE_INSTANCE", "SimpleCell", "AbstractCell", "Node", 
    "UnWeightedNode", "WeightedNode", "GridGraph",
    
    # Mapping
    "UnWeighted", "Weighted", "MappingGrid", "MappingResult",
    "map_graph", "embed_graph", "print_config", "map_config_back",
    
    # Path decomposition
    "pathwidth", "MinhThiTrick", "Greedy",
    
    # Utilities
    "unit_disk_graph", "is_independent_set",
    
    # QUBO mapping
    "map_qubo", "map_simple_wmis", "map_qubo_restricted", "map_qubo_square",
    "QUBOResult", "WMISResult", "RestrictedQUBOResult", "SquareQUBOResult",

    # Gadgets
    "Pattern", "Cross", "Turn", "Branch", "BranchFix", "WTurn", "BranchFixB",
    "TCon", "TrivialTurn", "EndTurn", "RotatedGadget", "ReflectedGadget",
    "rotated_and_reflected", "crossing_ruleset",
    
    # Advanced Gadgets
    "ComplexGadget", "EnhancedRotatedGadget", "EnhancedReflectedGadget",
    "enhanced_rotated", "enhanced_reflected", "enhanced_rotated_and_reflected",
    "StarPattern", "SpiralPattern", "DiagonalCross", "DoubleCross",
    "enhanced_crossing_ruleset", "complete_enhanced_crossing_ruleset",
    
    # Weighted Module
    "WeightedGadget", "weighted", "source_centers", "mapped_centers",
    "move_centers", "trace_centers", "map_weights", "map_configs_back",
    "simple_gadget_rule", "crossing_ruleset_weighted"
]