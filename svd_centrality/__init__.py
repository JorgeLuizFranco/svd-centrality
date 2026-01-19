#!/usr/bin/env python3
"""
SVD Centrality: A Python package for SVD-based centrality measures
=================================================================

This package implements SVD incidence centrality for both graphs and hypergraphs,
providing mathematically rigorous centrality measures with spectral foundations.

Main modules:
- core.svd_centrality: Core SVD centrality implementation for graphs
- hypergraph: SVD centrality extension for hypergraphs
- data_loader: Utilities for loading datasets

Examples:
---------
>>> from svd_centrality import SVDCentrality
>>> import networkx as nx
>>> 
>>> # Compute SVD centrality for a graph
>>> G = nx.karate_club_graph()
>>> svd = SVDCentrality()
>>> results = svd.compute_centralities(G)
>>> print(results['S_v'])  # Normalized vertex centralities
>>> 
>>> # For hypergraphs
>>> from svd_centrality.hypergraph import ManualHypergraph, HypergraphSVDCentrality
>>> H = ManualHypergraph()
>>> H.add_edge([1, 2, 3])
>>> H.add_edge([2, 3, 4])
>>> hsvd = HypergraphSVDCentrality()
>>> hresults = hsvd.compute_centralities(H)
"""

from .core.svd_centrality import SVDCentrality, ComputationStats
from .core.utils import (
    SVDCentralityError, 
    validate_graph, 
    compute_graph_properties,
    normalize_centralities,
    compare_centrality_rankings,
    create_test_graphs
)

# Import hypergraph functionality
try:
    from .hypergraph import (
        ManualHypergraph,
        HypergraphSVDCentrality,
        load_hypergraph_from_json,
        create_hypergraph_visualization
    )
    _HYPERGRAPH_AVAILABLE = True
except ImportError as e:
    _HYPERGRAPH_AVAILABLE = False
    print(f"Warning: Hypergraph functionality not available: {e}")

# Import data loading functionality
try:
    from .data_loader import (
        load_karate_club,
        load_biological_network,
        load_transportation_network,
        load_social_network
    )
    _DATA_LOADER_AVAILABLE = True
except ImportError as e:
    _DATA_LOADER_AVAILABLE = False
    print(f"Warning: Data loader functionality not available: {e}")

__version__ = "1.0.0"
__author__ = "Instituto Curvelo Research Team"
__email__ = "research@institutocurvelo.org"

__all__ = [
    # Core functionality
    'SVDCentrality',
    'ComputationStats', 
    'SVDCentralityError',
    'validate_graph',
    'compute_graph_properties',
    'normalize_centralities',
    'compare_centrality_rankings',
    'create_test_graphs'
]

# Add hypergraph exports if available
if _HYPERGRAPH_AVAILABLE:
    __all__.extend([
        'ManualHypergraph',
        'HypergraphSVDCentrality', 
        'load_hypergraph_from_json',
        'create_hypergraph_visualization'
    ])

# Add data loader exports if available  
if _DATA_LOADER_AVAILABLE:
    __all__.extend([
        'load_karate_club',
        'load_biological_network', 
        'load_transportation_network',
        'load_social_network'
    ])


def get_version():
    """Return the package version."""
    return __version__


def list_available_features():
    """List available package features."""
    features = ["Core SVD Centrality"]
    if _HYPERGRAPH_AVAILABLE:
        features.append("Hypergraph Support")
    if _DATA_LOADER_AVAILABLE:
        features.append("Data Loading Utilities")
    return features