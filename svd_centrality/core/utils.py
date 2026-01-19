#!/usr/bin/env python3
"""
Utility functions for SVD centrality computations.
"""

import numpy as np
import networkx as nx
from typing import Dict, Any, Union
import warnings


class SVDCentralityError(Exception):
    """Custom exception for SVD centrality computation errors."""
    pass


def validate_graph(graph: nx.Graph) -> None:
    """
    Validate input graph for SVD centrality computation.
    
    Parameters:
    -----------
    graph : nx.Graph
        Graph to validate
        
    Raises:
    -------
    SVDCentralityError
        If graph is invalid for computation
    """
    if not isinstance(graph, (nx.Graph, nx.DiGraph)):
        raise SVDCentralityError("Input must be a NetworkX Graph or DiGraph")
    
    if graph.number_of_nodes() == 0:
        raise SVDCentralityError("Graph must contain at least one node")
    
    # Check for self-loops (can cause numerical issues)
    self_loops = list(nx.selfloop_edges(graph))
    if self_loops:
        warnings.warn(f"Graph contains {len(self_loops)} self-loops which may affect results")
    
    # Check for isolated nodes
    isolated = list(nx.isolates(graph))
    if isolated:
        warnings.warn(f"Graph contains {len(isolated)} isolated nodes")


def compute_graph_properties(graph: nx.Graph) -> Dict[str, Any]:
    """
    Compute basic graph properties for analysis.
    
    Parameters:
    -----------
    graph : nx.Graph
        Input graph
        
    Returns:
    --------
    properties : Dict[str, Any]
        Dictionary of graph properties
    """
    properties = {
        "nodes": graph.number_of_nodes(),
        "edges": graph.number_of_edges(),
        "is_directed": graph.is_directed(),
        "density": nx.density(graph),
        "is_connected": nx.is_connected(graph.to_undirected()),
    }
    
    if graph.is_directed():
        properties.update({
            "is_weakly_connected": nx.is_weakly_connected(graph),
            "is_strongly_connected": nx.is_strongly_connected(graph),
            "number_weakly_connected_components": nx.number_weakly_connected_components(graph),
            "number_strongly_connected_components": nx.number_strongly_connected_components(graph)
        })
    else:
        properties.update({
            "number_connected_components": nx.number_connected_components(graph)
        })
    
    # Degree statistics
    degrees = dict(graph.degree())
    if degrees:
        degree_values = list(degrees.values())
        properties.update({
            "avg_degree": np.mean(degree_values),
            "max_degree": max(degree_values),
            "min_degree": min(degree_values),
            "degree_std": np.std(degree_values)
        })
    
    return properties


def ensure_numerical_stability(matrix: np.ndarray, 
                                tolerance: float = 1e-12) -> np.ndarray:
    """
    Apply numerical stability measures to a matrix.
    
    Parameters:
    -----------
    matrix : np.ndarray
        Input matrix
    tolerance : float
        Numerical tolerance for small values
        
    Returns:
    --------
    stable_matrix : np.ndarray
        Numerically stabilized matrix
    """
    # Set very small values to zero
    stable_matrix = matrix.copy()
    stable_matrix[np.abs(stable_matrix) < tolerance] = 0.0
    
    return stable_matrix


def normalize_centralities(centralities: Dict[Any, float], 
                           method: str = "sum") -> Dict[Any, float]:
    """
    Normalize centrality values.
    
    Parameters:
    -----------
    centralities : Dict[Any, float]
        Raw centrality values
    method : str
        Normalization method: "sum", "max", "l2"
        
    Returns:
    --------
    normalized : Dict[Any, float]
        Normalized centrality values
    """
    if not centralities:
        return centralities
    
    values = np.array(list(centralities.values()))
    
    if method == "sum":
        total = np.sum(values)
        if total > 0:
            normalized_values = values / total
        else:
            normalized_values = values
    elif method == "max":
        max_val = np.max(values)
        if max_val > 0:
            normalized_values = values / max_val
        else:
            normalized_values = values
    elif method == "l2":
        norm = np.linalg.norm(values)
        if norm > 0:
            normalized_values = values / norm
        else:
            normalized_values = values
    else:
        raise ValueError(f"Unknown normalization method: {method}")
    
    return {key: float(val) for key, val in zip(centralities.keys(), normalized_values)}


def compare_centrality_rankings(centrality1: Dict[Any, float],
                                centrality2: Dict[Any, float]) -> Dict[str, float]:
    """
    Compare two centrality measures through ranking analysis.
    
    Parameters:
    -----------
    centrality1 : Dict[Any, float]
        First centrality measure
    centrality2 : Dict[Any, float]
        Second centrality measure
        
    Returns:
    --------
    comparison : Dict[str, float]
        Statistical comparison metrics
    """
    common_keys = set(centrality1.keys()) & set(centrality2.keys())
    
    if not common_keys:
        return {"error": "No common keys between centrality measures"}
    
    values1 = [centrality1[key] for key in common_keys]
    values2 = [centrality2[key] for key in common_keys]
    
    # Correlations
    pearson_corr = np.corrcoef(values1, values2)[0, 1]
    
    # Ranking correlations
    rank1 = np.argsort(np.argsort(values1))
    rank2 = np.argsort(np.argsort(values2))
    rank_corr = np.corrcoef(rank1, rank2)[0, 1]
    
    # Spearman correlation
    spearman_corr = np.corrcoef(np.argsort(values1), np.argsort(values2))[0, 1]
    
    return {
        "pearson_correlation": float(pearson_corr),
        "rank_correlation": float(rank_corr),
        "spearman_correlation": float(spearman_corr),
        "sample_size": len(common_keys)
    }


def create_test_graphs() -> Dict[str, nx.Graph]:
    """
    Create standard test graphs for validation.
    
    Returns:
    --------
    test_graphs : Dict[str, nx.Graph]
        Dictionary of test graphs
    """
    graphs = {}
    
    # Simple directed graph
    G_directed = nx.DiGraph()
    G_directed.add_edges_from([(0, 1), (1, 2), (2, 0), (2, 3)])
    graphs["simple_directed"] = G_directed
    
    # Simple undirected graph
    G_undirected = nx.Graph()
    G_undirected.add_edges_from([(0, 1), (1, 2), (2, 3), (3, 0)])
    graphs["simple_undirected"] = G_undirected
    
    # Path graph
    graphs["path"] = nx.path_graph(5)
    
    # Star graph
    graphs["star"] = nx.star_graph(4)
    
    # Complete graph
    graphs["complete"] = nx.complete_graph(4)
    
    # Cycle graph
    graphs["cycle"] = nx.cycle_graph(5)
    
    return graphs