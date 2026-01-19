#!/usr/bin/env python3
"""
Hypergraph SVD Centrality Implementation
========================================

This module extends SVD centrality to hypergraphs, providing:
- Independent hypergraph data structures
- SVD centrality computation for hypergraphs
- Visualization capabilities
- Data loading from standard formats

Mathematical Foundation:
-----------------------
For hypergraph H with incidence matrix B ∈ ℝ^(n×m):
- B[i,j] = 1 if vertex i belongs to hyperedge j, 0 otherwise
- Vertex centrality: C_v(i) = [L_0^+]_{ii} where L_0 = BB^T
- Edge centrality: C_e(e) = [L_1^+]_{ee} where L_1 = B^TB
"""

import numpy as np
import networkx as nx
import matplotlib.pyplot as plt
import json
from pathlib import Path
from collections import defaultdict
from typing import Dict, List, Set, Tuple, Any, Optional
import warnings

from ..core.svd_centrality import SVDCentrality
from ..core.utils import SVDCentralityError


class ManualHypergraph:
    """
    Independent hypergraph implementation that avoids external dependencies.
    
    This class provides a clean, standalone hypergraph data structure
    suitable for SVD centrality computation without relying on XGI or
    other external hypergraph libraries.
    """
    
    def __init__(self):
        """Initialize empty hypergraph."""
        self.nodes = set()
        self.edges = {}  # edge_id -> set of nodes
        self.node_edges = defaultdict(set)  # node -> set of edges containing it
        
    def add_node(self, node: Any) -> None:
        """Add a node to the hypergraph."""
        self.nodes.add(node)
        
    def add_edge(self, nodes: List[Any], edge_id: Optional[Any] = None) -> Any:
        """
        Add a hyperedge to the hypergraph.
        
        Parameters:
        -----------
        nodes : List[Any]
            List of nodes in the hyperedge
        edge_id : Optional[Any]
            Edge identifier (auto-generated if None)
            
        Returns:
        --------
        edge_id : Any
            The edge identifier used
        """
        if edge_id is None:
            edge_id = len(self.edges)
            
        node_set = set(nodes)
        if len(node_set) == 0:
            warnings.warn(f"Empty edge {edge_id} ignored")
            return edge_id
            
        self.edges[edge_id] = node_set
        
        # Update node-edge mappings
        for node in node_set:
            self.nodes.add(node)
            self.node_edges[node].add(edge_id)
            
        return edge_id
        
    def num_nodes(self) -> int:
        """Return number of nodes."""
        return len(self.nodes)
        
    def num_edges(self) -> int:
        """Return number of hyperedges."""
        return len(self.edges)
        
    def degree(self, node: Any) -> int:
        """Return degree of a node (number of edges it belongs to)."""
        return len(self.node_edges[node])
        
    def edge_size(self, edge_id: Any) -> int:
        """Return size of an edge (number of nodes it contains)."""
        return len(self.edges.get(edge_id, set()))
        
    def get_incidence_matrix(self) -> Tuple[np.ndarray, List[Any], List[Any]]:
        """
        Compute the incidence matrix B where B[i,j] = 1 if node i is in edge j.
        
        Returns:
        --------
        B : np.ndarray
            Incidence matrix of shape (num_nodes, num_edges)
        nodes_list : List[Any]
            Ordered list of nodes corresponding to matrix rows
        edges_list : List[Any]
            Ordered list of edges corresponding to matrix columns
        """
        if not self.nodes or not self.edges:
            return np.array([]), [], []
            
        nodes_list = sorted(list(self.nodes))
        edges_list = sorted(list(self.edges.keys()))
        
        B = np.zeros((len(nodes_list), len(edges_list)))
        node_to_idx = {node: i for i, node in enumerate(nodes_list)}
        
        for j, edge_id in enumerate(edges_list):
            edge_nodes = self.edges[edge_id]
            for node in edge_nodes:
                if node in node_to_idx:
                    B[node_to_idx[node], j] = 1.0
                    
        return B, nodes_list, edges_list
        
    def to_networkx_projection(self) -> nx.Graph:
        """
        Create NetworkX graph projection for visualization.
        
        In the projection, two nodes are connected if they appear
        together in at least one hyperedge.
        """
        G = nx.Graph()
        G.add_nodes_from(self.nodes)
        
        # Add edges for each pair of nodes that appear together
        for edge_id, edge_nodes in self.edges.items():
            nodes_list = list(edge_nodes)
            for i in range(len(nodes_list)):
                for j in range(i+1, len(nodes_list)):
                    if G.has_edge(nodes_list[i], nodes_list[j]):
                        G[nodes_list[i]][nodes_list[j]]['weight'] += 1
                    else:
                        G.add_edge(nodes_list[i], nodes_list[j], weight=1)
                        
        return G
        
    def get_statistics(self) -> Dict[str, Any]:
        """Get basic statistics about the hypergraph."""
        if not self.edges:
            return {
                "num_nodes": self.num_nodes(),
                "num_edges": 0,
                "avg_degree": 0.0,
                "avg_edge_size": 0.0
            }
            
        degrees = [self.degree(node) for node in self.nodes]
        edge_sizes = [self.edge_size(eid) for eid in self.edges]
        
        return {
            "num_nodes": self.num_nodes(),
            "num_edges": self.num_edges(),
            "avg_degree": np.mean(degrees) if degrees else 0.0,
            "max_degree": max(degrees) if degrees else 0,
            "avg_edge_size": np.mean(edge_sizes) if edge_sizes else 0.0,
            "max_edge_size": max(edge_sizes) if edge_sizes else 0,
            "density": len(edge_sizes) / (self.num_nodes() * (self.num_nodes() - 1) / 2) if self.num_nodes() > 1 else 0.0
        }
        
    def __repr__(self) -> str:
        return f"ManualHypergraph(nodes={self.num_nodes()}, edges={self.num_edges()})"


class HypergraphSVDCentrality:
    """
    SVD centrality computation for hypergraphs.
    
    This class adapts the SVD centrality framework to work with hypergraphs
    by constructing the appropriate incidence matrix and applying the same
    mathematical principles.
    """
    
    def __init__(self, 
                 regularization: float = 1e-8,
                 numerical_tolerance: float = 1e-12):
        """
        Initialize hypergraph SVD centrality computer.
        
        Parameters:
        -----------
        regularization : float
            Regularization parameter for numerical stability
        numerical_tolerance : float
            Tolerance for considering singular values as zero
        """
        self.regularization = regularization
        self.numerical_tolerance = numerical_tolerance
        
    def compute_centralities(self, hypergraph: ManualHypergraph) -> Dict[str, Any]:
        """
        Compute vertex and edge centralities for a hypergraph.
        
        Parameters:
        -----------
        hypergraph : ManualHypergraph
            Input hypergraph
            
        Returns:
        --------
        results : Dict[str, Any]
            Dictionary containing centrality values and metadata
        """
        try:
            B, nodes_list, edges_list = hypergraph.get_incidence_matrix()
            
            if B.size == 0:
                return {
                    'vertex_centrality': {node: 0.0 for node in hypergraph.nodes},
                    'edge_centrality': {},
                    'normalized_vertex_centrality': {node: 0.0 for node in hypergraph.nodes},
                    'normalized_edge_centrality': {},
                    'nodes': list(hypergraph.nodes),
                    'edges': [],
                    'statistics': hypergraph.get_statistics()
                }
            
            # Add regularization for numerical stability
            epsilon = self.regularization
            B_reg = B + epsilon * np.random.randn(*B.shape)
            
            # Compute SVD
            try:
                U, sigma, Vt = np.linalg.svd(B_reg, full_matrices=False)
            except np.linalg.LinAlgError:
                # Fallback for problematic matrices
                warnings.warn("SVD failed, using pseudoinverse directly")
                B_pinv = np.linalg.pinv(B_reg)
                vertex_centralities = np.sum(np.abs(B_pinv), axis=1)
                edge_centralities = np.sum(np.abs(B_pinv), axis=0)
            else:
                # Filter out very small singular values
                valid_idx = sigma > self.numerical_tolerance
                
                if not np.any(valid_idx):
                    warnings.warn("All singular values below tolerance")
                    vertex_centralities = np.ones(len(nodes_list)) / len(nodes_list)
                    edge_centralities = np.ones(len(edges_list)) / len(edges_list)
                else:
                    # Compute pseudoinverse using SVD
                    sigma_inv = np.zeros_like(sigma)
                    sigma_inv[valid_idx] = 1.0 / sigma[valid_idx]
                    
                    # Moore-Penrose pseudoinverse: B+ = V Σ^+ U^T
                    B_pinv = Vt.T @ np.diag(sigma_inv) @ U.T
                    
                    # Centralities from pseudoinverse
                    vertex_centralities = np.sum(np.abs(B_pinv), axis=1)
                    edge_centralities = np.sum(np.abs(B_pinv), axis=0)
            
            # Create centrality dictionaries
            vertex_centrality = dict(zip(nodes_list, vertex_centralities))
            edge_centrality = dict(zip(edges_list, edge_centralities))
            
            # Normalize to [0, 1]
            max_vertex = max(vertex_centralities) if len(vertex_centralities) > 0 else 1.0
            max_edge = max(edge_centralities) if len(edge_centralities) > 0 else 1.0
            
            normalized_vertex = {node: val/max_vertex if max_vertex > 0 else 0.0 
                               for node, val in vertex_centrality.items()}
            normalized_edge = {edge: val/max_edge if max_edge > 0 else 0.0 
                             for edge, val in edge_centrality.items()}
            
            return {
                'vertex_centrality': vertex_centrality,
                'edge_centrality': edge_centrality,
                'normalized_vertex_centrality': normalized_vertex,
                'normalized_edge_centrality': normalized_edge,
                'nodes': nodes_list,
                'edges': edges_list,
                'incidence_matrix': B,
                'statistics': hypergraph.get_statistics(),
                'singular_values': sigma.tolist() if 'sigma' in locals() else []
            }
            
        except Exception as e:
            warnings.warn(f"SVD computation failed ({e}), using degree-based fallback")
            # Degree-based fallback
            degrees = {node: hypergraph.degree(node) for node in hypergraph.nodes}
            max_deg = max(degrees.values()) if degrees.values() else 1
            normalized_degrees = {node: deg/max_deg for node, deg in degrees.items()}
            
            return {
                'vertex_centrality': degrees,
                'edge_centrality': {},
                'normalized_vertex_centrality': normalized_degrees,
                'normalized_edge_centrality': {},
                'nodes': list(hypergraph.nodes),
                'edges': list(hypergraph.edges.keys()),
                'statistics': hypergraph.get_statistics(),
                'error': str(e)
            }


def load_hypergraph_from_json(filepath: Path) -> Tuple[ManualHypergraph, Dict[str, Any]]:
    """
    Load hypergraph from JSON file.
    
    Expected JSON format:
    {
        "node-data": {...},
        "edge-dict": {"edge_id": [node1, node2, ...]},
        "hypergraph-data": {...metadata...}
    }
    
    Parameters:
    -----------
    filepath : Path
        Path to JSON file
        
    Returns:
    --------
    hypergraph : ManualHypergraph
        Loaded hypergraph
    metadata : Dict[str, Any]
        Metadata from the file
    """
    if not filepath.exists():
        raise FileNotFoundError(f"Hypergraph dataset not found: {filepath}")
        
    with open(filepath, 'r') as f:
        data = json.load(f)
    
    H = ManualHypergraph()
    
    # Add nodes if specified
    if 'node-data' in data:
        for node_id in data['node-data'].keys():
            H.add_node(node_id)
    
    # Add edges
    if 'edge-dict' in data:
        for edge_id, node_list in data['edge-dict'].items():
            if len(node_list) > 0:
                H.add_edge(node_list, edge_id)
    
    metadata = data.get('hypergraph-data', {})
    
    return H, metadata


def create_hypergraph_visualization(hypergraph: ManualHypergraph, 
                                  centrality_results: Dict[str, Any],
                                  title: str = "Hypergraph Visualization",
                                  max_nodes: int = 50,
                                  max_edges_to_show: int = 20) -> plt.Figure:
    """
    Create visualization of hypergraph with centrality information.
    
    Parameters:
    -----------
    hypergraph : ManualHypergraph
        Hypergraph to visualize
    centrality_results : Dict[str, Any]
        Results from compute_centralities
    title : str
        Plot title
    max_nodes : int
        Maximum number of nodes to show (for large graphs)
    max_edges_to_show : int
        Maximum number of hyperedges to visualize
        
    Returns:
    --------
    fig : plt.Figure
        Matplotlib figure
    """
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 6))
    
    # Get centrality values
    vertex_centrality = centrality_results['normalized_vertex_centrality']
    
    # Create NetworkX projection for layout
    G = hypergraph.to_networkx_projection()
    
    if len(G.nodes()) == 0:
        ax1.text(0.5, 0.5, 'No nodes to display', ha='center', va='center', transform=ax1.transAxes)
        ax2.text(0.5, 0.5, 'No statistics to display', ha='center', va='center', transform=ax2.transAxes)
        return fig
    
    # Limit nodes for large graphs
    if len(G.nodes()) > max_nodes:
        # Keep top centrality nodes
        top_nodes = sorted(vertex_centrality.items(), key=lambda x: x[1], reverse=True)[:max_nodes]
        nodes_to_keep = {node for node, _ in top_nodes}
        G = G.subgraph(nodes_to_keep)
        vertex_centrality = {node: centrality for node, centrality in vertex_centrality.items() if node in nodes_to_keep}
    
    # Create layout
    try:
        pos = nx.spring_layout(G, k=2, iterations=50, seed=42)
    except:
        # Fallback for disconnected or problematic graphs
        pos = {node: (np.random.random(), np.random.random()) for node in G.nodes()}
    
    # Plot 1: Hypergraph visualization
    ax1.set_title(f"{title}\nNode Centrality")
    
    # Draw edges (sample for large hypergraphs)
    edges_to_draw = list(hypergraph.edges.items())[:max_edges_to_show]
    
    for i, (edge_id, edge_nodes) in enumerate(edges_to_draw):
        nodes_in_pos = [n for n in edge_nodes if n in pos]
        if len(nodes_in_pos) >= 2:
            # Draw convex hull or complete subgraph
            edge_pos = [pos[node] for node in nodes_in_pos]
            xs, ys = zip(*edge_pos)
            
            # Use different colors for different edges
            color = plt.cm.Set3(i / max(len(edges_to_draw), 1))
            
            if len(nodes_in_pos) == 2:
                ax1.plot(xs, ys, color=color, alpha=0.6, linewidth=2)
            else:
                # Draw polygon for larger edges
                try:
                    from scipy.spatial import ConvexHull
                    if len(edge_pos) >= 3:
                        hull = ConvexHull(edge_pos)
                        for simplex in hull.simplices:
                            ax1.plot([edge_pos[simplex[0]][0], edge_pos[simplex[1]][0]], 
                                   [edge_pos[simplex[0]][1], edge_pos[simplex[1]][1]], 
                                   color=color, alpha=0.4)
                except:
                    # Fallback: just connect all pairs
                    for j in range(len(edge_pos)):
                        for k in range(j+1, len(edge_pos)):
                            ax1.plot([edge_pos[j][0], edge_pos[k][0]], 
                                   [edge_pos[j][1], edge_pos[k][1]], 
                                   color=color, alpha=0.3, linewidth=1)
    
    # Draw nodes with size proportional to centrality
    if pos:
        nodes = list(pos.keys())
        node_sizes = [1000 * vertex_centrality.get(node, 0) + 100 for node in nodes]
        node_colors = [vertex_centrality.get(node, 0) for node in nodes]
        
        scatter = ax1.scatter([pos[node][0] for node in nodes], 
                            [pos[node][1] for node in nodes],
                            s=node_sizes, c=node_colors, cmap='viridis', 
                            alpha=0.8, edgecolors='black', linewidth=0.5)
        
        # Add colorbar
        plt.colorbar(scatter, ax=ax1, label='SVD Centrality')
    
    ax1.set_aspect('equal')
    ax1.axis('off')
    
    # Plot 2: Statistics and top nodes
    ax2.set_title("Centrality Statistics")
    
    if vertex_centrality:
        # Top nodes
        top_nodes = sorted(vertex_centrality.items(), key=lambda x: x[1], reverse=True)[:10]
        
        stats = centrality_results['statistics']
        stats_text = f"""Hypergraph Statistics:
Nodes: {stats['num_nodes']}
Hyperedges: {stats['num_edges']}
Avg Degree: {stats['avg_degree']:.2f}
Avg Edge Size: {stats['avg_edge_size']:.2f}

Top Centrality Nodes:"""
        
        for i, (node, centrality) in enumerate(top_nodes):
            stats_text += f"\n{i+1}. {node}: {centrality:.3f}"
        
        ax2.text(0.1, 0.9, stats_text, transform=ax2.transAxes, fontfamily='monospace',
                verticalalignment='top', fontsize=10)
    else:
        ax2.text(0.5, 0.5, 'No centrality data', ha='center', va='center', transform=ax2.transAxes)
    
    ax2.axis('off')
    
    plt.tight_layout()
    return fig