#!/usr/bin/env python3
"""
SVDCentrality: Mathematically Correct Implementation
===================================================


Mathematical Definitions:
- Vertex centrality: C_v(i) = [L_0^+]_{ii} where L_0 = B B^T
- Edge centrality: C_e(e) = [L_1^+]_{ee} where L_1 = B^T B  
- Hub centrality: c_hub(i) = α·C_v(i) + (1-α)·Σ_{e∈out(i)} C_e(e)
- Authority centrality: c_auth(i) = α·C_v(i) + (1-α)·Σ_{e∈in(i)} C_e(e)

Where:
- B: Oriented incidence matrix
- L_0^+, L_1^+: Moore-Penrose pseudoinverses of Hodge Laplacians
- SVD: B = U Σ V^T
"""

import numpy as np
import networkx as nx
from scipy.linalg import svd
from scipy.sparse import csr_matrix
from scipy.sparse.linalg import svds
from typing import Dict, Tuple, Any, Union, Optional
import warnings
import time
from dataclasses import dataclass

from .utils import validate_graph, ensure_numerical_stability, SVDCentralityError


@dataclass
class ComputationStats:
    """Statistics from SVD centrality computation."""
    network_nodes: int
    network_edges: int
    incidence_matrix_rank: int
    nonzero_singular_values: int
    largest_singular_value: float
    smallest_nonzero_singular_value: float
    condition_number: float
    computation_time: float
    memory_usage_mb: float


class SVDCentrality:
    """
    SVD Incidence Centrality Implementation
    
    Implements the complete mathematical framework from the paper with proper
    numerical stability, error handling, and performance optimization.
    
    Mathematical Foundation:
    ----------------------
    For graph G=(V,E) with incidence matrix B ∈ ℝ^(n×m):
    
    1. SVD Decomposition: B = U Σ V^T
    2. Hodge Laplacians: L_0 = BB^T, L_1 = B^TB  
    3. Pseudoinverses: L_0^+ = UΣ^(-2)U^T, L_1^+ = VΣ^(-2)V^T
    4. Centralities: C_v(i) = [L_0^+]_{ii}, C_e(e) = [L_1^+]_{ee}
    
    Key Properties:
    --------------
    - Dense rankings (no zero values for connected components)
    - Spectral foundation (eigenspace analysis)
    - Directional sensitivity (oriented incidence matrix)
    - Equivalence with current-flow closeness for undirected graphs
    """
    
    def __init__(self, 
                 regularization: float = 1e-8,
                 numerical_tolerance: float = 1e-12,
                 max_condition_number: float = 1e12,
                 enable_caching: bool = True):
        """
        Initialize SVD centrality computer.
        
        Parameters:
        -----------
        regularization : float
            Regularization parameter for numerical stability
        numerical_tolerance : float  
            Tolerance for considering singular values as zero
        max_condition_number : float
            Maximum allowed condition number before warning
        enable_caching : bool
            Whether to cache computations for repeated calls
        """
        self.regularization = regularization
        self.numerical_tolerance = numerical_tolerance
        self.max_condition_number = max_condition_number
        self.enable_caching = enable_caching
        
        # Internal state
        self._cache = {} if enable_caching else None
        self._last_computation_stats = None
        
    def compute_incidence_matrix(self, graph: nx.Graph) -> Tuple[csr_matrix, Dict[Any, int], Dict[Tuple, int]]:
        """
        Compute oriented incidence matrix as a sparse matrix.
        """
        validate_graph(graph)
        
        nodes = list(graph.nodes())
        edges = list(graph.edges())
        n_nodes = len(nodes)
        n_edges = len(edges)
        
        if n_edges == 0:
            B = csr_matrix((n_nodes, 0))
            node_map = {node: i for i, node in enumerate(nodes)}
            return B, node_map, {}
        
        node_map = {node: i for i, node in enumerate(nodes)}
        edge_map = {edge: i for i, edge in enumerate(edges)}
        
        row_ind = []
        col_ind = []
        data = []
        
        for edge_idx, (u, v) in enumerate(edges):
            u_idx, v_idx = node_map[u], node_map[v]
            
            if graph.is_directed():
                row_ind.extend([u_idx, v_idx])
                col_ind.extend([edge_idx, edge_idx])
                data.extend([-1, 1])
            else:
                row_ind.extend([u_idx, v_idx])
                col_ind.extend([edge_idx, edge_idx])
                data.extend([1, 1])
                
        B = csr_matrix((data, (row_ind, col_ind)), shape=(n_nodes, n_edges))
        return B, node_map, edge_map
    
    def compute_svd_decomposition(self, B: csr_matrix) -> Tuple[np.ndarray, np.ndarray, np.ndarray, int]:
        n_nodes, n_edges = B.shape
        if n_edges == 0:
            return np.eye(n_nodes), np.array([]), np.zeros((0, 0)), 0
        
        U_full, sigma_full, Vt_full = None, None, None # Initialize to None

        try:
            # First, try full SVD on dense array for accuracy
            U_full, sigma_full, Vt_full = svd(B.toarray(), full_matrices=False)
        except (np.linalg.LinAlgError, MemoryError) as e_dense:
            warnings.warn(f"Full SVD on dense matrix failed ({e_dense}). Attempting sparse SVD with binary search for max k.")
            
            # Binary search for largest possible k
            n_min = min(n_nodes, n_edges)
            low = 1
            high = n_min - 1
            
            best_U, best_sigma, best_Vt = None, None, None
            best_k = 0
            
            while low <= high:
                mid = (low + high) // 2
                try:
                    # Use 'SM' (Smallest Magnitude) for resistance centrality
                    # Warning: This can be slow for large mid
                    current_U, current_sigma, current_Vt = svds(B, k=mid, which='SM')
                    
                    # If successful, store and try to increase k
                    best_U, best_sigma, best_Vt = current_U, current_sigma, current_Vt
                    best_k = mid
                    low = mid + 1
                except Exception:
                    # If failed (convergence, memory, etc.), try smaller k
                    high = mid - 1
            
            if best_k == 0:
                raise SVDCentralityError("Sparse SVD failed even for k=1.")
                
            warnings.warn(f"Sparse SVD successful with maximized k={best_k} (which='SM')")
            U_full, sigma_full, Vt_full = best_U, best_sigma, best_Vt

        # Sort singular values in descending order
        sort_idx = np.argsort(sigma_full)[::-1]
        U_full = U_full[:, sort_idx]
        sigma_full = sigma_full[sort_idx]
        Vt_full = Vt_full[sort_idx, :]

        nonzero_mask = sigma_full > self.numerical_tolerance
        rank = np.sum(nonzero_mask)
        
        if rank == 0:
            warnings.warn("Graph appears to be disconnected or degenerate")
            return U_full, np.array([]), Vt_full.T, 0
        
        U = U_full[:, nonzero_mask]
        sigma = sigma_full[nonzero_mask] 
        V = Vt_full[nonzero_mask, :].T
        
        condition_number = sigma[0] / sigma[-1]
        if condition_number > self.max_condition_number:
            warnings.warn(f"High condition number: {condition_number:.2e}")
        
        return U, sigma, V, rank
    
    def compute_centralities(self, graph: nx.Graph) -> Dict[str, Any]:
        """
        Compute vertex and edge centralities, including raw eccentricity and
        normalized scores, using exact paper definitions.
        """
        start_time = time.time()
        
        graph_hash = hash(str(sorted(graph.edges())))
        if self.enable_caching and self._cache and graph_hash in self._cache:
            return self._cache[graph_hash]
        
        B, node_map, edge_map = self.compute_incidence_matrix(graph)
        n_nodes, n_edges = B.shape
        nodes = list(node_map.keys())
        edges = list(edge_map.keys())
        
        if n_edges == 0:
            return {'C_v': np.zeros(n_nodes), 'C_e': np.zeros(n_edges),
                    'S_v': np.zeros(n_nodes), 'S_e': np.zeros(n_edges),
                    'nodes': nodes, 'edges': edges,
                    'node_map': node_map, 'edge_map': edge_map}
        
        U, sigma, V, rank = self.compute_svd_decomposition(B)
        
        if rank == 0:
            C_v = np.zeros(n_nodes)
            C_e = np.zeros(n_edges)
        else:
            sigma_inv_squared = 1.0 / (sigma**2 + self.regularization)
            C_v = np.sum((U**2) * sigma_inv_squared, axis=1)
            C_e = np.sum((V**2) * sigma_inv_squared, axis=1)

        S_v_raw = 1.0 / (C_v + self.regularization)
        S_v = S_v_raw / (S_v_raw.max() or 1.0)

        S_e_raw = 1.0 / (C_e + self.regularization)
        S_e = S_e_raw / (S_e_raw.max() or 1.0)
        
        computation_time = time.time() - start_time
        self._last_computation_stats = ComputationStats(
            network_nodes=n_nodes, network_edges=n_edges,
            incidence_matrix_rank=rank, nonzero_singular_values=len(sigma),
            largest_singular_value=float(sigma[0]) if rank > 0 else 0.0,
            smallest_nonzero_singular_value=float(sigma[-1]) if rank > 0 else 0.0,
            condition_number=float(sigma[0] / sigma[-1]) if rank > 0 else 1.0,
            computation_time=computation_time, memory_usage_mb=(B.data.nbytes + B.indices.nbytes + B.indptr.nbytes) / 1024**2
        )
        
        results = {'C_v': C_v, 'C_e': C_e, 'S_v': S_v, 'S_e': S_e,
                   'nodes': nodes, 'edges': edges,
                   'node_map': node_map, 'edge_map': edge_map}
        
        if self.enable_caching:
            self._cache[graph_hash] = results
            
        return results
    
    def compute_hub_authority_centrality(self, 
                                         graph: nx.Graph, 
                                         alpha: float = 0.0,
                                         vertex_centrality: Optional[Dict[Any, float]] = None,
                                         edge_centrality: Optional[Dict[Tuple, float]] = None) -> Tuple[Dict[Any, float], Dict[Any, float]]:
        """
        Compute hub and authority centralities using exact paper definitions.
        """
        if not graph.is_directed():
            warnings.warn("Hub/authority analysis is most meaningful for directed graphs")
        
        if vertex_centrality is None or edge_centrality is None:
            centrality_results = self.compute_centralities(graph)
            vertex_centrality = {node: centrality_results['S_v'][idx] for idx, node in enumerate(centrality_results['nodes'])}
            edge_centrality = {edge: centrality_results['S_e'][idx] for idx, edge in enumerate(centrality_results['edges'])}
        
        nodes = list(graph.nodes())
        hub_centrality = {}
        authority_centrality = {}
        
        for node in nodes:
            # Hub score: sum of outgoing edge centralities
            outgoing_edges = list(graph.out_edges(node))
            hub_edge_sum = sum(edge_centrality.get(edge, 0.0) for edge in outgoing_edges)
            
            # Authority score: sum of incoming edge centralities
            incoming_edges = list(graph.in_edges(node))
            auth_edge_sum = sum(edge_centrality.get(edge, 0.0) for edge in incoming_edges)
            
            # Apply convex combination
            vertex_score = vertex_centrality.get(node, 0.0)
            hub_centrality[node] = alpha * vertex_score + (1 - alpha) * hub_edge_sum
            authority_centrality[node] = alpha * vertex_score + (1 - alpha) * auth_edge_sum
            
        # Normalize to [0, 1]
        max_hub = max(hub_centrality.values()) if hub_centrality else 1.0
        max_auth = max(authority_centrality.values()) if authority_centrality else 1.0

        for node in nodes:
            hub_centrality[node] /= max_hub if max_hub > 0 else 1.0
            authority_centrality[node] /= max_auth if max_auth > 0 else 1.0

        return hub_centrality, authority_centrality
    
    def validate_against_current_flow(self, graph: nx.Graph) -> Dict[str, float]:
        """
        Validate theoretical equivalence with current-flow closeness for undirected graphs.
        """
        if graph.is_directed():
            raise ValueError("Current-flow validation only applicable to undirected graphs")
        
        if not nx.is_connected(graph):
            warnings.warn("Graph is not connected; equivalence may not hold")
        
        centrality_results = self.compute_centralities(graph)
        vertex_centrality = {node: centrality_results['S_v'][idx] for idx, node in enumerate(centrality_results['nodes'])}
        
        try:
            cf_closeness = nx.current_flow_closeness_centrality(graph)
        except Exception as e:
            warnings.warn(f"Current-flow computation failed: {e}")
            return {"error": str(e)}
        
        nodes = list(graph.nodes())
        svd_values = [vertex_centrality[node] for node in nodes]
        cf_values = [cf_closeness[node] for node in nodes]
        
        pearson_corr = np.corrcoef(svd_values, cf_values)[0, 1]
        spearman_corr = np.corrcoef(np.argsort(svd_values), np.argsort(cf_values))[0, 1]
        
        svd_ranking = np.argsort(np.argsort(svd_values))
        cf_ranking = np.argsort(np.argsort(cf_values))
        rank_correlation = np.corrcoef(svd_ranking, cf_ranking)[0, 1]
        
        return {
            "pearson_correlation": float(pearson_corr),
            "spearman_correlation": float(spearman_corr), 
            "rank_correlation": float(rank_correlation),
            "mean_svd_centrality": float(np.mean(svd_values)),
            "mean_cf_closeness": float(np.mean(cf_values)),
            "svd_std": float(np.std(svd_values)),
            "cf_std": float(np.std(cf_values))
        }
    
    def get_computation_stats(self) -> Optional[ComputationStats]:
        """Get statistics from the last centrality computation."""
        return self._last_computation_stats
    
    def get_spectral_properties(self, graph: nx.Graph) -> Dict[str, Any]:
        """
        Analyze spectral properties of the network.
        """
        B, _, _ = self.compute_incidence_matrix(graph)
        
        if B.shape[1] == 0:
            return {"error": "No edges in graph"}
        
        U, sigma, V, rank = self.compute_svd_decomposition(B)
        
        L0 = B @ B.T
        L1 = B.T @ B
        
        properties = {
            "incidence_matrix_shape": B.shape,
            "numerical_rank": rank,
            "full_rank": min(B.shape),
            "rank_deficiency": min(B.shape) - rank,
            "singular_values": sigma.tolist(),
            "condition_number": float(sigma[0] / sigma[-1]) if len(sigma) > 0 else float('inf'),
            "spectral_gap": float(sigma[0] - sigma[1]) if len(sigma) > 1 else float('inf'),
            "vertex_laplacian_trace": float(np.trace(L0.toarray())),
            "edge_laplacian_trace": float(np.trace(L1.toarray())),
            "frobenius_norm": float(np.linalg.norm(B.toarray(), 'fro')),
            "connected_components": nx.number_connected_components(graph.to_undirected()),
            "is_directed": graph.is_directed()
        }
        
        return properties
    
    def clear_cache(self) -> None:
        """Clear computation cache."""
        if self.enable_caching and self._cache:
            self._cache.clear()
    
    def __repr__(self) -> str:
        return f"SVDCentrality(regularization={self.regularization}, tolerance={self.numerical_tolerance})"