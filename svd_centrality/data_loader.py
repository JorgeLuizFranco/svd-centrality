#!/usr/bin/env python3
"""
Data Loader for SVD Centrality Project
=======================================

This module centralizes loading for all datasets used in the paper, including:
- Real-world networks (from data/)
- Hypergraphs (from hypergraphDatasets/)
- Standard NetworkX benchmark graphs
- Synthetic graphs (e.g., hub-authority grid)
"""

import networkx as nx
import numpy as np
import json
from pathlib import Path

DATA_ROOT = Path(__file__).parent.parent.parent / 'SVD_incidence_centrality'

# --- NetworkX Benchmark Graphs ---

def load_benchmark_graph(name: str) -> nx.Graph:
    """Loads a standard NetworkX benchmark graph."""
    if name == 'karate':
        return nx.karate_club_graph()
    elif name == 'les_miserables':
        return nx.les_miserables_graph()
    elif name == 'florentine_families':
        return nx.florentine_families_graph()
    elif name == 'davis_southern_women':
        return nx.davis_southern_women_graph()
    else:
        raise ValueError(f"Unknown benchmark graph: {name}")

# --- Synthetic Graph Generators ---

def create_hub_authority_grid(rows=4, cols=4):
    """Builds a directed grid graph with specific hub and authority connections."""
    G = nx.DiGraph()
    positions = {(i, j): (j, -i) for i in range(rows) for j in range(cols)}
    G.add_nodes_from(positions.keys())
    nx.set_node_attributes(G, positions, 'pos')

    hub_node = (rows // 2, cols // 2)
    authority_node = (rows // 2, cols // 2 + 1)
    
    # Deterministic edge creation based on a fixed seed
    rng = np.random.default_rng(42)

    for i in range(rows):
        for j in range(cols):
            node = (i, j)
            # Connect to right neighbor
            if j + 1 < cols:
                G.add_edge(node, (i, j + 1)) if rng.random() > 0.5 else G.add_edge((i, j + 1), node)
            # Connect to bottom neighbor
            if i + 1 < rows:
                G.add_edge(node, (i + 1, j)) if rng.random() > 0.5 else G.add_edge((i + 1, j), node)

    # Emphasize hub/authority roles
    for node in G.nodes():
        if node != hub_node:
            G.add_edge(hub_node, node) # Hub points to others
        if node != authority_node:
            G.add_edge(node, authority_node) # Others point to authority

    return G

# --- Dutch School Network (Real Data) ---

def load_real_dutch_school_network(wave=3) -> nx.DiGraph:
    """
    Loads the REAL Dutch school friendship network from .dat files.
    
    Data source: T. Snijders, G. van de Bunt, and C. Steglich (2010).
    """
    data_dir = DATA_ROOT / "network_data/klas12b"
    filename = f"klas12b-net-{wave}.dat"
    filepath = data_dir / filename
    
    if not filepath.exists():
        raise FileNotFoundError(f"Dutch school data not found: {filepath}")
        
    # Read the adjacency matrix
    matrix = np.loadtxt(filepath, dtype=int)
    
    # Create directed graph
    G = nx.DiGraph()
    n_nodes = matrix.shape[0]
    
    # Add nodes (1-indexed as in original data)
    G.add_nodes_from(range(1, n_nodes + 1))
    
    # Add edges based on adjacency matrix: only value 1 indicates friendship
    for i in range(n_nodes):
        for j in range(n_nodes):
            if matrix[i, j] == 1:
                G.add_edge(i + 1, j + 1)
                
    # Remove isolated nodes to ensure stable spectral centrality
    G.remove_nodes_from(list(nx.isolates(G)))
                
    return G

# --- Real-world Networks from Files ---

def _load_edge_list(path: Path, directed: bool = True) -> nx.Graph:
    """Helper to load a graph from an edge list file."""
    G = nx.DiGraph() if directed else nx.Graph()
    if not path.exists():
        print(f"Warning: Data file not found at {path}")
        return G
    with open(path, 'r') as f:
        for line in f:
            line = line.strip()
            if not line or line.startswith('%') or line.startswith('#'):
                continue
            try:
                u, v = map(int, line.split()[:2])
                G.add_edge(u, v)
            except ValueError:
                continue
    
    # Return the largest weakly connected component for directed graphs
    if directed and G.number_of_nodes() > 0:
        largest_cc = max(nx.weakly_connected_components(G), key=len)
        return G.subgraph(largest_cc).copy()
    elif not directed and G.number_of_nodes() > 0:
        largest_cc = max(nx.connected_components(G), key=len)
        return G.subgraph(largest_cc).copy()
    return G

def load_real_world_network(name: str) -> nx.Graph:
    """Loads a real-world network from the data/ directory."""
    if name == 'c_elegans':
        path = DATA_ROOT / 'data/biological/bio-celegans-dir.edges'
        return _load_edge_list(path, directed=True)
    elif name == 'yeast':
        path = DATA_ROOT / 'data/biological/bio-yeast-protein-inter.edges'
        return _load_edge_list(path, directed=True)
    elif name == 'openflights':
        path = DATA_ROOT / 'data/transportation/inf-openflights.edges'
        return _load_edge_list(path, directed=True)
    elif name == 'euroroad':
        path = DATA_ROOT / 'data/transportation/inf-euroroad.edges'
        return _load_edge_list(path, directed=True)
    elif name == 'powergrid':
        path = DATA_ROOT / 'data/power/inf-power.mtx'
        return _load_edge_list(path, directed=False) # Powergrid is undirected
    else:
        raise ValueError(f"Unknown real-world network: {name}")

# --- Hypergraph Datasets ---
class ManualHypergraph:
    """A simple, independent hypergraph implementation."""
    def __init__(self):
        self.nodes = set()
        self.edges = {}
        self.node_edges = defaultdict(set)
    
    def add_edge(self, nodes, edge_id=None):
        if edge_id is None:
            edge_id = len(self.edges)
        node_set = set(nodes)
        self.edges[edge_id] = node_set
        for node in node_set:
            self.nodes.add(node)
            self.node_edges[node].add(edge_id)

def load_hypergraph_from_json(name: str) -> tuple[ManualHypergraph, dict]:
    """Loads a hypergraph dataset from the hypergraphDatasets/ directory."""
    filepath = DATA_ROOT / 'hypergraphDatasets' / f"{name}.json"
    if not filepath.exists():
        raise FileNotFoundError(f"Hypergraph dataset not found: {filepath}")

    with open(filepath, 'r') as f:
        data = json.load(f)
    
    H = ManualHypergraph()
    if 'node-data' in data:
        for node_id in data['node-data']:
            H.nodes.add(node_id)
    
    if 'edge-dict' in data:
        for edge_id, node_list in data['edge-dict'].items():
            if node_list:
                H.add_edge(node_list, edge_id)
    
    metadata = data.get('hypergraph-data', {})
    return H, metadata
