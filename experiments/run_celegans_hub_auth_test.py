#!/usr/bin/env python3
"""
Run the C. elegans Hub/Authority Test
======================================

This script runs a special test on the C. elegans network to visualize
SVD Hub/Authority centralities alongside edge centralities.
"""

import networkx as nx
import numpy as np
import matplotlib.pyplot as plt
import os

# Import refactored project components
from gemisvd.svd_centrality.data_loader import load_real_world_network
from gemisvd.svd_centrality.core.svd_centrality import SVDCentrality
from gemisvd.visualizations.main_visualizer import visualize_celegans_hub_auth_test

def run_test():
    """
    Main function to run the C. elegans test and generate visualizations.
    """
    print("--- Running C. elegans Hub/Authority Test ---")

    # 1. Load Data
    print("Step 1: Loading C. elegans network...")
    G = load_real_world_network('c_elegans')
    G.remove_edges_from(nx.selfloop_edges(G)) # Remove self-loops
    
    # Get largest weakly connected component
    largest_cc = max(nx.weakly_connected_components(G), key=len)
    G = G.subgraph(largest_cc).copy()
    
    pos = nx.spring_layout(G, seed=42)

    # 2. Compute SVD Centralities
    print("Step 2: Computing SVD centralities...")
    svd_computer = SVDCentrality()
    svd_results = svd_computer.compute_centralities(G)
    hub_svd, auth_svd = svd_computer.compute_hub_authority_centrality(G, alpha=0.1)
    
    # 3. Compute Baseline Centralities
    print("Step 3: Computing Betweenness and In-Degree centralities...")
    edge_betweenness = nx.edge_betweenness_centrality(G, normalized=True)
    node_betweenness = nx.betweenness_centrality(G, normalized=True)
    in_degree = {node: G.in_degree(node) for node in G.nodes()}
    
    # Align results with the ordered nodes/edges from svd_results
    nodes = svd_results['nodes']
    edges = svd_results['edges']
    
    aligned_edge_betweenness = np.array([edge_betweenness.get(edge, 0) for edge in edges])
    aligned_node_betweenness = np.array([node_betweenness.get(node, 0) for node in nodes])
    aligned_hub_svd = np.array([hub_svd.get(node, 0) for node in nodes])
    aligned_auth_svd = np.array([auth_svd.get(node, 0) for node in nodes])
    aligned_in_degree = np.array([in_degree.get(node, 0) for node in nodes])

    # 4. Visualization
    print("Step 4: Generating and saving visualization...")
    
    fig = visualize_celegans_hub_auth_test(
        G,
        pos,
        svd_results,
        {
            'hub': aligned_hub_svd,
            'authority': aligned_auth_svd,
            'node_betweenness': aligned_node_betweenness,
            'edge_betweenness': aligned_edge_betweenness,
            'in_degree': aligned_in_degree
        }
    )
    
    output_path_base = "gemisvd/outputs/figures/celegans_hub_auth_test"
    fig.savefig(f"{output_path_base}.png", dpi=300, bbox_inches='tight')
    fig.savefig(f"gemisvd/outputs/figures/svg/{os.path.basename(output_path_base)}.svg", bbox_inches='tight')
    fig.savefig(f"gemisvd/outputs/figures/pdf/{os.path.basename(output_path_base)}.pdf", bbox_inches='tight')
    print(f"   > Visualization saved to {output_path_base}.[png,svg,pdf]")

    print("\n--- C. elegans Hub/Authority Test Complete ---")

if __name__ == '__main__':
    # Ensure the output directory exists
    output_dir = "gemisvd/outputs/figures"
    os.makedirs(output_dir, exist_ok=True)
    os.makedirs(os.path.join(output_dir, "svg"), exist_ok=True)
    os.makedirs(os.path.join(output_dir, "pdf"), exist_ok=True)
    
    run_test()
