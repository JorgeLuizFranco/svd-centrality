#!/usr/bin/env python3
"""
Run the Dutch School Network Analysis Experiment
================================================

This script analyzes the Dutch school friendship network, computing and comparing
SVD-based node, edge, hub, and authority centralities with traditional measures.
It generates visualizations similar to those found in the paper's "Dutch School" section.
"""

import networkx as nx
import numpy as np
import matplotlib.pyplot as plt
import os

# Import refactored project components
from gemisvd.svd_centrality.data_loader import load_real_dutch_school_network
from gemisvd.svd_centrality.core.svd_centrality import SVDCentrality
from gemisvd.visualizations.main_visualizer import visualize_dutch_school_analysis

import argparse

def run_experiment():
    parser = argparse.ArgumentParser(description='Run Dutch School Network Analysis')
    parser.add_argument('--wave', type=int, default=3, help='Wave number (1-4)')
    args = parser.parse_args()
    
    wave = args.wave
    print(f"--- Running Dutch School Network Analysis (REAL DATA, WAVE {wave}) ---")

    # 1. Load Data
    print(f"Step 1: Loading REAL Dutch school network (Wave {wave})...")
    G = load_real_dutch_school_network(wave=wave)
    print(f"   Graph loaded: {G.number_of_nodes()} nodes, {G.number_of_edges()} edges")
    pos = nx.spring_layout(G, k=1.5, iterations=50, seed=42) # For consistent visualization

    # 2. Compute SVD Centralities
    print("Step 2: Computing SVD centralities using the core library...")
    svd_computer = SVDCentrality()
    svd_results = svd_computer.compute_centralities(G)
    
    # Compute Hub and Authority Centralities
    svd_hub_agg, svd_authority_agg = svd_computer.compute_hub_authority_centrality(G, alpha=0.0)

    # 3. Compute Baseline Centralities
    print("Step 3: Computing baseline centralities...")
    
    # Node Baselines
    node_betweenness = nx.betweenness_centrality(G, normalized=True)
    pagerank = nx.pagerank(G)
    
    # Edge Baselines
    edge_betweenness = nx.edge_betweenness_centrality(G, normalized=True)
    
    # Edge Degree Product: d_out(u) * d_in(v) for edge (u,v)
    edge_degree_product = {}
    for u, v in G.edges():
        prod = G.out_degree(u) * G.in_degree(v)
        edge_degree_product[(u, v)] = prod

    # Hub/Auth Baselines
    # hits() can sometimes fail to converge, use a try-except
    try:
        hits_hub, hits_authority = nx.hits(G, max_iter=1000, tol=1e-6)
    except:
        print("Warning: HITS failed to converge. Using zeros.")
        hits_hub = {n: 0.0 for n in G.nodes()}
        hits_authority = {n: 0.0 for n in G.nodes()}
        
    in_degree = dict(G.in_degree())
    out_degree = dict(G.out_degree())

    # --- Alignment ---
    nodes = svd_results['nodes']
    edges = svd_results['edges']
    
    # Convert dicts to aligned arrays
    data = {
        'svd_node': svd_results['S_v'],
        'bet_node': np.array([node_betweenness.get(n, 0) for n in nodes]),
        'pagerank': np.array([pagerank.get(n, 0) for n in nodes]),
        
        'svd_edge': svd_results['S_e'],
        'bet_edge': np.array([edge_betweenness.get(e, 0) for e in edges]),
        'edge_deg': np.array([edge_degree_product.get(e, 0) for e in edges]),
        
        'svd_auth': np.array([svd_authority_agg.get(n, 0) for n in nodes]),
        'hits_auth': np.array([hits_authority.get(n, 0) for n in nodes]),
        'in_degree': np.array([in_degree.get(n, 0) for n in nodes]),
        
        'svd_hub': np.array([svd_hub_agg.get(n, 0) for n in nodes]),
        'hits_hub': np.array([hits_hub.get(n, 0) for n in nodes]),
        'out_degree': np.array([out_degree.get(n, 0) for n in nodes])
    }

    # 4. Visualization
    print("Step 4: Generating and saving split visualizations...")
    
    fig_gen, fig_dir = visualize_dutch_school_analysis(G, pos, data)

    output_dir = "gemisvd/outputs/figures"
    
    # Save General Figure
    base_name_gen = "dutch_school_general"
    output_path_gen = os.path.join(output_dir, base_name_gen)
    fig_gen.savefig(f"{output_path_gen}.png", dpi=300, bbox_inches='tight')
    fig_gen.savefig(f"gemisvd/outputs/figures/svg/{base_name_gen}.svg", bbox_inches='tight')
    fig_gen.savefig(f"gemisvd/outputs/figures/pdf/{base_name_gen}.pdf", bbox_inches='tight')
    
    # Save Directional Figure
    base_name_dir = "dutch_school_directional"
    output_path_dir = os.path.join(output_dir, base_name_dir)
    fig_dir.savefig(f"{output_path_dir}.png", dpi=300, bbox_inches='tight')
    fig_dir.savefig(f"gemisvd/outputs/figures/svg/{base_name_dir}.svg", bbox_inches='tight')
    fig_dir.savefig(f"gemisvd/outputs/figures/pdf/{base_name_dir}.pdf", bbox_inches='tight')
    
    print(f"   > Visualizations saved: {base_name_gen}, {base_name_dir}")

    # 5. Print Top Nodes for Verification (Table Data)
    print("\n--- TABLE 1 DATA: Node Centrality ---")
    def print_top(label, values):
        indices = np.argsort(values)[::-1][:5]
        print(f"{label}: ", [(nodes[i], f"{values[i]:.3f}") for i in indices])

    print_top("SVD Node", data['svd_node'])
    print_top("Betweenness", data['bet_node'])
    print_top("PageRank", data['pagerank'])
    
    print("\n--- TABLE 2 DATA: Directional Roles ---")
    print_top("SVD Authority", data['svd_auth'])
    print_top("HITS Authority", data['hits_auth'])
    print_top("SVD Hub", data['svd_hub'])
    print_top("HITS Hub", data['hits_hub'])

    print("\n--- Dutch School Network Analysis Complete ---")

if __name__ == '__main__':
    # Ensure the output directory exists
    import os
    os.makedirs("gemisvd/outputs/figures", exist_ok=True)
    
    run_experiment()
