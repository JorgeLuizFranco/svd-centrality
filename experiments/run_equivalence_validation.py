#!/usr/bin/env python3
"""
Run the Equivalence Validation Experiment
=========================================

This script validates the theoretical equivalence between SVD centrality and
current-flow closeness centrality on the undirected Zachary's Karate Club graph,
and generates correlation plots for other benchmark networks.
"""

import networkx as nx
import numpy as np
import matplotlib.pyplot as plt
import os

# Import the refactored project components
from gemisvd.svd_centrality.data_loader import load_benchmark_graph
from gemisvd.svd_centrality.core.svd_centrality import SVDCentrality
from gemisvd.visualizations.main_visualizer import visualize_equivalence_validation, visualize_equivalence_correlations

def run_experiment():
    """
    Main function to run the equivalence validation and generate visualizations.
    """
    print("--- Running Equivalence Validation Experiment ---")
    
    output_dir = "gemisvd/outputs/figures"
    os.makedirs(f"{output_dir}/svg", exist_ok=True)
    os.makedirs(f"{output_dir}/pdf", exist_ok=True)

    # --- Define Networks ---
    networks = []

    # 1. Zachary's Karate Club
    print("Loading Zachary's Karate Club...")
    G_karate = load_benchmark_graph('karate')
    pos_karate = nx.spring_layout(G_karate, seed=42)
    networks.append({
        'name': "Zachary's Karate Club",
        'graph': G_karate,
        'pos': pos_karate,
        'type': 'karate'
    })

    # 2. Random ER(15, 0.3)
    print("Generating Random ER(15, 0.3)...")
    G_er = nx.erdos_renyi_graph(15, 0.3, seed=42)
    # Ensure connectivity for valid comparison
    while not nx.is_connected(G_er):
        G_er = nx.erdos_renyi_graph(15, 0.3, seed=np.random.randint(1000))
    pos_er = nx.spring_layout(G_er, seed=42)
    networks.append({
        'name': "Random ER(15,0.3)",
        'graph': G_er,
        'pos': pos_er,
        'type': 'er'
    })

    # 3. Path P8
    print("Generating Path P8...")
    G_path = nx.path_graph(8)
    pos_path = nx.kamada_kawai_layout(G_path)
    networks.append({
        'name': "Path P8",
        'graph': G_path,
        'pos': pos_path,
        'type': 'path'
    })

    # --- Processing ---
    svd_computer = SVDCentrality()
    correlation_results = []

    for net in networks:
        print(f"Processing {net['name']}...")
        G = net['graph']
        
        # Compute SVD Centrality
        svd_results = svd_computer.compute_centralities(G)
        S_v = svd_results['S_v']
        nodes = svd_results['nodes']

        # Compute Baseline Centrality
        # Use Current-Flow Closeness for connected undirected graphs, else standard Closeness
        is_directed = G.is_directed()
        is_connected = nx.is_strongly_connected(G) if is_directed else nx.is_connected(G)
        
        baseline_vals = []
        baseline_name = "Closeness Centrality"

        if not is_directed and is_connected:
             try:
                 cf_map = nx.current_flow_closeness_centrality(G)
                 baseline_vals = np.array([cf_map.get(n, 0) for n in nodes])
                 baseline_name = "Current-Flow Closeness"
             except:
                 # Fallback if CF fails (e.g. disconnected, though we checked)
                 cl_map = nx.closeness_centrality(G)
                 baseline_vals = np.array([cl_map.get(n, 0) for n in nodes])
        else:
             cl_map = nx.closeness_centrality(G)
             baseline_vals = np.array([cl_map.get(n, 0) for n in nodes])

        # Store for combined plot
        correlation_results.append({
            'name': net['name'],
            'svd': S_v,
            'closeness': baseline_vals,
            'baseline_name': baseline_name
        })

        # Calculate and print correlation for paper table
        from scipy.stats import pearsonr
        corr, _ = pearsonr(S_v, baseline_vals)
        print(f"RESULTS FOR {net['name']}:")
        print(f"  - Nodes: {G.number_of_nodes()}")
        print(f"  - Baseline: {baseline_name}")
        print(f"  - Pearson r: {corr:.4f}")
        print("-" * 30)

        # Generate Karate-Specific Visualization
        if net['type'] == 'karate':
            print(f"   Generating visualization for {net['name']}...")
            fig = visualize_equivalence_validation(
                G, net['pos'], svd_results, baseline_vals
            )
            
            output_path = f"{output_dir}/zacharys_karate_club_comparison"
            fig.savefig(f"{output_path}.png", dpi=300, bbox_inches='tight')
            fig.savefig(f"{output_dir}/svg/zacharys_karate_club_comparison.svg", bbox_inches='tight')
            fig.savefig(f"{output_dir}/pdf/zacharys_karate_club_comparison.pdf", bbox_inches='tight')
            print(f"   Saved to {output_path}.png")
            plt.close(fig)

    # --- Combined Correlation Plot ---
    print("Generating combined correlation plots...")
    fig_corr = visualize_equivalence_correlations(correlation_results)
    
    corr_path = f"{output_dir}/svd_vs_closeness_comparison"
    fig_corr.savefig(f"{corr_path}.png", dpi=300, bbox_inches='tight')
    fig_corr.savefig(f"{output_dir}/svg/svd_vs_closeness_comparison.svg", bbox_inches='tight')
    fig_corr.savefig(f"{output_dir}/pdf/svd_vs_closeness_comparison.pdf", bbox_inches='tight')
    print(f"   Saved to {corr_path}.png")
    plt.close(fig_corr)

    print("\n--- Equivalence Validation Complete ---")

if __name__ == '__main__':
    run_experiment()
