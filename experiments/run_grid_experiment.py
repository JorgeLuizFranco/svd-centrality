#!/usr/bin/env python3
"""
Run the Controlled Grid Experiment
==================================

This script generates a synthetic 4x4 grid with a pre-defined hub and authority,
computes all SVD and baseline centralities, and saves the comparative
visualizations.
"""

import networkx as nx
import numpy as np
import matplotlib.pyplot as plt
import os

# Import the refactored project components
from gemisvd.svd_centrality.data_loader import create_hub_authority_grid
from gemisvd.svd_centrality.core.svd_centrality import SVDCentrality
from gemisvd.visualizations.main_visualizer import visualize_grid_experiment, visualize_grid_hub_authority, visualize_grid_raw_svd_vs_betweenness

def run_experiment():
    """
    Main function to run the grid experiment and generate visualizations.
    """
    print("--- Running Controlled Grid Experiment ---")

    # 1. Load Data
    print("Step 1: Creating the synthetic 4x4 hub-authority grid...")
    G = create_hub_authority_grid(rows=4, cols=4)
    pos = nx.get_node_attributes(G, 'pos')

    # 2. Compute SVD Centralities
    print("Step 2: Computing SVD centralities using the core library...")
    svd_computer = SVDCentrality()
    svd_results = svd_computer.compute_centralities(G)
    hub_svd, auth_svd = svd_computer.compute_hub_authority_centrality(G, alpha=0.0)

    # 3. Compute Baseline Centralities
    print("Step 3: Computing baseline (Betweenness) centralities...")
    node_betweenness = nx.betweenness_centrality(G, normalized=True)
    edge_betweenness = nx.edge_betweenness_centrality(G, normalized=True)

    # Align results with the ordered nodes/edges from svd_results
    nodes = svd_results['nodes']
    edges = svd_results['edges']
    
    aligned_node_betweenness = np.array([node_betweenness.get(node, 0) for node in nodes])
    aligned_edge_betweenness = np.array([edge_betweenness.get(edge, 0) for edge in edges])
    aligned_hub_svd = np.array([hub_svd.get(node, 0) for node in nodes])
    aligned_auth_svd = np.array([auth_svd.get(node, 0) for node in nodes])

    # 4. Visualization
    print("Step 4: Generating and saving visualizations...")
    
    fig1 = visualize_grid_experiment(
        G,
        pos,
        svd_results,
        {'node': aligned_node_betweenness, 'edge': aligned_edge_betweenness}
    )
    output_path1_base = "gemisvd/outputs/figures/grid_experiment_comparison"
    fig1.savefig(f"{output_path1_base}.png", dpi=300, bbox_inches='tight')
    fig1.savefig(f"gemisvd/outputs/figures/svg/{os.path.basename(output_path1_base)}.svg", bbox_inches='tight')
    fig1.savefig(f"gemisvd/outputs/figures/pdf/{os.path.basename(output_path1_base)}.pdf", bbox_inches='tight')
    print(f"   > Visualizations saved to {output_path1_base}.[png,svg,pdf]")

    fig2 = visualize_grid_hub_authority(
        G,
        pos,
        svd_results,
        {
            'hub': aligned_hub_svd,
            'authority': aligned_auth_svd,
            'node_betweenness': aligned_node_betweenness
        }
    )
    output_path2_base = "gemisvd/outputs/figures/grid_hub_authority_comparison"
    fig2.savefig(f"{output_path2_base}.png", dpi=300, bbox_inches='tight')
    fig2.savefig(f"gemisvd/outputs/figures/svg/{os.path.basename(output_path2_base)}.svg", bbox_inches='tight')
    fig2.savefig(f"gemisvd/outputs/figures/pdf/{os.path.basename(output_path2_base)}.pdf", bbox_inches='tight')
    print(f"   > Visualizations saved to {output_path2_base}.[png,svg,pdf]")

    fig3 = visualize_grid_raw_svd_vs_betweenness(
        G,
        pos,
        {'S_v': svd_results['S_v'], 'S_e': svd_results['S_e']},
        {
            'edge': aligned_edge_betweenness
        }
    )
    output_path3_base = "gemisvd/outputs/figures/grid_raw_svd_vs_betweenness"
    fig3.savefig(f"{output_path3_base}.png", dpi=300, bbox_inches='tight')
    fig3.savefig(f"gemisvd/outputs/figures/svg/{os.path.basename(output_path3_base)}.svg", bbox_inches='tight')
    fig3.savefig(f"gemisvd/outputs/figures/pdf/{os.path.basename(output_path3_base)}.pdf", bbox_inches='tight')
    print(f"   > Visualizations saved to {output_path3_base}.[png,svg,pdf]")

    print("\n--- Grid Experiment Complete ---")

if __name__ == '__main__':
    # Create output directory if it doesn't exist
    import os
    os.makedirs("gemisvd/outputs/figures", exist_ok=True)

    run_experiment()
