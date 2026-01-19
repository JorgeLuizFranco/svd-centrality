#!/usr/bin/env python3
"""
Run the Real-World Network Analysis Experiment
==============================================

This script analyzes various real-world networks (e.g., C. elegans PPI,
Yeast PPI, OpenFlights, Euroroad) to compare SVD centrality with other
traditional centrality measures. It generates visualizations similar to
those found in the paper's "Real-World Network Analysis" section.
"""

import networkx as nx
import numpy as np
import matplotlib.pyplot as plt
import os

# Import refactored project components
from gemisvd.svd_centrality.data_loader import load_real_world_network
from gemisvd.svd_centrality.core.svd_centrality import SVDCentrality
from gemisvd.visualizations.main_visualizer import visualize_real_world_analysis

def run_experiment():
    """
    Main function to run the real-world network analysis and generate visualizations.
    """
    print("--- Running Real-World Network Analysis ---")

    networks_to_analyze = {
        'C. elegans PPI': 'c_elegans',
        'Yeast PPI': 'yeast',
        'OpenFlights': 'openflights',
        'Euroroad': 'euroroad',
        # 'US Power Grid': 'powergrid' # Power grid is undirected, skip for now if focusing on directed graphs for some comparisons
    }
    
    all_network_results = []

    for display_name, internal_name in networks_to_analyze.items():
        print(f"\nAnalyzing network: {display_name}")

        # 1. Load Data
        print(f"Step 1: Loading {display_name} network...")
        G = load_real_world_network(internal_name)
        
        if G.number_of_nodes() == 0:
            print(f"   > Warning: Network {display_name} is empty or not found. Skipping.")
            continue

        # --- Preprocessing (Match C. elegans test) ---
        # 1. Remove self-loops
        G.remove_edges_from(nx.selfloop_edges(G))
        
        # 2. Extract largest weakly connected component
        if nx.is_directed(G):
            if not nx.is_weakly_connected(G):
                largest_cc = max(nx.weakly_connected_components(G), key=len)
                G = G.subgraph(largest_cc).copy()
        else:
            if not nx.is_connected(G):
                largest_cc = max(nx.connected_components(G), key=len)
                G = G.subgraph(largest_cc).copy()
        # ---------------------------------------------

        # Use a consistent layout for each graph, matching the test script exactly
        # The test script uses: pos = nx.spring_layout(G, seed=42)
        print("   > Computing layout...")
        pos = nx.spring_layout(G, seed=42)

        # 2. Compute SVD Centralities
        print("Step 2: Computing SVD centralities using the core library...")
        svd_computer = SVDCentrality()
        svd_results = svd_computer.compute_centralities(G)
        
        # Compute Hub/Authority SVD
        hub_svd_dict, auth_svd_dict = svd_computer.compute_hub_authority_centrality(G, alpha=0.1)
        
        # 3. Compute Baseline Centralities
        print("Step 3: Computing baseline centralities (Betweenness, PageRank, HITS, Closeness)...")
        nodes = svd_results['nodes']
        edges = svd_results['edges']
        
        # Align Hub/Authority SVD
        hub_svd = np.array([hub_svd_dict.get(node, 0) for node in nodes])
        auth_svd = np.array([auth_svd_dict.get(node, 0) for node in nodes])

        # Node Betweenness
        node_betweenness_dict = nx.betweenness_centrality(G, normalized=True)
        node_betweenness = np.array([node_betweenness_dict.get(node, 0) for node in nodes])

        # Edge Betweenness
        edge_betweenness_dict = nx.edge_betweenness_centrality(G, normalized=True)
        edge_betweenness = np.array([edge_betweenness_dict.get(edge, 0) for edge in edges])

        # PageRank
        try:
            pagerank_dict = nx.pagerank(G)
            pagerank = np.array([pagerank_dict.get(node, 0) for node in nodes])
        except Exception as e:
            print(f"   > Warning: PageRank computation failed for {display_name}: {e}. Setting to zeros.")
            pagerank = np.zeros(len(nodes))

        # HITS
        try:
            hits_hub, hits_authority = nx.hits(G)
            hits_hub_aligned = np.array([hits_hub.get(node, 0) for node in nodes])
            hits_authority_aligned = np.array([hits_authority.get(node, 0) for node in nodes])
        except Exception as e:
            print(f"   > Warning: HITS computation failed for {display_name}: {e}. Setting to zeros.")
            hits_hub_aligned = np.zeros(len(nodes))
            hits_authority_aligned = np.zeros(len(nodes))

        # Closeness Centrality (for comparison with undirected graphs if applicable)
        # SVD Centrality is equivalent to Current-Flow Closeness for undirected graphs
        # For directed graphs, standard closeness is less directly comparable
        closeness_dict = nx.closeness_centrality(G)
        closeness = np.array([closeness_dict.get(node, 0) for node in nodes])

        # In-Degree Centrality
        in_degree_dict = {node: G.in_degree(node) for node in G.nodes()}
        in_degree = np.array([in_degree_dict.get(node, 0) for node in nodes])

        all_network_results.append({
            'graph': G,
            'display_name': display_name,
            'pos': pos,
            'svd_results': svd_results,
            'svd_derived': {
                'hub_svd': hub_svd,
                'auth_svd': auth_svd
            },
            'baselines': {
                'node_betweenness': node_betweenness,
                'edge_betweenness': edge_betweenness,
                'pagerank': pagerank,
                'hits_hub': hits_hub_aligned,
                'hits_authority': hits_authority_aligned,
                'closeness': closeness,
                'in_degree': in_degree # Add in-degree to baselines
            }
        })
    
    # 4. Visualization
    print("\nStep 4: Generating and saving visualizations...")
    
    output_dir = "gemisvd/outputs/figures"
    generated_figures = visualize_real_world_analysis(all_network_results)
    for fig_name, fig_obj in generated_figures.items():
        output_path_base = os.path.join(output_dir, f"real_world_{fig_name}")
        fig_obj.savefig(f"{output_path_base}.png", dpi=300, bbox_inches='tight')
        fig_obj.savefig(f"gemisvd/outputs/figures/svg/real_world_{fig_name}.svg", bbox_inches='tight')
        fig_obj.savefig(f"gemisvd/outputs/figures/pdf/real_world_{fig_name}.pdf", bbox_inches='tight')
        print(f"   > Visualizations saved to {output_path_base}.[png,svg,pdf]")

    print("\n--- Real-World Network Analysis Complete ---")

if __name__ == '__main__':
    # Ensure the output directory exists
    import os
    os.makedirs("gemisvd/outputs/figures", exist_ok=True)
    
    run_experiment()
