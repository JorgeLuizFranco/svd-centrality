
import networkx as nx
import numpy as np
import scipy.stats as stats
import warnings
from gemisvd.svd_centrality.data_loader import load_real_world_network
from gemisvd.svd_centrality.core.svd_centrality import SVDCentrality

def analyze_correlations():
    networks = {
        'C. elegans': 'c_elegans',
        'Yeast': 'yeast',
        'OpenFlights': 'openflights',
        'Euroroad': 'euroroad'
    }

    print(f"{'Network':<15} | {'SVD vs Closeness (Pearson)':<25} | {'SVD vs Betweenness (Spearman)':<25}")
    print("-" * 70)

    for name, key in networks.items():
        try:
            # Load and Preprocess
            G = load_real_world_network(key)
            G.remove_edges_from(nx.selfloop_edges(G))
            
            if nx.is_directed(G):
                if not nx.is_weakly_connected(G):
                    largest_cc = max(nx.weakly_connected_components(G), key=len)
                    G = G.subgraph(largest_cc).copy()
            else:
                if not nx.is_connected(G):
                    largest_cc = max(nx.connected_components(G), key=len)
                    G = G.subgraph(largest_cc).copy()

            # Compute SVD
            svd_computer = SVDCentrality()
            res = svd_computer.compute_centralities(G)
            svd_vals = res['S_v']
            nodes = res['nodes']
            
            # Compute Baselines
            # Closeness (Standard)
            closeness = nx.closeness_centrality(G)
            close_vals = [closeness[n] for n in nodes]
            
            # Betweenness
            betweenness = nx.betweenness_centrality(G)
            bet_vals = [betweenness[n] for n in nodes]
            
            # Correlations
            p_close, _ = stats.pearsonr(svd_vals, close_vals)
            s_bet, _ = stats.spearmanr(svd_vals, bet_vals)
            
            print(f"{name:<15} | {p_close:<25.4f} | {s_bet:<25.4f}")

        except Exception as e:
            print(f"{name:<15} | Error: {str(e)}")

if __name__ == "__main__":
    analyze_correlations()
