#!/usr/bin/env python3
"""
SVD Centrality Examples
======================

This script demonstrates the main features of the SVD centrality package
with practical examples on real networks.
"""

import networkx as nx
import matplotlib.pyplot as plt
import numpy as np
from svd_centrality import SVDCentrality
from svd_centrality.hypergraph import ManualHypergraph, HypergraphSVDCentrality, create_hypergraph_visualization


def example_graph_centrality():
    """Demonstrate SVD centrality on Zachary's Karate Club."""
    print("=== Graph Centrality Example ===")
    print("Analyzing Zachary's Karate Club network...")
    
    # Load the famous karate club graph
    G = nx.karate_club_graph()
    print(f"Network: {G.number_of_nodes()} nodes, {G.number_of_edges()} edges")
    
    # Compute SVD centrality
    svd = SVDCentrality()
    results = svd.compute_centralities(G)
    
    # Get top 5 most central nodes
    centrality_pairs = [(node, results['S_v'][i]) for i, node in enumerate(results['nodes'])]
    centrality_pairs.sort(key=lambda x: x[1], reverse=True)
    
    print("\nTop 5 most central nodes:")
    for i, (node, centrality) in enumerate(centrality_pairs[:5]):
        print(f"{i+1}. Node {node}: {centrality:.4f}")
    
    # Validate against current-flow closeness
    validation = svd.validate_against_current_flow(G)
    print(f"\nCorrelation with current-flow closeness: {validation['pearson_correlation']:.4f}")
    
    # Get computation statistics
    stats = svd.get_computation_stats()
    print(f"Computation time: {stats.computation_time:.4f} seconds")
    print(f"Memory usage: {stats.memory_usage_mb:.2f} MB")
    
    return results


def example_directed_graph():
    """Demonstrate hub/authority analysis on a directed graph."""
    print("\n=== Directed Graph Example ===")
    print("Creating a directed citation network...")
    
    # Create a simple directed graph (citation network)
    G = nx.DiGraph()
    edges = [
        (1, 2), (1, 3), (2, 4), (3, 4), (4, 5), 
        (2, 5), (6, 1), (6, 2), (7, 1), (7, 3)
    ]
    G.add_edges_from(edges)
    print(f"Network: {G.number_of_nodes()} nodes, {G.number_of_edges()} edges")
    
    # Compute SVD centrality
    svd = SVDCentrality()
    results = svd.compute_centralities(G)
    
    # Compute hub and authority scores
    hub_scores, auth_scores = svd.compute_hub_authority_centrality(G)
    
    print("\nHub scores (outgoing influence):")
    for node, score in sorted(hub_scores.items(), key=lambda x: x[1], reverse=True):
        print(f"  Node {node}: {score:.4f}")
    
    print("\nAuthority scores (incoming prestige):")
    for node, score in sorted(auth_scores.items(), key=lambda x: x[1], reverse=True):
        print(f"  Node {node}: {score:.4f}")
    
    return hub_scores, auth_scores


def example_hypergraph_centrality():
    """Demonstrate SVD centrality on a hypergraph."""
    print("\n=== Hypergraph Centrality Example ===")
    print("Creating a research collaboration hypergraph...")
    
    # Create a hypergraph representing research collaborations
    H = ManualHypergraph()
    
    # Add research groups (hyperedges)
    H.add_edge(['Alice', 'Bob', 'Charlie'], 'Paper1')          # 3-author paper
    H.add_edge(['Alice', 'David'], 'Paper2')                   # 2-author paper  
    H.add_edge(['Bob', 'Charlie', 'Eve', 'Frank'], 'Paper3')   # 4-author paper
    H.add_edge(['David', 'Eve'], 'Paper4')                     # 2-author paper
    H.add_edge(['Alice', 'Eve', 'Grace'], 'Paper5')            # 3-author paper
    H.add_edge(['Frank', 'Grace'], 'Paper6')                   # 2-author paper
    
    print(f"Hypergraph: {H.num_nodes()} researchers, {H.num_edges()} papers")
    
    # Get basic statistics
    stats = H.get_statistics()
    print(f"Average degree: {stats['avg_degree']:.2f}")
    print(f"Average paper size: {stats['avg_edge_size']:.2f}")
    
    # Compute hypergraph centrality
    hsvd = HypergraphSVDCentrality()
    results = hsvd.compute_centralities(H)
    
    print("\nResearcher centralities:")
    vertex_centrality = results['normalized_vertex_centrality']
    for researcher, centrality in sorted(vertex_centrality.items(), 
                                       key=lambda x: x[1], reverse=True):
        print(f"  {researcher}: {centrality:.4f}")
    
    print("\nPaper centralities:")
    edge_centrality = results['normalized_edge_centrality']
    for paper, centrality in sorted(edge_centrality.items(), 
                                  key=lambda x: x[1], reverse=True):
        print(f"  {paper}: {centrality:.4f}")
    
    # Create visualization
    try:
        fig = create_hypergraph_visualization(H, results, 
                                            title="Research Collaboration Network")
        plt.savefig('hypergraph_example.png', dpi=150, bbox_inches='tight')
        print("\nVisualization saved as 'hypergraph_example.png'")
        plt.close()
    except Exception as e:
        print(f"Visualization failed: {e}")
    
    return results


def example_comparison():
    """Compare SVD centrality with other centrality measures."""
    print("\n=== Centrality Comparison Example ===")
    print("Comparing SVD centrality with classical measures...")
    
    # Use a well-known graph
    G = nx.barabasi_albert_graph(50, 3, seed=42)
    
    # Compute various centrality measures
    svd = SVDCentrality()
    svd_results = svd.compute_centralities(G)
    svd_centrality = {node: svd_results['S_v'][i] for i, node in enumerate(svd_results['nodes'])}
    
    betweenness = nx.betweenness_centrality(G)
    closeness = nx.closeness_centrality(G) 
    pagerank = nx.pagerank(G)
    eigenvector = nx.eigenvector_centrality(G)
    
    # Compute correlations
    nodes = list(G.nodes())
    svd_values = [svd_centrality[n] for n in nodes]
    bet_values = [betweenness[n] for n in nodes]
    clo_values = [closeness[n] for n in nodes]
    pr_values = [pagerank[n] for n in nodes]
    eig_values = [eigenvector[n] for n in nodes]
    
    print("\nCorrelations with SVD centrality:")
    print(f"  Betweenness:   {np.corrcoef(svd_values, bet_values)[0,1]:.4f}")
    print(f"  Closeness:     {np.corrcoef(svd_values, clo_values)[0,1]:.4f}")
    print(f"  PageRank:      {np.corrcoef(svd_values, pr_values)[0,1]:.4f}")
    print(f"  Eigenvector:   {np.corrcoef(svd_values, eig_values)[0,1]:.4f}")
    
    # Find nodes with highest differences
    differences = []
    for node in nodes:
        svd_rank = sorted(svd_values, reverse=True).index(svd_centrality[node])
        bet_rank = sorted(bet_values, reverse=True).index(betweenness[node])
        diff = abs(svd_rank - bet_rank)
        differences.append((node, diff, svd_rank, bet_rank))
    
    differences.sort(key=lambda x: x[1], reverse=True)
    
    print("\nNodes with largest ranking differences (SVD vs Betweenness):")
    for node, diff, svd_rank, bet_rank in differences[:3]:
        print(f"  Node {node}: SVD rank {svd_rank}, Betweenness rank {bet_rank} (diff: {diff})")


def main():
    """Run all examples."""
    print("SVD Centrality Package Examples")
    print("=" * 40)
    
    try:
        # Graph examples
        example_graph_centrality()
        example_directed_graph()
        
        # Hypergraph example
        example_hypergraph_centrality()
        
        # Comparison example
        example_comparison()
        
        print("\n" + "=" * 40)
        print("All examples completed successfully!")
        print("For more information, see the documentation at:")
        print("https://svd-centrality.readthedocs.io")
        
    except Exception as e:
        print(f"\nError running examples: {e}")
        print("Please ensure all dependencies are installed:")
        print("pip install networkx matplotlib scipy numpy")


if __name__ == "__main__":
    main()