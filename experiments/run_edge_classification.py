#!/usr/bin/env python3
"""
Run the Edge Classification Experiment
======================================

This script reproduces the machine learning experiment from the paper,
evaluating how well different edge centrality measures perform as features
for classifying important "bridge" edges in a network.
"""

import networkx as nx
import numpy as np
import matplotlib.pyplot as plt
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, roc_auc_score
import os

# Import the refactored project components
from gemisvd.svd_centrality.data_loader import load_benchmark_graph
from gemisvd.svd_centrality.core.svd_centrality import SVDCentrality
from gemisvd.visualizations.main_visualizer import visualize_edge_classification

def create_bridge_detection_task(G: nx.Graph) -> dict:
    """
    Creates ground truth labels for the bridge detection task.
    An edge is labeled 1 if it is a bridge, 0 otherwise.
    """
    # A bridge is an edge whose removal increases the number of connected components.
    bridges = set(nx.bridges(G))
    edge_labels = {}
    for edge in G.edges():
        # Ensure canonical edge order for undirected graphs
        u, v = sorted(edge)
        is_bridge = (u, v) in bridges
        edge_labels[(u, v)] = 1 if is_bridge else 0
    return edge_labels

def run_single_experiment(G: nx.Graph, graph_name: str):
    """
    Runs the classification experiment for a single graph.
    """
    print(f"--- Running Edge Classification for: {graph_name} ---")

    # 1. Create Ground Truth
    print("Step 1: Creating ground truth labels (bridge detection)...")
    edge_labels_map = create_bridge_detection_task(G)

    # 2. Compute Centralities
    print("Step 2: Computing SVD and Betweenness edge centralities...")
    # Use the correct core library for SVD
    svd_computer = SVDCentrality()
    svd_results = svd_computer.compute_centralities(G)
    
    # Baseline: Edge Betweenness Centrality
    betweenness_results = nx.edge_betweenness_centrality(G, normalized=True)

    # 3. Prepare ML Features and Labels
    print("Step 3: Preparing features and labels for machine learning...")
    
    # Ensure consistent edge ordering
    edges = svd_results['edges']
    
    X_svd = svd_results['S_e'].reshape(-1, 1)
    X_bet = np.array([betweenness_results.get(edge, 0) for edge in edges]).reshape(-1, 1)
    y = np.array([edge_labels_map.get(tuple(sorted(edge)), 0) for edge in edges])

    if len(set(y)) < 2:
        print("   > Warning: Only one class present. Skipping experiment.")
        return None
    
    # Check if any class has less than 2 samples, which would break stratify
    class_counts = np.bincount(y)
    if (class_counts < 2).any():
        print(f"   > Warning: Not enough samples in all classes for stratification (counts: {class_counts}). Skipping ML for this graph.")
        return None

    # 4. Train and Evaluate Models
    print("Step 4: Training and evaluating classifiers...")
    X_svd_train, X_svd_test, y_train, y_test = train_test_split(X_svd, y, test_size=0.3, random_state=42, stratify=y)
    X_bet_train, X_bet_test, _, _ = train_test_split(X_bet, y, test_size=0.3, random_state=42, stratify=y)

    # Model using SVD feature
    clf_svd = RandomForestClassifier(n_estimators=100, random_state=42)
    clf_svd.fit(X_svd_train, y_train)
    svd_pred = clf_svd.predict(X_svd_test)
    svd_prob = clf_svd.predict_proba(X_svd_test)[:, 1]
    svd_accuracy = accuracy_score(y_test, svd_pred)
    svd_auc = roc_auc_score(y_test, svd_prob)

    # Model using Betweenness feature
    clf_bet = RandomForestClassifier(n_estimators=100, random_state=42)
    clf_bet.fit(X_bet_train, y_train)
    bet_pred = clf_bet.predict(X_bet_test)
    bet_prob = clf_bet.predict_proba(X_bet_test)[:, 1]
    bet_accuracy = accuracy_score(y_test, bet_pred)
    bet_auc = roc_auc_score(y_test, bet_prob)
    
    print(f"   > SVD Edge Centrality:        Accuracy={svd_accuracy:.3f}, AUC={svd_auc:.3f}")
    print(f"   > Betweenness Edge Centrality: Accuracy={bet_accuracy:.3f}, AUC={bet_auc:.3f}")

    return {
        'graph': G,
        'graph_name': graph_name,
        'task_name': 'Bridge Detection',
        'svd_accuracy': svd_accuracy,
        'betweenness_accuracy': bet_accuracy,
        'svd_auc': svd_auc,
        'betweenness_auc': bet_auc,
        'accuracy_advantage': svd_accuracy - bet_accuracy,
        'auc_advantage': svd_auc - bet_auc,
        'edge_labels': edge_labels_map
    }

def main():
    """
    Main function to run all edge classification experiments.
    """
    graphs_to_test = {
        'Karate Club': load_benchmark_graph('karate'),
        'Les Misérables': load_benchmark_graph('les_miserables'),
        'Florentine Families': load_benchmark_graph('florentine_families')
    }
    
    all_results = []
    for name, G in graphs_to_test.items():
        result = run_single_experiment(G, name)
        if result:
            all_results.append(result)
            
    if all_results:
        print("\nStep 5: Generating and saving final visualization...")
        fig = visualize_edge_classification(all_results)
        output_path_base = "gemisvd/outputs/figures/edge_classification_results"
        fig.savefig(f"{output_path_base}.png", dpi=300, bbox_inches='tight')
        fig.savefig(f"gemisvd/outputs/figures/svg/{os.path.basename(output_path_base)}.svg", bbox_inches='tight')
        fig.savefig(f"gemisvd/outputs/figures/pdf/{os.path.basename(output_path_base)}.pdf", bbox_inches='tight')
        print(f"   > Visualizations saved to {output_path_base}.[png,svg,pdf]")
    
    print("\n--- Edge Classification Experiment Complete ---")

if __name__ == '__main__':
    # Ensure the output directory exists
    import os
    os.makedirs("gemisvd/outputs/figures", exist_ok=True)
    main()
