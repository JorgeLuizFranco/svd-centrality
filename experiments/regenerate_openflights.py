
import networkx as nx
import numpy as np
import matplotlib.pyplot as plt
import os
import sys

# Ensure project root is in python path
sys.path.append(os.getcwd())

from gemisvd.svd_centrality.data_loader import load_real_world_network
from gemisvd.svd_centrality.core.svd_centrality import SVDCentrality
from gemisvd.visualizations.main_visualizer import visualize_real_world_analysis

def regenerate_openflights():
    print("--- Regenerating OpenFlights PDF (Isolated Run) ---")
    
    # 1. Load Data
    display_name = 'OpenFlights'
    internal_name = 'openflights'
    print(f"Step 1: Loading {display_name} network...")
    G = load_real_world_network(internal_name)
    
    # --- Preprocessing (Matching standard pipeline) ---
    print("   > Preprocessing (removing self-loops, largest CC)...")
    G.remove_edges_from(nx.selfloop_edges(G))
    
    if nx.is_directed(G):
        if not nx.is_weakly_connected(G):
            largest_cc = max(nx.weakly_connected_components(G), key=len)
            G = G.subgraph(largest_cc).copy()
    else:
        if not nx.is_connected(G):
            largest_cc = max(nx.connected_components(G), key=len)
            G = G.subgraph(largest_cc).copy()
            
    # Calculate layout explicitely to pass to visualizer, ensuring it exists
    # The visualizer has logic for >2000 nodes, but we want to ensure it runs
    # OpenFlights has ~3000 nodes. main_visualizer uses kamada_kawai for >2000.
    # We will let main_visualizer handle the layout generation to be consistent 
    # with the PNG, or we can pre-calculate here if we want to be safe.
    # To match the PNG exactly, we rely on the logic inside main_visualizer.
    # However, to be safe against timeouts, we can pre-calc here.
    # BUT, main_visualizer ignores passed 'pos' if n_nodes > 2000. 
    # So we strictly rely on main_visualizer's internal layout logic.
    pos = None 

    # 2. Compute SVD Centralities
    print("Step 2: Computing SVD centralities...")
    svd_computer = SVDCentrality()
    # This will trigger the binary search for k automatically
    svd_results = svd_computer.compute_centralities(G)
    
    # Compute Hub/Authority
    print("   > Computing Hub/Authority (alpha=0.1)...")
    hub_svd_dict, auth_svd_dict = svd_computer.compute_hub_authority_centrality(G, alpha=0.1)
    
    # Align results
    nodes = svd_results['nodes']
    edges = svd_results['edges']
    
    hub_svd = np.array([hub_svd_dict.get(node, 0) for node in nodes])
    auth_svd = np.array([auth_svd_dict.get(node, 0) for node in nodes])

    # 3. Compute Baselines (Only those needed for Viz: Betw, Degree)
    print("Step 3: Computing baseline centralities...")
    
    # Node Betweenness
    print("   > Node Betweenness...")
    node_betweenness_dict = nx.betweenness_centrality(G, normalized=True)
    node_betweenness = np.array([node_betweenness_dict.get(node, 0) for node in nodes])

    # Edge Betweenness (Expensive!)
    print("   > Edge Betweenness (this is the bottleneck)...")
    edge_betweenness_dict = nx.edge_betweenness_centrality(G, normalized=True)
    edge_betweenness = np.array([edge_betweenness_dict.get(edge, 0) for edge in edges])

    # In-Degree
    print("   > In-Degree...")
    in_degree_dict = {node: G.in_degree(node) for node in G.nodes()}
    in_degree = np.array([in_degree_dict.get(node, 0) for node in nodes])

    # Prepare Data Object
    results = [{
        'graph': G,
        'display_name': display_name,
        'pos': pos, # Visualizer handles layout for OpenFlights internally
        'svd_results': svd_results,
        'svd_derived': {
            'hub_svd': hub_svd,
            'auth_svd': auth_svd
        },
        'baselines': {
            'node_betweenness': node_betweenness,
            'edge_betweenness': edge_betweenness,
            'in_degree': in_degree
        }
    }]
    
    # 4. Generate and Save
    print("Step 4: Generating visualization...")
    figs = visualize_real_world_analysis(results)
    
    output_dir = "gemisvd/outputs/figures"
    pdf_dir = os.path.join(output_dir, "pdf")
    os.makedirs(pdf_dir, exist_ok=True)
    
    fig = figs['openflights']
    
    # Save PDF specifically
    pdf_path = os.path.join(pdf_dir, "real_world_openflights.pdf")
    print(f"   > Saving PDF to {pdf_path}...")
    fig.savefig(pdf_path, bbox_inches='tight')
    
    # Save PNG just to be sure we have a sync pair
    png_path = os.path.join(output_dir, "real_world_openflights.png")
    print(f"   > Saving PNG to {png_path}...")
    fig.savefig(png_path, dpi=300, bbox_inches='tight')
    
    print("Done.")

if __name__ == '__main__':
    regenerate_openflights()
