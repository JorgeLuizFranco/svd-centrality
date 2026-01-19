#!/usr/bin/env python3
"""
Clean Paper-Ready Real Hypergraph Visualizations
===============================================

Creates clean hypergraph visualizations for real datasets following paper style:
- No titles (captions handle context)
- Continuous color scales (light to dark)
- Separate node and hyperedge centrality figures
- Gray/opaque elements when not the focus
- Uses real datasets downloaded from XGI-DATA
"""

import json
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from matplotlib.collections import PatchCollection
import networkx as nx
from pathlib import Path
from collections import defaultdict

# Try to import scipy for ConvexHull
try:
    from scipy.spatial import ConvexHull
except ImportError:
    print("Warning: scipy not available, using simplified polygon drawing")
    def ConvexHull(points):
        class SimpleHull:
            def __init__(self, points):
                self.vertices = range(len(points))
        return SimpleHull(points)

class CleanHypergraph:
    """Clean hypergraph implementation for visualization"""
    
    def __init__(self):
        self.nodes = set()
        self.edges = {}  # edge_id -> set of nodes
        self.node_edges = defaultdict(set)  # node -> set of edges
    
    def add_node(self, node):
        self.nodes.add(node)
    
    def add_edge(self, nodes, edge_id=None):
        if edge_id is None:
            edge_id = len(self.edges)
        
        node_set = set(nodes)
        self.edges[edge_id] = node_set
        
        for node in node_set:
            self.nodes.add(node)
            self.node_edges[node].add(edge_id)
    
    def get_nodes(self):
        return sorted(list(self.nodes))
    
    def get_edges(self):
        return list(self.edges.values())
    
    def degree(self, node):
        return len(self.node_edges[node])

def load_hypergraph_from_json(filepath):
    """Load hypergraph from JSON file"""
    with open(filepath, 'r') as f:
        data = json.load(f)
    
    H = CleanHypergraph()
    
    # Add nodes
    if 'node-data' in data:
        for node_id in data['node-data'].keys():
            H.add_node(node_id)
    
    # Add edges
    if 'edge-dict' in data:
        for edge_id, node_list in data['edge-dict'].items():
            if len(node_list) > 0:
                H.add_edge(node_list, edge_id)
    
    metadata = data.get('hypergraph-data', {})
    return H, metadata

def compute_hypergraph_centralities(hypergraph):
    """Compute SVD centralities for nodes and hyperedges"""
    nodes = hypergraph.get_nodes()
    edges = hypergraph.get_edges()
    
    n_nodes = len(nodes)
    n_edges = len(edges)
    
    if n_nodes == 0 or n_edges == 0:
        return np.zeros(n_nodes), np.zeros(n_edges)
    
    # Build incidence matrix B (nodes x edges)
    B = np.zeros((n_nodes, n_edges))
    node_to_idx = {node: i for i, node in enumerate(nodes)}
    
    # Fill incidence matrix - note: edges are in the same order as hypergraph.get_edges()
    for j, edge_nodes in enumerate(edges):
        for node in edge_nodes:
            if node in node_to_idx:
                i = node_to_idx[node]
                B[i, j] = 1
    
    # Add regularization for numerical stability
    epsilon = 1e-10
    B_reg = B + epsilon * np.random.randn(*B.shape)
    
    try:
        # Node centralities from incidence matrix pseudoinverse
        B_pinv = np.linalg.pinv(B_reg)
        node_centralities = np.sum(np.abs(B_pinv), axis=0)  # Sum over edge dimension to get node centralities
        
        # Hyperedge centralities from transpose pseudoinverse  
        Bt_pinv = np.linalg.pinv(B_reg.T)
        hyperedge_centralities = np.sum(np.abs(Bt_pinv), axis=0)  # Sum over node dimension to get edge centralities
        
    except:
        # Fallback to degree-based centrality
        print("    Using degree-based fallback centrality")
        degrees = [hypergraph.degree(node) for node in nodes]
        max_deg = max(degrees) if degrees else 1
        node_centralities = np.array([d/max_deg for d in degrees])
        
        edge_sizes = [len(edge) for edge in edges]
        max_size = max(edge_sizes) if edge_sizes else 1
        hyperedge_centralities = np.array([s/max_size for s in edge_sizes])
    
    # Normalize to [0, 1]
    if np.max(node_centralities) > 0:
        node_centralities = node_centralities / np.max(node_centralities)
    if np.max(hyperedge_centralities) > 0:
        hyperedge_centralities = hyperedge_centralities / np.max(hyperedge_centralities)
    
    return node_centralities, hyperedge_centralities

def get_largest_connected_component(hypergraph):
    """Extract largest connected component from hypergraph"""
    nodes = hypergraph.get_nodes()
    edges = hypergraph.get_edges()
    
    if not nodes or not edges:
        return hypergraph
    
    # Create auxiliary graph to find connected components
    G = nx.Graph()
    G.add_nodes_from(nodes)
    
    # Add edges between nodes that appear in same hyperedge
    for edge_nodes in edges:
        edge_list = list(edge_nodes)
        for i, node1 in enumerate(edge_list):
            for node2 in edge_list[i+1:]:
                G.add_edge(node1, node2)
    
    # Find all connected components
    components = list(nx.connected_components(G))
    
    if not components:
        return hypergraph
    
    # Get largest connected component
    largest_component = max(components, key=len)
    
    print(f"    Found {len(components)} components. Largest has {len(largest_component)} nodes (was {len(nodes)})")
    
    # Create new hypergraph with only largest component
    connected_hypergraph = CleanHypergraph()
    
    # Add nodes from largest component
    for node in largest_component:
        connected_hypergraph.add_node(node)
    
    # Add only edges that have all nodes in largest component
    # Use new sequential edge IDs for clean indexing
    edge_count = 0
    for edge_id, edge_nodes in hypergraph.edges.items():
        if edge_nodes.issubset(largest_component):
            connected_hypergraph.add_edge(edge_nodes, edge_count)  # Sequential edge IDs
            edge_count += 1
    
    print(f"    Kept {edge_count} edges (was {len(edges)})")
    
    return connected_hypergraph

def create_spring_layout(hypergraph, seed=42):
    """Create spring layout for hypergraph nodes"""
    nodes = hypergraph.get_nodes()
    edges = hypergraph.get_edges()
    
    # Create auxiliary graph for layout
    G = nx.Graph()
    G.add_nodes_from(nodes)
    
    # Add edges between nodes that appear in same hyperedge
    for edge_nodes in edges:
        edge_list = list(edge_nodes)
        for i, node1 in enumerate(edge_list):
            for node2 in edge_list[i+1:]:
                if G.has_edge(node1, node2):
                    G[node1][node2]['weight'] += 1
                else:
                    G.add_edge(node1, node2, weight=1)
    
    # Use spring layout with good parameters
    if len(nodes) > 0:
        pos = nx.spring_layout(G, seed=seed, k=3/np.sqrt(len(nodes)), iterations=100)
    else:
        pos = {}
    
    return pos

def draw_hypergraph_node_centrality(hypergraph, node_centralities, pos, dataset_name):
    """Draw hypergraph with node centrality coloring (paper style)"""
    fig, ax = plt.subplots(figsize=(8, 6))
    
    nodes = hypergraph.get_nodes()
    edges = hypergraph.get_edges()
    
    # Normalize node centralities for color mapping (0 to 1)
    if len(node_centralities) > 0 and np.max(node_centralities) > np.min(node_centralities):
        norm_node_cent = (node_centralities - np.min(node_centralities)) / (np.max(node_centralities) - np.min(node_centralities))
    else:
        norm_node_cent = np.ones(len(node_centralities))
    
    # Draw hyperedges as gray/opaque polygons (not the focus)
    edges_drawn = 0
    for edge_nodes in edges:
        edge_list = list(edge_nodes)
        
        # Skip edges with nodes not in position dict
        if not all(node in pos for node in edge_list):
            continue
            
        if len(edge_list) >= 3:
            # Draw as convex hull for 3+ nodes
            hyperedge_pos = np.array([pos[node] for node in edge_list])
            try:
                hull = ConvexHull(hyperedge_pos)
                polygon = patches.Polygon(hyperedge_pos[hull.vertices], 
                                        facecolor='lightgray', alpha=0.3, 
                                        edgecolor='gray', linewidth=1)
                ax.add_patch(polygon)
                edges_drawn += 1
            except Exception as e:
                # Fallback for collinear points or other issues
                x_coords = [pos[node][0] for node in edge_list]
                y_coords = [pos[node][1] for node in edge_list]
                if len(x_coords) >= 2:
                    # Draw connecting lines
                    for i in range(len(x_coords)):
                        for k in range(i+1, len(x_coords)):
                            ax.plot([x_coords[i], x_coords[k]], [y_coords[i], y_coords[k]], 
                                   'gray', alpha=0.3, linewidth=1)
                    edges_drawn += 1
        elif len(edge_list) == 2:
            # Draw as edge for 2 nodes
            node1, node2 = edge_list
            x_coords = [pos[node1][0], pos[node2][0]]
            y_coords = [pos[node1][1], pos[node2][1]]
            ax.plot(x_coords, y_coords, 'gray', alpha=0.5, linewidth=2)
            edges_drawn += 1
        elif len(edge_list) == 1:
            # Single node hyperedge - draw as light circle around node
            node = edge_list[0]
            circle = patches.Circle(pos[node], radius=0.03, facecolor='lightgray', alpha=0.3, zorder=0)
            ax.add_patch(circle)
            edges_drawn += 1
    
    print(f"    Drew {edges_drawn} background hyperedges")
    
    # Draw nodes with continuous color scale (light to dark blue)
    node_size = 400 if len(nodes) <= 20 else (200 if len(nodes) <= 100 else 100)
    
    # Create mapping from node to centrality value
    node_to_idx = {node: i for i, node in enumerate(nodes)}
    
    for node in nodes:
        if node in pos:
            x, y = pos[node]
            # Get centrality value for this specific node
            node_idx = node_to_idx[node]
            if node_idx < len(norm_node_cent):
                color_intensity = norm_node_cent[node_idx]
            else:
                color_intensity = 0.0  # Default for missing centrality
                
            # Use blues colormap: light (low centrality) to dark (high centrality)
            color = plt.cm.Blues(0.3 + 0.7 * color_intensity)  # Avoid too light colors
            
            ax.scatter(x, y, s=node_size, c=[color], alpha=0.9, 
                      edgecolors='black', linewidths=1.5, zorder=5)
            
            # Only show node labels for small datasets
            if len(nodes) <= 30:
                ax.text(x, y, str(node), ha='center', va='center', fontsize=8, 
                       fontweight='bold', zorder=6)
    
    # Add colorbar for node centralities
    sm = plt.cm.ScalarMappable(cmap=plt.cm.Blues, 
                              norm=plt.Normalize(vmin=np.min(node_centralities), 
                                               vmax=np.max(node_centralities)))
    sm.set_array([])
    cbar = plt.colorbar(sm, ax=ax, shrink=0.8, pad=0.02)
    cbar.set_label('Node Centrality', rotation=270, labelpad=15, fontsize=12)
    
    # Set limits with some padding
    if pos:
        all_x = [pos[node][0] for node in nodes if node in pos]
        all_y = [pos[node][1] for node in nodes if node in pos]
        padding = 0.1
        ax.set_xlim([min(all_x) - padding, max(all_x) + padding])
        ax.set_ylim([min(all_y) - padding, max(all_y) + padding])
    
    ax.set_aspect('equal')
    ax.axis('off')
    
    plt.tight_layout()
    return fig

def draw_hypergraph_edge_centrality(hypergraph, hyperedge_centralities, pos, dataset_name):
    """Draw hypergraph with hyperedge centrality coloring (paper style)"""
    fig, ax = plt.subplots(figsize=(8, 6))
    
    nodes = hypergraph.get_nodes()
    edges = hypergraph.get_edges()
    
    print(f"    DEBUG: Drawing {len(edges)} edges with {len(hyperedge_centralities)} centrality values")
    
    # Ensure we have the right number of centrality values
    if len(hyperedge_centralities) != len(edges):
        print(f"    ERROR: Centrality array length ({len(hyperedge_centralities)}) != edges count ({len(edges)})")
        return fig
    
    # Normalize hyperedge centralities for color mapping
    if len(hyperedge_centralities) > 0 and np.max(hyperedge_centralities) > np.min(hyperedge_centralities):
        norm_edge_cent = (hyperedge_centralities - np.min(hyperedge_centralities)) / (np.max(hyperedge_centralities) - np.min(hyperedge_centralities))
    else:
        norm_edge_cent = np.ones(len(hyperedge_centralities))
    
    # Draw nodes as gray/opaque (not the focus)
    node_size = 300 if len(nodes) <= 20 else (150 if len(nodes) <= 100 else 80)
    for node in nodes:
        if node in pos:
            x, y = pos[node]
            ax.scatter(x, y, s=node_size, c='lightgray', alpha=0.7, 
                      edgecolors='gray', linewidths=1, zorder=3)
            
            # Only show node labels for small datasets
            if len(nodes) <= 30:
                ax.text(x, y, str(node), ha='center', va='center', fontsize=7, 
                       color='black', zorder=4)
    
    # Draw ALL hyperedges with continuous color scale (light to dark red)
    edges_drawn = 0
    for j, edge_nodes in enumerate(edges):
        # Ensure j is within bounds
        if j >= len(norm_edge_cent):
            print(f"    Warning: Edge index {j} exceeds centrality array bounds")
            break
            
        color_intensity = norm_edge_cent[j]
        # Use reds colormap: light (low centrality) to dark (high centrality)
        color = plt.cm.Reds(0.3 + 0.7 * color_intensity)
        
        edge_list = list(edge_nodes)
        
        # Skip edges with nodes not in position dict
        if not all(node in pos for node in edge_list):
            continue
            
        if len(edge_list) >= 3:
            # Draw as convex hull for 3+ nodes
            hyperedge_pos = np.array([pos[node] for node in edge_list])
            try:
                hull = ConvexHull(hyperedge_pos)
                polygon = patches.Polygon(hyperedge_pos[hull.vertices], 
                                        facecolor=color, alpha=0.6, 
                                        edgecolor='darkred', linewidth=2, zorder=1)
                ax.add_patch(polygon)
                edges_drawn += 1
            except Exception as e:
                # Fallback for collinear points or other issues
                x_coords = [pos[node][0] for node in edge_list]
                y_coords = [pos[node][1] for node in edge_list]
                if len(x_coords) >= 2:
                    # Draw connecting lines
                    for i in range(len(x_coords)):
                        for k in range(i+1, len(x_coords)):
                            ax.plot([x_coords[i], x_coords[k]], [y_coords[i], y_coords[k]], 
                                   color=color, linewidth=2, alpha=0.4, zorder=1)
                    edges_drawn += 1
        elif len(edge_list) == 2:
            # Draw as thick edge for 2 nodes
            node1, node2 = edge_list
            x_coords = [pos[node1][0], pos[node2][0]]
            y_coords = [pos[node1][1], pos[node2][1]]
            ax.plot(x_coords, y_coords, color=color, linewidth=4, alpha=0.8, zorder=2)
            edges_drawn += 1
        elif len(edge_list) == 1:
            # Single node hyperedge - draw as circle around node
            node = edge_list[0]
            circle = patches.Circle(pos[node], radius=0.05, facecolor=color, alpha=0.6, zorder=1)
            ax.add_patch(circle)
            edges_drawn += 1
    
    print(f"    Drew {edges_drawn} out of {len(edges)} hyperedges")
    
    # Add colorbar for hyperedge centralities
    sm = plt.cm.ScalarMappable(cmap=plt.cm.Reds, 
                              norm=plt.Normalize(vmin=np.min(hyperedge_centralities), 
                                               vmax=np.max(hyperedge_centralities)))
    sm.set_array([])
    cbar = plt.colorbar(sm, ax=ax, shrink=0.8, pad=0.02)
    cbar.set_label('Hyperedge Centrality', rotation=270, labelpad=15, fontsize=12)
    
    # Set limits with some padding
    if pos:
        all_x = [pos[node][0] for node in nodes if node in pos]
        all_y = [pos[node][1] for node in nodes if node in pos]
        padding = 0.1
        ax.set_xlim([min(all_x) - padding, max(all_x) + padding])
        ax.set_ylim([min(all_y) - padding, max(all_y) + padding])
    
    ax.set_aspect('equal')
    ax.axis('off')
    
    plt.tight_layout()
    return fig

def process_real_datasets():
    """Process all real datasets and create clean paper-ready visualizations"""
    print("Creating Clean Paper-Ready Real Dataset Visualizations")
    print("=" * 60)
    
    datasets_folder = Path('hypergraphDatasets')
    output_folder = Path('data/clean_paper_figures')
    output_folder.mkdir(parents=True, exist_ok=True)
    
    # Set random seed for reproducible layouts
    np.random.seed(42)
    
    results = []
    
    # Process datasets that are good for clean visualization
    target_datasets = [
        'plant-pollinator-mpl-046.json',  # Perfect size: 16 nodes, 44 edges
        'diseasome.json',                 # Medium size but manageable: 516 nodes
        'senate-bills.json'               # Large but interesting: 294 nodes
    ]
    
    for json_file_name in target_datasets:
        json_file = datasets_folder / json_file_name
        if not json_file.exists():
            print(f"⚠️ Skipping {json_file_name}: file not found")
            continue
            
        print(f"\n📊 Processing {json_file.name}...")
        
        try:
            # Load original hypergraph
            hypergraph_full, metadata = load_hypergraph_from_json(json_file)
            
            num_nodes_orig = len(hypergraph_full.get_nodes())
            num_edges_orig = len(hypergraph_full.get_edges())
            
            print(f"    Original: {num_nodes_orig} nodes, {num_edges_orig} edges")
            
            if num_nodes_orig == 0 or num_edges_orig == 0:
                print(f"    ⚠️ Skipping: empty dataset")
                continue
            
            dataset_name = metadata.get('name', json_file.stem)
            
            # First, check if dataset has disconnected components
            print(f"    Checking connectivity...")
            hypergraph_connected = get_largest_connected_component(hypergraph_full)
            num_nodes_connected = len(hypergraph_connected.get_nodes())
            num_edges_connected = len(hypergraph_connected.get_edges())
            
            has_disconnected_components = (num_nodes_orig != num_nodes_connected) or (num_edges_orig != num_edges_connected)
            
            if has_disconnected_components:
                print(f"    🔥 HAS DISCONNECTED COMPONENTS! Will generate both FULL and CONNECTED versions")
                print(f"       Largest component: {num_nodes_connected} nodes, {num_edges_connected} edges")
                print(f"       Will remove: {num_nodes_orig - num_nodes_connected} nodes, {num_edges_orig - num_edges_connected} edges")
            else:
                print(f"    ✅ ALREADY FULLY CONNECTED - FULL and CONNECTED versions will be identical")
                print(f"       Generating only FULL version to avoid duplicates")
            
            # ==================== FULL DATASET VERSION ====================
            print(f"    Creating FULL dataset visualizations...")
            
            # Skip if too large for visualization
            if num_nodes_orig <= 600:  # Only process if reasonable size
                # Use the ORIGINAL hypergraph without any component extraction
                print(f"    Computing SVD centralities (full - ALL nodes/edges)...")
                node_centralities_full, hyperedge_centralities_full = compute_hypergraph_centralities(hypergraph_full)
                
                print(f"    Full - Node centrality range: [{np.min(node_centralities_full):.4f}, {np.max(node_centralities_full):.4f}]")
                print(f"    Full - Edge centrality range: [{np.min(hyperedge_centralities_full):.4f}, {np.max(hyperedge_centralities_full):.4f}]")
                print(f"    Full - Computing centralities for {len(hypergraph_full.get_nodes())} nodes and {len(hypergraph_full.get_edges())} edges")
                
                # Create layout for FULL original dataset (no filtering)
                print(f"    Creating spring layout (full - original dataset)...")
                pos_full = create_spring_layout(hypergraph_full, seed=42)  # Use seed 42 for FULL
                
                # Create FULL dataset visualizations (with ALL original nodes and edges)
                print(f"    Creating FULL node centrality figure (ALL {len(hypergraph_full.get_nodes())} nodes)...")
                fig1_full = draw_hypergraph_node_centrality(hypergraph_full, node_centralities_full, pos_full, dataset_name)
                node_output_full = output_folder / f"{json_file.stem}_node_centrality_FULL.png"
                fig1_full.savefig(node_output_full, dpi=300, bbox_inches='tight', facecolor='white')
                plt.close(fig1_full)
                
                print(f"    Creating FULL hyperedge centrality figure (ALL {len(hypergraph_full.get_edges())} edges)...")
                fig2_full = draw_hypergraph_edge_centrality(hypergraph_full, hyperedge_centralities_full, pos_full, dataset_name)
                edge_output_full = output_folder / f"{json_file.stem}_edge_centrality_FULL.png"
                fig2_full.savefig(edge_output_full, dpi=300, bbox_inches='tight', facecolor='white')
                plt.close(fig2_full)
                
                print(f"    ✅ FULL dataset visualizations saved (preserved all {num_nodes_orig} nodes, {num_edges_orig} edges)")
            else:
                print(f"    ⚠️ Skipping FULL version: too large ({num_nodes_orig} nodes)")
                node_output_full = None
                edge_output_full = None
            
            # =============== CONNECTED COMPONENT VERSION =================
            # Only create CONNECTED version if dataset has disconnected components
            if has_disconnected_components:
                print(f"    Creating CONNECTED COMPONENT visualizations...")
                
                if num_nodes_connected == 0 or num_edges_connected == 0:
                    print(f"    ⚠️ Skipping connected component: empty")
                    node_output_conn = None
                    edge_output_conn = None
                elif num_nodes_connected > 600:
                    print(f"    ⚠️ Skipping connected component: still too large ({num_nodes_connected} nodes)")
                    node_output_conn = None
                    edge_output_conn = None
                else:
                    # Compute centralities for connected component
                    print(f"    Computing SVD centralities (connected)...")
                    node_centralities_conn, hyperedge_centralities_conn = compute_hypergraph_centralities(hypergraph_connected)
                    
                    print(f"    Connected - Node centrality range: [{np.min(node_centralities_conn):.4f}, {np.max(node_centralities_conn):.4f}]")
                    print(f"    Connected - Edge centrality range: [{np.min(hyperedge_centralities_conn):.4f}, {np.max(hyperedge_centralities_conn):.4f}]")
                    
                    # Create layout for connected component
                    print(f"    Creating spring layout (connected)...")
                    pos_connected = create_spring_layout(hypergraph_connected, seed=123)  # Different seed for CONNECTED
                    
                    # Create CONNECTED COMPONENT visualizations
                    print(f"    Creating CONNECTED node centrality figure...")
                    fig1_conn = draw_hypergraph_node_centrality(hypergraph_connected, node_centralities_conn, pos_connected, dataset_name)
                    node_output_conn = output_folder / f"{json_file.stem}_node_centrality_CONNECTED.png"
                    fig1_conn.savefig(node_output_conn, dpi=300, bbox_inches='tight', facecolor='white')
                    plt.close(fig1_conn)
                    
                    print(f"    Creating CONNECTED hyperedge centrality figure...")
                    fig2_conn = draw_hypergraph_edge_centrality(hypergraph_connected, hyperedge_centralities_conn, pos_connected, dataset_name)
                    edge_output_conn = output_folder / f"{json_file.stem}_edge_centrality_CONNECTED.png"
                    fig2_conn.savefig(edge_output_conn, dpi=300, bbox_inches='tight', facecolor='white')
                    plt.close(fig2_conn)
                    
                    print(f"    ✅ CONNECTED component visualizations saved")
            else:
                print(f"    ⏭️ Skipping CONNECTED version (dataset already fully connected)")
                node_output_conn = None
                edge_output_conn = None
            
            # Store results
            results.append({
                'dataset': json_file.stem,
                'name': dataset_name,
                'nodes_orig': num_nodes_orig,
                'edges_orig': num_edges_orig,
                'nodes_connected': num_nodes_connected,
                'edges_connected': num_edges_connected,
                'has_disconnected': has_disconnected_components,
                'node_file_full': node_output_full.name if node_output_full else None,
                'edge_file_full': edge_output_full.name if edge_output_full else None,
                'node_file_connected': node_output_conn.name if node_output_conn else None,
                'edge_file_connected': edge_output_conn.name if edge_output_conn else None
            })
            
        except Exception as e:
            print(f"    ❌ Error: {e}")
            import traceback
            traceback.print_exc()
    
    # Summary
    print(f"\n{'=' * 60}")
    print("DUAL VISUALIZATION ANALYSIS COMPLETE")
    print(f"{'=' * 60}")
    
    print("✓ Both FULL and CONNECTED component versions generated")
    print("✓ Choose the best visualization for each dataset")
    print("✓ No titles (captions provide context)")
    print("✓ Continuous color scales (light to dark)")
    print("✓ Separate node and hyperedge focus")
    print("✓ Gray/opaque non-focus elements")
    print("✓ Professional colorbars")
    print("✓ Real datasets from XGI-DATA/Zenodo")
    
    for result in results:
        print(f"\n📈 {result['name']} ({result['dataset']}):")
        print(f"   Original size: {result['nodes_orig']} nodes, {result['edges_orig']} edges")
        print(f"   Connected size: {result['nodes_connected']} nodes, {result['edges_connected']} edges")
        
        if result['has_disconnected']:
            print(f"   🔥 HAS DISCONNECTED COMPONENTS")
        else:
            print(f"   ✅ ALREADY FULLY CONNECTED")
        
        if result['node_file_full']:
            print(f"   FULL - Node: {result['node_file_full']}")
            print(f"   FULL - Edge: {result['edge_file_full']}")
        else:
            print(f"   FULL - Skipped (too large)")
            
        if result['node_file_connected']:
            print(f"   CONNECTED - Node: {result['node_file_connected']}")
            print(f"   CONNECTED - Edge: {result['edge_file_connected']}")
        else:
            if result['has_disconnected']:
                print(f"   CONNECTED - Skipped (too large or empty)")
            else:
                print(f"   CONNECTED - Skipped (already fully connected, would be identical)")
    
    print(f"\n🎨 All figures saved to: {output_folder}")
    print("📊 Compare FULL vs CONNECTED versions to choose the best!")
    
    return results

if __name__ == "__main__":
    results = process_real_datasets()