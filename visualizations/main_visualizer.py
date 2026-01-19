#!/usr/bin/env python3
"""
Main Visualizer for SVD Centrality Project
==========================================

This module contains reusable functions for generating all visualizations
for the paper. Each function is designed to take graph and centrality
data as input and return a matplotlib figure.
"""

import networkx as nx
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patheffects as path_effects
from matplotlib.colors import Normalize
import seaborn as sns
from scipy.stats import pearsonr, spearmanr

# Set style for publication-quality plots
plt.style.use('default')
plt.rcParams["font.family"] = "serif"
plt.rcParams["font.serif"] = ["DejaVu Serif", "Computer Modern Roman", "serif"]
sns.set_palette("husl")

# --- Utility Functions ---

def _create_color_mapping(values, cmap=plt.cm.viridis):
    """
    Creates a list of colors from a set of values by normalizing them.
    """
    if len(values) == 0:
        return []
    
    vmin = np.min(values)
    vmax = np.max(values)
    
    # If all values are the same, return a medium-intensity color from the map.
    if vmin == vmax:
        return [cmap(0.5)] * len(values)

    norm = Normalize(vmin=vmin, vmax=vmax)
    return [cmap(norm(val)) for val in values]

# --- Experiment-Specific Visualizations ---

def visualize_grid_experiment(
    G: nx.DiGraph,
    pos: dict,
    svd_results: dict,
    betweenness_results: dict
):
    """
    Generates the comprehensive 6-panel visualization for the controlled
    grid experiment, comparing SVD and Betweenness centralities.

    Args:
        G: The grid graph.
        pos: The node positions for plotting.
        svd_results: The dictionary output from SVDCentrality.compute_centralities().
        betweenness_results: A dict with 'node' and 'edge' betweenness arrays.

    Returns:
        A matplotlib Figure object.
    """
    S_v = svd_results['S_v']
    S_e = svd_results['S_e']
    nodes = svd_results['nodes']
    bet_node_cent = betweenness_results['node']
    bet_edge_cent = betweenness_results['edge']

    fig, axes = plt.subplots(2, 3, figsize=(16, 8))


    node_cmap = plt.cm.viridis
    edge_cmap = plt.cm.plasma

    # --- Row 1: Node Centralities ---

    # Panel 1.1: SVD Node Centrality (S_v)
    ax = axes[0, 0]
    node_colors = _create_color_mapping(S_v, node_cmap)
    nx.draw_networkx(G, pos, ax=ax, node_color=node_colors, node_size=500, with_labels=False,
                    edgecolors='black', linewidths=1,
                    edge_color='lightgray', alpha=0.6, arrows=True, arrowsize=15, arrowstyle='->')
    sm = plt.cm.ScalarMappable(cmap=node_cmap, norm=Normalize(vmin=min(S_v), vmax=max(S_v)))
    plt.colorbar(sm, ax=ax, shrink=0.8).set_label("Centrality")

    # Panel 1.2: Betweenness Node Centrality
    ax = axes[0, 1]
    node_colors = _create_color_mapping(bet_node_cent, node_cmap)
    nx.draw_networkx(G, pos, ax=ax, node_color=node_colors, node_size=500, with_labels=False,
                    edgecolors='black', linewidths=1,
                    edge_color='lightgray', alpha=0.6, arrows=True, arrowsize=15, arrowstyle='->')
    sm = plt.cm.ScalarMappable(cmap=node_cmap, norm=Normalize(vmin=min(bet_node_cent), vmax=max(bet_node_cent)))
    plt.colorbar(sm, ax=ax, shrink=0.8).set_label("Centrality")

    # Panel 1.3: Node Correlation
    ax = axes[0, 2]
    corr, _ = pearsonr(S_v, bet_node_cent)
    sns.regplot(x=S_v, y=bet_node_cent, ax=ax, ci=None,
                scatter_kws={'alpha': 0.7, 's': 80},
                line_kws={'color': 'red', 'linestyle': '--'})
    ax.set_xlabel('SVD Node Centrality', fontweight='bold')
    ax.set_ylabel('Betweenness Node Centrality', fontweight='bold')
    ax.grid(True)
    
    # --- Row 2: Edge Centralities ---

    # Panel 2.1: SVD Edge Centrality (S_e)
    ax = axes[1, 0]
    edge_colors = _create_color_mapping(S_e, edge_cmap)
    edge_widths = 1 + 4 * S_e
    nx.draw_networkx(G, pos, ax=ax, with_labels=False, node_size=100, node_color='lightgray',
                    edge_color=edge_colors, width=edge_widths, alpha=0.8, arrows=True, arrowsize=15, arrowstyle='->')
    sm = plt.cm.ScalarMappable(cmap=edge_cmap, norm=Normalize(vmin=min(S_e), vmax=max(S_e)))
    plt.colorbar(sm, ax=ax, shrink=0.8).set_label("Centrality")
    
    # Panel 2.2: Betweenness Edge Centrality
    ax = axes[1, 1]
    edge_colors = _create_color_mapping(bet_edge_cent, plt.cm.Greens)
    edge_widths = 1 + 4 * (bet_edge_cent / (bet_edge_cent.max() or 1.0))
    nx.draw_networkx(G, pos, ax=ax, with_labels=False, node_size=100, node_color='lightgray',
                    edge_color=edge_colors, width=edge_widths, alpha=0.8, arrows=True, arrowsize=15, arrowstyle='->')
    sm = plt.cm.ScalarMappable(cmap=edge_cmap, norm=Normalize(vmin=min(bet_edge_cent), vmax=max(bet_edge_cent)))
    plt.colorbar(sm, ax=ax, shrink=0.8).set_label("Centrality")

    # Panel 2.3: Edge Correlation
    ax = axes[1, 2]
    corr, _ = pearsonr(S_e, bet_edge_cent)
    sns.regplot(x=S_e, y=bet_edge_cent, ax=ax, ci=None,
                scatter_kws={'alpha': 0.7, 's': 80, 'color': 'purple'},
                line_kws={'color': 'red', 'linestyle': '--'})
    ax.set_xlabel('SVD Edge Centrality', fontweight='bold')
    ax.set_ylabel('Betweenness Edge Centrality', fontweight='bold')
    ax.grid(True)
    
    plt.tight_layout(rect=[0, 0, 1, 0.96])
    return fig

def visualize_equivalence_validation(
    G: nx.Graph,
    pos: dict,
    svd_results: dict,
    cf_closeness: np.ndarray
):
    """
    Generates a high-quality 2-panel visualization for the equivalence validation
    experiment (e.g., on Zachary's Karate Club).

    Args:
        G: The graph.
        pos: The node positions for plotting.
        svd_results: The dictionary output from SVDCentrality.compute_centralities().
        cf_closeness: An array of current-flow closeness values.

    Returns:
        A matplotlib Figure object.
    """
    S_v = svd_results['S_v']
    
    fig, axes = plt.subplots(1, 2, figsize=(14, 6))

    node_size = 300
    edge_alpha = 0.4
    edge_width = 1.0
    font_size = 12

    # Panel 1: SVD Node Centrality
    ax = axes[0]
    node_colors_svd = _create_color_mapping(S_v, plt.cm.Blues)
    
    # Draw edges first
    nx.draw_networkx_edges(G, pos, ax=ax, edge_color='gray', width=edge_width, alpha=edge_alpha)
    
    # Draw nodes
    nx.draw_networkx_nodes(G, pos, ax=ax, node_color=node_colors_svd, node_size=node_size,
                           edgecolors='black', linewidths=0.5)
    
    # Remove axis
    ax.axis('off')

    sm_svd = plt.cm.ScalarMappable(cmap=plt.cm.Blues, norm=Normalize(vmin=min(S_v), vmax=max(S_v)))
    sm_svd.set_array([])
    cbar_svd = plt.colorbar(sm_svd, ax=ax, shrink=0.8, orientation='vertical', pad=0.02)
    cbar_svd.set_label("SVD Centrality", fontsize=font_size)

    # Panel 2: Current-Flow Closeness Centrality
    ax = axes[1]
    # Normalize cf_closeness for better color mapping
    cf_closeness_normalized = cf_closeness / (np.max(cf_closeness) or 1.0)
    node_colors_cf = _create_color_mapping(cf_closeness_normalized, plt.cm.Greens)
    
    # Draw edges first
    nx.draw_networkx_edges(G, pos, ax=ax, edge_color='gray', width=edge_width, alpha=edge_alpha)
    
    # Draw nodes
    nx.draw_networkx_nodes(G, pos, ax=ax, node_color=node_colors_cf, node_size=node_size,
                           edgecolors='black', linewidths=0.5)
    
    # Remove axis
    ax.axis('off')

    sm_cf = plt.cm.ScalarMappable(cmap=plt.cm.Greens, norm=Normalize(vmin=min(cf_closeness_normalized), vmax=max(cf_closeness_normalized)))
    sm_cf.set_array([])
    cbar_cf = plt.colorbar(sm_cf, ax=ax, shrink=0.8, orientation='vertical', pad=0.02)
    cbar_cf.set_label("Current-Flow Closeness", fontsize=font_size)
    
    plt.tight_layout()
    return fig

def visualize_equivalence_correlations(results_list: list):
    """
    Generates scatter plots comparing SVD Centrality vs Traditional Closeness
    for multiple networks. Filters out constant/NaN results.

    Args:
        results_list: List of dicts, each containing:
                      {'name': str, 'svd': array, 'closeness': array}

    Returns:
        A matplotlib Figure object.
    """
    # Filter valid results (non-constant, non-NaN)
    valid_results = []
    for res in results_list:
        svd = np.array(res['svd'])
        clo = np.array(res['closeness'])
        
        # Check for NaNs
        if np.isnan(svd).any() or np.isnan(clo).any():
            continue
            
        # Check for constant values (std dev close to 0)
        if np.std(svd) < 1e-9 or np.std(clo) < 1e-9:
            continue
            
        valid_results.append(res)
    
    n_plots = len(valid_results)
    if n_plots == 0:
        return plt.figure() # Return empty figure if no valid plots

    # Determine grid layout
    cols = min(n_plots, 3)
    rows = (n_plots + cols - 1) // cols
    
    fig, axes = plt.subplots(rows, cols, figsize=(5 * cols, 4 * rows))
    if n_plots == 1:
        axes = np.array([axes])
    axes = axes.flatten()

    # Distinct colors for each dataset
    colors = ['indigo', 'teal', 'darkorange', 'firebrick']

    for i, res in enumerate(valid_results):
        ax = axes[i]
        svd = res['svd']
        clo = res['closeness']
        name = res['name']
        
        # Cycle through colors
        plot_color = colors[i % len(colors)]
        
        corr, _ = pearsonr(svd, clo)
        
        sns.regplot(x=svd, y=clo, ax=ax, ci=None,
                    scatter_kws={'alpha': 0.7, 's': 80, 'color': plot_color},
                    line_kws={'color': 'red', 'linestyle': '--', 'linewidth': 1.5})
        
        # Consistent styling with Grid experiment: Serif labels, NO bold
        ax.set_xlabel('SVD Centrality')
        ax.set_ylabel('Current-Flow Closeness')
        ax.grid(True, linestyle='--', alpha=0.3)
        
        # Add text label (optional, can be removed if strictly no text desired)
        # ax.text(0.05, 0.95, f"{name}\n$r = {corr:.3f}$", ...)

    # Hide unused subplots
    for j in range(i + 1, len(axes)):
        axes[j].axis('off')

    plt.tight_layout()
    return fig

def visualize_edge_classification(results_list: list):
    """
    Generates the 4-panel visualization for the edge classification experiment.
    """
    fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(16, 12))

    # Panel 1: Network with classified edges (Karate Club example)
    karate_result = next((r for r in results_list if r['graph_name'] == 'Karate Club'), None)
    if karate_result:
        G = karate_result['graph']
        edge_labels = karate_result['edge_labels']
        pos = nx.spring_layout(G, seed=42, k=1.2, iterations=50) # Harmonized layout
        
        important_edges = [e for e, label in edge_labels.items() if label == 1]
        regular_edges = [e for e, label in edge_labels.items() if label == 0]

        nx.draw_networkx_nodes(G, pos, node_color='lightblue', node_size=200, ax=ax1)
        nx.draw_networkx_edges(G, pos, edgelist=regular_edges, edge_color='gray', width=1, alpha=0.6, ax=ax1)
        nx.draw_networkx_edges(G, pos, edgelist=important_edges, edge_color='red', width=2, alpha=0.8, ax=ax1)
        nx.draw_networkx_labels(G, pos, font_size=6, ax=ax1) # Harmonized font size
        ax1.text(0.01, 0.01, f"{len(important_edges)} bridge edges (red)", transform=ax1.transAxes, fontsize=10)

    # Panel 2: Accuracy comparison
    networks = [r['graph_name'] for r in results_list]
    svd_accs = [r['svd_accuracy'] for r in results_list]
    bet_accs = [r['betweenness_accuracy'] for r in results_list]
    
    x = np.arange(len(networks))
    width = 0.35
    
    bars1 = ax2.bar(x - width/2, svd_accs, width, label='SVD Centrality', color='coral', alpha=0.8) # Harmonized alpha
    bars2 = ax2.bar(x + width/2, bet_accs, width, label='Betweenness Centrality', color='lightblue', alpha=0.8) # Harmonized alpha
    ax2.set_ylabel('Accuracy', fontweight='bold')
    ax2.set_xticks(x)
    ax2.set_xticklabels(networks, rotation=15, ha="right", fontsize=8) # Harmonized rotation and font size
    ax2.legend()
    ax2.grid(axis='y', linestyle='--', alpha=0.7)
    ax2.set_ylim(bottom=0.5)
    # Add value labels
    for bars in [bars1, bars2]:
        for bar in bars:
            height = bar.get_height()
            ax2.text(bar.get_x() + bar.get_width()/2., height + 0.01,
                     f'{height:.3f}', ha='center', va='bottom', fontsize=8) # Harmonized font size

    # Panel 3: AUC comparison
    svd_aucs = [r['svd_auc'] for r in results_list]
    bet_aucs = [r['betweenness_auc'] for r in results_list]

    ax3.bar(x - width/2, svd_aucs, width, label='SVD Centrality', color='coral', alpha=0.8) # Harmonized alpha
    bars2 = ax3.bar(x + width/2, bet_aucs, width, label='Betweenness Centrality', color='lightblue', alpha=0.8) # Harmonized alpha
    ax3.set_ylabel('Area Under Curve (AUC)', fontweight='bold')
    ax3.set_xticks(x)
    ax3.set_xticklabels(networks, rotation=15, ha="right", fontsize=8) # Harmonized rotation and font size
    ax3.legend()
    ax3.grid(axis='y', linestyle='--', alpha=0.7)
    ax3.set_ylim(0, 1)
    # Add value labels
    for bars in [bars1, bars2]:
        for bar in bars:
            height = bar.get_height()
            ax3.text(bar.get_x() + bar.get_width()/2., height + 0.02,
                     f'{height:.3f}', ha='center', va='bottom', fontsize=8) # Harmonized font size

    # Panel 4: Results table
    ax4.axis('off')
    table_data = []
    for r in results_list:
        table_data.append([
            r['graph_name'],
            f"{r['task_name'].replace(' Detection', '')}", # Harmonized task name
            f"{r['svd_accuracy']:.3f}",
            f"{r['betweenness_accuracy']:.3f}",
            f"{r['accuracy_advantage']:+.3f}", # Added accuracy_advantage
            f"{r['auc_advantage']:+.3f}" # Added auc_advantage
        ])
    
    table = ax4.table(
        cellText=table_data,
        colLabels=['Network', 'Task', 'SVD\nAcc.', 'Betw.\nAcc.', 'Acc.\nAdv.', 'AUC\nAdv.'], # Harmonized column labels
        cellLoc='center',
        loc='center',
        bbox=[0, 0.25, 1, 0.5] # Harmonized bbox
    )
    table.auto_set_font_size(False)
    table.set_fontsize(9) # Harmonized font size
    table.scale(1, 2.5) # Harmonized scale
    
    # Color header (Harmonized)
    for i in range(6):
        table[(0, i)].set_facecolor('#E6E6FA')
        table[(0, i)].set_text_props(weight='bold')
    
    # Color advantage columns (Harmonized)
    for i in range(1, len(table_data) + 1):
        acc_adv = float(table_data[i-1][4])
        auc_adv = float(table_data[i-1][5])
        
        if acc_adv > 0:
            table[(i, 4)].set_facecolor('#E8F5E8')
        elif acc_adv < 0:
            table[(i, 4)].set_facecolor('#FFE8E8')
            
        if auc_adv > 0:
            table[(i, 5)].set_facecolor('#E8F5E8')
        elif auc_adv < 0:
            table[(i, 5)].set_facecolor('#FFE8E8')

    
    plt.tight_layout()
    return fig

def visualize_dutch_school_analysis(
    G: nx.DiGraph,
    pos: dict,
    data: dict
):
    """
    Generates two comprehensive 2x3 grid visualizations for the Dutch school network.
    
    Returns:
        (fig_general, fig_directional)
        
    Figure 1 (General):
        Row 1: Node Centrality (SVD, Betweenness, PageRank)
        Row 2: Edge Centrality (SVD, Betweenness, Edge Degree Product)
        
    Figure 2 (Directional):
        Row 1: Authority (SVD, HITS, In-Degree)
        Row 2: Hub (SVD, HITS, Out-Degree)
    """
    # Extract data
    nodes = list(G.nodes())
    edges = list(G.edges())
    
    # Helper to draw individual panels (reused for both figures)
    def draw_panel(ax, values, cmap, label, is_edge=False):
        if is_edge:
            # Edge-centric view
            edge_colors = _create_color_mapping(values, cmap)
            v_min, v_max = np.min(values), np.max(values)
            if v_max > v_min:
                widths = [0.5 + 3.5 * ((v - v_min) / (v_max - v_min)) for v in values]
            else:
                widths = [1.5] * len(values)
            
            nx.draw_networkx_edges(G, pos, ax=ax, edge_color=edge_colors, width=widths, 
                                 arrows=True, arrowsize=12, arrowstyle='-|>', 
                                 connectionstyle='arc3,rad=0.1')
            nx.draw_networkx_nodes(G, pos, ax=ax, node_size=150, node_color='white', 
                                 edgecolors='gray', linewidths=0.5, alpha=0.8)
            sm = plt.cm.ScalarMappable(cmap=cmap, norm=Normalize(vmin=v_min, vmax=v_max))
            
        else:
            # Node-centric view
            node_colors = _create_color_mapping(values, cmap)
            nx.draw_networkx_edges(G, pos, ax=ax, edge_color='lightgray', alpha=0.4, 
                                 width=0.8, arrows=True, arrowsize=10, connectionstyle='arc3,rad=0.1')
            nx.draw_networkx_nodes(G, pos, ax=ax, node_color=node_colors, node_size=400,
                                 edgecolors='black', linewidths=0.8)
            
            # Label top 3 nodes with halo
            top_indices = np.argsort(values)[-3:]
            for idx in top_indices:
                n = nodes[idx]
                x, y = pos[n]
                text = ax.text(x, y, str(n), fontsize=9, fontweight='bold', 
                              horizontalalignment='center', verticalalignment='center',
                              color='black', zorder=10)
                text.set_path_effects([path_effects.withStroke(linewidth=3, foreground='white')])

            sm = plt.cm.ScalarMappable(cmap=cmap, norm=Normalize(vmin=np.min(values), vmax=np.max(values)))

        ax.axis('off')
        sm.set_array([])
        cbar = plt.colorbar(sm, ax=ax, shrink=0.8, pad=0.02)
        cbar.set_label(label, fontsize=11)
        cbar.ax.tick_params(labelsize=9)

    # --- Figure 1: General Centrality (Node & Edge) ---
    fig_gen, axes_gen = plt.subplots(2, 3, figsize=(15, 10))
    
    # Row 1: Node (Blues)
    draw_panel(axes_gen[0,0], data['svd_node'], plt.cm.Blues, "SVD Node ($S_v$)")
    draw_panel(axes_gen[0,1], data['bet_node'], plt.cm.Blues, "Betweenness")
    draw_panel(axes_gen[0,2], data['pagerank'], plt.cm.Blues, "PageRank")
    
    # Row 2: Edge (Oranges)
    draw_panel(axes_gen[1,0], data['svd_edge'], plt.cm.Oranges, "SVD Edge ($S_e$)", is_edge=True)
    draw_panel(axes_gen[1,1], data['bet_edge'], plt.cm.Oranges, "Edge Betw.", is_edge=True)
    draw_panel(axes_gen[1,2], data['edge_deg'], plt.cm.Oranges, "Deg. Prod.", is_edge=True)
    
    plt.tight_layout()

    # --- Figure 2: Directional Roles (Hub & Auth) ---
    fig_dir, axes_dir = plt.subplots(2, 3, figsize=(15, 10))
    
    # Row 1: Authority (Greens)
    draw_panel(axes_dir[0,0], data['svd_auth'], plt.cm.Greens, "SVD Auth.")
    draw_panel(axes_dir[0,1], data['hits_auth'], plt.cm.Greens, "HITS Auth.")
    draw_panel(axes_dir[0,2], data['in_degree'], plt.cm.Greens, "In-Degree")
    
    # Row 2: Hub (Reds)
    draw_panel(axes_dir[1,0], data['svd_hub'], plt.cm.Reds, "SVD Hub")
    draw_panel(axes_dir[1,1], data['hits_hub'], plt.cm.Reds, "HITS Hub")
    draw_panel(axes_dir[1,2], data['out_degree'], plt.cm.Reds, "Out-Degree")

    plt.tight_layout()
    
    return fig_gen, fig_dir

def visualize_grid_hub_authority(
    G: nx.DiGraph,
    pos: dict,
    svd_results: dict,
    other_centralities: dict,
    alpha: float = 0.0
):
    """
    Generates the hub and authority comparison visualization for the controlled grid experiment.
    """
    nodes = svd_results['nodes']
    hub_svd = other_centralities['hub']
    auth_svd = other_centralities['authority']
    bet_node = other_centralities['node_betweenness']
    
    fig, axes = plt.subplots(1, 3, figsize=(18, 6))

    # SVD Hub Centrality
    ax = axes[0]
    node_colors = _create_color_mapping(hub_svd, plt.cm.Reds)
    nx.draw_networkx(G, pos, ax=ax, node_color=node_colors, node_size=500, with_labels=True, font_size=8,
                    edgecolors='black', linewidths=1, edge_color='lightgray', alpha=0.6, arrows=True, arrowsize=15)
    sm = plt.cm.ScalarMappable(cmap=plt.cm.Reds, norm=Normalize(vmin=min(hub_svd), vmax=max(hub_svd)))
    plt.colorbar(sm, ax=ax, shrink=0.8).set_label("SVD Hub Centrality")

    # SVD Authority Centrality
    ax = axes[1]
    node_colors = _create_color_mapping(auth_svd, plt.cm.Blues)
    nx.draw_networkx(G, pos, ax=ax, node_color=node_colors, node_size=500, with_labels=True, font_size=8,
                    edgecolors='black', linewidths=1, edge_color='lightgray', alpha=0.6, arrows=True, arrowsize=15)
    sm = plt.cm.ScalarMappable(cmap=plt.cm.Blues, norm=Normalize(vmin=min(auth_svd), vmax=max(auth_svd)))
    plt.colorbar(sm, ax=ax, shrink=0.8).set_label("SVD Authority Centrality")

    # Betweenness Centrality
    ax = axes[2]
    node_colors = _create_color_mapping(bet_node, plt.cm.Greens)
    nx.draw_networkx(G, pos, ax=ax, node_color=node_colors, node_size=500, with_labels=True, font_size=8,
                    edgecolors='black', linewidths=1, edge_color='lightgray', alpha=0.6, arrows=True, arrowsize=15)
    sm = plt.cm.ScalarMappable(cmap=plt.cm.Greens, norm=Normalize(vmin=min(bet_node), vmax=max(bet_node)))
    plt.colorbar(sm, ax=ax, shrink=0.8).set_label("Betweenness Centrality")
    
    plt.tight_layout()
    return fig

def visualize_grid_raw_svd_vs_betweenness(
    G: nx.DiGraph,
    pos: dict,
    svd_results: dict,
    betweenness_results: dict
):
    """
    Generates a 3-panel visualization for the controlled grid experiment,
    showing SVD raw node centrality (C_v), SVD edge centrality (S_e),
    and Betweenness edge centrality.
    """
    S_v = svd_results['S_v'] # Use S_v as clarified by user
    S_e = svd_results['S_e']
    bet_edge_cent = betweenness_results['edge']

    fig, axes = plt.subplots(1, 3, figsize=(18, 6))

    node_cmap = plt.cm.Reds
    edge_cmap = plt.cm.Purples
    bet_edge_cmap= plt.cm.Greens

    # Panel 1: SVD Node Centrality (S_v)
    ax = axes[0]
    node_colors = _create_color_mapping(S_v, node_cmap)
    nx.draw_networkx(G, pos, ax=ax, node_color=node_colors, node_size=500, with_labels=True, font_size=8,
                    edgecolors='black', linewidths=1, edge_color='lightgray', alpha=0.6, arrows=True, arrowsize=15)
    sm = plt.cm.ScalarMappable(cmap=node_cmap, norm=Normalize(vmin=min(S_v), vmax=max(S_v)))
    plt.colorbar(sm, ax=ax, shrink=0.8).set_label("SVD Node Centrality")

    # Panel 2: SVD Edge Centrality (S_e)
    ax = axes[1]
    edge_colors = _create_color_mapping(S_e, edge_cmap)
    nx.draw_networkx(G, pos, ax=ax, with_labels=False, node_size=100, node_color='lightgray',
                    edge_color=edge_colors, width=1 + 3 * S_e, alpha=0.8, arrows=True, arrowsize=15)
    sm = plt.cm.ScalarMappable(cmap=edge_cmap, norm=Normalize(vmin=min(S_e), vmax=max(S_e)))
    plt.colorbar(sm, ax=ax, shrink=0.8).set_label("SVD Edge Centrality")

    # Panel 3: Betweenness Edge Centrality
    ax = axes[2]
    edge_colors = _create_color_mapping(bet_edge_cent, bet_edge_cmap)
    nx.draw_networkx(G, pos, ax=ax, with_labels=False, node_size=100, node_color='lightgray',
                    edge_color=edge_colors, width=1 + 3 * (bet_edge_cent / (bet_edge_cent.max() or 1.0)), alpha=0.8, arrows=True, arrowsize=15)
    sm = plt.cm.ScalarMappable(cmap=bet_edge_cmap, norm=Normalize(vmin=min(bet_edge_cent), vmax=max(bet_edge_cent)))
    plt.colorbar(sm, ax=ax, shrink=0.8).set_label("Betweenness Edge Centrality")
    
    plt.tight_layout()
    return fig

def visualize_celegans_hub_auth_test(
    G: nx.DiGraph,
    pos: dict,
    svd_results: dict,
    other_centralities: dict,
    alpha: float = 0.0
):
    """
    Generates a special 2x3 visualization for the C. elegans network,
    comparing SVD hub/authority with SVD/betweenness edge centralities.
    """
    S_e = svd_results['S_e']
    nodes = np.array(svd_results['nodes'])
    edges = np.array(svd_results['edges'])
    
    hub_svd = other_centralities['hub']
    auth_svd = other_centralities['authority']
    node_bet = other_centralities['node_betweenness']
    edge_bet = other_centralities['edge_betweenness']
    in_degree = other_centralities['in_degree']

    fig, axes = plt.subplots(2, 3, figsize=(18, 12))

    node_size = 60
    edge_alpha = 0.2
    edge_width_factor = 1.2
    font_size = 12
    
    # --- Row 1: Hub, Authority, and Betweenness Node ---
    # 1. SVD Hub Centrality
    ax = axes[0, 0]
    node_colors_hub = _create_color_mapping(hub_svd, plt.cm.Reds)
    node_alphas_hub = [0.1 + 0.9 * cent for cent in hub_svd]
    
    node_indices = sorted(range(len(hub_svd)), key=lambda i: hub_svd[i], reverse=True)
    for idx in node_indices:
        node = nodes[idx]
        if node in G.nodes():
            nx.draw_networkx_nodes(G, pos, ax=ax, nodelist=[node],
                                  node_color=[node_colors_hub[idx]], node_size=node_size,
                                  edgecolors='black', linewidths=0.5, alpha=node_alphas_hub[idx])

    nx.draw_networkx_edges(G, pos, ax=ax, edge_color='lightgray',
                          alpha=edge_alpha, arrows=True, arrowsize=8, arrowstyle='->')
    ax.axis('off')

    sm_hub = plt.cm.ScalarMappable(cmap=plt.cm.Reds, norm=Normalize(np.min(hub_svd), np.max(hub_svd)))
    sm_hub.set_array([])
    cbar_hub = plt.colorbar(sm_hub, ax=ax, shrink=0.8)
    cbar_hub.set_label('SVD Hub Centrality', fontsize=font_size)
    
    # 2. SVD Authority Centrality
    ax = axes[0, 1]
    node_colors_auth = _create_color_mapping(auth_svd, plt.cm.Blues)
    node_alphas_auth = [0.1 + 0.9 * cent for cent in auth_svd]
    
    node_indices = sorted(range(len(auth_svd)), key=lambda i: auth_svd[i], reverse=True)
    for idx in node_indices:
        node = nodes[idx]
        if node in G.nodes():
            nx.draw_networkx_nodes(G, pos, ax=ax, nodelist=[node],
                                  node_color=[node_colors_auth[idx]], node_size=node_size,
                                  edgecolors='black', linewidths=0.5, alpha=node_alphas_auth[idx])

    nx.draw_networkx_edges(G, pos, ax=ax, edge_color='lightgray',
                          alpha=edge_alpha, arrows=True, arrowsize=8, arrowstyle='->')
    ax.axis('off')

    sm_auth = plt.cm.ScalarMappable(cmap=plt.cm.Blues, norm=Normalize(np.min(auth_svd), np.max(auth_svd)))
    sm_auth.set_array([])
    cbar_auth = plt.colorbar(sm_auth, ax=ax, shrink=0.8)
    cbar_auth.set_label('SVD Authority Centrality', fontsize=font_size)

    # 3. Betweenness Node Centrality
    ax = axes[0, 2]
    node_colors_bet = _create_color_mapping(node_bet, plt.cm.Greens)
    bet_max = max(node_bet) if len(node_bet) > 0 else 1
    node_alphas_bet = [0.1 + 0.9 * (cent / bet_max) if bet_max > 0 else 0.1 for cent in node_bet]
    
    sorted_nodes_bet = sorted(range(len(node_bet)), key=lambda i: node_bet[i], reverse=True)
    for idx in sorted_nodes_bet:
        node = nodes[idx]
        if node in G.nodes():
            nx.draw_networkx_nodes(G, pos, ax=ax, nodelist=[node],
                                  node_color=[node_colors_bet[idx]], node_size=node_size,
                                  edgecolors='black', linewidths=0.5, alpha=node_alphas_bet[idx])

    nx.draw_networkx_edges(G, pos, ax=ax, edge_color='lightgray',
                          alpha=edge_alpha, arrows=True, arrowsize=8, arrowstyle='->')
    ax.axis('off')
    
    sm_bet = plt.cm.ScalarMappable(cmap=plt.cm.Greens, norm=Normalize(np.min(node_bet), np.max(node_bet)))
    sm_bet.set_array([])
    cbar_bet = plt.colorbar(sm_bet, ax=ax, shrink=0.8)
    cbar_bet.set_label('Betweenness Node Centrality', fontsize=font_size)

    # --- Row 2: Edge and In-Degree Centralities ---
    # 1. SVD Edge Centrality
    ax = axes[1, 0]
    edge_colors_se = _create_color_mapping(S_e, plt.cm.Purples)
    edge_widths_se = [1 + edge_width_factor*cent for cent in S_e]
    edge_alphas_se = [0.1 + 0.9 * cent for cent in S_e]

    nx.draw_networkx_nodes(G, pos, ax=ax, node_color='white',
                          node_size=node_size//4, edgecolors='lightgray', linewidths=0.3,
                          alpha=0.0)
    nx.draw_networkx_edges(G, pos, ax=ax, edge_color=edge_colors_se,
                          width=edge_widths_se, alpha=edge_alphas_se, arrows=True,
                          arrowsize=8, arrowstyle='->')
    ax.axis('off')

    sm_svd_edge = plt.cm.ScalarMappable(cmap=plt.cm.Purples, norm=Normalize(np.min(S_e), np.max(S_e)))
    sm_svd_edge.set_array([])
    cbar_svd_edge = plt.colorbar(sm_svd_edge, ax=ax, shrink=0.8)
    cbar_svd_edge.set_label('SVD Edge Centrality', fontsize=font_size)

    # 2. In-Degree Centrality
    ax = axes[1, 1]
    node_colors_degree = _create_color_mapping(in_degree, plt.cm.YlOrBr)
    degree_max = max(in_degree) if len(in_degree) > 0 else 1
    node_alphas_degree = [0.1 + 0.9 * (cent / degree_max) if degree_max > 0 else 0.1 for cent in in_degree]
    
    sorted_nodes_degree = sorted(range(len(in_degree)), key=lambda i: in_degree[i], reverse=True)
    for idx in sorted_nodes_degree:
        node = nodes[idx]
        if node in G.nodes():
            nx.draw_networkx_nodes(G, pos, ax=ax, nodelist=[node],
                                  node_color=[node_colors_degree[idx]], node_size=node_size,
                                  edgecolors='black', linewidths=0.5, alpha=node_alphas_degree[idx])

    nx.draw_networkx_edges(G, pos, ax=ax, edge_color='lightgray',
                          alpha=edge_alpha, arrows=True, arrowsize=8, arrowstyle='->')
    ax.axis('off')
    
    sm_degree = plt.cm.ScalarMappable(cmap=plt.cm.YlOrBr, norm=Normalize(np.min(in_degree), np.max(in_degree)))
    sm_degree.set_array([])
    cbar_degree = plt.colorbar(sm_degree, ax=ax, shrink=0.8)
    cbar_degree.set_label('In-Degree Centrality', fontsize=font_size)

    # 3. Betweenness Edge Centrality
    ax = axes[1, 2]
    bet_edge_max = max(edge_bet) if len(edge_bet) > 0 else 1
    edge_colors_bet = _create_color_mapping(edge_bet, plt.cm.Oranges)
    edge_widths_bet = [max(1, 1 + edge_width_factor*cent) for cent in edge_bet]
    edge_alphas_bet = [0.1 + 0.9 * (cent / bet_edge_max) if bet_edge_max > 0 else 0.1 for cent in edge_bet]

    nx.draw_networkx_nodes(G, pos, ax=ax, node_color='white',
                          node_size=node_size//4, edgecolors='lightgray', linewidths=0.3,
                          alpha=0.0)
    nx.draw_networkx_edges(G, pos, ax=ax, edgelist=list(edges), edge_color=edge_colors_bet,
                          width=edge_widths_bet, alpha=edge_alphas_bet, arrows=True,
                          arrowsize=8, arrowstyle='->')
    ax.axis('off')

    sm_bet_edge = plt.cm.ScalarMappable(cmap=plt.cm.Oranges, norm=Normalize(np.min(edge_bet), np.max(edge_bet)))
    sm_bet_edge.set_array([])
    cbar_bet_edge = plt.colorbar(sm_bet_edge, ax=ax, shrink=0.8)
    cbar_bet_edge.set_label('Betweenness Edge', fontsize=font_size)
    
    plt.tight_layout()
    return fig

def visualize_real_world_analysis(all_network_results: list):
    """
    Generates a set of comparison visualizations for various real-world networks.
    Uses the improved 6-panel layout from the C. elegans hub/authority test.
    
    Layout:
    Row 1: SVD Hub, SVD Authority, Node Betweenness
    Row 2: SVD Edge, In-Degree, Edge Betweenness
    """
    output_figures = {}
    
    # Consistent font size for all text elements
    font_size = 12

    for result in all_network_results:
        G = result['graph']
        display_name = result['display_name']
        S_v = result['svd_results']['S_v']
        S_e = result['svd_results']['S_e']
        nodes = np.array(result['svd_results']['nodes'])
        edges = np.array(result['svd_results']['edges'])
        
        # Unpack baselines and SVD derived
        node_bet = result['baselines']['node_betweenness']
        edge_bet = result['baselines']['edge_betweenness']
        in_degree = result['baselines']['in_degree']
        
        hub_svd = result.get('svd_derived', {}).get('hub_svd', np.zeros(len(nodes)))
        auth_svd = result.get('svd_derived', {}).get('auth_svd', np.zeros(len(nodes)))

        # Use the original graph for visualization to preserve directionality
        viz_graph = G
        
        # Adaptive layout selection based on network size and type
        n_nodes = viz_graph.number_of_nodes()
        if display_name == "US Power Grid":
            pos = nx.spring_layout(viz_graph, seed=42, k=3.0, iterations=150)
            node_size = 30
            edge_alpha = 0.1
            edge_width_factor = 0.8
        elif display_name == "C. elegans PPI":
            # Match the exact settings from run_celegans_hub_auth_test.py
            pos = nx.spring_layout(viz_graph, seed=42)
            node_size = 60
            edge_alpha = 0.2
            edge_width_factor = 1.2
        elif display_name == "Yeast PPI":
            pos = nx.spring_layout(viz_graph, seed=42, k=2.5, iterations=120)
            node_size = 60
            edge_alpha = 0.2
            edge_width_factor = 1.2
        elif display_name == "Euroroad":
            pos = nx.spring_layout(viz_graph, seed=42, k=2.8, iterations=130)
            node_size = 80
            edge_alpha = 0.25
            edge_width_factor = 1.3
        elif n_nodes > 2000:
            pos = nx.kamada_kawai_layout(viz_graph)
            node_size = 50
            edge_alpha = 0.1
            edge_width_factor = 1
        elif n_nodes > 500:
            pos = nx.spring_layout(viz_graph, seed=42, k=1.5, iterations=100)
            node_size = 100
            edge_alpha = 0.3
            edge_width_factor = 1.5
        else:
            pos = nx.spring_layout(viz_graph, seed=42, k=2, iterations=50)
            node_size = 200
            edge_alpha = 0.5
            edge_width_factor = 2

        fig, axes = plt.subplots(2, 3, figsize=(18, 12)) # 2x3 grid

        # --- Row 1: Hub, Authority, Node Betweenness ---
        
        # 1. SVD Hub Centrality (Reds)
        ax = axes[0, 0]
        node_colors_hub = _create_color_mapping(hub_svd, plt.cm.Reds)
        hub_max = max(hub_svd) if len(hub_svd) > 0 else 1.0
        node_alphas_hub = [0.1 + 0.9 * (cent / hub_max) if hub_max > 0 else 0.1 for cent in hub_svd]
        
        node_indices = sorted(range(len(hub_svd)), key=lambda i: hub_svd[i], reverse=True)
        for idx in node_indices:
            node = nodes[idx]
            if node in viz_graph.nodes():
                nx.draw_networkx_nodes(viz_graph, pos, ax=ax, nodelist=[node],
                                      node_color=[node_colors_hub[idx]], node_size=node_size,
                                      edgecolors='black', linewidths=0.5, alpha=node_alphas_hub[idx])

        nx.draw_networkx_edges(viz_graph, pos, ax=ax, edge_color='lightgray',
                              alpha=edge_alpha, arrows=True, arrowsize=8, arrowstyle='->')
        ax.axis('off')

        sm_hub = plt.cm.ScalarMappable(cmap=plt.cm.Reds, norm=Normalize(np.min(hub_svd), np.max(hub_svd)))
        sm_hub.set_array([])
        cbar_hub = plt.colorbar(sm_hub, ax=ax, shrink=0.8)
        cbar_hub.set_label('SVD Hub Centrality', fontsize=font_size)
        
        # 2. SVD Authority Centrality (Blues)
        ax = axes[0, 1]
        node_colors_auth = _create_color_mapping(auth_svd, plt.cm.Blues)
        auth_max = max(auth_svd) if len(auth_svd) > 0 else 1.0
        node_alphas_auth = [0.1 + 0.9 * (cent / auth_max) if auth_max > 0 else 0.1 for cent in auth_svd]
        
        node_indices = sorted(range(len(auth_svd)), key=lambda i: auth_svd[i], reverse=True)
        for idx in node_indices:
            node = nodes[idx]
            if node in viz_graph.nodes():
                nx.draw_networkx_nodes(viz_graph, pos, ax=ax, nodelist=[node],
                                      node_color=[node_colors_auth[idx]], node_size=node_size,
                                      edgecolors='black', linewidths=0.5, alpha=node_alphas_auth[idx])

        nx.draw_networkx_edges(viz_graph, pos, ax=ax, edge_color='lightgray',
                              alpha=edge_alpha, arrows=True, arrowsize=8, arrowstyle='->')
        ax.axis('off')

        sm_auth = plt.cm.ScalarMappable(cmap=plt.cm.Blues, norm=Normalize(np.min(auth_svd), np.max(auth_svd)))
        sm_auth.set_array([])
        cbar_auth = plt.colorbar(sm_auth, ax=ax, shrink=0.8)
        cbar_auth.set_label('SVD Authority Centrality', fontsize=font_size)

        # 3. Betweenness Node Centrality (Greens)
        ax = axes[0, 2]
        bet_max = max(node_bet) if len(node_bet) > 0 else 1.0
        node_colors_bet = _create_color_mapping(node_bet, plt.cm.Greens)
        node_alphas_bet = [0.1 + 0.9 * (cent / bet_max) if bet_max > 0 else 0.1 for cent in node_bet]

        sorted_nodes_bet = sorted(range(len(node_bet)), key=lambda i: node_bet[i], reverse=True)
        for idx in sorted_nodes_bet:
            node = nodes[idx]
            if node in viz_graph.nodes():
                nx.draw_networkx_nodes(viz_graph, pos, ax=ax, nodelist=[node],
                                      node_color=[node_colors_bet[idx]], node_size=node_size,
                                      edgecolors='black', linewidths=0.5, alpha=node_alphas_bet[idx])

        nx.draw_networkx_edges(viz_graph, pos, ax=ax, edge_color='lightgray',
                              alpha=edge_alpha, arrows=True, arrowsize=8, arrowstyle='->')
        ax.axis('off')

        sm_bet = plt.cm.ScalarMappable(cmap=plt.cm.Greens, norm=Normalize(np.min(node_bet), np.max(node_bet)))
        sm_bet.set_array([])
        cbar_bet = plt.colorbar(sm_bet, ax=ax, shrink=0.8)
        cbar_bet.set_label('Betweenness Node Centrality', fontsize=font_size)


        # --- Row 2: Edge centralities and In-Degree ---
        
        # 1. SVD Edge Centrality (Purples)
        ax = axes[1, 0]
        edge_colors_se = _create_color_mapping(S_e, plt.cm.Purples)
        edge_widths_se = [1 + edge_width_factor*cent for cent in S_e]
        edge_alphas_se = [0.1 + 0.9 * cent for cent in S_e]

        nx.draw_networkx_nodes(viz_graph, pos, ax=ax, node_color='white',
                              node_size=node_size//4, edgecolors='lightgray', linewidths=0.3,
                              alpha=0.0)
        nx.draw_networkx_edges(viz_graph, pos, ax=ax, edge_color=edge_colors_se,
                              width=edge_widths_se, alpha=edge_alphas_se, arrows=True,
                              arrowsize=8, arrowstyle='->')
        ax.axis('off')

        sm_svd_edge = plt.cm.ScalarMappable(cmap=plt.cm.Purples, norm=Normalize(np.min(S_e), np.max(S_e)))
        sm_svd_edge.set_array([])
        cbar_svd_edge = plt.colorbar(sm_svd_edge, ax=ax, shrink=0.8)
        cbar_svd_edge.set_label('SVD Edge Centrality', fontsize=font_size)

        # 2. In-Degree Centrality (YlOrBr)
        ax = axes[1, 1]
        degree_max = max(in_degree) if len(in_degree) > 0 else 1.0
        node_colors_degree = _create_color_mapping(in_degree, plt.cm.YlOrBr)
        node_alphas_degree = [0.1 + 0.9 * (cent / degree_max) if degree_max > 0 else 0.1 for cent in in_degree]
        
        sorted_nodes_degree = sorted(range(len(in_degree)), key=lambda i: in_degree[i], reverse=True)
        for idx in sorted_nodes_degree:
            node = nodes[idx]
            if node in viz_graph.nodes():
                nx.draw_networkx_nodes(viz_graph, pos, ax=ax, nodelist=[node],
                                      node_color=[node_colors_degree[idx]], node_size=node_size,
                                      edgecolors='black', linewidths=0.5, alpha=node_alphas_degree[idx])

        nx.draw_networkx_edges(viz_graph, pos, ax=ax, edge_color='lightgray',
                              alpha=edge_alpha, arrows=True, arrowsize=8, arrowstyle='->')
        ax.axis('off')
        
        sm_degree = plt.cm.ScalarMappable(cmap=plt.cm.YlOrBr, norm=Normalize(np.min(in_degree), np.max(in_degree)))
        sm_degree.set_array([])
        cbar_degree = plt.colorbar(sm_degree, ax=ax, shrink=0.8)
        cbar_degree.set_label('In-Degree Centrality', fontsize=font_size)

        # 3. Betweenness Edge Centrality (Oranges)
        ax = axes[1, 2]
        bet_edge_max = max(edge_bet) if len(edge_bet) > 0 else 1.0
        edge_colors_bet = _create_color_mapping(edge_bet, plt.cm.Oranges)
        edge_widths_bet = [max(1, 1 + edge_width_factor*cent) for cent in edge_bet]
        edge_alphas_bet = [0.1 + 0.9 * (cent / bet_edge_max) if bet_edge_max > 0 else 0.1 for cent in edge_bet]

        nx.draw_networkx_nodes(viz_graph, pos, ax=ax, node_color='white',
                              node_size=node_size//4, edgecolors='lightgray', linewidths=0.3,
                              alpha=0.0)
        nx.draw_networkx_edges(viz_graph, pos, ax=ax, edgelist=list(edges), edge_color=edge_colors_bet,
                              width=edge_widths_bet, alpha=edge_alphas_bet, arrows=True,
                              arrowsize=8, arrowstyle='->')
        ax.axis('off')

        sm_bet_edge = plt.cm.ScalarMappable(cmap=plt.cm.Oranges, norm=Normalize(np.min(edge_bet), np.max(edge_bet)))
        sm_bet_edge.set_array([])
        cbar_bet_edge = plt.colorbar(sm_bet_edge, ax=ax, shrink=0.8)
        cbar_bet_edge.set_label('Betweenness Edge Centrality', fontsize=font_size)
        
        plt.tight_layout()
        output_figures[display_name.lower().replace(' ', '_').replace('.', '')] = fig
        
    return output_figures
