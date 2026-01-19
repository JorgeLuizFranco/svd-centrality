# SVD Centrality: Spectral Centrality Measures for Networks and Hypergraphs

[![Python 3.8+](https://img.shields.io/badge/python-3.8+-blue.svg)](https://www.python.org/downloads/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![DOI](https://img.shields.io/badge/DOI-10.xxxx%2Fxxxxxx-blue)](https://doi.org/10.xxxx/xxxxxx)

A Python package implementing **SVD incidence centrality** for both graphs and hypergraphs, providing mathematically rigorous centrality measures with strong spectral foundations.

## 🔬 About

This package implements the SVD centrality framework described in our research paper. The method provides:

- **Dense rankings**: No zero centrality values for connected components
- **Spectral foundation**: Based on singular value decomposition of incidence matrices  
- **Directional sensitivity**: Handles directed graphs naturally
- **Theoretical guarantees**: Equivalent to current-flow closeness for undirected graphs
- **Hypergraph extension**: Native support for higher-order network structures

## 🚀 Quick Start

### Installation

```bash
pip install svd-centrality
```

Or install from source:
```bash
git clone https://github.com/instituto-curvelo/svd-centrality
cd svd-centrality
pip install -e .
```

### Basic Usage

#### For Standard Graphs

```python
import networkx as nx
from svd_centrality import SVDCentrality

# Load a graph
G = nx.karate_club_graph()

# Compute SVD centrality
svd = SVDCentrality()
results = svd.compute_centralities(G)

# Access centrality scores
vertex_centrality = results['S_v']  # Normalized vertex centralities
edge_centrality = results['S_e']    # Normalized edge centralities

# Print top 5 most central nodes
for i, node in enumerate(results['nodes']):
    print(f"Node {node}: {vertex_centrality[i]:.3f}")
```

#### For Hypergraphs

```python
from svd_centrality.hypergraph import ManualHypergraph, HypergraphSVDCentrality

# Create a hypergraph
H = ManualHypergraph()
H.add_edge([1, 2, 3])      # 3-way interaction
H.add_edge([2, 3, 4, 5])   # 4-way interaction  
H.add_edge([1, 4])         # pairwise interaction

# Compute hypergraph centrality
hsvd = HypergraphSVDCentrality()
results = hsvd.compute_centralities(H)

print("Vertex centralities:", results['normalized_vertex_centrality'])
print("Hyperedge centralities:", results['normalized_edge_centrality'])
```

#### Hub and Authority Analysis

```python
# For directed graphs, compute hub/authority scores
G_directed = nx.DiGraph([(1,2), (2,3), (3,1), (1,3)])
hub_scores, auth_scores = svd.compute_hub_authority_centrality(G_directed)

print("Hub scores:", hub_scores)
print("Authority scores:", auth_scores)
```

## 📊 Advanced Features

### Validation Against Current-Flow Closeness

```python
# Validate theoretical equivalence for undirected graphs
validation = svd.validate_against_current_flow(G)
print(f"Correlation with current-flow closeness: {validation['pearson_correlation']:.3f}")
```

### Spectral Analysis

```python
# Analyze spectral properties
properties = svd.get_spectral_properties(G)
print(f"Matrix rank: {properties['numerical_rank']}")
print(f"Condition number: {properties['condition_number']:.2e}")
print(f"Spectral gap: {properties['spectral_gap']:.3f}")
```

### Performance Statistics

```python
# Get computation statistics
stats = svd.get_computation_stats()
print(f"Computation time: {stats.computation_time:.3f} seconds")
print(f"Memory usage: {stats.memory_usage_mb:.1f} MB")
```

## 🔬 Mathematical Foundation

The SVD centrality framework is based on the following mathematical principles:

### For Graphs

Given a graph $G=(V,E)$ with oriented incidence matrix $B \in \mathbb{R}^{n \times m}$:

1. **SVD Decomposition**: $B = U \Sigma V^T$
2. **Hodge Laplacians**: 
   - $L_0 = BB^T$ (vertex Laplacian)
   - $L_1 = B^TB$ (edge Laplacian)
3. **Pseudoinverses**: 
   - $L_0^+ = U\Sigma^{-2}U^T$
   - $L_1^+ = V\Sigma^{-2}V^T$
4. **Centralities**:
   - Vertex: $C_v(i) = [L_0^+]_{ii}$
   - Edge: $C_e(e) = [L_1^+]_{ee}$

### For Hypergraphs

The framework extends naturally to hypergraphs by constructing the hypergraph incidence matrix $B$ where $B_{ij} = 1$ if vertex $i$ belongs to hyperedge $j$.

### Key Properties

- **Theoretical equivalence**: For undirected graphs, vertex centrality equals current-flow closeness centrality
- **Dense rankings**: Connected components have no zero centrality values
- **Spectral interpretation**: Centrality reflects position in the network's harmonic eigenspace
- **Computational efficiency**: $O(nm)$ sparse matrix operations

## 📈 Applications

This package has been successfully applied to:

- **Social networks**: Identifying influential individuals and communities
- **Biological networks**: Finding critical proteins and pathways
- **Transportation networks**: Locating strategic hubs and routes
- **Collaboration networks**: Discovering key researchers and projects
- **Hypergraph analysis**: Understanding higher-order interactions

## 🛠 Development

### Requirements

- Python 3.8+
- NumPy >= 1.19.0
- SciPy >= 1.5.0  
- NetworkX >= 2.5
- Matplotlib >= 3.3.0 (for visualizations)

### Running Tests

```bash
python -m pytest tests/ -v
```

### Building Documentation

```bash
cd docs/
make html
```

## 📚 Citation

If you use this package in your research, please cite our paper:

```bibtex
@article{svd_centrality_2026,
  title={SVD Incidence Centrality: A Spectral Approach to Network Analysis},
  author={Instituto Curvelo Research Team},
  journal={Journal of Network Analysis},
  year={2026},
  doi={10.xxxx/xxxxxx}
}
```

## 🤝 Contributing

We welcome contributions! Please see [CONTRIBUTING.md](CONTRIBUTING.md) for guidelines.

### Reporting Issues

Please use our [issue tracker](https://github.com/instituto-curvelo/svd-centrality/issues) for:
- Bug reports
- Feature requests  
- Documentation improvements

## 📄 License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## 🙏 Acknowledgments

- The NetworkX team for providing excellent graph analysis tools
- The SciPy community for robust linear algebra implementations
- Our research collaborators and the broader network science community

## 📞 Contact

- **Instituto Curvelo**: [research@institutocurvelo.org](mailto:research@institutocurvelo.org)
- **GitHub**: [https://github.com/instituto-curvelo/svd-centrality](https://github.com/instituto-curvelo/svd-centrality)
- **Documentation**: [https://svd-centrality.readthedocs.io](https://svd-centrality.readthedocs.io)

---

**Keywords**: network analysis, centrality measures, singular value decomposition, spectral graph theory, hypergraphs, current-flow closeness, effective resistance