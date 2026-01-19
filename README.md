# SVD Incidence Centrality

Implementation of SVD incidence centrality for graphs and hypergraphs.

## Description

This repository contains the implementation of the SVD centrality framework described in:

*"SVD Incidence Centrality: A Unified Spectral Framework for Node and Edge Analysis in Directed Networks and Hypergraphs"*

The method computes vertex and edge centralities through the pseudoinverse of Hodge Laplacians derived from the SVD of the incidence matrix.

## Installation

## Installation

```bash
git clone https://github.com/JorgeLuizFranco/svd-centrality
cd svd-centrality
pip install -e .
```

## Usage

### Basic Example

```python
from svd_centrality import SVDCentrality
import networkx as nx

# Load graph
G = nx.karate_club_graph()

# Compute centralities
svd = SVDCentrality()
results = svd.compute_centralities(G)

vertex_centrality = results['S_v']
edge_centrality = results['S_e']
```

### Hypergraphs

```python
from svd_centrality.hypergraph import ManualHypergraph, HypergraphSVDCentrality

H = ManualHypergraph()
H.add_edge([1, 2, 3])
H.add_edge([2, 3, 4, 5])

hsvd = HypergraphSVDCentrality()
results = hsvd.compute_centralities(H)
```

## Dependencies

- numpy
- scipy  
- networkx
- matplotlib

## Structure

```
svd_centrality/
├── __init__.py           # Core SVD centrality implementation
├── hypergraph/          # Hypergraph extension
├── experiments/         # Reproduction scripts
└── examples/            # Usage examples
```