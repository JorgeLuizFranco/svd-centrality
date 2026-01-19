#!/usr/bin/env python3
"""
Setup script for SVD Centrality package.
"""

from setuptools import setup, find_packages
import os

# Read README for long description
with open("README.md", "r", encoding="utf-8") as fh:
    long_description = fh.read()

# Read requirements
requirements = [
    "numpy>=1.19.0",
    "scipy>=1.5.0", 
    "networkx>=2.5",
    "matplotlib>=3.3.0",
]

# Optional dependencies
extras_require = {
    "dev": [
        "pytest>=6.0",
        "pytest-cov>=2.10",
        "black>=21.0",
        "flake8>=3.8",
        "mypy>=0.812",
    ],
    "docs": [
        "sphinx>=3.0",
        "sphinx-rtd-theme>=0.5",
        "sphinx-autodoc-typehints>=1.11",
    ],
    "jupyter": [
        "jupyter>=1.0",
        "ipywidgets>=7.6",
        "seaborn>=0.11",
    ]
}

# All optional dependencies
extras_require["all"] = list(set().union(*extras_require.values()))

setup(
    name="svd-centrality",
    version="1.0.0",
    author="Instituto Curvelo Research Team",
    author_email="research@institutocurvelo.org",
    description="SVD incidence centrality for graphs and hypergraphs",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/instituto-curvelo/svd-centrality",
    project_urls={
        "Bug Tracker": "https://github.com/instituto-curvelo/svd-centrality/issues",
        "Documentation": "https://svd-centrality.readthedocs.io",
        "Paper": "https://doi.org/10.xxxx/xxxxxx",
        "Source": "https://github.com/instituto-curvelo/svd-centrality",
    },
    packages=find_packages(),
    classifiers=[
        "Development Status :: 5 - Production/Stable",
        "Intended Audience :: Science/Research",
        "Intended Audience :: Developers",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: 3.8",
        "Programming Language :: Python :: 3.9",
        "Programming Language :: Python :: 3.10",
        "Programming Language :: Python :: 3.11",
        "Programming Language :: Python :: 3.12",
        "Topic :: Scientific/Engineering :: Mathematics",
        "Topic :: Scientific/Engineering :: Information Analysis",
        "Topic :: Software Development :: Libraries :: Python Modules",
        "Typing :: Typed",
    ],
    python_requires=">=3.8",
    install_requires=requirements,
    extras_require=extras_require,
    keywords=[
        "network analysis",
        "centrality measures", 
        "singular value decomposition",
        "spectral graph theory",
        "hypergraphs",
        "current-flow closeness",
        "effective resistance",
    ],
    entry_points={
        "console_scripts": [
            "svd-centrality=svd_centrality.cli:main",
        ],
    },
    include_package_data=True,
    package_data={
        "svd_centrality": ["data/*.json", "data/*.csv"],
    },
    zip_safe=False,
)