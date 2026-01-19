#!/usr/bin/env python3
"""
Reproduce All Paper Results
===========================

This master script runs all refactored experiments in the 'gemisvd' project
to reproduce the figures and analyses presented in the paper.

This script will:
1. Run the Controlled Grid Experiment.
2. Run the Equivalence Validation Experiment (Zachary's Karate Club).
3. Run the Edge Classification Experiment.
4. Run the Dutch School Network Analysis.
5. Run the Real-World Network Analysis.

All output figures will be saved in 'gemisvd/outputs/figures/'.
"""

import os

# Import the main function from each experiment script
from gemisvd.experiments import run_grid_experiment
from gemisvd.experiments import run_equivalence_validation
from gemisvd.experiments import run_edge_classification
from gemisvd.experiments import run_dutch_school_analysis
from gemisvd.experiments import run_real_world_analysis

def main():
    """
    Main function to run all experiments.
    """
    print("==================================================")
    print("  Reproducing All Paper Figures and Analyses")
    print("==================================================")
    
    # Ensure the output directory exists
    output_dir = "gemisvd/outputs/figures"
    os.makedirs(output_dir, exist_ok=True)
    os.makedirs(os.path.join(output_dir, "svg"), exist_ok=True) # Create svg subdirectory
    os.makedirs(os.path.join(output_dir, "pdf"), exist_ok=True) # Create pdf subdirectory
    print(f"All figures will be saved in: {output_dir}\n")

    # --- Run Experiment 1: Controlled Grid ---
    print("\n[1/5] Running Controlled Grid Experiment...")
    try:
        run_grid_experiment.run_experiment()
        print("[1/5] Controlled Grid Experiment COMPLETED")
    except Exception as e:
        print(f"[1/5] Controlled Grid Experiment FAILED: {e}")

    # --- Run Experiment 2: Equivalence Validation ---
    print("\n[2/5] Running Equivalence Validation Experiment...")
    try:
        run_equivalence_validation.run_experiment()
        print("[2/5] Equivalence Validation Experiment COMPLETED")
    except Exception as e:
        print(f"[2/5] Equivalence Validation Experiment FAILED: {e}")

    # --- Run Experiment 3: Edge Classification ---
    print("\n[3/5] Running Edge Classification Experiment...")
    try:
        run_edge_classification.main() # This experiment's entry point is main()
        print("[3/5] Edge Classification Experiment COMPLETED")
    except Exception as e:
        print(f"[3/5] Edge Classification Experiment FAILED: {e}")
        
    # --- Run Experiment 4: Dutch School Analysis ---
    print("\n[4/5] Running Dutch School Network Analysis...")
    try:
        run_dutch_school_analysis.run_experiment()
        print("[4/5] Dutch School Network Analysis COMPLETED")
    except Exception as e:
        print(f"[4/5] Dutch School Network Analysis FAILED: {e}")

    # --- Run Experiment 5: Real-World Network Analysis ---
    print("\n[5/5] Running Real-World Network Analysis...")
    try:
        run_real_world_analysis.run_experiment()
        print("[5/5] Real-World Network Analysis COMPLETED")
    except Exception as e:
        print(f"[5/5] Real-World Network Analysis FAILED: {e}")

    print("\n==================================================")
    print("  All experiments have been executed.")
    print("==================================================")

if __name__ == '__main__':
    main()
