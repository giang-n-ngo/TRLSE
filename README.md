# TRLSE - Trust Region Level Set Estimation

This repository contains the implementation of Trust Region-based Level Set Estimation algorithms.

## Overview

This project implements various optimization and level set estimation methods including:
- Trust Region Level Set Estimation (TR-LSE)
- Baseline Level Set Estimation models
- High-dimensional LSE methods

## Structure

- `TR_lse.py` - Main Trust Region LSE implementation
- `baseline_lse.py` - Baseline LSE algorithms
- `TR_models.py` - Trust Region model definitions
- `baseline_models.py` - Baseline model definitions
- `utils.py` - Utility functions
- `dataloaders.py` - Data loading utilities
- `results/` - Experimental results
- `plot_figures/` - Generated plots and visualizations

## Requirements

See `requirements.txt` for dependencies.

## Usage

Run experiments using the provided shell scripts:
- `run_baseline.sh` - Run baseline experiments
- `run_TR_lse_boundary.sh` - Run TR-LSE with boundary exploration
- `run_TR_lse_exploit.sh` - Run TR-LSE with exploitation strategy

## Results

Results are stored in the `results/` directory, organized by problem dimension and method.

## License

See LICENSE file for details.
