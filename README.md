# Machine Learning Framework for Predicting Enthalpy of Vaporization of Liquid Cooling Materials

This repository contains the curated dataset, machine learning models, and analysis scripts supporting the manuscript:

**"Interpretable Machine Learning for the Rational Design of Liquid Cooling Materials with High Enthalpy of Vaporization"**  
(submitted to *ACS Applied Energy Materials*).

The goal of this work is to develop an efficient and interpretable machine learning framework for predicting the enthalpy of vaporization (Hvap) of organic compounds and to accelerate the discovery of high-performance liquid cooling (LC) materials for heat pipe applications.

## Repository Structure

The repository is organized as follows:

- `data/` – Curated datasets including SMILES strings and molecular descriptors.
- `scripts/` – Python scripts for descriptor generation, model training, interpretation, and analysis.
- `models/` – Trained machine learning models used in this work.
- `results/` – Numerical results corresponding to the main findings reported in the manuscript.

## Key Results

The main results reported in the manuscript can be found in the `results/` and `figures/` directories, including:
- Predicted enthalpy of vaporization of candidate LC materials
- Applicability domain analysis
- Nonlinear descriptor contributions based on SHAP analysis
- Scaffold-based model validation
- PCA analysis of the molecular descriptor space

## Reproducibility

The provided data, scripts, and trained models enable full reproduction of the machine learning workflow and results reported in the manuscript.


