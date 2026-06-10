# CE-EGNN for QM9

Code accompanying the paper [*Counterfactual Explanations for Equivariant Graph Neural Networks: CE-EGNN for Molecular Property Regression*](https://edbticdt2026.github.io/workshop_papers/XAI4Science-paper7.pdf).

## Summary

This repository provides a QM9 regression pipeline based on an Equivariant Graph Neural Network (EGNN) and a counterfactual explainer that learns sparse edge masks while keeping the predictor fixed.

## Contents

- `src/egnn/`: EGNN predictor, data loading, configuration, and training code.
- `src/cf_explainer/`: counterfactual wrapper and optimization loop.
- `src/scripts/run_cf.py`: batch counterfactual generation on the QM9 test split.

## Requirements

Install the Python dependencies with:

```bash
pip install -r requirements.txt
```

The QM9 dataset is downloaded through `torch-geometric` on first use and stored under `data/`.

## Training

Train a property-specific predictor with:

```bash
python -m src.egnn.train_custom --property mu
```

The script saves the best checkpoint and configuration under `models/<property>/`.

## Counterfactual generation

Generate regression-target counterfactuals on the test split with:

```bash
python -m src.scripts.run_cf --property mu --alpha 0.25 --tau 0.15
```

Important arguments:

- `--property`: QM9 target property (`mu`, `alpha`, `zpve`, ...)
- `--alpha`: relative target shift
- `--tau`: tolerance around the target value
- `--epochs`: nbr. of training epochs

Results are written to `outputs/counterfactuals/` as JSON.

