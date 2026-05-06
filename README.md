# pypuf-temp

## Project Overview

`pypuf-temp` is a research-ready fork of the original [pypuf](https://github.com/berndpfrommer/pypuf) library. It extends the core Python PUF simulation package with environmental sweep analysis, reliability metrics, and machine learning attack experiments for Arbiter, XOR, Feed-Forward, Interpose, Lightweight Secure, and Permutation PUF families.

The repository contains:

- `pypuf/`: the core PUF package and simulation framework.
- `USE_CASE/`: example scripts and notebooks for reliability, metric sweeps, and plot generation.
- `machine learning attacks/`: end-to-end machine learning attack experiments and evaluation scripts.
- `PHYSICAL_FACTORS_ANALYSIS.md`: design notes and analysis of temperature/voltage effects on PUF security.

## Repository Structure

- `pypuf/`: Python package for PUF simulation, CRP generation, attacks, and metrics.
- `USE_CASE/`: analysis scripts, notebooks, and saved plots.
- `machine learning attacks/`: experiments that reproduce MLP, logistic regression, LMN, and least-squares attacks on PUF families.

## Installation

1. Create and activate a Python virtual environment.

   ```bash
   python -m venv .venv
   .\.venv\Scripts\Activate.ps1  # PowerShell
   # or .\.venv\Scripts\activate  # CMD
   ```

2. Install the required packages.

   ```bash
   python -m pip install --upgrade pip
   python -m pip install -r pypuf/requirements.txt
   python -m pip install -e pypuf
   ```

3. Install Jupyter if you want to run notebooks interactively.

   ```bash
   python -m pip install jupyter
   ```

## Reproduce Machine Learning Attacks

The machine learning attacks are implemented as scripts and notebooks.

### Run the main attack sweep

```bash
python "machine learning attacks/run_experiment.py"
```

This script trains an `MLPAttack2021` model for each PUF type and sweeps over `m` and `alpha` parameters. It produces:

- `*_m_alpha_accuracy.csv`
- `*_m_alpha_heatmap.png`
- `PUF_parameter_sensitivity_summary.csv`

### Run the FeedForward interpose experiment

```bash
python "machine learning attacks/run_experiment_interpose.py"
```

This script trains and evaluates a `FeedForwardArbiterPUF` under parameter variation and saves:

- `feedforward_m_alpha_accuracy.csv`

### Recreate attack notebooks

Open the following notebooks in Jupyter to reproduce interactive plots:

- `machine learning attacks/mlp_attacks.ipynb`
- `machine learning attacks/logistic_attacks.ipynb`
- `machine learning attacks/leastSquares_regression_attacks.ipynb`
- `machine learning attacks/LMN_attacks.ipynb`

## Reproduce Reliability and Metric Plots

The following example scripts generate the repository's metric visuals:

- `python USE_CASE/plot_metrics_sweeps.py`
- `python USE_CASE/verify_reliability_plot.py`
- `python USE_CASE/manual_metrics/PUF_performance_reliability.py`
- `python USE_CASE/manual_metrics/PUF_performance_reliability_architechtures.py`

Generated plot folders include:

- `USE_CASE/graphs_metrics/`
- `USE_CASE/manual_metrics/graphs/`
- `USE_CASE/manual_metrics/graphs_reliability/`

## Notes on Reproducibility

- The experiments use fixed seeds for PUF initialization and challenge generation.
- The attack scripts use `MLPAttack2021` from `pypuf.attack` with consistent network architecture.
- Environment variation is simulated through temperature and voltage sweeps.

## Viewing Embedded Figures

Several notebooks now include inline example images so you can see representative output directly when the notebook is viewed in GitHub or Jupyter.

## License and Acknowledgements

This repository is based on the original `pypuf` library by Bernd Pfrommer and contributors. Review `pypuf/LICENSE.txt` for license details.
