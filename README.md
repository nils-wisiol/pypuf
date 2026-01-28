# pypuf-temp

## Overview

This project is based on the original [pypuf](https://github.com/berndpfrommer/pypuf) library, which provides Python implementations and simulation tools for various Physical Unclonable Functions (PUFs), including Arbiter PUFs, XOR Arbiter PUFs, Feed-Forward Arbiter PUFs, and Bistable Ring PUFs. The library is designed for research and educational purposes, enabling users to simulate, analyze, and evaluate the performance of PUF architectures under different environmental conditions.

## Modifications and Enhancements

This fork includes several enhancements and new example scripts for advanced PUF reliability analysis:

- **Fine-grained Environmental Sweeps:** Added scripts to sweep temperature and voltage with fine increments for detailed reliability analysis.
- **Consistent Reliability Metrics:** Improved reliability calculation logic to ensure fair comparison across different PUF sizes and environmental conditions.
- **Interactive and High-Resolution Plotting:** Example scripts now support both interactive plotting (for use in Spyder/Jupyter) and high-resolution image saving for publication-quality figures.
- **Seed and Challenge Control:** Ensured reproducibility by fixing seeds and using consistent challenge sets for all sweeps.
- **Batch Analysis:** Example scripts can analyze multiple PUF sizes (n) in a single run, producing comparative plots.

## Example Files in `USE_CASE` Folder

The `USE_CASE` folder contains example scripts demonstrating advanced analysis and usage of the library:

- **PUF_performance_reliability_architechtures.py**
  - Analyzes reliability of ArbiterPUF for different numbers of challenge bits (n) under voltage and temperature sweeps.
  - Supports interactive plotting for visual analysis and high-resolution saving for publication.
  - Uses fixed seeds and consistent challenge sets for fair comparison.

- **PUF_performance_reliability.py**
  - Provides additional reliability and bias analysis for various PUF architectures (Arbiter, XOR, FeedForward, etc.).
  - Includes both voltage and temperature sweep modes, with CSV export for further analysis.

- **try.py / try.ipynb**
  - Quick test scripts and notebooks for experimenting with PUF instantiation, evaluation, and plotting.

## Usage

1. Install dependencies from `requirements.txt`.
2. Run example scripts from the `USE_CASE` folder:
   ```bash
   python PUF_performance_reliability_architechtures.py
   # or
   python PUF_performance_reliability.py
   ```
3. For interactive analysis, open the scripts in Spyder or Jupyter and use the provided plotting functions.

## Acknowledgements

Original library by Bernd Pfrommer and contributors. See `LICENSE.txt` for details.

---

For more details, see the original [pypuf documentation](https://github.com/berndpfrommer/pypuf) and the example scripts in the `USE_CASE` folder.
