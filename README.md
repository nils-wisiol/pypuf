pypuf ![Master Branch Build Status Indicator](https://travis-ci.org/nils-wisiol/pypuf.svg?branch=master)
=============

Python package to simulate and learn Physically Unclonable Functions.

Conda Environment Setup For Linux
----------------------------
1. Download the miniconda installer from [here](https://conda.io/miniconda.html).
2. Make the installer executable and run the script. Do not forget to add Minicoda to your PATH variable(this could be done during installation)!
```
# Make the installer executable
chmod +x Miniconda3-latest-Linux-WHATEVER.sh
# Run the installer
./Miniconda3-latest-Linux-WHATEVER.sh
```
3. Switch to your repository and create the virtual environmant.
```
# Switch to your repository
cd /pypuf
# Create the environment
conda env create -f environment.yml
```
4. Activate the created environment.
```
# Activate the virtual environment
source activate pypuf
```
Introduction
------------

Please see `sim_learn.py` for example usage.

