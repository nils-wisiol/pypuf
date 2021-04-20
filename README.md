# pypuf: Cryptanalysis of Physically Unclonable Functions

[![DOI](https://zenodo.org/badge/DOI/10.5281/zenodo.3901410.svg)](https://doi.org/10.5281/zenodo.3901410)
![](https://github.com/nils-wisiol/pypuf/workflows/Doc%20Tests/badge.svg?branch=main)
[![pypi](https://img.shields.io/pypi/v/pypuf.svg)](https://pypi.python.org/pypi/pypuf)

pypuf is a toolbox for simulation, testing, and attacking Physically Unclonable Functions.

## Getting Started

Please check out the [pypuf hello world](https://pypuf.readthedocs.io/en/latest/#getting-started) in the
[documentation](https://pypuf.readthedocs.org).

## Studies and Results

pypuf is used in the following projects:

- 2020, Wisiol et al.: [ Splitting the Interpose PUF: A Novel Modeling Attack Strategy](https://eprint.iacr.org/2019/1473):
  Modeling attacks on the Interpose PUF using Logistic Regression in a Divide-and-Conquer strategy.
- 2020, Wisiol et al.: [Short Paper: XOR Arbiter PUFs have Systematic Response Bias](https://eprint.iacr.org/2019/1091):
  Empirical and theoretical study of XOR Arbiter PUF response bias for unbiased arbiter chains.
- 2019, Wisiol et al.: [Breaking the Lightweight Secure PUF: Understanding the Relation of Input Transformations and Machine Learning Resistance](https://eprint.iacr.org/2019/799):
  An advanced machine learning attack on the Lightweight Secure PUF.
- 2019, Wisiol et al.: [Why Attackers Lose: Design and Security Analysis of Arbitrarily Large XOR Arbiter PUFs](https://doi.org/10.1007/s13389-019-00204-8):
  Simulation of the stabiltiy of Majority Vote XOR Arbiter PUFs.

Please check out the [archived version of pypuf v1](https://github.com/nils-wisiol/pypuf/tree/v1) to find the
original code used in these projects.

## Citation

To refer to pypuf, please use DOI `10.5281/zenodo.3901410`.
pypuf is published [via Zenodo](https://zenodo.org/badge/latestdoi/87066421).
Please cite this work as

> Nils Wisiol, Christoph Gräbnitz, Christopher Mühl, Benjamin Zengin, Tudor Soroceanu, & Niklas Pirnay.
> pypuf: Cryptanalysis of Physically Unclonable Functions (Version v2, March 2021). Zenodo.
> https://doi.org/10.5281/zenodo.3901410

or use the following BibTeX:

```
@software{pypuf,
  author       = {Nils Wisiol and
                  Christoph Gräbnitz and
                  Christopher Mühl and
                  Benjamin Zengin and
                  Tudor Soroceanu and
                  Niklas Pirnay},
  title        = {{pypuf: Cryptanalysis of Physically Unclonable
                   Functions}},
  year         = 2021,
  publisher    = {Zenodo},
  version      = {v2},
  doi          = {10.5281/zenodo.3901410},
  url          = {https://doi.org/10.5281/zenodo.3901410}
}
```

## Contribute

Testing, linting, licensing.

### Run Tests

1. install `sphinx-build xdoctest`
1. `xdoctest pypuf`
1. `cd docs`
1. `make clean`
1. `make doctest && make html`
1. `cd` to project root
1. `python3 -m pytest test`

### Maintainer: Prepare New Release

1. Make sure docs are testing and building without error (see above)
1. Commit all changes
1. Clean up `dist/` folder
1. Set up new release version: `RELEASE=x.y.z`
1. Update version to `x.y.z` in `setup.py` and `docs/conf.py`
1. Commit with message "Release Version vx.y.z": `git commit -p -m "Release Version v$RELEASE"`
1. Tag commit using `git tag -as v$RELEASE -m "Release Version v$RELEASE"`
1. If applicable, adjust `dev` and/or `stable` tags.
1. Push
    1. branch: `git push`
    1. tag: `git push origin v$RELEASE`
1. Set environment variables `GITHUB_TOKEN` to a GitHub token, `TWINE_USERNAME` and `TWINE_PASSWORD` to PyPi
    credentials.
1. Publish using `publish nils-wisiol pypuf`
1. At zenodo.org, make sure the author list is up to date.
