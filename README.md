# pypuf: Cryptanalysis of Physically Unclonable Functions

[![DOI](https://zenodo.org/badge/DOI/10.5281/zenodo.3901435.svg)](https://doi.org/10.5281/zenodo.3901435)

pypuf is a toolbox for simulation, testing, and attacking Physically Unclonable Functions.

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

## Using pypuf

To get started, please check out the [documentation](https://pypuf.readthedocs.org).

## Contribute

Testing, linting, licensing.

### Update Documentation and Check Doc Tests

1. install `sphinx-build`
1. `cd docs`
1. `make clean`
1. `make doctest && make html`

### Maintainer: Prepare New Release

1. Make sure docs are testing and building without error (see above)
1. Commit all changes
1. Clean up `dist/` folder
1. Set up new release version: `RELEASE=x.y.z`
1. Update version to `x.y.z` in `setup.py` and `docs/conf.py`
1. Commit with message "Release Version vx.y.z": `git commit -p -m "Release Version v$RELEASE"`
1. Tag commit using `git tag -as v$RELEASE -m "Release Version v$RELEASE"`
1. Push
    1. branch: `git push`
    1. tag: `git push origin v$RELEASE`
1. Set environment variables `GITHUB_TOKEN` to a GitHub token, `TWINE_USERNAME` and `TWINE_PASSWORD` to PyPi
    credentials.
1. Publish using `publish nils-wisiol pypuf`

## Citation

pypuf is published [via Zenodo](https://zenodo.org/record/3901435). Please cite this work as (update date and version
as appropriate)

> Nils Wisiol, Christopher Gräbnitz, Christopher Mühl, Benjamin Zengin, Tudor Soroceanu, & Niklas Pirnay. (2020, June 19). pypuf: Cryptanalysis of Physically Unclonable Functions (Version v0.0.6). Zenodo. http://doi.org/10.5281/zenodo.3901435

or [download BibTeX directly from Zenodo](https://zenodo.org/record/3901435/export/hx).
