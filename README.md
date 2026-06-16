[![DOI](https://zenodo.org/badge/DOI/10.5281/zenodo.17805725.svg)](https://doi.org/10.5281/zenodo.17805725)
# FEniQS
A library for finite element (FE) modelling of static/quasi-static structural mechanics problems with _legacy_ [FEniCS](https://fenicsproject.org/), which contains the following main modules:

**Structure:**

For defining a structural mechanics experiment, including **geometry**, **mesh**, **boundary conditions** (BCs), and _time-varying_ **loadings**. The time is _quasi-static_, i.e. no dynamic (inertia) effects will be accounted for in the problem module below.

**Material:**

For handling **constitutive laws** such as _elasticity_, _gradient damage_, _plasticity_, etc.

**Problem:**

For establishing **structural mechanics problems** for desired structures and material laws coming from the two above modules, and **solving** the problems built up. These can be performed for two main cases: _static_ (no time-evolution) that also includes _homogenization_, and _quasi-static_ (QS).

# Installation
After installing Anaconda, clone the repository and run the following command from the root (of the cloned repository):
```shell
conda env create -f ./environment.yml
```
. This creates a conda environment named as 'feniqs' to be activated and used. Within that environment the package is accessible as 'feniQS'. NOTE: Using [mamba](https://mamba.readthedocs.io/en/latest/index.html) instead of conda is faster/easier.
