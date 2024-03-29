[![StandWithPalestineBadge](https://raw.githubusercontent.com/saedyousef/StandWithPalestine/main/badges/flat/IStandWithPalestine.svg)](https://techforpalestine.org/learn-more)

# The Cellular Potts Model using Cython


![](https://github.com/vandasari/mt1mmp_paper/blob/main/Movie3_Figure3.gif)


## Description

This repository contains Cython code to simulate individual cell migration driven by the dynamics of MT1-MMP using the Cellular Potts Model, for the paper below.

## Get Started

Each folder has files (`mt1mmp.pyx`, `setup.py`, and `main.py`) for Figures 2 and 3 in the paper.

Go to each directory. 

To compile the code:<br>
```python
$ python setup.py build_ext --inplace
```

To run:<br>
```python
$ python main.py 
```

The `main.py` file contains:
* CPM parameter values
* Options to display and save images

## Citation

To cite the code: <br>

Vivi Andasari and Muhammad Zaman<br>
Multiscale Modeling of MT1-MMP-Mediated Cell Migration: Destabilization of Cell-Matrix Adhesion<br>
bioRxiv (2022), doi: https://www.biorxiv.org/content/10.1101/2022.10.12.511909
