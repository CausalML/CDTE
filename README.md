# CDTE

Tools for estimating quantile and super-quantile conditional distributional treatment effects.

Replication code for [Robust and Agnostic Learning of Conditional Distributional Treatment Effects](https://arxiv.org/abs/2205.11486). 

## Requirements

* [econml](https://github.com/microsoft/EconML)
* [doubleml](https://github.com/DoubleML/doubleml-for-py)

## Replication code

* For Figure 1 & 3, run `python cdte_sims.py`
* For Figure 2 & Table 1, run `python 401k.py`

_Note:_ the original results were obtained using an Amazon Web Services instance with 32 vCPUs and 64 GiB of RAM. These results might take longer to run on other machines.
