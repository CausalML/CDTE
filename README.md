# CDTE

Tools for estimating quantile, super-quantile and f-risk conditional distributional treatment effects.

Replication code for [Robust and Agnostic Learning of Conditional Distributional Treatment Effects](https://arxiv.org/abs/2205.11486). 

## Requirements

* [econml](https://github.com/microsoft/EconML)
* [doubleml](https://github.com/DoubleML/doubleml-for-py)
* [sklearn-quantile](https://pypi.org/project/sklearn-quantile/)

## Replication code

* For Figure 2, run `python csqte_sims.py`
* For Figure 3, Figure 6 & Table 1, run `python 401k.py`
* For Figure 4, run `python cqte_sims.py`
* For Figure 5, run `python cklrte_sims.py`

_Note:_ the original results were obtained using an Amazon Web Services instance with 32 vCPUs and 64 GiB of RAM. These results might take longer to run on other machines.
