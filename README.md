# Quality-Weighted Vendi Score

This repository contains the implementation of the Quality-Weighted Vendi Score (qVS), a diversity metric that accounts for the quality of individual items built on top of the previously proposed [Vend Score](https://github.com/vertaix/Vendi-).
The input of the metric is a collection of samples, a pairwise similarity function, and a score function.
The output is a number, which can be interpreted as the effective quality sum of the samples in the collection.
Specifically, given a positive semi-definite matrix $K \in \mathbb{R}^{n \times n}$ of similarity values and a score vector $\boldsymbol{s}$, the qVS is defined to be:
$$\mathrm{qVS}(K, \boldsymbol{s}) = \left( \sum_i s_i / n \right) \exp(-\mathrm{tr}(K/n \log K/n)) = \left( \sum_i s_i / n \right) \exp(-\sum_{i=1}^n \lambda_i \log \lambda_i),$$
where $\lambda_i$ are the eigenvalues of $K/n$ and $0 \log 0 = 0$.

## Usage

The input to `q_vendi.score` is a list of samples, a similarity function `k`, and a score function `s`.
`k` should be symmetric and `k(x, x) = 1`.
```python
>>> import numpy as np
>>> from q_vendi import *

>>> samples = [0, 0, 2, 2, 4, 4]
>>> k = lambda a, b: np.exp(-np.abs(a - b))
>>> s = lambda a: np.exp(-np.abs(a - 2))

>>> score(samples[:3], k, s)

0.793705659274703
```

You can find the subset that maximizes the qVS:
```python
>>> selected_samples, qVS = sequential_maximize_score(samples, k, s, 3)

>>> selected_samples
[2, 0, 2]

>>> qVS
1.2551553872451062
```

An example in 2d is included in a Jupyter notebook in the `examples/` folder.

## Experimental Design

The qVS is used for experimental design and active learning tasks, specifically active search and Bayesian optimization aiming at making diverse discoveries.
See the respective subdirectories `diverse_search` and `diverse_bayesopt` for more details.

## Citation
```bibtex
@article{nguyen2024quality,
title={Quality-Weighted Vendi Score for Diverse Experimental Design},
author={Nguyen, Quan and Dieng, Adji Bousso},
journal={arXiv preprint arXiv:[PENDING]},
year={2024}
}
```
