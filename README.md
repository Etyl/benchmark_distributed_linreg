# Distributed Optimization Benchmark

![Build Status](https://github.com/Etyl/benchmark_distributed_linreg/actions/workflows/main.yml/badge.svg)
![Python 3.10+](https://img.shields.io/badge/python-3.10%2B-blue)

This benchmark is dedicated to distributed optimizers.
The simulated dataset is a linear regression problem distributed over multiple workers.

$$
\min_{W} || W X - Y ||^2,
$$

where ($X$) is the matrix of data, ($Y$) is the target matrix, and
($W$) is the weight matrix.

The data is split across multiple workers, each worker having access to a subset of samples.
These samples can be heterogeneously distributed across workers, where each ($X_i$) is sampled
from an offset Gaussian distribution.

## Install

This benchmark can be run using the following commands:

```bash
pip install -U benchopt
git clone https://github.com/Etyl/benchmark_distributed_linreg
benchopt run benchmark_distributed_linreg --config config.yml
```

Use `benchopt run -h` for more details about the options, or visit
[https://benchopt.github.io/api.html](https://benchopt.github.io/api.html).


## Solvers

The following solvers are implemented in this benchmark:

### All-reduce

Each worker computes the local gradient on its data, ($\nabla f_i(W)$), then
all workers communicate to compute the global gradient

$$
\nabla f(W) = \frac{1}{N} \sum_{i=1}^N \nabla f_i(W),
$$

and finally each worker updates its local model using the global gradient.

### EF21

Each worker computes the local gradient on its data, ($\nabla f_i(W)$), and compresses it using a
compression operator ($C(\cdot)$) (e.g., TopK, RandomK, PowerSGD).

We apply EF21:

$$
c_i^t = C(\nabla f_i(W) - g^t)
$$

We then communicate the compressed gradients with all-reduce across all workers:

$$
g^{t+1} = g^t + \frac{1}{N} \sum_{i=1}^N c_i^t
$$

Finally, each worker updates its local model using the global gradient estimate
($g^{t+1}$).

### Local SGD

Each worker computes the local gradient on its data, ($\nabla f_i(W)$), and updates its local
model using this local gradient.
After a fixed number of local updates, all workers communicate to average their models.

### Pairwise SGD

Similar to Local SGD, but instead of averaging models across all workers,
each worker selects a neighbouring worker to exchange models with.
