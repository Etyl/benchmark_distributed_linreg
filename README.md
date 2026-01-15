# Distributed Optimization Benchmark

![Build Status](https://github.com/Etyl/benchmark_distributed_linreg/actions/workflows/main.yml/badge.svg)
![Python 3.10+](https://img.shields.io/badge/python-3.10%2B-blue)

This benchmark is dedicated to distributed optimization algorithms.
The simulated dataset corresponds to a linear regression problem distributed across multiple workers.

The optimization problem is

$$
\min_{W} \lVert W X - Y \rVert^2
$$

where:

* $X$ is the data matrix,
* $Y$ is the target matrix,
* $W$ is the weight matrix.

The dataset is split across multiple workers, each worker having access to a subset of samples.
The data can be heterogeneously distributed across workers, where each local dataset $X_i$
is sampled from an offset Gaussian distribution.


## Installation

This benchmark can be run using the following commands:

```bash
pip install -U benchopt
git clone https://github.com/Etyl/benchmark_distributed_linreg
benchopt run benchmark_distributed_linreg --config config.yml
```

Use
`bash
benchopt run -h
`
for more details about the available options, or visit
[https://benchopt.github.io/api.html](https://benchopt.github.io/api.html).


## Solvers

The following solvers are implemented in this benchmark.

### All-reduce

Each worker computes the local gradient on its data, denoted by $\nabla f_i(W)$.
All workers then communicate to compute the global gradient

$$
\nabla f(W) = \frac{1}{N} \sum_{i=1}^N \nabla f_i(W)
$$

Each worker finally updates its local model using this global gradient.

### EF21

Each worker computes its local gradient $\nabla f_i(W)$ and compresses it using a compression operator
$C(\cdot)$, such as TopK, RandomK, or PowerSGD.

EF21 applies the following update:

$$
c_i^t = C\left(\nabla f_i(W) - g^t\right)
$$

The compressed gradients are then aggregated using all-reduce:

$$
g^{t+1} = g^t + \frac{1}{N} \sum_{i=1}^N c_i^t
$$

Each worker updates its local model using the global gradient estimate gᵗ⁺¹.


### Local SGD

Each worker computes the local gradient on its data and updates its model using this local gradient.
After a fixed number of local updates, all workers communicate to average their models.


### Pairwise SGD

Pairwise SGD is similar to Local SGD, but instead of averaging models across all workers,
each worker periodically exchanges models with a neighbouring worker.

