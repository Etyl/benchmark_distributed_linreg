==================================
Distributed Optimization Benchmark
==================================
|Build Status| |Python 3.10+|

This benchmark is dedicated to distributed optimizers.
The simulated dataset is a linear regression problem distributed over multiple workers.

.. math::

   \min_{W} \| W X - Y \|^2,

where :math:`X` is the matrix of data, :math:`Y` is the target matrix, and
:math:`W` is the weight matrix.
The data is split across multiple workers, each worker having access to a subset of samples.
These samples can be heterogeneously distributed across workers, where each :math:`X_i` is sampled
from an offset Gaussian distribution.

Solvers
=======

The following solvers are implemented in this benchmark:

All-reduce
----------

Each worker computes the local gradient on its data, :math:`\nabla f_i(W)`, then
all workers communicate to compute the global gradient

.. math::

   \nabla f(W) = \frac{1}{N} \sum_{i=1}^N \nabla f_i(W),

and finally each worker updates its local model using the global gradient.

EF21
----

Each worker computes the local gradient on its data, :math:`\nabla f_i(W)`, and compresses it using a
compression operator :math:`C(\cdot)` (e.g., TopK, RandomK, PowerSGD).

We apply EF21:

.. math::

   c_i^t = C(\nabla f_i(W) - g^t)

We then communicate the compressed gradients with all-reduce across all workers:

.. math::

   g^{t+1} = g^t + \frac{1}{N} \sum_{i=1}^N c_i^t

Finally, each worker updates its local model using the global gradient estimate
:math:`g^{t+1}`.

Local SGD
---------

Each worker computes the local gradient on its data, :math:`\nabla f_i(W)`, and updates its local
model using this local gradient.
After a fixed number of local updates, all workers communicate to average their models.

Pairwise SGD
------------

Similar to Local SGD, but instead of averaging models across all workers,
each worker selects a neighbouring worker to exchange models with.

Install
=======

This benchmark can be run using the following commands:

.. code-block:: bash

   pip install -U benchopt
   git clone https://github.com/Etyl/benchmark_distributed_linreg
   benchopt run benchmark_distributed_linreg

Apart from the problem, options can be passed to ``benchopt run``, to restrict the benchmarks
to some solvers or datasets, e.g.:

.. code-block:: bash

   benchopt run benchmark_distributed_linreg -s solver1 -d dataset2 --max-runs 10 --n-repetitions 10

Use ``benchopt run -h`` for more details about these options, or visit
https://benchopt.github.io/api.html.

.. |Build Status| image:: https://github.com/Etyl/benchmark_distributed_linreg/actions/workflows/main.yml/badge.svg
   :target: https://github.com/Etyl/benchmark_distributed_linreg/actions
.. |Python 3.10+| image:: https://img.shields.io/badge/python-3.10%2B-blue
   :target: https://www.python.org/downloads/release/python-3100/
