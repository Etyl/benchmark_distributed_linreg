from benchopt.stopping_criterion import SufficientProgressCriterion
from collections import defaultdict
import time
from mpi4py import MPI
import numpy as np
from numpy.lib.format import open_memmap


from benchmark_utils.mpi_solver import DistributedMPISolver


class Solver(DistributedMPISolver):
    name = "all-reduce"

    parameters = {
        "n_workers": [1, 4, 16],
        "batch_size": [64],
        "lr": [1e-3],
    }

    requirements = ["numpy", "mpi4py"]

    stopping_criterion = SufficientProgressCriterion(
        eps=1e-10, patience=3, strategy="iteration"
    )

    @classmethod
    def init_worker(cls, args, comm, rank, world_size):
        """
        Initialize the worker environment, clear logs, and load data.
        Returns the local data tensor (X_local).
        """
        x = open_memmap(args.x_path)
        y = open_memmap(args.y_path)

        assert x.shape[0] == y.shape[0], (
            "Number of samples in x and y do not match."
        )
        n_samples = x.shape[0]
        samples_per_worker = n_samples // world_size

        x_local = x[rank * samples_per_worker:(rank + 1) * samples_per_worker]
        y_local = y[rank * samples_per_worker:(rank + 1) * samples_per_worker]

        return x_local, y_local

    @classmethod
    def worker_run(
        cls, n_iter, worker_ctx, args, comm, rank, world_size
    ):
        x_local, y_local = worker_ctx
        d1, d2 = x_local.shape[1], y_local.shape[1]
        logs = defaultdict(list)

        # Re-init weights for every run
        rng = np.random.RandomState(0)
        W = rng.randn(d1, d2)
        G_global = np.zeros_like(W)

        for _ in range(n_iter):
            # Sampling
            indices = np.random.randint(0, len(x_local), (args.batch_size,))
            x_batch = x_local[indices]
            y_batch = y_local[indices]

            # Local Computation
            t_start = time.perf_counter()
            y_pred = x_batch @ W
            dW = (-2/args.batch_size) * x_batch.T @ (y_batch - y_pred)
            logs['compute_time'].append(time.perf_counter() - t_start)

            # Communication
            t_start = time.perf_counter()
            comm.Allreduce(dW, G_global, op=MPI.SUM)
            G_global /= world_size
            logs['comm_time'].append(time.perf_counter() - t_start)

            # Global Update
            t_start = time.perf_counter()
            W -= G_global * args.lr
            logs['update_time'].append(time.perf_counter() - t_start)

        return dict(W=W, logs=dict(logs))


if __name__ == "__main__":
    Solver.entry_point()
