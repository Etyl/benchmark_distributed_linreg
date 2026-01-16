from benchopt.stopping_criterion import SufficientProgressCriterion
from collections import defaultdict
import time
from mpi4py import MPI
import numpy as np
from numpy.lib.format import open_memmap

from benchmark_utils.mpi_solver import DistributedMPISolver


class Solver(DistributedMPISolver):
    name = "slowmo"

    parameters = {
        "n_workers": [1, 4, 16],
        "batch_size": [32],
        "merge_every": [4],
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
        W0 = rng.randn(d1, d2)
        W = W0.copy()

        # Adam Parameters and State initialization
        beta1 = 0.9
        beta2 = 0.999
        epsilon = 1e-8
        m = np.zeros_like(W)  # First moment vector
        v = np.zeros_like(W)  # Second moment vector

        for k in range(n_iter):
            # Sampling
            start_idx = ((k+1) * args.batch_size) % x_local.shape[0]
            start_idx = start_idx - args.batch_size
            x_batch = x_local[start_idx:start_idx + args.batch_size]
            y_batch = y_local[start_idx:start_idx + args.batch_size]

            # Local Computation
            t_start = time.perf_counter()
            y_pred = x_batch @ W
            dW = (-2/args.batch_size) * x_batch.T @ (y_batch - y_pred)
            logs['compute_time'].append(time.perf_counter() - t_start)

            # Update
            t_start = time.perf_counter()
            W -= dW * args.lr
            logs['update_time'].append(time.perf_counter() - t_start)

            # Communication
            t_start = time.perf_counter()
            if (
                (args.merge_every > 0 and (k + 1) % args.merge_every == 0)
                or k == n_iter - 1
            ):
                comm.Allreduce(MPI.IN_PLACE, W, op=MPI.SUM)
                W /= world_size
                dW = W - W0    
                t = k + 1
                m = beta1 * m + (1 - beta1) * dW
                v = beta2 * v + (1 - beta2) * (dW ** 2)

                m_hat = m / (1 - beta1 ** t)
                v_hat = v / (1 - beta2 ** t)

                W -= args.lr * m_hat / (np.sqrt(v_hat) + epsilon)

                W = W0.copy()
            logs['comm_time'].append(time.perf_counter() - t_start)

        return dict(W=W, logs=dict(logs))


if __name__ == "__main__":
    Solver.entry_point()
