from benchopt.stopping_criterion import SufficientProgressCriterion
from collections import defaultdict
import time
from mpi4py import MPI
import numpy as np
from numpy.lib.format import open_memmap

from benchmark_utils.mpi_solver import DistributedMPISolver


class Solver(DistributedMPISolver):
    name = "pairwise"

    parameters = {
        "n_workers": [1, 4, 16],
        "batch_size": [4, 32],
        "lr": [1e-3],
        "mixing": [0.5],
        "moments": [False]
    }

    requirements = ["numpy", "mpi4py"]

    stopping_criterion = SufficientProgressCriterion(
        eps=1e-10, patience=3, strategy="iteration"
    )

    @classmethod
    def init_worker(cls, args, comm, rank, world_size):
        """
        Initialize the worker environment and load data.
        Identical to the base implementation.
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
        rng = np.random.RandomState(42)
        W = rng.randn(d1, d2)
        W_neighbor = np.zeros_like(W)

        for k in range(n_iter):
            start_idx = ((k+1) * args.batch_size) % x_local.shape[0]
            start_idx = start_idx - args.batch_size

            x_batch = x_local[start_idx:start_idx + args.batch_size]
            y_batch = y_local[start_idx:start_idx + args.batch_size]

            # Local Computation (Gradient Step)
            t_start = time.perf_counter()
            y_pred = x_batch @ W
            dW = (-2/args.batch_size) * x_batch.T @ (y_batch - y_pred)
            logs['compute_time'].append(time.perf_counter() - t_start)

            # Update local model with local gradient immediately
            t_start = time.perf_counter()
            W -= args.lr * dW
            logs['update_time'].append(time.perf_counter() - t_start)

            if world_size > 1:
                # Sparse Communication (Gossip)
                t_start = time.perf_counter()
                # Shift changes every iteration 'k'
                shift = k if (k % world_size) != 0 else k+1
                dest_rank = (rank + shift) % world_size

                req = comm.Isend(W, dest=dest_rank, tag=k)
                comm.Recv(W_neighbor, source=MPI.ANY_SOURCE, tag=k)
                req.Wait()
                logs['comm_time'].append(time.perf_counter() - t_start)

                # Global Update
                t_start = time.perf_counter()
                W = (1 - args.mixing) * W + args.mixing * W_neighbor
                logs['update_time'][-1] += time.perf_counter() - t_start

        return dict(W=W, logs=dict(logs))


if __name__ == "__main__":
    Solver.entry_point()
