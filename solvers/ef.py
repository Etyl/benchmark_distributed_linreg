from benchopt.stopping_criterion import SufficientProgressCriterion
from collections import defaultdict
import time
import numpy as np
from numpy.lib.format import open_memmap

from benchmark_utils.mpi_solver import DistributedMPISolver
from benchmark_utils.compressors import TopK, PowerSGD


class Solver(DistributedMPISolver):
    name = "ef"

    parameters = {
        "n_workers": [1, 16],
        "batch_size": [32],
        "lr": [1e-3],
        "compressor": ["top-10", "power-10"],
    }

    requirements = ["numpy", "mpi4py"]

    stopping_criterion = SufficientProgressCriterion(
        eps=1e-10, patience=3, strategy="iteration"
    )

    @classmethod
    def init_worker(cls, args, comm, rank, world_size):
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

        # Re-init weights
        rng = np.random.RandomState(0)
        W = rng.randn(d1, d2)
        g = np.zeros((d1, d2))
        error = np.zeros((d1, d2))

        compressor_name = args.compressor.split('-')[0]
        k = int(args.compressor.split('-')[1])
        if compressor_name == "top":
            compressor = TopK(k, comm)
        elif compressor_name == "power":
            compressor = PowerSGD(k, comm)
        else:
            raise ValueError(f"Unknown compressor: {compressor_name}")

        for iter in range(n_iter):
            # Sampling
            start_idx = ((iter+1) * args.batch_size) % x_local.shape[0]
            start_idx = start_idx - args.batch_size
            x_batch = x_local[start_idx:start_idx + args.batch_size]
            y_batch = y_local[start_idx:start_idx + args.batch_size]

            # Local Computation
            t_start = time.perf_counter()
            y_pred = x_batch @ W
            grad_local = (-2/args.batch_size) * x_batch.T @ (y_batch - y_pred)

            # Error Feedback: Compress the difference
            grad_local = grad_local + error
            logs['compute_time'].append(time.perf_counter() - t_start)

            # Communication
            t_start = time.perf_counter()
            g_compressed = compressor.communicate(
                grad_local,
                return_compressed=True
            )
            error = grad_local - g_compressed
            logs['comm_time'].append(time.perf_counter() - t_start)

            # Global Update
            t_start = time.perf_counter()
            avg_update = compressor.decompress()
            g += avg_update
            W -= args.lr * g
            logs['update_time'].append(time.perf_counter() - t_start)

        return dict(W=W, logs=dict(logs))


if __name__ == "__main__":
    Solver.entry_point()
