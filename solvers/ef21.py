from benchopt.stopping_criterion import SufficientProgressCriterion
from collections import defaultdict
import time
from mpi4py import MPI
import numpy as np
from numpy.lib.format import open_memmap

from benchmark_utils.mpi_solver import DistributedMPISolver



def compress(tensor, compressor, k):
    """
    Returns a sparse representation (values, indices) of the top-k elements.
    """
    if compressor == "top":
        flat_tensor = tensor.flatten()
        d = flat_tensor.size
        
        # Safety check if k is larger than dimension
        if k >= d:
            return flat_tensor, np.arange(d, dtype=np.int32)
        
        # Efficiently find indices of top-k absolute values
        # argpartition is O(n), much faster than full sort
        indices = np.argpartition(np.abs(flat_tensor), -k)[-k:]
        values = flat_tensor[indices]
        
        return values, indices.astype(np.int32)
    else:
        raise ValueError(f"Unknown compressor: {compressor}")


def decompress(sparse_data, compressor, shape):
    """
    Reconstructs the dense tensor from sparse (values, indices).
    """
    if compressor == "top":
        values, indices = sparse_data
        
        # Create empty dense tensor
        tensor = np.zeros(np.prod(shape), dtype=values.dtype)
        
        # Fill in the non-zero values
        # np.add.at is used here in case of repeated indices, though unlikely in simple top-k
        np.put(tensor, indices, values)
        
        return tensor.reshape(shape)
    else:
        raise ValueError(f"Unknown compressor: {compressor}")


class Solver(DistributedMPISolver):
    name = "ef21"

    parameters = {
        "n_workers": [1, 16],
        "batch_size": [32],
        "lr": [1e-3],
        "compressor": ["top-10"],
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
        
        compressor_name = args.compressor.split('-')[0]
        k = int(args.compressor.split('-')[1])

        # Pre-allocate buffers for Allgather
        # We will receive 'k' values and 'k' indices from EVERY worker
        recv_values = np.empty((world_size, k), dtype=W.dtype)
        recv_indices = np.empty((world_size, k), dtype=np.int32)

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
            diff = grad_local - g
            logs['compute_time'].append(time.perf_counter() - t_start)

            # Communication
            t_start = time.perf_counter()
            
            # 1. Compress to sparse format
            local_vals, local_idxs = compress(diff, compressor_name, k)
            
            # 2. Gather all sparse updates (Values and Indices)
            # Note: Allgather is used because we cannot sum indices.
            comm.Allgather(local_vals, recv_values)
            comm.Allgather(local_idxs, recv_indices)
            
            logs['comm_time'].append(time.perf_counter() - t_start)

            # Global Update
            t_start = time.perf_counter()
            
            # 3. Aggregate (Decompress and Sum)
            # We reconstruct the updates from all workers locally
            avg_update = np.zeros((d1, d2))
            
            for i in range(world_size):
                # Decompress individual worker updates
                worker_update = decompress(
                    (recv_values[i], recv_indices[i]), 
                    compressor_name, 
                    (d1, d2)
                )
                avg_update += worker_update
            
            avg_update /= world_size
            
            # EF21 Update rule
            g += avg_update
            W -= args.lr * g
            
            logs['update_time'].append(time.perf_counter() - t_start)

        return dict(W=W, logs=dict(logs))


if __name__ == "__main__":
    Solver.entry_point()