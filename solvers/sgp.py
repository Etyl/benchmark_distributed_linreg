from benchopt.stopping_criterion import SufficientProgressCriterion
from collections import defaultdict
import time
from mpi4py import MPI
import numpy as np
from numpy.lib.format import open_memmap

from benchmark_utils.mpi_solver import DistributedMPISolver

# Implementation of Stochastic Gradient Push (SGP)
# https://arxiv.org/pdf/1811.10792


class Solver(DistributedMPISolver):
    name = "sgp"

    parameters = {
        "n_workers": [4, 16],
        "batch_size": [32],
        "lr": [1e-3],
        "overlap": [0, 1],
    }

    requirements = ["numpy", "mpi4py"]

    stopping_criterion = SufficientProgressCriterion(
        eps=1e-10, patience=3, strategy="iteration"
    )

    @classmethod
    def init_worker(cls, args, comm, rank, world_size):
        x = open_memmap(args.x_path)
        y = open_memmap(args.y_path)

        n_samples = x.shape[0]
        samples_per_worker = n_samples // world_size

        x_local = x[rank * samples_per_worker:(rank + 1) * samples_per_worker]
        y_local = y[rank * samples_per_worker:(rank + 1) * samples_per_worker]

        return x_local, y_local

    @classmethod
    def worker_run(cls, n_iter, worker_ctx, args, comm, rank, world_size):
        x_local, y_local = worker_ctx
        d1, d2 = x_local.shape[1], y_local.shape[1]
        logs = defaultdict(list)

        rng = np.random.RandomState(0)

        # SGP State
        W = rng.randn(d1, d2)
        ps_weight = 1.0  # Scalar Push-Sum weight

        # Requests handles for non-blocking communication
        req_send = None
        req_recv = None

        for k in range(n_iter):
            start_idx = ((k+1) * args.batch_size) % x_local.shape[0]
            x_batch = x_local[start_idx:start_idx + args.batch_size]
            y_batch = y_local[start_idx:start_idx + args.batch_size]

            # --- 1. Compute Gradient (Always happens) ---
            t_start = time.perf_counter()

            # De-bias model for computation: W_model = W / ps_weight
            W_model = W / ps_weight

            y_pred = x_batch @ W_model
            grads = (-2/args.batch_size) * x_batch.T @ (y_batch - y_pred)

            logs['compute_time'].append(time.perf_counter() - t_start)

            # --- 2. Local Update ---
            t_start = time.perf_counter()
            # Standard SGD update on the numerator
            W -= args.lr * grads
            logs['update_time'].append(time.perf_counter() - t_start)

            # --- 3. Communication Strategy ---
            t_start = time.perf_counter()

            if world_size > 1:
                # Logic for Standard Blocking SGP (overlap=0)
                if args.overlap == 0:
                    shift = k if (k % world_size) != 0 else k+1
                    dest_rank = (rank + shift) % world_size

                    # Prepare Payload
                    to_send_W = W * 0.5
                    to_send_w = ps_weight * 0.5

                    # Update local mass immediately
                    W *= 0.5
                    ps_weight *= 0.5
                    # Blocking Send/Recv
                    comm.send((to_send_W, to_send_w), dest=dest_rank, tag=k)
                    W_neighbor, w_neighbor = comm.recv(
                        source=MPI.ANY_SOURCE,
                        tag=k
                    )

                    # Aggregate
                    W += W_neighbor
                    ps_weight += w_neighbor

                # Logic for tau-OSGP (overlap > 0)
                else:
                    # We check if we need to sync based on tau
                    is_comm_step = (k % args.overlap == 0)

                    if is_comm_step:
                        # A. Finalize Previous Communication (if any)
                        if req_recv is not None:
                            # Wait for data from tau iterations ago
                            W_neighbor, w_neighbor = req_recv.wait()
                            req_send.wait()  # Ensure send is also done

                            # Aggregate (Late Mixing)
                            W += W_neighbor
                            ps_weight += w_neighbor

                            req_recv = None
                            req_send = None

                        # B. Initiate New Communication
                        shift = k if (k % world_size) != 0 else k+1
                        dest_rank = (rank + shift) % world_size

                        # Prepare Payload
                        to_send_W = (W * 0.5).copy()
                        to_send_w = ps_weight * 0.5

                        # Update local mass
                        W *= 0.5
                        ps_weight *= 0.5

                        # Non-blocking Start
                        req_send = comm.isend(
                            (to_send_W, to_send_w),
                            dest=dest_rank,
                            tag=k
                        )
                        req_recv = comm.irecv(source=MPI.ANY_SOURCE, tag=k)

            logs['comm_time'].append(time.perf_counter() - t_start)

        # Cleanup: If we exit loop with pending requests, wait for them
        if req_recv is not None:
            req_recv.wait()
            req_send.wait()

        return dict(W=W / ps_weight, logs=dict(logs))


if __name__ == "__main__":
    Solver.entry_point()
