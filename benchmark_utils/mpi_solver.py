import os
import sys
import subprocess
import argparse
import inspect
import socket
import pickle
import struct
import atexit
from benchopt import BaseSolver

from benchmark_utils import ACTIVE_SOLVERS


class DistributedMPISolver(BaseSolver):
    """
    Persistent MPI Solver using Socket Communication.
    Handles the infrastructure (Workers, Sockets, MPI Loop).
    """

    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.worker_process = None
        self.server_socket = None
        self.connection = None
        # Ensure workers are killed if the script exits abruptly
        atexit.register(self.cleanup)

    def __del__(self):
        # Ensure cleanup is called when the solver object is destroyed
        self.cleanup()

    def set_objective(self, x_path, y_path):
        # Store parameters (Launch logic moved to warm_up)
        self.x_path = x_path
        self.y_path = y_path

    def warm_up(self):
        """
        Launch the workers and run one iteration to warm up the system.
        """
        # Ensure any previous workers are cleaned up
        for solver in ACTIVE_SOLVERS:
            solver.cleanup()
        ACTIVE_SOLVERS.clear()

        # Setup Socket Server
        self.server_socket = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        self.server_socket.bind(("0.0.0.0", 0))
        self.server_socket.listen(1)

        driver_host = socket.gethostname()
        _, driver_port = self.server_socket.getsockname()

        # Launch Workers (Non-Blocking)
        child_file_path = inspect.getfile(self.__class__)
        cpus_per_task = os.environ.get("SLURM_CPUS_PER_TASK")
        cmd = [
            "mpiexec",
            "-n", str(self.n_workers),
            "python", child_file_path,
            "--worker",
            "--x_path", self.x_path,
            "--y_path", self.y_path,
            "--driver_host", str(driver_host),
            "--driver_port", str(driver_port),
        ]
        for param in self.parameters:
            cmd.append(f"--{param}")
            cmd.append(str(getattr(self, param)))

        print(f"Driver: Launching persistent MPI cluster on {child_file_path}")

        env = os.environ.copy()
        pythonpath = os.getcwd() + os.pathsep + env.get("PYTHONPATH", "")
        env["PYTHONPATH"] = pythonpath
        env["OMP_NUM_THREADS"] = cpus_per_task
        env["MKL_NUM_THREADS"] = cpus_per_task
        env["OPENBLAS_NUM_THREADS"] = cpus_per_task
        env["VECLIB_MAXIMUM_THREADS"] = cpus_per_task
        env["NUMEXPR_NUM_THREADS"] = cpus_per_task

        self.worker_process = subprocess.Popen(
            cmd,
            stdout=sys.stdout,
            stderr=sys.stderr,
            env=env
        )

        print(
            f"[{self.name}] Launching worker on {driver_host}:{driver_port}..."
        )

        # Register cleanup to ensure process is killed at exit
        ACTIVE_SOLVERS.append(self)
        atexit.register(self.cleanup)
        self.server_socket.settimeout(60)
        try:
            self.connection, addr = self.server_socket.accept()
            print(f"Driver: Connected to worker at {addr}")
        except socket.timeout:
            self.cleanup()
            raise RuntimeError("Timed out waiting for MPI workers to connect.")

    def run(self, n_iter):
        if not self.connection:
            raise RuntimeError(
                "No active connection to workers. Please call warm_up() first."
            )

        # Send RUN command
        msg = {"command": "RUN", "n_iter": n_iter}
        self._send_msg(self.connection, msg)

        # Wait for Result
        response = self._recv_msg(self.connection)

        if response and response.get("status") == "DONE":
            self.result = response.get("result")
        else:
            raise RuntimeError(f"Unexpected response from worker: {response}")

    def get_result(self):
        return self.result

    def cleanup(self):
        """Terminate worker and close sockets."""
        # Unregister from shared state and atexit
        if self in ACTIVE_SOLVERS:
            ACTIVE_SOLVERS.remove(self)
        atexit.unregister(self.cleanup)

        # Send EXIT command to workers
        if getattr(self, 'connection', None):
            try:
                self._send_msg(self.connection, {"command": "EXIT"})
                self.connection.close()
            except Exception:
                pass
            self.connection = None

        # Close Server Socket
        if getattr(self, 'server_socket', None):
            try:
                self.server_socket.close()
            except Exception:
                pass
            self.server_socket = None

        # Kill Process
        if getattr(self, 'worker_process', None):
            try:
                self.worker_process.wait(timeout=2)
            except subprocess.TimeoutExpired:
                self.worker_process.kill()
            self.worker_process = None

    # --- Abstract Methods for Concrete Solvers ---

    @classmethod
    def init_worker(cls, args, comm, rank, world_size):
        raise NotImplementedError

    @classmethod
    def worker_run(cls, n_iter, worker_ctx, args, comm, rank, world_size):
        raise NotImplementedError

    # --- Worker Entry Point (Generic) ---

    @classmethod
    def worker(cls, args):
        from mpi4py import MPI

        # 1. Init Environment & MPI
        comm = MPI.COMM_WORLD
        rank = comm.Get_rank()
        world_size = comm.Get_size()

        print(
            f"Worker initialized: Rank {rank}/{world_size} "
            f"on {socket.gethostname()}"
        )
        sys.stdout.flush()

        # 2. Solver-Specific Initialization
        worker_ctx = cls.init_worker(args, comm, rank, world_size)

        # 3. Connect to Driver (Rank 0 only)
        sock = None
        if rank == 0:
            try:
                sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
                sock.connect((args.driver_host, args.driver_port))
                print("Worker 0: Connected to Driver.")
            except Exception as e:
                print(f"Worker 0: Connection failed: {e}")
                sys.exit(1)

        # 4. Command Loop
        while True:
            cmd_data = None

            # Rank 0 receives command
            if rank == 0:
                try:
                    cmd_data = DistributedMPISolver._recv_msg(sock)
                except Exception:
                    cmd_data = {"command": "EXIT"}

            # Broadcast to all ranks
            cmd_data = comm.bcast(cmd_data, root=0)

            if cmd_data is None:
                break

            cmd = cmd_data.get("command")

            if cmd == "EXIT":
                break

            if cmd == "RUN":
                n_iter = cmd_data.get("n_iter", 0)

                # Execute Solver Logic
                result_worker = cls.worker_run(
                    n_iter, worker_ctx, args, comm, rank, world_size
                )

                # Send Result (Rank 0)
                if rank == 0:
                    result = {
                        "status": "DONE",
                        "result": result_worker
                    }
                    DistributedMPISolver._send_msg(sock, result)

            comm.Barrier()

        if rank == 0 and sock:
            sock.close()

    # --- Socket Helpers ---

    @staticmethod
    def _send_msg(sock, msg):
        try:
            data = pickle.dumps(msg)
            sock.sendall(struct.pack('>I', len(data)) + data)
        except (OSError, BrokenPipeError):
            pass

    @staticmethod
    def _recv_msg(sock):
        try:
            raw_msglen = DistributedMPISolver._recvall(sock, 4)
            if not raw_msglen:
                return None
            msglen = struct.unpack('>I', raw_msglen)[0]
            data = DistributedMPISolver._recvall(sock, msglen)
            return pickle.loads(data)
        except (OSError, struct.error):
            return None

    @staticmethod
    def _recvall(sock, n):
        data = bytearray()
        while len(data) < n:
            try:
                packet = sock.recv(n - len(data))
                if not packet:
                    return None
                data.extend(packet)
            except OSError:
                return None
        return data

    @classmethod
    def entry_point(cls):
        parser = argparse.ArgumentParser()
        parser.add_argument("--worker", action="store_true")
        parser.add_argument("--x_path", type=str)
        parser.add_argument("--y_path", type=str)
        parser.add_argument("--driver_host", type=str)
        parser.add_argument("--driver_port", type=int)

        for param, values in cls.parameters.items():
            param_type = type(values[0]) if values else str
            parser.add_argument(f"--{param}", type=param_type)

        args, _ = parser.parse_known_args()

        if args.worker:
            cls.worker(args)
