import mpi4py.MPI as MPI
from abc import ABC, abstractmethod
import numpy as np


class Compressor(ABC):
    def __init__(self, comm):
        self.comm = comm

    @abstractmethod
    def communicate(self, comm, return_compressed=False):
        pass

    @abstractmethod
    def decompress(self):
        pass


class TopK(Compressor):
    def __init__(self, k, comm):
        super().__init__(comm)
        self.k = k
        self.comm = comm
        self.shape = None

        # Internal buffers
        self.indices = None
        self.values = None

        # Communication buffers (pre-allocated)
        self.world_size = comm.Get_size()
        self.recv_values = None
        self.recv_indices = None

    def communicate(self, tensor, return_compressed=False):
        self.shape = tensor.shape
        flat_tensor = tensor.ravel()

        if flat_tensor.size <= self.k:
            self.indices = np.arange(flat_tensor.size, dtype=np.int32)
            self.values = flat_tensor
        else:
            idx = np.argpartition(np.abs(flat_tensor), -self.k)[-self.k:]
            self.indices = idx.astype(np.int32)
            self.values = flat_tensor[idx]

        if self.recv_values is None:
            self.recv_values = np.empty(
                (self.world_size, self.k),
                dtype=self.values.dtype
            )
            self.recv_indices = np.empty(
                (self.world_size, self.k),
                dtype=np.int32
            )

        self.comm.Allgather(self.values, self.recv_values)
        self.comm.Allgather(self.indices, self.recv_indices)

        if return_compressed:
            tensor_compressed = np.zeros_like(flat_tensor)
            tensor_compressed[self.indices] = self.values
            return tensor_compressed.reshape(self.shape)

    def decompress(self):
        D = np.prod(self.shape)
        result = np.zeros(D, dtype=self.values.dtype)

        all_indices = self.recv_indices.ravel()
        all_values = self.recv_values.ravel()

        np.add.at(result, all_indices, all_values)

        result /= self.world_size
        return result.reshape(self.shape)


class PowerSGD(Compressor):
    def __init__(self, rank, comm):
        super().__init__(comm)
        self.r = rank  # The low rank 'r'
        self.comm = comm
        self.world_size = comm.Get_size()

        # Local compressed matrices
        self.P = None
        self.Q = None

    def communicate(self, tensor, return_compressed=False):
        if self.Q is None:
            d2 = tensor.shape[1]
            rng = np.random.RandomState(0)
            self.Q = rng.randn(d2, self.r)
            self.Q, _ = np.linalg.qr(self.Q)

        P_local = tensor @ self.Q
        if self.P is None:
            self.P = np.empty_like(P_local)
        self.comm.Allreduce(P_local, self.P, op=MPI.SUM)
        self.P /= self.world_size
        self.P, _ = np.linalg.qr(self.P)

        Q_local = tensor.T @ self.P
        self.comm.Allreduce(Q_local, self.Q, op=MPI.SUM)
        self.Q /= self.world_size

        if return_compressed:
            return tensor @ self.Q @ self.Q.T

    def decompress(self):
        """
        Reconstructs the global gradient.
        We want to compute: Avg( P_i @ Q_i.T ) for i in workers.
        """
        return self.P @ self.Q.T
