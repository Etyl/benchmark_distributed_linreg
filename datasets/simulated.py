from benchopt import BaseDataset
from benchmark_utils import get_data_path
import numpy as np
import os


class Dataset(BaseDataset):
    name = "Simulated"

    parameters = {
        'n': [16*1024],
        'd1': [800],
        'd2': [800],
        'iid': [True, False],
        'n_blocks': [16],
        'noise': [1.],
    }
    requirements = ["numpy"]

    def get_data(self):
        data_dir = get_data_path("simulated")
        os.makedirs(data_dir, exist_ok=True)
        param_string = "_".join(
            [f"{k}-{getattr(self, k)}" for k in self.parameters]
        )
        X_name = f"X_{param_string}.npy"
        x_path = os.path.join(data_dir, X_name)
        Y_name = f"Y_{param_string}.npy"
        y_path = os.path.join(data_dir, Y_name)

        if os.path.exists(x_path) and os.path.exists(y_path):
            return dict(x_path=x_path, y_path=y_path)

        rng = np.random.RandomState(42)

        n_per_block = self.n // self.n_blocks

        W_linear = rng.randn(self.d1, self.d2)

        X_blocks = []
        Y_blocks = []

        for _ in range(self.n_blocks):
            # Different distribution per block
            mean_shift = rng.randn(self.d1) * 2.0
            Xb = rng.randn(n_per_block, self.d1) + mean_shift

            # Multivariate quadratic regression
            Yb = (
                Xb @ (W_linear + rng.randn(self.d1, self.d2) * 0.1)
                + self.noise * rng.randn(n_per_block, self.d2)
            )

            X_blocks.append(Xb)
            Y_blocks.append(Yb)

        X = np.vstack(X_blocks)
        Y = np.vstack(Y_blocks)

        # IID version = shuffle samples
        if self.iid:
            perm = rng.permutation(self.n)
            X = X[perm]
            Y = Y[perm]

        np.save(x_path, X)
        np.save(y_path, Y)

        return dict(x_path=x_path, y_path=y_path)
