from benchopt import BaseObjective
import numpy as np
from numpy.lib.format import open_memmap


def _compute_loss(X, y, W):
    residuals = X @ W - y
    loss = np.mean(residuals ** 2)
    return loss


class Objective(BaseObjective):
    name = "Linear Regression"
    min_benchopt_version = "1.7"

    parameters = {
    }

    def set_data(self, x_path, y_path):
        self.x_path, self.y_path = x_path, y_path

    def get_one_result(self):
        x = open_memmap(self.x_path)
        y = open_memmap(self.y_path)
        return dict(W=np.zeros((x.shape[1], y.shape[1])))

    def evaluate_result(self, W, logs={}):
        x = open_memmap(self.x_path)
        y = open_memmap(self.y_path)
        train_loss = _compute_loss(
            x, y, W
        )

        return {
            "value": train_loss,
            **{
                k: np.mean(v)
                for k, v in logs.items()
            }
        }

    def get_objective(self):
        return dict(
            x_path=self.x_path, y_path=self.y_path
        )
