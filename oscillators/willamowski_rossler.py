from collections.abc import Callable

import numpy as np


# TODO: Write docs
def willamowski_rossler_model(
    b1: float = 80, b2: float = 20, d1: float = 0.16, d2: float = 0.13, d3: float = 16
) -> Callable[[np.ndarray], np.ndarray]:
    def willamowski_rossler_model(x_state: np.ndarray) -> np.ndarray:
        x = x_state[0]
        y = x_state[1]
        z = x_state[2]
        x_dot = x * (b1 - d1 * x - y - z)
        y_dot = y * (b2 - d2 * y - x)
        z_dot = z * (x - d3)
        return np.array([x_dot, y_dot, z_dot])

    # NOTE: Type ignored because of mypy and numba limitations
    return willamowski_rossler_model  # type:ignore


def willamowski_rossler_jacobian_model(
    b1: float = 80, b2: float = 20, d1: float = 0.16, d2: float = 0.13, d3: float = 16
) -> Callable[[np.ndarray], np.ndarray]:
    def willamowski_rossler_jacobian(state: np.ndarray) -> np.ndarray:
        x = state[0]
        y = state[1]
        z = state[2]
        fxx = b1 - 2 * d1 * x - y - z
        fxy = -x
        fxz = -x
        fyx = -y
        fyy = b2 - 2 * d2 * y - x
        fyz = 0
        fzx = z
        fzy = 0
        fzz = x - d3
        return np.array([[fxx, fxy, fxz], [fyx, fyy, fyz], [fzx, fzy, fzz]])

    # NOTE: Type ignored because of mypy and numba limitations
    return willamowski_rossler_jacobian


def willamowski_rossler_perturbated_model(
    b1: float = 80, b2: float = 20, d1: float = 0.16, d2: float = 0.13, d3: float = 16
) -> Callable[[np.ndarray, np.ndarray], np.ndarray]:
    willamowski = willamowski_rossler_model(b1, b2, d1, d2, d3)

    def willamowski_rossler_perturbated(
        x_state: np.ndarray, perturbation: np.ndarray
    ) -> np.ndarray:
        res = willamowski(x_state) + perturbation
        return res

    # NOTE: Type ignored because of mypy and numba limitations
    return willamowski_rossler_perturbated  # type:ignore
