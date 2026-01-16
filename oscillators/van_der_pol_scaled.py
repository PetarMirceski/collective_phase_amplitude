from collections.abc import Callable

import numpy as np

# TODO: Write docs for this file

# NOTE: x * x * x is faster than x**3


def van_der_pol_model_scaled(
    mi: float, x0: float, y0: float, d: float
) -> Callable[[np.ndarray], np.ndarray]:
    def van_der_pol_model(x_state: np.ndarray) -> np.ndarray:
        x = x_state[0]
        y = x_state[1]
        x_dot = d * (mi * x - (x * x * x) / 3 - y + x0)
        y_dot = d * (x + y0)
        return np.array([x_dot, y_dot])

    # NOTE: Type ignored because of mypy and numba limitations
    return van_der_pol_model  # type:ignore


def perturbated_van_der_pol_model_scaled(
    mi: float, x0: float, y0: float, d: float
) -> Callable[[np.ndarray, np.ndarray], np.ndarray]:
    van_model = van_der_pol_model_scaled(mi, x0, y0, d)

    def perturbated_van_der_pol_model(
        x_state: np.ndarray, perturbation: np.ndarray
    ) -> np.ndarray:
        x_dot: np.ndarray = van_model(x_state) + perturbation
        return x_dot

    # NOTE: Type ignored because of mypy and numba limitations
    return perturbated_van_der_pol_model  # type:ignore


def van_der_pol_jacobian_model_scaled(
    mi: float, d: float
) -> Callable[[np.ndarray], np.ndarray]:
    def van_der_pol_jacobian(state: np.ndarray) -> np.ndarray:
        x = state[0]
        return np.array([[d * (mi - x**2), -d], [d, 0]])

    # NOTE: Type ignored because of mypy and numba limitations
    return van_der_pol_jacobian
