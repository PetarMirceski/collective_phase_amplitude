from collections.abc import Callable

import numpy as np


def van_der_pol_model(
    mi: float, x0: float, y0: float
) -> Callable[[np.ndarray], np.ndarray]:
    """Function that returns a Van der Pol oscillator. It's form is described by
    the equation:

    \\begin{align}
    & \\dot{x} = \\mu  x - \\frac{x^3}{3} - y +x_0 \\\\
    & \\dot{y} = x + y_0
    \\end{align}


    Args:
        mi (float): Scalar parameter indicating the nonlinearity and strength of
            the dampening
        x0 (float): x-offset
        y0 (float): y-offset

    Returns:
        Callable[[np.ndarray], np.ndarray]: A function that represents the Van
            der Pol's vector field with the above mentioned parameters set
    """

    def van_der_pol_model(x_state: np.ndarray) -> np.ndarray:
        x = x_state[0]
        y = x_state[1]
        x_dot = mi * x - (x**3) / 3 - y + x0
        y_dot = x + y0
        return np.array([x_dot, y_dot])

    # NOTE: Type ignored because of mypy and numba limitations
    return van_der_pol_model  # type:ignore


def perturbated_van_der_pol_model(
    mi: float, x0: float, y0: float
) -> Callable[[np.ndarray, float], np.ndarray]:
    van_der_pol = van_der_pol_model(mi, x0, y0)

    def perturbated_van_der_pol_model(
        x_state: np.ndarray, perturbation: float
    ) -> np.ndarray:
        x_dot: np.ndarray = van_der_pol(x_state) + perturbation
        return x_dot

    # NOTE: Type ignored because of mypy and numba limitations
    return perturbated_van_der_pol_model  # type:ignore


def van_der_pol_jacobian_model(mi: float) -> Callable[[np.ndarray], np.ndarray]:
    def van_der_pol_jacobian(state: np.ndarray) -> np.ndarray:
        x = state[0, 0]
        return np.array([[mi - x**2, -1], [1, 0]])

    # NOTE: Type ignored because of mypy and numba limitations
    return van_der_pol_jacobian
