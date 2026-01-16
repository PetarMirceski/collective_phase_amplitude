from collections.abc import Callable

import numba
import numpy as np
import numpy.typing as npt


def fitz_element_model(
    a_parameter: float,
    b_parameter: float,
    e_parameter: float,
    excitation: float,
) -> Callable[[npt.NDArray[np.float64]], npt.NDArray[np.float64]]:
    @numba.jit(nopython=True)
    def fitz_model(input_state: npt.NDArray[np.float64]) -> np.ndarray:
        u_state = input_state[0]  # even index elements 0 2 4 6
        v_state = input_state[1]  # odd index elements 1 3 5 7
        v_3 = v_state * v_state * v_state

        du_vector = e_parameter * (v_state + a_parameter - b_parameter * u_state)
        dv_vector = v_state - v_3 / 3 - u_state + excitation
        return np.array([du_vector, dv_vector])

    # NOTE: Type ignored because of mypy and numba limitations
    return fitz_model  # type:ignore


def fitz_element_jacobian_model(
    a_parameter: float,
    b_parameter: float,
    e_parameter: float,
    excitation: float,
) -> Callable[[np.ndarray], np.ndarray]:
    @numba.jit(nopython=True)
    def fitz_jacobian(state: np.ndarray) -> np.ndarray:
        v_state = state[1]
        v_2 = v_state * v_state
        return np.array([[-b_parameter * e_parameter, e_parameter], [-1, 1 - v_2]])

    return fitz_jacobian  # type:ignore


def fitz_element_perturbated_model(
    a_parameter: float,
    b_parameter: float,
    e_parameter: float,
    excitation: float,
) -> Callable[[np.ndarray, np.ndarray], np.ndarray]:
    fitz_model = fitz_element_model(a_parameter, b_parameter, e_parameter, excitation)

    @numba.jit(nopython=True)
    def perturbated_fitz_element(
        x_state: np.ndarray, perturbation: np.ndarray
    ) -> np.ndarray:
        x_dot: np.ndarray = fitz_model(x_state) + perturbation
        return x_dot

    # NOTE: Type ignored because of mypy and numba limitations
    return perturbated_fitz_element  # type:ignore
