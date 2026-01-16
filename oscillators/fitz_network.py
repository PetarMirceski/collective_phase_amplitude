from collections.abc import Callable

import numpy as np
import numpy.typing as npt


def fitz_model(
    a_parameters: npt.NDArray[np.float64],
    b_parameters: npt.NDArray[np.float64],
    e_parameters: npt.NDArray[np.float64],
    excitations: npt.NDArray[np.float64],
    coupling_parameters: npt.NDArray[np.float64],
) -> Callable[[npt.NDArray[np.float64]], npt.NDArray[np.float64]]:
    # Pre-compute row sums for the coupling term
    # Coupling[i] = sum_j K[i,j] * (v[j] - v[i])
    #   = (K @ v)[i] - (sum_j K[i,j]) * v[i]
    #   = (K @ v)[i] - row_sum[i] * v[i]
    row_sum = coupling_parameters.sum(axis=1)

    def fitz_model_inner(input_state: npt.NDArray[np.float64]) -> np.ndarray:
        u_state = input_state[::2]  # even index elements 0 2 4 6
        v_state = input_state[1::2]  # odd index elements 1 3 5 7

        # Coupling: sum_j K[i,j] * (v[j] - v[i]) for each i
        coupling = coupling_parameters @ v_state - row_sum * v_state

        du_vector = e_parameters * (v_state + a_parameters - b_parameters * u_state)
        dv_vector = v_state - v_state**3 / 3 - u_state + excitations + coupling

        result = np.empty_like(input_state)
        result[::2] = du_vector
        result[1::2] = dv_vector
        return result

    return fitz_model_inner  # type:ignore


def fitz_jacobian_model(
    b_parameters: np.ndarray,
    e_parameters: np.ndarray,
    coupling_parameters: np.ndarray,
) -> Callable[[np.ndarray], np.ndarray]:
    # Pre-compute row sums for diagonal coupling term
    # Coupling[i] = sum_j K[i,j] * (v[j] - v[i])
    # d(coupling[i])/d(v[i]) = -sum_j K[i,j] = -row_sum[i]
    # d(coupling[i])/d(v[j]) = K[i,j] for j != i
    row_sums = coupling_parameters.sum(axis=1)

    def fitz_jacobian(state: np.ndarray) -> np.ndarray:
        n = state.shape[0]
        jacobian = np.zeros((n, n))

        v_state = state[1::2]
        even_idx = np.arange(0, n, 2)
        odd_idx = np.arange(1, n, 2)

        # d(du_i)/d(u_i) = -e_i * b_i
        jacobian[even_idx, even_idx] = -e_parameters * b_parameters
        # d(du_i)/d(v_i) = e_i
        jacobian[even_idx, odd_idx] = e_parameters
        # d(dv_i)/d(u_i) = -1
        jacobian[odd_idx, even_idx] = -1.0
        # d(dv_i)/d(v_i) = 1 - v_i^2 - row_sum[i]
        jacobian[odd_idx, odd_idx] = 1.0 - v_state * v_state - row_sums

        # Off-diagonal coupling: d(dv_i)/d(v_j) = K[i,j]
        jacobian[np.ix_(odd_idx, odd_idx)] += coupling_parameters

        return jacobian

    return fitz_jacobian  # type:ignore


def fitz_perturbated_model(
    a_parameters: np.ndarray,
    b_parameters: np.ndarray,
    e_parameters: np.ndarray,
    excitations: np.ndarray,
    coupling_parameters: np.ndarray,
) -> Callable[[np.ndarray, np.ndarray], np.ndarray]:
    fitz_net = fitz_model(
        a_parameters, b_parameters, e_parameters, excitations, coupling_parameters
    )

    def fitz_perturbated(x_state: np.ndarray, perturbation: np.ndarray) -> np.ndarray:
        x_dot: np.ndarray = fitz_net(x_state) + perturbation
        return x_dot

    return fitz_perturbated  # type:ignore
