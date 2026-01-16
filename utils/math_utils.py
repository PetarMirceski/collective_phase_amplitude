from typing import Any

import numpy as np


# TODO: WRITE DOCS
def _identity(matrix: Any) -> Any:
    """
    This function is used to satisfy some typing hacks and programming callable tricks
    done in the left and right floquet finding functions used only as transform as func(x) = x.
    Do not use it extensively in the code.
    Parameters:
        matrix: np.ndarray of any shape
    Returns:
        out: np.ndarray of any shape where out === matrix
    """
    return matrix


def _transpose(matrix: np.ndarray) -> np.ndarray:
    """
    This function is used to satisfy some typing hacks and programming callable tricks
    done in the left and right floquet finding functions used only as transform as func(x) = x.T
    Do not use it extensively in the code.
    Parameters:
        matrix: np.ndarray of any shape
    Returns:
        out: np.ndarray of any shape where out === matrix
    """
    return matrix.T


def conjugate(exponent: np.complex128 | float) -> np.complex128:
    """
    This function is used to satisfy some typing hacks and programming callable tricks
    done in the left and right floquet finding functions used only as transform as
    func([a1+jb1, a2-jb2, ...]) = [[a1-jb1, a2-jb2, ...], ...]
    Do not use it extensively in the code.
    Parameters:
        matrix: np.ndarray of any shape
    Returns:
        out: np.ndarray of any shape where out === matrix
    """

    result: np.complex128 = np.conjugate(exponent)
    return result


def inner_product_of_vector_arrays(
    vector_arr1: np.ndarray, vector_arr2: np.ndarray
) -> np.ndarray:
    """
    Calculates the inner product of two arrays of vectors  as |v1(i)*T @ v2(i)|
    Parameters:
        vector_1: First vector of shape [n_dim, N]
        vector_1: Second vector of shape [n_dim, N]
    Returns:
        product: the |v1*T @ v2| a array of shape [N, ]
    """
    res = np.sum(np.conj(vector_arr1) * vector_arr2, axis=1)
    return np.array(res)


def mean_over_one_period(
    phase: np.ndarray, simu_len: float, phase_len: int
) -> np.ndarray:
    num_sub_samples = int(phase.shape[0] / (simu_len / phase_len))
    list_mean = [phase[0]]
    for i in range(int(phase.shape[0] / num_sub_samples)):
        list_mean.append(
            np.mean(phase[i * num_sub_samples : (i + 1) * num_sub_samples])
        )
    return np.array(list_mean)
