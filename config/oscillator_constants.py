from __future__ import annotations

import numpy as np


class VanDerPolScaled:
    """Class for the scaled Van Der Pol parameters."""

    MI: float = 0.3
    X0: float = 0
    Y0: float = 0
    D: float = 10
    initial_conditions = np.array([2.0, 1.0], dtype=np.float64)
    name: str = "van_der_pol"


class WillamowskiRossler:
    """Class for the scaled Van Der Pol parameters."""

    B1: float = 80
    B2: float = 20
    D1: float = 0.16
    D2: float = 0.13
    D3: float = 16
    initial_conditions = np.array([1.0, 1.0, 1.0], dtype=np.float64)
    name: str = "willamowski_rossler"


class FitzNagumo:
    n: int = 20
    a_param: float = 0.7
    b_param: float = 0.8
    e_param: float = 0.08
    coupling_parameters_k = [
        [0.000, 0.409, -0.176, -0.064, -0.218, 0.464, -0.581, 0.101, -0.409, -0.140],
        [0.229, 0.000, 0.480, -0.404, -0.409, 0.040, 0.125, 0.099, -0.276, -0.131],
        [-0.248, 0.291, 0.000, -0.509, -0.114, 0.429, 0.530, 0.195, 0.416, -0.597],
        [-0.045, 0.039, 0.345, 0.000, 0.579, -0.232, 0.121, 0.130, -0.345, 0.463],
        [-0.234, -0.418, -0.195, -0.135, 0.000, 0.304, 0.124, 0.038, -0.049, 0.183],
        [-0.207, 0.536, -0.158, 0.533, -0.591, 0.000, -0.273, -0.571, 0.110, -0.354],
        [0.453, -0.529, -0.287, -0.237, 0.470, -0.002, 0.000, -0.256, 0.438, 0.211],
        [-0.050, 0.552, 0.330, -0.148, -0.326, -0.175, -0.240, 0.000, 0.263, 0.079],
        [0.389, -0.131, 0.383, 0.413, -0.383, 0.532, -0.090, 0.025, 0.000, 0.496],
        [0.459, 0.314, -0.121, 0.226, 0.314, -0.114, -0.450, -0.018, -0.333, 0.000],
    ]
    num_of_oscillators: int = n // 2
    coupling_parameters: np.ndarray = np.array(coupling_parameters_k)
    a_parameters: np.ndarray = np.ones(num_of_oscillators) * a_param
    b_parameters: np.ndarray = np.ones(num_of_oscillators) * b_param
    e_parameters: np.ndarray = np.ones(num_of_oscillators) * e_param
    excitations: np.ndarray = np.array(
        [0.2, 0.2, 0.2, 0.2, 0.2, 0.2, 0.2, 0.8, 0.8, 0.8]
    )
    initial_conditions: np.ndarray = np.ones((n,))
    name: str = "fitz_nagumo"


class FitzNagumoRing:
    n: int = 20
    a_param: float = 0.7
    b_param: float = 0.8
    e_param: float = 0.08
    coupling_parameters_k = [
        [0, +0.3, 0, 0, 0, 0, 0, 0, 0, -0.3],
        [-0.3, 0, +0.3, 0, 0, 0, 0, 0, 0, 0],
        [0, -0.3, 0, +0.3, 0, 0, 0, 0, 0, 0],
        [0, 0, -0.3, 0, +0.3, 0, 0, 0, 0, 0],
        [0, 0, 0, -0.3, 0, +0.3, 0, 0, 0, 0],
        [0, 0, 0, 0, -0.3, 0, +0.3, 0, 0, 0],
        [0, 0, 0, 0, 0, -0.3, 0, +0.3, 0, 0],
        [0, 0, 0, 0, 0, 0, -0.3, 0, +0.3, 0],
        [0, 0, 0, 0, 0, 0, 0, -0.3, 0, +0.3],
        [+0.3, 0, 0, 0, 0, 0, 0, 0, -0.3, 0],
    ]
    num_of_oscillators: int = n // 2
    coupling_parameters: np.ndarray = np.array(coupling_parameters_k)
    a_parameters: np.ndarray = np.ones(n // 2) * a_param
    b_parameters: np.ndarray = np.ones(n // 2) * b_param
    e_parameters: np.ndarray = np.ones(n // 2) * e_param
    excitations: np.ndarray = np.array(
        [0.32, 0.32, 0.32, 0.32, 0.32, 0.32, 0.32, 0.32, 0.32, 0.32]
    )
    initial_conditions: np.ndarray = np.array(
        [1.0 if id == 0 else 0.0 for id in range(n)]
    )
    name: str = "fitz_nagumo_ring"


class FitzNagumoStar:
    n: int = 8
    a_param: float = 0.7
    b_param: float = 0.8
    e_param: float = 0.08
    coupling_parameters_k = [
        [0.0, 0.28230, 0.49230, -0.05960],
        [0.424674, 0.0, 0.0, 0.0],
        [-0.430378, 0.0, 0.0, 0.0],
        [-0.295792, 0.0, 0.0, 0.0],
    ]

    num_of_oscillators: int = n // 2
    coupling_parameters: np.ndarray = np.array(coupling_parameters_k).T
    a_parameters: np.ndarray = np.ones(n // 2) * a_param
    b_parameters: np.ndarray = np.ones(n // 2) * b_param
    e_parameters: np.ndarray = np.ones(n // 2) * e_param
    excitations: np.ndarray = np.array([0.8, 0.8, 0.8, 0.8])
    initial_conditions: np.ndarray = np.array(
        [1.0 if id == 0 else 0.0 for id in range(n)]
    )
    name: str = "fitz_nagumo_star"
