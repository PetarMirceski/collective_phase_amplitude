from collections.abc import Callable

import numpy as np
from scipy.interpolate import interp1d
from tqdm import tqdm


def phase_model(
    delta: float, gamma_f: Callable[[float], float]
) -> Callable[[float], float]:
    def phase_model(phi: float) -> float:
        d_phi = delta + gamma_f(phi)
        return float(d_phi)

    return phase_model


def simulate(
    gamma: np.ndarray, dt: float, phi: float, sim_time: float
) -> tuple[np.ndarray, np.ndarray]:
    gamma_x = gamma[:, 0]
    gamma_y = gamma[:, 1]
    gamma_f = interp1d(gamma_x, gamma_y)
    model = phase_model(0, gamma_f)

    steps = int(sim_time / dt)
    phi_array = np.zeros(steps)
    time_array = np.zeros(steps)
    for i in tqdm(range(0, steps)):
        d_phi = model(phi)
        phi = phi + d_phi * dt
        phi_array[i] = phi
        time_array[i] = i * dt
    return time_array, phi_array
