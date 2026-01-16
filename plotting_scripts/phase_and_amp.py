from collections.abc import Callable

import matplotlib.pyplot as plt
import numpy as np
import numpy.typing as npt
from matplotlib.axes import Axes
from scipy.integrate import solve_ivp
from tqdm import tqdm

from config.constants import STEP_FITZ, OscillatorParameters
from config.oscillator_constants import FitzNagumo, FitzNagumoRing
from config.paths import FIGURE_PATH
from oscillators.fitz_network import fitz_model
from solvers.period_finder import find_period

plt.rcParams["figure.figsize"] = [27, 13]

BASE_DIR = FIGURE_PATH / "phase_simulation/"
BASE_DIR.mkdir(exist_ok=True, parents=True)

NUM_CYCLES = 2
T_FINAL = 300
R = complex(1, 0)


def amplitude_equation(exponent: complex) -> Callable[[complex], complex]:
    def model(r: complex) -> complex:
        dr = r * exponent
        return dr

    return model


def euler_step(
    model: Callable[[complex], complex], dt: float, state: complex
) -> complex:
    return complex(state + dt * model(state))


def get_states_and_phase(
    net: Callable[[npt.NDArray[np.float64]], npt.NDArray[np.float64]],
    initial_conditions: np.ndarray,
) -> tuple[np.ndarray, np.ndarray, int]:
    """Get limit cycle states and phase using solve_ivp."""

    def system(t: float, y: np.ndarray) -> np.ndarray:
        return net(y)

    # First, evolve to converge to the limit cycle
    sol = solve_ivp(
        system,
        (0, T_FINAL),
        initial_conditions,
        method="RK45",
        rtol=1e-7,
        atol=1e-10,
    )
    converged_state = sol.y[:, -1]

    # Find the natural period using modern period finder
    period, zero_phase_state = find_period(
        system=system,
        initial_state=converged_state,
        threshold_value=0,
        state_index=1,
        direction=-1,
    )
    num_of_itter = int(period / STEP_FITZ)

    # Integrate over multiple cycles
    t_final_cycles = period * NUM_CYCLES
    t_eval = np.arange(0, t_final_cycles, STEP_FITZ)
    sol_cycles = solve_ivp(
        system,
        (0, t_final_cycles),
        zero_phase_state,
        method="RK45",
        t_eval=t_eval,
        rtol=1e-7,
        atol=1e-10,
    )
    states = sol_cycles.y.T
    raw_phase = np.arange(states.shape[0])

    phase = raw_phase % num_of_itter
    return states, phase, num_of_itter


def simulate_phase_and_amplitude(
    net: Callable[[npt.NDArray[np.float64]], npt.NDArray[np.float64]],
    initial_conditions: np.ndarray,
    omega: float,
    exponent: complex,
) -> tuple[Axes, Axes]:
    _, phase, num_of_itter = get_states_and_phase(net, initial_conditions)
    time_vector = np.arange(phase.shape[0]) * STEP_FITZ
    y_max = np.max(phase)
    phase = (phase / y_max) * (2 * np.pi)  # scale phase to 0-2pi

    N = int(T_FINAL / STEP_FITZ)
    amps = np.zeros((N,), dtype=np.complex128)
    amps[0] = R

    model = amplitude_equation(exponent)
    for i in tqdm(range(1, N)):
        amps[i] = euler_step(model, STEP_FITZ, amps[i - 1])
    time_vector_amps = np.arange(0, T_FINAL, STEP_FITZ)

    offset_d = 0.3
    offset_x = int(num_of_itter * 0.1)
    offset = offset_d / np.sqrt(2)

    # Parallel Line Coordinates
    x1 = time_vector[offset_x] - offset
    x2 = time_vector[num_of_itter - offset_x - 1] - offset

    y1 = phase[offset_x] + offset
    y2 = phase[num_of_itter - 1 - offset_x] + offset

    txt_cen_x = (x2 + x1) / 2 - 2 * offset
    txt_cen_y = (y2 + y1) / 2 + offset

    fig, (ax1, ax2) = plt.subplots(1, 2)
    ax1.plot(time_vector_amps, amps)
    ax1.grid()
    ax1.set_ylabel(r"$R_{1}$")

    ax2.text(txt_cen_x, txt_cen_y, rf"$\omega = {omega:.3f}$", ha="left", rotation=65)
    ax2.plot(time_vector, phase)
    ax2.plot([x1, x2], [y1, y2])
    ax2.grid()
    ax2.set_yticks(
        [0, np.pi / 2, np.pi, 3 * np.pi / 2, 2 * np.pi],
        ["0", r"$\frac{\pi}{2}$", r"$\pi$", r"$\frac{3\pi}{2}$", r"$2\pi$"],
    )
    ax2.set_ylim(0, 2 * np.pi)
    ax2.set_xlabel("$t$")
    ax2.set_ylabel(r"$\theta$")
    return (ax1, ax2)


def main() -> None:
    fitz_net = fitz_model(
        FitzNagumo.a_parameters,
        FitzNagumo.b_parameters,
        FitzNagumo.e_parameters,
        FitzNagumo.excitations,
        FitzNagumo.coupling_parameters,
    )
    fitz_data = OscillatorParameters.load(FitzNagumo.name)
    simulate_phase_and_amplitude(
        fitz_net,
        FitzNagumo.initial_conditions,
        fitz_data.natural_freq,
        fitz_data.floquet_exponents[1],
    )
    plt.savefig(BASE_DIR / "fitz_random_phase_amp.png")
    plt.show()

    fitz_net = fitz_model(
        FitzNagumoRing.a_parameters,
        FitzNagumoRing.b_parameters,
        FitzNagumoRing.e_parameters,
        FitzNagumoRing.excitations,
        FitzNagumoRing.coupling_parameters,
    )
    fitz_data = OscillatorParameters.load(FitzNagumoRing.name)
    simulate_phase_and_amplitude(
        fitz_net,
        FitzNagumoRing.initial_conditions,
        fitz_data.natural_freq,
        fitz_data.floquet_exponents[1],
    )
    plt.savefig(BASE_DIR / "fitz_ring_phase_amp.png")


if __name__ == "__main__":
    main()
