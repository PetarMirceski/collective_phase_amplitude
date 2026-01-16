import matplotlib.pyplot as plt
import numpy as np
from scipy import stats
from scipy.integrate import solve_ivp

from config.constants import NUM_PHASE_POINTS, STEP_FITZ, OscillatorParameters
from config.oscillator_constants import FitzNagumoRing
from config.paths import FIGURE_PATH
from oscillators.fitz_network import fitz_model
from solvers.phase_solver import PhaseSolver

plt.rcParams["figure.figsize"] = [27, 13]
NUM_OF_PERIODS = 3

BASE_DIR = FIGURE_PATH / "phase_amp_near_lc/"
BASE_DIR.mkdir(exist_ok=True, parents=True)


def get_offset_line(
    x_line: np.ndarray, y_line: np.ndarray, offset_d: float, num_of_itter: int
) -> tuple[np.ndarray, np.ndarray, float, float]:
    offset_x = int(num_of_itter * 0.1)
    offset = offset_d / np.sqrt(2)

    x1 = x_line[offset_x] - offset
    x2 = x_line[num_of_itter - offset_x - 1] - offset

    y1 = y_line[offset_x] + offset
    y2 = y_line[num_of_itter - 1 - offset_x] + offset

    txt_cen_x = (x2 + x1) / 2 - 1 * offset
    txt_cen_y = (y2 + y1) / 2 + offset
    return (np.array([x1, x2]), np.array([y1, y2]), txt_cen_x, txt_cen_y)


def main() -> None:
    fitz_data = OscillatorParameters.load(FitzNagumoRing.name)
    fitz_net = fitz_model(
        FitzNagumoRing.a_parameters,
        FitzNagumoRing.b_parameters,
        FitzNagumoRing.e_parameters,
        FitzNagumoRing.excitations,
        FitzNagumoRing.coupling_parameters,
    )

    isf = fitz_data.left_floquet_vectors[1]
    limit_cycle = fitz_data.limit_cycle
    oscillator_phase_len = fitz_data.number_of_itters
    x_mod = limit_cycle[50] + 0.02

    simulation_steps = fitz_data.number_of_itters * NUM_OF_PERIODS
    nth_point = simulation_steps // NUM_PHASE_POINTS
    phase_saving_arr = np.zeros((NUM_PHASE_POINTS,), dtype=np.float64)
    amplitude_saving_arr = np.zeros((NUM_PHASE_POINTS,), dtype=np.complex128)

    phase_solver = PhaseSolver(fitz_net, oscillator_phase_len, STEP_FITZ)

    # Wrap system for solve_ivp
    def system(t: float, y: np.ndarray) -> np.ndarray:
        return fitz_net(y)

    # Pre-compute the full trajectory using solve_ivp
    t_final = simulation_steps * STEP_FITZ
    t_eval = np.linspace(0, t_final, simulation_steps)
    sol = solve_ivp(
        system,
        (0, t_final),
        x_mod,
        method="RK45",
        t_eval=t_eval,
        rtol=1e-7,
        atol=1e-10,
    )
    trajectory = sol.y.T

    # Sample phase and amplitude at specific points
    idx = 0
    for i in range(simulation_steps):
        if i % nth_point == 0 and idx < NUM_PHASE_POINTS:
            x_current = trajectory[i]
            relaxed_phase = phase_solver.relax_state(np.array([x_current]), 5)
            phase = int(phase_solver.calculate_phase(relaxed_phase, limit_cycle).item())
            i_theta = isf[phase]
            x_theta = limit_cycle[phase]
            dx = x_theta - x_current
            amplitude = np.dot(i_theta, dx)
            phase_saving_arr[idx] = phase
            amplitude_saving_arr[idx] = amplitude
            idx += 1

    phase_saving_arr = (phase_saving_arr / np.max(phase_saving_arr)) * (2 * np.pi)
    time_vector = np.arange(0, NUM_PHASE_POINTS) * nth_point * STEP_FITZ
    num_of_itter = time_vector.shape[0] // NUM_OF_PERIODS

    # Real and Complex amplitude logarithm
    real_log = np.log(amplitude_saving_arr).real
    imag_log = np.log(amplitude_saving_arr).imag
    # Estimate freq and amplitude

    omega_res = stats.linregress(time_vector[:40], phase_saving_arr[:40])
    omega = omega_res.slope

    real_res = stats.linregress(time_vector[:40], real_log[:40])
    real_r = real_res.slope

    imag_res = stats.linregress(time_vector[:40], imag_log[:40])
    imag_r = imag_res.slope

    # AMPLITUDE PLOT
    fig, ax = plt.subplots()

    ax.plot(time_vector, real_log, label="$Re(r)$")
    x_line, y_line, txt_x, txt_y = get_offset_line(
        time_vector, real_log, 0.3, num_of_itter
    )
    rotation = np.rad2deg(np.arctan(real_r))
    ax.text(
        txt_x,
        txt_y,
        rf"$Re(r)={real_r:.3f}$",
        ha="center",
        rotation=rotation,
        rotation_mode="anchor",
        transform_rotates_text=True,
    )
    ax.plot(x_line, y_line, c="blue")

    ax.plot(time_vector, imag_log, label="$Im(r)$")
    x_line, y_line, txt_x, txt_y = get_offset_line(
        time_vector, imag_log, 0.3, num_of_itter
    )

    rotation = np.rad2deg(np.arctan(imag_r))
    ax.text(
        txt_x,
        txt_y,
        rf"$Im(r)= {imag_r:.3f}$",
        ha="center",
        rotation=rotation,
        rotation_mode="anchor",
        transform_rotates_text=True,
    )
    ax.plot(x_line, y_line, c="blue")
    ax.grid()
    ax.set_ylabel(r"$\ln{R_{1}}$")
    ax.set_xlabel("$time(s)$")
    ax.legend()
    plt.savefig(BASE_DIR / "fitz_ring_amp.png")
    plt.show()

    # PHASE PLOT
    fig, ax = plt.subplots()

    x_line, y_line, txt_x, txt_y = get_offset_line(
        time_vector, phase_saving_arr, 0.3, num_of_itter
    )
    rotation = np.rad2deg(np.arctan(omega))
    ax.text(
        txt_x,
        txt_y,
        rf"$\omega = {omega:.3f}$",
        ha="center",
        rotation=rotation,
        rotation_mode="anchor",
        transform_rotates_text=True,
    )

    ax.plot(time_vector, phase_saving_arr)
    ax.plot(x_line, y_line, c="blue")
    ax.grid()
    ax.set_yticks(
        [0, np.pi / 2, np.pi, 3 * np.pi / 2, 2 * np.pi],
        ["0", r"$\frac{\pi}{2}$", r"$\pi$", r"$\frac{3\pi}{2}$", r"$2\pi$"],
    )
    ax.set_ylim(0, 2 * np.pi)
    ax.set_xlabel("$time(s)$")
    ax.set_ylabel(r"$\theta$")

    plt.savefig(BASE_DIR / "fitz_ring_phase.png")
    plt.show()

    # LIMIT CYCLE PLOT
    lim_cycle = np.vstack((limit_cycle, limit_cycle, limit_cycle))
    lim_time_vector = np.arange(0, lim_cycle.shape[0]) * STEP_FITZ
    fig, ax = plt.subplots()
    ax.plot(lim_time_vector, lim_cycle[:, 1::2])
    ax.set_xlabel("$time(s)$")
    ax.set_ylabel(r"$v_{i}$")
    plt.axvline(x=limit_cycle.shape[0] * STEP_FITZ, color="black", linestyle="--")
    plt.axvline(x=2 * limit_cycle.shape[0] * STEP_FITZ, color="black", linestyle="--")
    ax.grid()
    plt.savefig(BASE_DIR / "fitz_ring_limit_cycle_phase_amp.png")
    plt.show()


if __name__ == "__main__":
    main()
