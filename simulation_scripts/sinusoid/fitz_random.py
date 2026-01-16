import numpy as np
from tqdm import tqdm

from config.constants import OptimizationParameters, OscillatorParameters
from config.oscillator_constants import FitzNagumo
from oscillators.fitz_network import fitz_model, fitz_perturbated_model
from solvers.control_algorithms.phase_method import OptimalEntrainmentSolver
from solvers.phase_solver import PhaseSolver
from utils.timing import ExecutionTimer

CONFIG = [
    ([1, 3, 5], 1e-5, 3800.0),
    ([1, 3, 5], 5e-4, 3800.0),
    ([1], 5e-5, 4500.0),
    ([1], 5e-4, 4500.0),
]

DIFF = 1 / 30


def main() -> None:
    fitz_data = OscillatorParameters.load(FitzNagumo.name)
    fitz_net = fitz_model(
        FitzNagumo.a_parameters,
        FitzNagumo.b_parameters,
        FitzNagumo.e_parameters,
        FitzNagumo.excitations,
        FitzNagumo.coupling_parameters,
    )
    fitz_net_perturbated = fitz_perturbated_model(
        FitzNagumo.a_parameters,
        FitzNagumo.b_parameters,
        FitzNagumo.e_parameters,
        FitzNagumo.excitations,
        FitzNagumo.coupling_parameters,
    )

    psf = fitz_data.left_floquet_vectors[0].real
    psf_diff = fitz_data.v0_diff
    limit_cycle = fitz_data.limit_cycle
    natural_freq = fitz_data.natural_freq
    oscillator_phase_len = fitz_data.number_of_itters
    step = fitz_data.natural_period / oscillator_phase_len

    for config, power, sim_time in tqdm(CONFIG):
        num_steps = int(sim_time / step)
        t = np.linspace(0.0, sim_time, num_steps, endpoint=False)
        sine_amp = power * np.sqrt(2)
        sine_wave = sine_amp * np.sin(natural_freq * t) / len(config)
        sine_wave = np.array([sine_wave] * len(config)).T

        psf_sol = psf[:, config]
        psf_diff_sol = psf_diff[:, config]
        input_freq = natural_freq - 0
        solver = OptimalEntrainmentSolver(
            psf=psf_sol,
            psf_diff=psf_diff_sol,
            power=power,
            delta=0,
            omega=natural_freq,
            input_freq=input_freq,
            simulation_time=sim_time,
            dt=step,
            phase_len=oscillator_phase_len,
        )

        state_point = limit_cycle[int(limit_cycle.shape[0] * DIFF)]

        phase_solver = PhaseSolver(fitz_net, oscillator_phase_len, step)
        with ExecutionTimer("Optimal Simulation started"):
            _, external_force_phase = solver.get_input_and_phase()
        optimal_input = np.zeros((sine_wave.shape[0], psf.shape[1]))

        for idxs, state in enumerate(config):
            optimal_input[:, state] = sine_wave[:, idxs]

        gamma = PhaseSolver.calculate_phase_coupling_function_fast(
            0, psf, optimal_input, step, fitz_data.natural_period
        )

        limit_cycle_with_input = solver.apply_input(
            fitz_net_perturbated, state_point, optimal_input
        )

        # slow operation
        idx = np.arange(
            0, limit_cycle_with_input.shape[0], oscillator_phase_len, dtype=np.int32
        )
        limit_cycle_input_cut = limit_cycle_with_input[idx]
        efp_cut = external_force_phase[idx]
        relaxed_phase = phase_solver.relax_state(limit_cycle_input_cut, 3)
        phase = phase_solver.calculate_phase(relaxed_phase, limit_cycle)

        mask = ~np.isnan(phase.astype(float))
        phase = phase[mask]
        efp_cut = efp_cut[mask]
        phase_min_pi_pi = np.unwrap(
            ((phase - efp_cut) / oscillator_phase_len) * 2 * np.pi
        )
        # import matplotlib.pyplot as plt

        # plt.plot(phase_min_pi_pi)
        # plt.show()

        optimization_config = OptimizationParameters(
            solver.nu,
            solver.mu,
            config,
            gamma,
            optimal_input,
            limit_cycle_with_input,
            phase_min_pi_pi,
            simulation_time=sim_time,
            delta=0,
            init_phase=DIFF,
            power=power,
            name=FitzNagumo.name,
            type="sine",
        )
        optimization_config.dump()


if __name__ == "__main__":
    main()
