import numpy as np
from tqdm import tqdm

from config.constants import OptimizationParameters, OscillatorParameters
from config.oscillator_constants import FitzNagumoStar
from config.simulation_parameters.fitz_star import optimization_configurations
from oscillators.fitz_network import fitz_model, fitz_perturbated_model
from solvers.control_algorithms.phase_method import OptimalEntrainmentSolver
from solvers.phase_solver import PhaseSolver
from utils.timing import ExecutionTimer


def main() -> None:
    fitz_data = OscillatorParameters.load(FitzNagumoStar.name)
    fitz_net = fitz_model(
        FitzNagumoStar.a_parameters,
        FitzNagumoStar.b_parameters,
        FitzNagumoStar.e_parameters,
        FitzNagumoStar.excitations,
        FitzNagumoStar.coupling_parameters,
    )
    fitz_net_perturbated = fitz_perturbated_model(
        FitzNagumoStar.a_parameters,
        FitzNagumoStar.b_parameters,
        FitzNagumoStar.e_parameters,
        FitzNagumoStar.excitations,
        FitzNagumoStar.coupling_parameters,
    )

    psf = fitz_data.left_floquet_vectors[0].real
    psf_diff = fitz_data.v0_diff
    limit_cycle = fitz_data.limit_cycle
    natural_freq = fitz_data.natural_freq
    oscillator_phase_len = fitz_data.number_of_itters
    step = fitz_data.natural_period / oscillator_phase_len

    for config in tqdm(optimization_configurations):
        num_steps = int(config.simulation_time / step)
        t = np.linspace(0.0, config.simulation_time, num_steps, endpoint=False)
        sine_amp = config.power * np.sqrt(2)
        sine_wave = sine_amp * np.sin(natural_freq * t)
        sine_wave = np.array([sine_wave] * len(config.states)).T

        psf_sol = psf[:, config.states]
        psf_diff_sol = psf_diff[:, config.states]
        input_freq = natural_freq - 0
        solver = OptimalEntrainmentSolver(
            psf=psf_sol,
            psf_diff=psf_diff_sol,
            power=config.power,
            delta=0,
            omega=natural_freq,
            input_freq=input_freq,
            simulation_time=config.simulation_time,
            dt=step,
            phase_len=oscillator_phase_len,
        )

        state_point = limit_cycle[int(limit_cycle.shape[0] * config.initial_phase)]

        phase_solver = PhaseSolver(fitz_net, oscillator_phase_len, step)
        with ExecutionTimer("Optimal Simulation started"):
            _, external_force_phase = solver.get_input_and_phase()
        optimal_input = np.zeros((sine_wave.shape[0], psf.shape[1]))

        sine_wave = np.roll(
            sine_wave, -int(limit_cycle.shape[0] * config.initial_phase) + 350
        )
        for idxs, state in enumerate(config.states):
            optimal_input[:, state] = -sine_wave[:, idxs]

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

        optimization_config = OptimizationParameters(
            solver.nu,
            solver.mu,
            config.states,
            gamma,
            optimal_input,
            limit_cycle_with_input,
            phase_min_pi_pi,
            simulation_time=config.simulation_time,
            delta=0,
            init_phase=config.initial_phase,
            power=config.power,
            name=FitzNagumoStar.name,
            type="sine",
        )
        optimization_config.dump()


if __name__ == "__main__":
    main()
