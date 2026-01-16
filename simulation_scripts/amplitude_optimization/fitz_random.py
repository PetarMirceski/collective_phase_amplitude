import numpy as np

from config.constants import OptimizationParameters, OscillatorParameters
from config.oscillator_constants import FitzNagumo
from config.simulation_parameters.fitz_random import optimization_configuration
from oscillators.fitz_network import fitz_model, fitz_perturbated_model
from solvers.control_algorithms.phase_method_amplitude import (
    OptimalEntrainmentSolverPenalty,
)
from solvers.phase_solver import PhaseSolver
from utils.timing import ExecutionTimer


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
    isf = fitz_data.left_floquet_vectors[1]
    # Compute step from stored data to ensure consistency
    step = fitz_data.natural_period / fitz_data.number_of_itters

    for config in optimization_configuration:
        psf_sol = psf[:, config.states]
        psf_diff_sol = psf_diff[:, config.states]
        isf_sol = isf[:, config.states]

        with ExecutionTimer("Minimizing nu and mu"):
            input_freq = natural_freq - config.delta
            solver = OptimalEntrainmentSolverPenalty(
                psf=psf_sol,
                psf_diff=psf_diff_sol,
                isf=isf_sol,
                power=config.power,
                delta=config.delta,
                omega=natural_freq,
                input_freq=input_freq,
                optimization_weight=config.k,
                simulation_time=config.simulation_time,
                dt=step,
                phase_len=oscillator_phase_len,
            )

        state_point = limit_cycle[int(limit_cycle.shape[0] * config.initial_phase)]
        phase_solver = PhaseSolver(fitz_net, oscillator_phase_len, step)

        with ExecutionTimer("Computing optimal input"):
            optimal_input_sol, external_force_phase = solver.get_input_and_phase()

        optimal_input = np.zeros((optimal_input_sol.shape[0], psf.shape[1]))
        optimal_input[:, config.states] = optimal_input_sol

        gamma = PhaseSolver.calculate_phase_coupling_function_fast(
            config.delta, psf, optimal_input, step, fitz_data.natural_period
        )

        limit_cycle_with_input = solver.apply_input(
            fitz_net_perturbated, state_point, optimal_input
        )

        idx = np.arange(
            0, limit_cycle_with_input.shape[0], oscillator_phase_len, dtype=np.int32
        )
        limit_cycle_input_cut = limit_cycle_with_input[idx]
        efp_cut = external_force_phase[idx]
        relaxed_phase = phase_solver.relax_state(limit_cycle_input_cut, 5)
        phase = phase_solver.calculate_phase(relaxed_phase, limit_cycle)

        mask = ~np.isnan(phase.astype(float))
        phase = phase[mask]
        efp_cut = efp_cut[mask]
        phase_min_pi_pi = np.unwrap(
            phase_solver.minus_pi_pi_range_transform(phase - efp_cut)
        )

        optimization_config = OptimizationParameters(
            nu=solver.nu,
            mu=solver.mu,
            input_index=config.states,
            gamma=gamma,
            input=optimal_input,
            limit_cycle_input=limit_cycle_with_input,
            phase_diff=phase_min_pi_pi,
            simulation_time=config.simulation_time,
            delta=config.delta,
            init_phase=config.initial_phase,
            power=config.power,
            name=FitzNagumo.name,
            type="amplitude",
        )
        optimization_config.dump()


if __name__ == "__main__":
    main()
