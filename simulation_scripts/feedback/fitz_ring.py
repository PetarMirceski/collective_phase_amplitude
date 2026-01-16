import numpy as np

from config.constants import OptimizationParameters, OscillatorParameters
from config.oscillator_constants import FitzNagumoRing
from config.simulation_parameters.fitz_ring import optimization_configurations
from oscillators.fitz_network import fitz_model, fitz_perturbated_model
from solvers.control_algorithms.feedback_method import OptimalEntrainmentSolverFeedback
from solvers.control_algorithms.phase_method import OptimalEntrainmentSolver
from solvers.phase_solver import PhaseSolver
from utils.timing import ExecutionTimer

# Set to None to use the same time step as the optimal input (step)
# Or set to a value like 1e-2 for faster simulation with interpolation
SIMULATION_DT = None


def main() -> None:
    fitz_data = OscillatorParameters.load(FitzNagumoRing.name)
    fitz_net = fitz_model(
        FitzNagumoRing.a_parameters,
        FitzNagumoRing.b_parameters,
        FitzNagumoRing.e_parameters,
        FitzNagumoRing.excitations,
        FitzNagumoRing.coupling_parameters,
    )
    fitz_net_perturbated = fitz_perturbated_model(
        FitzNagumoRing.a_parameters,
        FitzNagumoRing.b_parameters,
        FitzNagumoRing.e_parameters,
        FitzNagumoRing.excitations,
        FitzNagumoRing.coupling_parameters,
    )

    psf = fitz_data.left_floquet_vectors[0].real
    psf_diff = fitz_data.v0_diff
    limit_cycle = fitz_data.limit_cycle
    natural_freq = fitz_data.natural_freq
    phase_len = fitz_data.number_of_itters
    step = fitz_data.natural_period / fitz_data.number_of_itters

    # Compute simulation step to have ~POINTS_PER_PERIOD points per period
    simulation_dt = SIMULATION_DT

    for config in optimization_configurations:
        psf_sol = psf[:, config.states]
        psf_diff_sol = psf_diff[:, config.states]
        input_freq = natural_freq - config.delta

        solver = OptimalEntrainmentSolver(
            psf=psf_sol,
            psf_diff=psf_diff_sol,
            power=config.power,
            delta=config.delta,
            omega=natural_freq,
            input_freq=input_freq,
            simulation_time=config.simulation_time,
            dt=step,
            phase_len=phase_len,
        )

        x_phase = int(limit_cycle.shape[0] * config.initial_phase)
        state_point = limit_cycle[x_phase]

        phase_solver = PhaseSolver(fitz_net, phase_len, step)
        feedback_solver = OptimalEntrainmentSolverFeedback(
            fitz_net,
            fitz_net_perturbated,
            limit_cycle,
            psf,
            config.simulation_time,
            step,
            phase_len,
            alpha=config.alpha,
            simulation_dt=simulation_dt,
        )

        with ExecutionTimer("Optimal Input inference"):
            optimal_input_sol, external_force_phase = solver.get_input_and_phase()

        optimal_input = np.zeros((optimal_input_sol.shape[0], psf.shape[1]))
        optimal_input[:, config.states] = optimal_input_sol

        with ExecutionTimer("Optimal Simulation started"):
            limit_cycle_with_input, applied_input = (
                feedback_solver.apply_input_fixed_step(
                    optimal_input, state_point, x_phase
                )
            )

        # Calculate phase at each period
        resampled_phase_len = feedback_solver.resampled_phase_len
        idx = np.arange(
            0, limit_cycle_with_input.shape[0], resampled_phase_len, dtype=np.int32
        )
        limit_cycle_input_cut = limit_cycle_with_input[idx]

        # Compute external force phase at the actual sampling times
        # idx represents time steps in the resampled simulation
        time_at_idx = idx * feedback_solver.dt
        # Convert time to equivalent step indices at original resolution
        # Then use same formula as phase_and_interpolation_parameters: efp = inp_freq * indexes / omega
        equivalent_indexes = time_at_idx / step
        efp_cut = (input_freq * equivalent_indexes / natural_freq) % phase_len

        relaxed_phase = phase_solver.relax_state(limit_cycle_input_cut, 5)
        phase = phase_solver.calculate_phase(relaxed_phase, limit_cycle)

        mask = ~np.isnan(phase.astype(float))
        phase = phase[mask]
        efp_cut = efp_cut[mask]
        phase_min_pi_pi = np.unwrap(
            phase_solver.minus_pi_pi_range_transform(phase - efp_cut)
        )
        import matplotlib.pyplot as plt

        plt.plot(phase_min_pi_pi)
        plt.show()

        gamma = np.ones((1, 3))

        optimization_config = OptimizationParameters(
            nu=solver.nu,
            mu=solver.mu,
            input_index=config.states,
            gamma=gamma,
            input=applied_input,
            limit_cycle_input=limit_cycle_with_input,
            phase_diff=phase_min_pi_pi,
            simulation_time=config.simulation_time,
            delta=config.delta,
            init_phase=config.initial_phase,
            power=config.power,
            name=FitzNagumoRing.name,
            type="feedback",
        )
        optimization_config.dump()


if __name__ == "__main__":
    main()
