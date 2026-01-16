import matplotlib.pyplot as plt
import numpy as np

from config.constants import (
    NUM_PHASE_POINTS,
    OptimizationParameters,
    OscillatorParameters,
    VanConfig,
)
from config.oscillator_constants import VanDerPolScaled
from oscillators.van_der_pol_scaled import (
    perturbated_van_der_pol_model_scaled,
    van_der_pol_model_scaled,
)
from solvers.control_algorithms.phase_method import OptimalEntrainmentSolver
from solvers.phase_solver import PhaseSolver
from utils.timing import ExecutionTimer


def main() -> None:
    van_data = OscillatorParameters.load(VanDerPolScaled.name)
    van_der_pol = van_der_pol_model_scaled(
        mi=VanDerPolScaled.MI,
        x0=VanDerPolScaled.X0,
        y0=VanDerPolScaled.Y0,
        d=VanDerPolScaled.D,
    )
    van_der_pol_perturbated = perturbated_van_der_pol_model_scaled(
        mi=VanDerPolScaled.MI,
        x0=VanDerPolScaled.X0,
        y0=VanDerPolScaled.Y0,
        d=VanDerPolScaled.D,
    )

    psf = van_data.left_floquet_vectors[0].real
    psf_diff = van_data.v0_diff
    limit_cycle = van_data.limit_cycle
    natural_freq = van_data.natural_freq
    oscillator_phase_len = van_data.number_of_itters
    # Compute step from stored data to ensure consistency
    step = van_data.natural_period / van_data.number_of_itters

    control_states = VanConfig.states

    for power in VanConfig.power_list:
        for cont_states in control_states:
            psf_sol = psf[:, cont_states]
            psf_diff_sol = psf_diff[:, cont_states]
            input_freq = natural_freq - VanConfig.delta
            solver = OptimalEntrainmentSolver(
                psf=psf_sol,
                psf_diff=psf_diff_sol,
                power=power,
                delta=VanConfig.delta,
                omega=natural_freq,
                input_freq=input_freq,
                simulation_time=VanConfig.simulation_time,
                dt=step,
                phase_len=oscillator_phase_len,
            )

            state_point = limit_cycle[
                int(limit_cycle.shape[0] * VanConfig.initial_phase)
            ]

            phase_solver = PhaseSolver(van_der_pol, oscillator_phase_len, step)
            with ExecutionTimer("Optimal Simulation started"):
                optimal_input_sol, external_force_phase = solver.get_input_and_phase()

            optimal_input = np.zeros((optimal_input_sol.shape[0], psf.shape[1]))
            optimal_input[:, cont_states] = optimal_input_sol

            gamma = phase_solver.calculate_phase_coupling_function(
                VanConfig.delta, psf, optimal_input
            )

            with ExecutionTimer("Applying Input"):
                limit_cycle_with_input = solver.apply_input(
                    van_der_pol_perturbated, state_point, optimal_input
                )

            # slow operation
            idx = np.linspace(
                0, limit_cycle_with_input.shape[0] - 1, NUM_PHASE_POINTS, dtype=np.int32
            )
            limit_cycle_input_cut = limit_cycle_with_input[idx]
            efp_cut = external_force_phase[idx]

            with ExecutionTimer("Relax State"):
                relaxed_phase = phase_solver.relax_state(limit_cycle_input_cut, 5)

            with ExecutionTimer("Calculating phase"):
                phase = phase_solver.calculate_phase(relaxed_phase, limit_cycle)

            mask = ~np.isnan(phase.astype(float))
            phase = phase[mask]
            efp_cut = efp_cut[mask]
            phase_min_pi_pi = np.unwrap(
                phase_solver.minus_pi_pi_range_transform(phase - efp_cut)
            )
            plt.plot(phase_min_pi_pi)
            plt.show()

            plt.plot(limit_cycle_with_input[:, 0], limit_cycle_with_input[:, 1])
            plt.plot(limit_cycle[:, 0], limit_cycle[:, 1])
            plt.show()

            optimization_config = OptimizationParameters(
                nu=solver.nu,
                mu=solver.mu,
                input_index=cont_states,
                gamma=gamma,
                input=optimal_input,
                limit_cycle_input=limit_cycle_with_input,
                phase_diff=phase_min_pi_pi,
                simulation_time=VanConfig.simulation_time,
                delta=VanConfig.delta,
                init_phase=VanConfig.initial_phase,
                power=power,
                name=VanDerPolScaled.name,
                type="simple",
            )
            optimization_config.dump()


if __name__ == "__main__":
    main()
