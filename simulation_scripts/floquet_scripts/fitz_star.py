import matplotlib.pyplot as plt
import numpy as np
from scipy.integrate import solve_ivp

from config.constants import OscillatorParameters
from config.oscillator_constants import FitzNagumoStar
from oscillators.fitz_network import fitz_jacobian_model, fitz_model
from solvers.floquet import FloquetResult, FloquetSolver, check_orthogonality
from solvers.period_finder import find_period
from utils.timing import ExecutionTimer


def floquet_result_to_oscillator_params(
    result: FloquetResult, oscillator_name: str
) -> OscillatorParameters:
    """Convert FloquetResult to OscillatorParameters for persistence."""
    return OscillatorParameters(
        oscillator_name=oscillator_name,
        natural_freq=result.omega,
        natural_period=result.period,
        number_of_itters=result.limit_cycle.shape[0],
        limit_cycle=result.limit_cycle,
        left_floquet_vectors=result.left_vectors,
        right_floquet_vectors=result.right_vectors,
        floquet_exponents=result.exponents,
        v0_diff=result.v0_diff,
    )


def main() -> None:
    fitz_net = fitz_model(
        FitzNagumoStar.a_parameters,
        FitzNagumoStar.b_parameters,
        FitzNagumoStar.e_parameters,
        FitzNagumoStar.excitations,
        FitzNagumoStar.coupling_parameters,
    )
    fitz_jacobian = fitz_jacobian_model(
        FitzNagumoStar.b_parameters,
        FitzNagumoStar.e_parameters,
        FitzNagumoStar.coupling_parameters,
    )

    # Wrap the oscillator to match solve_ivp signature (t, y) -> dy
    def system(t: float, y: np.ndarray) -> np.ndarray:
        return fitz_net(y)

    # Evolve to limit cycle
    evolve_time = 500.0
    with ExecutionTimer("Evolution to limit cycle"):
        sol = solve_ivp(
            system,
            (0, evolve_time),
            FitzNagumoStar.initial_conditions,
            method="RK45",
            rtol=1e-7,
            atol=1e-10,
        )
        converged_state = sol.y[:, -1]

    # Plot limit cycle (v1 vs v2 projection)
    t_plot = np.linspace(0, 100, 10000)
    sol_plot = solve_ivp(
        system, (0, 100), converged_state, t_eval=t_plot, method="RK45"
    )
    plt.figure(figsize=(8, 6))
    # Plot v1 vs v2 (indices 1 and 3 for odd elements)
    plt.plot(sol_plot.y[1], sol_plot.y[3], label="FitzHugh-Nagumo Star limit cycle")
    plt.scatter([sol_plot.y[1, 0]], [sol_plot.y[3, 0]], color="red", s=50, zorder=5)
    plt.xlabel("v1")
    plt.ylabel("v2")
    plt.title("FitzHugh-Nagumo Star Limit Cycle (v1-v2 projection)")
    plt.grid(True)
    plt.legend()
    plt.show()

    # Find period
    with ExecutionTimer("Period finding"):
        period, zero_phase_state = find_period(
            system=system,
            initial_state=converged_state,
            threshold_value=0,
            state_index=1,
            direction=-1,
        )
    print(f"Natural period: {period:.6f}")
    print(f"Natural frequency: {2 * np.pi / period:.6f}")

    # Compute Floquet vectors
    floquet_solver = FloquetSolver(
        system=fitz_net,
        jacobian=fitz_jacobian,
        rtol=1e-9,
        atol=1e-12,
    )

    with ExecutionTimer("Floquet analysis"):
        result = floquet_solver.solve(
            zero_phase_state=zero_phase_state,
            period=period,
            num_points=10000,
            num_floquet_modes=1,
            num_rotations=20,
        )

    print(f"\nFloquet exponents: {result.exponents}")
    print(f"λ_0 = {result.exponents[0]:.6f} (should be 0)")
    print(f"λ_1 = {result.exponents[1]:.6f} (should be negative)")

    # Check orthogonality
    print("\nChecking bi-orthogonality...")
    check_orthogonality(result.left_vectors, result.right_vectors)

    # Extract vectors for plotting
    u0_vec = result.right_vectors[0]
    u1_vec = result.right_vectors[1]
    v0_vec = result.left_vectors[0]
    v1_vec = result.left_vectors[1]

    # Plot v1 (ISF) components
    plt.figure(figsize=(12, 5))
    for i in range(v1_vec.shape[1]):
        plt.plot(result.t_eval, v1_vec[:, i], label=f"v1_{i}")
    plt.xlabel("t")
    plt.ylabel("v1")
    plt.title("First Left Floquet Vector (ISF)")
    plt.legend(bbox_to_anchor=(1.05, 1), loc="upper left")
    plt.grid(True)
    plt.tight_layout()
    plt.show()

    # Plotting all Floquet vectors
    fig, axs = plt.subplots(2, 2, figsize=(14, 10))

    num_points = u0_vec.shape[0]
    theta_ticks = [0, num_points // 4, num_points // 2, 3 * num_points // 4, num_points]
    theta_labels = ["0", r"$\pi/2$", r"$\pi$", r"$3\pi/2$", r"$2\pi$"]

    for ax in axs.flat:
        ax.set_xticks(theta_ticks)
        ax.set_xticklabels(theta_labels)
        ax.set_xlabel(r"$\theta$", fontsize=14)
        ax.grid(True)

    # u0
    for i in range(u0_vec.shape[1]):
        axs[0, 0].plot(u0_vec[:, i], label=f"$u_0^{{({i})}}$")
    axs[0, 0].set_ylabel(r"$u_0$", fontsize=14)
    axs[0, 0].set_title(r"$u_0(\theta)$ - Tangent to limit cycle")

    # u1
    for i in range(u1_vec.shape[1]):
        axs[0, 1].plot(u1_vec[:, i], label=f"$u_1^{{({i})}}$")
    axs[0, 1].set_ylabel(r"$u_1$", fontsize=14)
    axs[0, 1].set_title(r"$u_1(\theta)$ - First right Floquet vector")

    # v0
    for i in range(v0_vec.shape[1]):
        axs[1, 0].plot(v0_vec[:, i], label=f"$v_0^{{({i})}}$")
    axs[1, 0].set_ylabel(r"$v_0$", fontsize=14)
    axs[1, 0].set_title(r"$v_0(\theta)$ - Phase sensitivity function (PSF)")

    # v1
    for i in range(v1_vec.shape[1]):
        axs[1, 1].plot(v1_vec[:, i], label=f"$v_1^{{({i})}}$")
    axs[1, 1].set_ylabel(r"$v_1$", fontsize=14)
    axs[1, 1].set_title(r"$v_1(\theta)$ - Isostable sensitivity function (ISF)")

    plt.tight_layout()
    plt.show()

    # Print summary
    print("\n" + "=" * 50)
    print("Floquet Analysis Summary")
    print("=" * 50)
    print(f"Period T = {result.period:.6f}")
    print(f"Angular frequency ω = {result.omega:.6f}")
    print(
        f"Floquet exponents: λ_0 = {result.exponents[0]:.6f}, "
        f"λ_1 = {result.exponents[1]:.6f}"
    )
    print(f"System dimension: {FitzNagumoStar.n}")
    print(f"Number of oscillators: {FitzNagumoStar.num_of_oscillators}")
    print(f"Limit cycle shape: {result.limit_cycle.shape}")
    print(f"Right vectors shape: {result.right_vectors.shape}")
    print(f"Left vectors shape: {result.left_vectors.shape}")

    # Save results
    oscillator_params = floquet_result_to_oscillator_params(result, FitzNagumoStar.name)
    oscillator_params.dump()
    print(f"\nFloquet results saved for oscillator: {FitzNagumoStar.name}")


if __name__ == "__main__":
    main()
