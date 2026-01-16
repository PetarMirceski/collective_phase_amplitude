import matplotlib.pyplot as plt
import numpy as np
from scipy.integrate import solve_ivp

from config.oscillator_constants import WillamowskiRossler
from oscillators.willamowski_rossler import (
    willamowski_rossler_jacobian_model,
    willamowski_rossler_model,
)
from solvers.floquet import FloquetSolver, check_orthogonality
from solvers.period_finder import find_period
from utils.timing import ExecutionTimer


def main() -> None:
    willamowski = willamowski_rossler_model(
        b1=WillamowskiRossler.B1,
        b2=WillamowskiRossler.B2,
        d1=WillamowskiRossler.D1,
        d2=WillamowskiRossler.D2,
        d3=WillamowskiRossler.D3,
    )
    willamowski_jacobian = willamowski_rossler_jacobian_model(
        b1=WillamowskiRossler.B1,
        b2=WillamowskiRossler.B2,
        d1=WillamowskiRossler.D1,
        d2=WillamowskiRossler.D2,
        d3=WillamowskiRossler.D3,
    )

    # Wrap the oscillator to match solve_ivp signature (t, y) -> dy
    def system(t: float, y: np.ndarray) -> np.ndarray:
        return willamowski(y)

    # Evolve to limit cycle
    evolve_time = 500.0
    with ExecutionTimer("Evolution to limit cycle"):
        sol = solve_ivp(
            system,
            (0, evolve_time),
            WillamowskiRossler.initial_conditions,
            method="RK45",
            rtol=1e-7,
            atol=1e-10,
        )
        converged_state = sol.y[:, -1]

    # Plot limit cycle
    t_plot = np.linspace(0, 50, 5000)
    sol_plot = solve_ivp(system, (0, 50), converged_state, t_eval=t_plot, method="RK45")
    plt.figure(figsize=(8, 6))
    plt.plot(sol_plot.y[0], sol_plot.y[1], label="Willamowski-Rossler limit cycle")
    plt.scatter([sol_plot.y[0, 0]], [sol_plot.y[1, 0]], color="red", s=50, zorder=5)
    plt.xlabel("x")
    plt.ylabel("y")
    plt.title("Willamowski-Rossler Limit Cycle (x-y projection)")
    plt.grid(True)
    plt.legend()
    plt.show()

    # Find period
    with ExecutionTimer("Period finding"):
        period, zero_phase_state = find_period(
            system=system,
            initial_state=converged_state,
            threshold_value=15,
            state_index=0,
            direction=-1,
        )
    print(f"Natural period: {period:.6f}")
    print(f"Natural frequency: {2 * np.pi / period:.6f}")

    # Compute Floquet vectors
    floquet_solver = FloquetSolver(
        system=willamowski,
        jacobian=willamowski_jacobian,
        rtol=1e-9,
        atol=1e-12,
    )

    with ExecutionTimer("Floquet analysis"):
        result = floquet_solver.solve(
            zero_phase_state=zero_phase_state,
            period=period,
            num_points=10000,
            num_floquet_modes=2,
            num_rotations=30,
        )

    print(f"\nFloquet exponents: {result.exponents}")
    print(f"λ_0 = {result.exponents[0]:.6f} (should be 0)")
    print(f"λ_1 = {result.exponents[1]:.6f} (should be negative)")
    print(f"λ_2 = {result.exponents[2]:.6f} (should be negative)")

    # Check orthogonality
    print("\nChecking bi-orthogonality...")
    check_orthogonality(result.left_vectors, result.right_vectors)

    # Extract vectors for plotting
    u0_vec = result.right_vectors[0]
    u1_vec = result.right_vectors[1]
    u2_vec = result.right_vectors[2]
    v0_vec = result.left_vectors[0]
    v1_vec = result.left_vectors[1]
    v2_vec = result.left_vectors[2]

    # Plot v1 (ISF)
    plt.figure(figsize=(10, 4))
    plt.plot(result.t_eval, v1_vec[:, 0], label="v1_x")
    plt.plot(result.t_eval, v1_vec[:, 1], label="v1_y")
    plt.plot(result.t_eval, v1_vec[:, 2], label="v1_z")
    plt.xlabel("t")
    plt.ylabel("v1")
    plt.title("First Left Floquet Vector (ISF)")
    plt.legend()
    plt.grid(True)
    plt.show()

    # Plotting all Floquet vectors (2x3 for 3D system)
    fig, axs = plt.subplots(2, 3, figsize=(18, 10))

    num_points = u0_vec.shape[0]
    theta_ticks = [0, num_points // 4, num_points // 2, 3 * num_points // 4, num_points]
    theta_labels = ["0", r"$\pi/2$", r"$\pi$", r"$3\pi/2$", r"$2\pi$"]

    for ax in axs.flat:
        ax.set_xticks(theta_ticks)
        ax.set_xticklabels(theta_labels)
        ax.set_xlabel(r"$\theta$", fontsize=14)
        ax.grid(True)

    # u0
    axs[0, 0].plot(u0_vec[:, 0], color="r", label=r"$u_0^{(1)}$")
    axs[0, 0].plot(u0_vec[:, 1], "--", color="g", label=r"$u_0^{(2)}$")
    axs[0, 0].plot(u0_vec[:, 2], ":", color="b", label=r"$u_0^{(3)}$")
    axs[0, 0].set_ylabel(r"$u_0$", fontsize=14)
    axs[0, 0].set_title(r"$u_0(\theta)$ - Tangent to limit cycle")
    axs[0, 0].legend()

    # u1
    axs[0, 1].plot(u1_vec[:, 0], color="r", label=r"$u_1^{(1)}$")
    axs[0, 1].plot(u1_vec[:, 1], "--", color="g", label=r"$u_1^{(2)}$")
    axs[0, 1].plot(u1_vec[:, 2], ":", color="b", label=r"$u_1^{(3)}$")
    axs[0, 1].set_ylabel(r"$u_1$", fontsize=14)
    axs[0, 1].set_title(r"$u_1(\theta)$ - First right Floquet vector")
    axs[0, 1].legend()

    # u2
    axs[0, 2].plot(u2_vec[:, 0], color="r", label=r"$u_2^{(1)}$")
    axs[0, 2].plot(u2_vec[:, 1], "--", color="g", label=r"$u_2^{(2)}$")
    axs[0, 2].plot(u2_vec[:, 2], ":", color="b", label=r"$u_2^{(3)}$")
    axs[0, 2].set_ylabel(r"$u_2$", fontsize=14)
    axs[0, 2].set_title(r"$u_2(\theta)$ - Second right Floquet vector")
    axs[0, 2].legend()

    # v0
    axs[1, 0].plot(v0_vec[:, 0], color="r", label=r"$v_0^{(1)}$")
    axs[1, 0].plot(v0_vec[:, 1], "--", color="g", label=r"$v_0^{(2)}$")
    axs[1, 0].plot(v0_vec[:, 2], ":", color="b", label=r"$v_0^{(3)}$")
    axs[1, 0].set_ylabel(r"$v_0$", fontsize=14)
    axs[1, 0].set_title(r"$v_0(\theta)$ - Phase sensitivity function (PSF)")
    axs[1, 0].legend()

    # v1
    axs[1, 1].plot(v1_vec[:, 0], color="r", label=r"$v_1^{(1)}$")
    axs[1, 1].plot(v1_vec[:, 1], "--", color="g", label=r"$v_1^{(2)}$")
    axs[1, 1].plot(v1_vec[:, 2], ":", color="b", label=r"$v_1^{(3)}$")
    axs[1, 1].set_ylabel(r"$v_1$", fontsize=14)
    axs[1, 1].set_title(r"$v_1(\theta)$ - First isostable sensitivity function (ISF)")
    axs[1, 1].legend()

    # v2
    axs[1, 2].plot(v2_vec[:, 0], color="r", label=r"$v_2^{(1)}$")
    axs[1, 2].plot(v2_vec[:, 1], "--", color="g", label=r"$v_2^{(2)}$")
    axs[1, 2].plot(v2_vec[:, 2], ":", color="b", label=r"$v_2^{(3)}$")
    axs[1, 2].set_ylabel(r"$v_2$", fontsize=14)
    axs[1, 2].set_title(r"$v_2(\theta)$ - Second isostable sensitivity function (ISF)")
    axs[1, 2].legend()

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
        f"λ_1 = {result.exponents[1]:.6f}, λ_2 = {result.exponents[2]:.6f}"
    )
    print(f"Limit cycle shape: {result.limit_cycle.shape}")
    print(f"Right vectors shape: {result.right_vectors.shape}")
    print(f"Left vectors shape: {result.left_vectors.shape}")


if __name__ == "__main__":
    main()
