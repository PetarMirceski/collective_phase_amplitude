import matplotlib.pyplot as plt
import numpy as np
from scipy.integrate import solve_ivp

from oscillators.fitz_element import fitz_element_jacobian_model, fitz_element_model
from solvers.floquet import FloquetSolver, check_orthogonality
from solvers.period_finder import find_period
from utils.timing import ExecutionTimer

# FitzHugh-Nagumo single element parameters
A_PARAM = 0.7
B_PARAM = 0.8
E_PARAM = 0.08
EXCITATION = 0.8
INITIAL_CONDITIONS = np.array([0.0, 1.0])


def main() -> None:
    fitz_el = fitz_element_model(
        a_parameter=A_PARAM,
        b_parameter=B_PARAM,
        e_parameter=E_PARAM,
        excitation=EXCITATION,
    )
    fitz_el_jac = fitz_element_jacobian_model(
        a_parameter=A_PARAM,
        b_parameter=B_PARAM,
        e_parameter=E_PARAM,
        excitation=EXCITATION,
    )

    # Wrap the oscillator to match solve_ivp signature (t, y) -> dy
    def system(t: float, y: np.ndarray) -> np.ndarray:
        return fitz_el(y)

    # Evolve to limit cycle
    evolve_time = 500.0
    with ExecutionTimer("Evolution to limit cycle"):
        sol = solve_ivp(
            system,
            (0, evolve_time),
            INITIAL_CONDITIONS,
            method="RK45",
            rtol=1e-7,
            atol=1e-10,
        )
        converged_state = sol.y[:, -1]

    # Plot limit cycle
    t_plot = np.linspace(0, 100, 10000)
    sol_plot = solve_ivp(
        system, (0, 100), converged_state, t_eval=t_plot, method="RK45"
    )
    plt.figure(figsize=(8, 6))
    plt.plot(sol_plot.y[0], sol_plot.y[1], label="FitzHugh-Nagumo limit cycle")
    plt.scatter([sol_plot.y[0, 0]], [sol_plot.y[1, 0]], color="red", s=50, zorder=5)
    plt.xlabel("u")
    plt.ylabel("v")
    plt.title("FitzHugh-Nagumo Single Element Limit Cycle")
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
        system=fitz_el,
        jacobian=fitz_el_jac,
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

    # Plot v1 (ISF)
    plt.figure(figsize=(10, 4))
    plt.plot(result.t_eval, v1_vec[:, 0], label="v1_u")
    plt.plot(result.t_eval, v1_vec[:, 1], label="v1_v")
    plt.xlabel("t")
    plt.ylabel("v1")
    plt.title("First Left Floquet Vector (ISF)")
    plt.legend()
    plt.grid(True)
    plt.show()

    # Plotting all Floquet vectors
    fig, axs = plt.subplots(2, 2, figsize=(14, 10))

    num_points = u0_vec.shape[0]
    theta_ticks = [0, num_points // 4, num_points // 2, 3 * num_points // 4, num_points]
    theta_labels = ["0", r"$\pi/2$", r"$\pi$", r"$3\pi/2$", r"$2\pi$"]

    for ax in axs.flat:
        ax.set_xticks(theta_ticks)
        ax.set_xticklabels(theta_labels)
        ax.set_ylim(-2, 2)
        ax.set_xlabel(r"$\theta$", fontsize=14)
        ax.grid(True)

    # u0
    axs[0, 0].plot(u0_vec[:, 0], color="r", label=r"$u_0^{(1)}$")
    axs[0, 0].plot(u0_vec[:, 1], "--", color="g", label=r"$u_0^{(2)}$")
    axs[0, 0].set_ylabel(r"$u_0$", fontsize=14)
    axs[0, 0].set_title(r"$u_0(\theta)$ - Tangent to limit cycle")
    axs[0, 0].legend()

    # u1
    axs[0, 1].plot(u1_vec[:, 0], color="r", label=r"$u_1^{(1)}$")
    axs[0, 1].plot(u1_vec[:, 1], "--", color="g", label=r"$u_1^{(2)}$")
    axs[0, 1].set_ylabel(r"$u_1$", fontsize=14)
    axs[0, 1].set_title(r"$u_1(\theta)$ - First right Floquet vector")
    axs[0, 1].legend()

    # v0
    axs[1, 0].plot(v0_vec[:, 0], color="r", label=r"$v_0^{(1)}$")
    axs[1, 0].plot(v0_vec[:, 1], "--", color="g", label=r"$v_0^{(2)}$")
    axs[1, 0].set_ylabel(r"$v_0$", fontsize=14)
    axs[1, 0].set_title(r"$v_0(\theta)$ - Phase sensitivity function (PSF)")
    axs[1, 0].legend()

    # v1
    axs[1, 1].plot(v1_vec[:, 0], color="r", label=r"$v_1^{(1)}$")
    axs[1, 1].plot(v1_vec[:, 1], "--", color="g", label=r"$v_1^{(2)}$")
    axs[1, 1].set_ylabel(r"$v_1$", fontsize=14)
    axs[1, 1].set_title(r"$v_1(\theta)$ - Isostable sensitivity function (ISF)")
    axs[1, 1].legend()

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
    print(f"Limit cycle shape: {result.limit_cycle.shape}")
    print(f"Right vectors shape: {result.right_vectors.shape}")
    print(f"Left vectors shape: {result.left_vectors.shape}")


if __name__ == "__main__":
    main()
