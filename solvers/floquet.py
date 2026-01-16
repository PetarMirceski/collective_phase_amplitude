from collections.abc import Callable
from dataclasses import dataclass

import numpy as np
from scipy.integrate import solve_ivp
from scipy.linalg import eig

type StateVector = np.ndarray
type Trajectory = np.ndarray
type JacobianMatrix = np.ndarray
type VectorField = Callable[[StateVector], StateVector]
type JacobianFunc = Callable[[StateVector], JacobianMatrix]


@dataclass
class FloquetResult:
    """Container for Floquet analysis results.

    Attributes:
        limit_cycle: State trajectory along the limit cycle, shape (num_points, dim).
        t_eval: Time points corresponding to limit_cycle, shape (num_points,).
        period: Natural period of the oscillation.
        omega: Natural angular frequency (2*pi/period).
        exponents: Floquet exponents, shape (num_exponents,).
        right_vectors: Right Floquet vectors u_i(t), shape (num_vectors, num_points, dim).
        left_vectors: Left Floquet vectors v_i(t), shape (num_vectors, num_points, dim).
        v0_diff: Derivative of the zeroth left Floquet vector, shape (num_points, dim).
    """

    limit_cycle: Trajectory
    t_eval: np.ndarray
    period: float
    omega: float
    exponents: np.ndarray
    right_vectors: np.ndarray
    left_vectors: np.ndarray
    v0_diff: np.ndarray


class PeriodicInterpolator:
    """Linear interpolator for periodic trajectories.

    Provides efficient interpolation of state vectors along a periodic
    trajectory, handling wraparound at the period boundary.
    """

    def __init__(self, t_eval: np.ndarray, data: Trajectory, period: float) -> None:
        """Initialize the interpolator.

        Args:
            t_eval: Time points of the data, shape (num_points,).
            data: Trajectory data, shape (num_points, dim).
            period: Period of the trajectory.
        """
        self._t_eval = t_eval
        self._data = data
        self._period = period
        self._num_points = len(t_eval)

    def __call__(self, t: float) -> StateVector:
        """Interpolate the trajectory at time t.

        Args:
            t: Time at which to interpolate.

        Returns:
            Interpolated state vector.
        """
        t_mod = t % self._period
        idx = np.searchsorted(self._t_eval, t_mod)

        if idx == 0:
            return self._data[0]
        if idx >= self._num_points:
            return self._data[-1]

        t0, t1 = self._t_eval[idx - 1], self._t_eval[idx]
        alpha = (t_mod - t0) / (t1 - t0)
        return (1 - alpha) * self._data[idx - 1] + alpha * self._data[idx]


class FloquetSolver:
    """Solver for Floquet vectors and exponents of limit cycle oscillators.

    This class computes the Floquet decomposition of a limit cycle oscillator,
    providing both left (v) and right (u) Floquet vectors along with their
    corresponding exponents.

    The right vectors satisfy: du_i/dt = [J(x_0(t)) - λ_i] u_i
    The left vectors satisfy:  dv_i/dt = -[J^T(x_0(t)) - λ_i*] v_i

    With bi-orthogonality: <v_i, u_j> = δ_ij
    """

    # Default tolerances for numerical integration
    DEFAULT_RTOL = 1e-9
    DEFAULT_ATOL = 1e-12

    def __init__(
        self,
        system: VectorField,
        jacobian: JacobianFunc,
        rtol: float = DEFAULT_RTOL,
        atol: float = DEFAULT_ATOL,
    ) -> None:
        """Initialize the Floquet solver.

        Args:
            system: Vector field function F(x) -> dx/dt.
            jacobian: Jacobian function J(x) -> dF/dx.
            rtol: Relative tolerance for integration.
            atol: Absolute tolerance for integration.
        """
        self.system = system
        self.jacobian = jacobian
        self.rtol = rtol
        self.atol = atol

    def _integrate(
        self,
        rhs: Callable,
        y0: np.ndarray,
        t_span: tuple[float, float],
        t_eval: np.ndarray | None = None,
        dense_output: bool = False,
    ):
        """Wrapper for solve_ivp with default tolerances.

        Args:
            rhs: Right-hand side function.
            y0: Initial condition.
            t_span: Integration interval (t0, tf).
            t_eval: Times at which to store the solution.
            dense_output: Whether to compute a continuous solution.

        Returns:
            OdeResult object from solve_ivp.
        """
        return solve_ivp(
            rhs,
            t_span,
            y0,
            method="RK45",
            t_eval=t_eval,
            rtol=self.rtol,
            atol=self.atol,
            dense_output=dense_output,
        )

    def compute_limit_cycle(
        self,
        zero_phase_state: StateVector,
        period: float,
        num_points: int,
    ) -> tuple[np.ndarray, Trajectory]:
        """Integrate the system over one period to obtain the limit cycle.

        Args:
            zero_phase_state: Initial state at the zero-phase point.
            period: Oscillation period.
            num_points: Number of points to sample.

        Returns:
            Tuple of (t_eval, limit_cycle) where limit_cycle has shape (num_points, dim).
        """
        t_eval = np.linspace(0, period, num_points, endpoint=False)

        sol = self._integrate(
            lambda t, y: self.system(y),
            zero_phase_state,
            (0, period),
            t_eval=t_eval,
        )

        return sol.t, sol.y.T

    def compute_u0(self, limit_cycle: Trajectory, omega: float) -> Trajectory:
        """Compute the zeroth right Floquet vector u_0 = F(x_0) / ω.

        Args:
            limit_cycle: Limit cycle trajectory, shape (num_points, dim).
            omega: Natural angular frequency.

        Returns:
            u_0 vector along the limit cycle, shape (num_points, dim).
        """
        return np.array([self.system(x) / omega for x in limit_cycle])

    def compute_v0(
        self,
        limit_cycle: Trajectory,
        t_eval: np.ndarray,
        u0: Trajectory,
        omega: float,
        num_rotations: int = 30,
    ) -> tuple[Trajectory, Trajectory]:
        """Compute the zeroth left Floquet vector v_0 (phase sensitivity function).

        Solves: dv_0/dt = -J^T(x_0(t)) v_0 backwards in time, with normalization
        <v_0, u_0> = 1.

        Args:
            limit_cycle: Limit cycle trajectory, shape (num_points, dim).
            t_eval: Time points for the limit cycle.
            u0: Zeroth right Floquet vector.
            omega: Natural angular frequency.
            num_rotations: Number of rotations for convergence.

        Returns:
            Tuple of (v0, v0_derivative) both with shape (num_points, dim).
        """
        num_points, dim = limit_cycle.shape
        period = t_eval[-1] + (t_eval[1] - t_eval[0])

        interp = PeriodicInterpolator(t_eval, limit_cycle, period)

        def adjoint_rhs(t: float, v: StateVector) -> StateVector:
            x = interp(period - t)  # Backward time
            return self.jacobian(x).T @ v

        # Iterate to convergence starting from uniform initial guess
        v0_current = np.ones(dim) / np.sqrt(dim)

        for _ in range(num_rotations):
            sol = self._integrate(adjoint_rhs, v0_current, (0, period))
            v0_final = sol.y[:, -1]
            # Normalize: <v0, F/ω> = 1
            norm = np.dot(v0_final, self.system(limit_cycle[0]) / omega)
            v0_current = v0_final / norm

        # Compute v0 at all points along the limit cycle
        v0_vec = np.zeros((num_points, dim))
        v0_diff = np.zeros((num_points, dim))

        sol = self._integrate(
            adjoint_rhs,
            v0_current,
            (0, period),
            t_eval=np.linspace(0, period, num_points, endpoint=False),
            dense_output=True,
        )

        for i in range(num_points):
            t_backward = period - t_eval[i]
            v0_vec[i] = sol.sol(t_backward if t_backward < period else 0)
            # Normalize at each point
            norm = np.dot(v0_vec[i], self.system(limit_cycle[i]) / omega)
            v0_vec[i] = v0_vec[i] / norm

        # Compute derivative using the adjoint equation
        for i in range(num_points):
            v0_diff[i] = -self.jacobian(limit_cycle[i]).T @ v0_vec[i]

        return v0_vec, v0_diff

    def compute_monodromy(
        self,
        limit_cycle: Trajectory,
        t_eval: np.ndarray,
    ) -> JacobianMatrix:
        """Compute the monodromy matrix M = Φ(T) where Φ satisfies dΦ/dt = J(x_0(t))Φ.

        Args:
            limit_cycle: Limit cycle trajectory, shape (num_points, dim).
            t_eval: Time points for the limit cycle.

        Returns:
            Monodromy matrix of shape (dim, dim).
        """
        _, dim = limit_cycle.shape
        period = t_eval[-1] + (t_eval[1] - t_eval[0])

        interp = PeriodicInterpolator(t_eval, limit_cycle, period)

        def variational_rhs(t: float, phi_flat: np.ndarray) -> np.ndarray:
            phi = phi_flat.reshape((dim, dim))
            dphi = self.jacobian(interp(t)) @ phi
            return dphi.flatten()

        sol = self._integrate(
            variational_rhs,
            np.eye(dim).flatten(),
            (0, period),
        )

        return sol.y[:, -1].reshape((dim, dim))

    def compute_exponents_and_eigenvectors(
        self,
        monodromy: JacobianMatrix,
        period: float,
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        """Compute Floquet exponents and initial eigenvectors from monodromy matrix.

        Args:
            monodromy: Monodromy matrix.
            period: Oscillation period.

        Returns:
            Tuple of (exponents, left_eigenvectors, right_eigenvectors).
            The zeroth (λ=0) mode is excluded and results are sorted by Re(λ).
        """
        eigenvalues, left_vecs, right_vecs = eig(monodromy, left=True, right=True)  # pyright: ignore

        exponents = np.log(eigenvalues.astype(np.complex128)) / period

        # Find and remove the zero exponent (λ_0 = 0)
        zero_idx = np.argmin(np.abs(np.real(exponents)))
        if np.abs(exponents[zero_idx]) > 1e-3:
            print(f"Warning: λ_0 = {exponents[zero_idx]}, expected ~0")

        mask = np.ones(len(exponents), dtype=bool)
        mask[zero_idx] = False

        exponents = exponents[mask]
        left_vecs = left_vecs[:, mask]
        right_vecs = right_vecs[:, mask]

        # Sort by Re(λ) descending (least negative first)
        sort_idx = np.argsort(-np.real(exponents))
        exponents = exponents[sort_idx]
        left_vecs = left_vecs[:, sort_idx]
        right_vecs = right_vecs[:, sort_idx]

        # Normalize for bi-orthogonality: <v_i, u_i> = 1
        for i in range(left_vecs.shape[1]):
            inner = np.conj(right_vecs[:, i]) @ left_vecs[:, i]
            left_vecs[:, i] = left_vecs[:, i] / inner

        return exponents, left_vecs, right_vecs

    def compute_right_vector(
        self,
        limit_cycle: Trajectory,
        t_eval: np.ndarray,
        exponent: complex,
        u_init: StateVector,
        u_previous: Trajectory,
        v_previous: Trajectory,
        num_rotations: int = 30,
    ) -> Trajectory:
        """Compute right Floquet vector u_i(t) along the limit cycle.

        Solves: du_i/dt = [J(x_0(t)) - λ_i] u_i with orthogonalization against
        all previously computed Floquet vectors u_0, u_1, ..., u_{i-1}.

        The orthogonalization uses the bi-orthogonality relation:
        u_i = u_i - Σ_{j=0}^{i-1} <v_j, u_i> u_j

        Args:
            limit_cycle: Limit cycle trajectory.
            t_eval: Time points.
            exponent: Floquet exponent λ_i.
            u_init: Initial eigenvector from monodromy matrix.
            u_previous: All previously computed right Floquet vectors,
                shape (i, num_points, dim) where i is the current mode index.
            v_previous: All previously computed left Floquet vectors,
                shape (i, num_points, dim).
            num_rotations: Iterations for convergence.

        Returns:
            Right Floquet vector u_i(t), shape (num_points, dim).
        """
        num_points, dim = limit_cycle.shape
        period = t_eval[-1] + (t_eval[1] - t_eval[0])

        state_interp = PeriodicInterpolator(t_eval, limit_cycle, period)

        # Create interpolators for all previous vectors
        u_interps = [
            PeriodicInterpolator(t_eval, u_previous[j], period)
            for j in range(u_previous.shape[0])
        ]
        v_interps = [
            PeriodicInterpolator(t_eval, v_previous[j], period)
            for j in range(v_previous.shape[0])
        ]

        def rhs(t: float, u: StateVector) -> StateVector:
            return (self.jacobian(state_interp(t)) - exponent * np.eye(dim)) @ u

        # Normalize initial vector and iterate to convergence
        u = u_init / np.linalg.norm(u_init)

        for _ in range(num_rotations):
            # Remove components along all previous u vectors
            for j in range(len(u_interps)):
                proj = np.dot(v_interps[j](0), u)
                u = u - proj * u_interps[j](0)

            sol = self._integrate(rhs, u, (0, period))
            u = sol.y[:, -1]
            u = u / np.linalg.norm(u)

        # Compute u(t) at all points
        # Remove components along all previous u vectors at initial point
        for j in range(u_previous.shape[0]):
            proj = np.dot(v_previous[j, 0], u)
            u = u - proj * u_previous[j, 0]
        u = u / np.linalg.norm(u)

        sol = self._integrate(rhs, u, (0, period), t_eval=t_eval)

        u_vec = np.zeros((num_points, dim), dtype=np.complex128)
        for i in range(len(t_eval)):
            u_vec[i] = sol.y[:, i]

        return u_vec

    def compute_left_vector(
        self,
        limit_cycle: Trajectory,
        t_eval: np.ndarray,
        exponent: complex,
        v_init: StateVector,
        u_previous: Trajectory,
        v_previous: Trajectory,
        u_i: Trajectory,
        num_rotations: int = 30,
    ) -> Trajectory:
        """Compute left Floquet vector v_i(t) along the limit cycle.

        Solves: dv_i/dt = -[J^T(x_0(t)) - λ_i*] v_i backwards, with orthogonalization
        against all previously computed Floquet vectors.

        The orthogonalization uses the bi-orthogonality relation:
        v_i = v_i - Σ_{j=0}^{i-1} <v_i, u_j> v_j

        Args:
            limit_cycle: Limit cycle trajectory.
            t_eval: Time points.
            exponent: Floquet exponent λ_i.
            v_init: Initial eigenvector from monodromy matrix.
            u_previous: All previously computed right Floquet vectors,
                shape (i, num_points, dim) where i is the current mode index.
            v_previous: All previously computed left Floquet vectors,
                shape (i, num_points, dim).
            u_i: The corresponding right Floquet vector u_i(t) for this mode.
            num_rotations: Iterations for convergence.

        Returns:
            Left Floquet vector v_i(t), shape (num_points, dim).
        """
        num_points, dim = limit_cycle.shape
        period = t_eval[-1] + (t_eval[1] - t_eval[0])

        state_interp = PeriodicInterpolator(t_eval, limit_cycle, period)

        # Create interpolators for all previous vectors
        u_interps = [
            PeriodicInterpolator(t_eval, u_previous[j], period)
            for j in range(u_previous.shape[0])
        ]
        v_interps = [
            PeriodicInterpolator(t_eval, v_previous[j], period)
            for j in range(v_previous.shape[0])
        ]

        exponent_conj = np.conj(exponent)

        def adjoint_rhs(t: float, v: StateVector) -> StateVector:
            # Backward integration: t goes 0 -> period, actual time is period - t
            return self.jacobian(state_interp(period - t)).T @ v - exponent_conj * v

        # Normalize initial vector and iterate to convergence
        v = v_init / np.linalg.norm(v_init)

        for _ in range(num_rotations):
            # Normalize by <v, u_i(0)> = 1
            v = v / np.dot(v, np.conj(u_i[0]))

            sol = self._integrate(adjoint_rhs, v, (0, period))
            v = sol.y[:, -1]

            # Remove components along all previous v vectors
            for j in range(len(u_interps)):
                proj = np.dot(np.conj(u_interps[j](0)), v)
                v = v - proj * v_interps[j](0)

        # Normalize final
        v = v / np.dot(v, np.conj(u_i[0]))

        # Compute v(t) at all points (backward integration)
        sol = self._integrate(
            adjoint_rhs,
            v,
            (0, period),
            t_eval=np.linspace(0, period, num_points, endpoint=False),
        )

        v_vec = np.zeros((num_points, dim), dtype=np.complex128)
        for i in range(num_points):
            # Map backward time to forward time index
            v_vec[num_points - 1 - i] = sol.y[:, i]

        return v_vec

    def _solve_2d(
        self,
        limit_cycle: Trajectory,
        t_eval: np.ndarray,
        period: float,
        u0: Trajectory,
        v0: Trajectory,
        num_rotations: int,
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        """Specialized solver for 2D systems using direct iteration.

        For 2D systems, there's only one non-trivial Floquet mode, and we can
        compute it efficiently without the full monodromy matrix.

        Args:
            limit_cycle: Limit cycle trajectory, shape (num_points, 2).
            t_eval: Time points for the limit cycle.
            period: Oscillation period.
            omega: Natural angular frequency.
            u0: Zeroth right Floquet vector.
            v0: Zeroth left Floquet vector.
            num_rotations: Number of iterations for convergence.

        Returns:
            Tuple of (exponents, right_vectors, left_vectors).
        """
        num_points = limit_cycle.shape[0]
        dim = 2

        state_interp = PeriodicInterpolator(t_eval, limit_cycle, period)
        v0_interp = PeriodicInterpolator(t_eval, v0, period)
        u0_interp = PeriodicInterpolator(t_eval, u0, period)

        def u1_rhs(t: float, u: StateVector) -> StateVector:
            return self.jacobian(state_interp(t)) @ u

        # Compute u1 by forward integration
        u1 = np.ones(dim) / np.sqrt(dim)
        norm = 1.0

        for _ in range(num_rotations):
            # Remove u0 component
            proj = np.dot(v0_interp(0), u1)
            u1 = u1 - proj * u0_interp(0)

            sol = self._integrate(u1_rhs, u1, (0, period))
            u1 = sol.y[:, -1]
            norm = float(np.linalg.norm(u1))
            u1 = u1 / norm

        lambda1 = np.log(norm) / period

        # Compute u1(t) at all points
        # Remove u0 component and normalize
        proj = np.dot(v0[0], u1)
        u1 = u1 - proj * u0[0]
        u1 = u1 / np.linalg.norm(u1)

        def u1_rhs_shifted(t: float, u: StateVector) -> StateVector:
            return (self.jacobian(state_interp(t)) - lambda1 * np.eye(dim)) @ u

        sol = self._integrate(u1_rhs_shifted, u1, (0, period), t_eval=t_eval)
        u1_vec = sol.y.T

        # Compute v1 by backward integration
        def v1_adjoint_rhs(t: float, v: StateVector) -> StateVector:
            return self.jacobian(state_interp(period - t)).T @ v - lambda1 * v

        v1 = np.ones(dim) / np.sqrt(dim)

        for _ in range(num_rotations):
            # Normalize by <v1, u1(0)> = 1
            v1 = v1 / np.dot(v1, u1_vec[0])

            sol = self._integrate(v1_adjoint_rhs, v1, (0, period))
            v1 = sol.y[:, -1]

            # Remove v0 component
            proj = np.dot(u0[0], v1)
            v1 = v1 - proj * v0[0]

        # Normalize
        v1 = v1 / np.dot(v1, u1_vec[0])

        # Compute v1(t) at all points
        sol = self._integrate(
            v1_adjoint_rhs,
            v1,
            (0, period),
            t_eval=np.linspace(0, period, num_points, endpoint=False),
        )

        v1_vec = np.zeros((num_points, dim))
        # Map backward time to indices
        for i in range(num_points):
            v1_vec[num_points - 1 - i] = sol.y[:, i]

        # Assemble results
        exponents = np.array([0.0, lambda1])
        right_vectors = np.array([u0, u1_vec])
        left_vectors = np.array([v0, v1_vec])

        return exponents, right_vectors, left_vectors

    def solve(
        self,
        zero_phase_state: StateVector,
        period: float,
        num_points: int = 10000,
        num_floquet_modes: int = 1,
        num_rotations: int = 30,
    ) -> FloquetResult:
        """Perform complete Floquet analysis.

        Args:
            zero_phase_state: Initial state at zero phase.
            period: Oscillation period.
            num_points: Number of points to sample along limit cycle.
            num_floquet_modes: Number of Floquet modes to compute (excluding zeroth).
            num_rotations: Number of iterations for vector convergence.

        Returns:
            FloquetResult containing all computed quantities.
        """
        omega = 2 * np.pi / period
        dim = zero_phase_state.shape[0]

        # Compute limit cycle
        t_eval, limit_cycle = self.compute_limit_cycle(
            zero_phase_state, period, num_points
        )

        # Compute zeroth Floquet vectors
        u0 = self.compute_u0(limit_cycle, omega)
        v0, v0_diff = self.compute_v0(limit_cycle, t_eval, u0, omega, num_rotations)

        if dim == 2 and num_floquet_modes >= 1:
            # For 2D systems, use direct iteration method
            exponents, right_vectors, left_vectors = self._solve_2d(
                limit_cycle, t_eval, period, u0, v0, num_rotations
            )
        else:
            # For higher-dimensional systems, use monodromy matrix
            exponents, right_vectors, left_vectors = self._solve_nd(
                limit_cycle,
                t_eval,
                period,
                dim,
                u0,
                v0,
                num_floquet_modes,
                num_rotations,
            )

        return FloquetResult(
            limit_cycle=limit_cycle,
            t_eval=t_eval,
            period=period,
            omega=omega,
            exponents=exponents,
            right_vectors=right_vectors.real
            if np.allclose(right_vectors.imag, 0)
            else right_vectors,
            left_vectors=left_vectors.real
            if np.allclose(left_vectors.imag, 0)
            else left_vectors,
            v0_diff=v0_diff,
        )

    def _solve_nd(
        self,
        limit_cycle: Trajectory,
        t_eval: np.ndarray,
        period: float,
        dim: int,
        u0: Trajectory,
        v0: Trajectory,
        num_floquet_modes: int,
        num_rotations: int,
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        """Solve for Floquet modes in N-dimensional systems using monodromy matrix.

        Args:
            limit_cycle: Limit cycle trajectory.
            t_eval: Time evaluation points.
            period: Oscillation period.
            dim: System dimension.
            u0: Zeroth right Floquet vector.
            v0: Zeroth left Floquet vector.
            num_floquet_modes: Number of modes to compute.
            num_rotations: Iterations for convergence.

        Returns:
            Tuple of (exponents, right_vectors, left_vectors).
        """
        num_points = limit_cycle.shape[0]

        monodromy = self.compute_monodromy(limit_cycle, t_eval)
        exponents, left_init, right_init = self.compute_exponents_and_eigenvectors(
            monodromy, period
        )

        # Limit to requested number of modes
        num_modes = min(num_floquet_modes, len(exponents))

        right_vectors = np.zeros((num_modes + 1, num_points, dim), dtype=np.complex128)
        left_vectors = np.zeros((num_modes + 1, num_points, dim), dtype=np.complex128)

        right_vectors[0] = u0
        left_vectors[0] = v0

        for i in range(num_modes):
            # Pass all previously computed vectors (indices 0 to i) for orthogonalization
            # When computing u_{i+1} and v_{i+1}, we need u_0..u_i and v_0..v_i
            u_previous = right_vectors[: i + 1]  # shape: (i+1, num_points, dim)
            v_previous = left_vectors[: i + 1]  # shape: (i+1, num_points, dim)

            right_vectors[i + 1] = self.compute_right_vector(
                limit_cycle,
                t_eval,
                exponents[i],
                right_init[:, i],
                u_previous,
                v_previous,
                num_rotations,
            )
            left_vectors[i + 1] = self.compute_left_vector(
                limit_cycle,
                t_eval,
                exponents[i],
                left_init[:, i],
                u_previous,
                v_previous,
                right_vectors[i + 1],
                num_rotations,
            )

        exponents = np.concatenate([[0], exponents[:num_modes]])

        return exponents, right_vectors, left_vectors


def check_orthogonality(
    left_vectors: np.ndarray,
    right_vectors: np.ndarray,
    tolerance: float = 0.1,
) -> bool:
    """Check bi-orthogonality of Floquet vectors: <v_i, u_j> ≈ δ_ij.

    Args:
        left_vectors: Left Floquet vectors, shape (num_modes, num_points, dim).
        right_vectors: Right Floquet vectors, shape (num_modes, num_points, dim).
        tolerance: Tolerance for orthogonality check.

    Returns:
        True if vectors satisfy bi-orthogonality within tolerance.
    """
    num_modes = left_vectors.shape[0]
    num_points = left_vectors.shape[1]

    is_orthogonal = True

    for i in range(num_modes):
        for j in range(num_modes):
            # Compute average inner product along limit cycle
            inner_products = np.array(
                [
                    np.dot(np.conj(left_vectors[i, k]), right_vectors[j, k])
                    for k in range(num_points)
                ]
            )
            mean_inner = np.mean(np.abs(inner_products))
            expected = 1.0 if i == j else 0.0

            if np.abs(mean_inner - expected) > tolerance:
                print(
                    f"Orthogonality violation: <v_{i}, u_{j}> = {mean_inner:.4f}, "
                    f"expected {expected}"
                )
                is_orthogonal = False

    return is_orthogonal
