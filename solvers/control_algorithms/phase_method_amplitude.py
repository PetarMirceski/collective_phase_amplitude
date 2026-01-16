import numpy as np
from scipy import optimize

from solvers.control_algorithms.optimal_entrainment_abstractor import (
    OptimalEntrainmentAbstractor,
)


class OptimalEntrainmentSolverPenalty(OptimalEntrainmentAbstractor):
    """Optimal entrainment solver with amplitude penalty.

    Computes optimal input that minimizes amplitude deviation while achieving
    phase-locking. The optimal input takes the form:

        q(θ) = (1/2) * (νE + k * I(θ) @ I(θ)*)^{-1} @ (-Z'(θ)/ω + μ*Z(θ))

    where Z is the PSF, I is the ISF, and ν, μ are Lagrange multipliers.

    Attributes:
        psf: Phase sensitivity function, shape (phase_len, dim).
        psf_diff: Derivative of PSF, shape (phase_len, dim).
        isf: Isostable sensitivity function, shape (phase_len, dim).
        omega: Natural frequency of the oscillator.
        delta: Frequency detuning parameter.
        power: Power constraint for the optimization.
        optimization_weight: Weight (k) for amplitude penalty term.
    """

    def __init__(
        self,
        psf: np.ndarray,
        psf_diff: np.ndarray,
        isf: np.ndarray,
        power: float,
        delta: float,
        omega: float,
        input_freq: float,
        optimization_weight: float,
        simulation_time: float,
        dt: float,
        phase_len: int,
    ) -> None:
        num_steps = int(simulation_time / dt)
        super().__init__(dt, phase_len, omega, input_freq, num_steps)

        self.psf = psf
        self.psf_diff = psf_diff
        self.isf = isf
        self.omega = omega
        self.delta = delta
        self.power = power
        self.optimization_weight = optimization_weight
        self.phase_len = phase_len

        # Pre-compute inverse matrices for all phase points
        self._inverses = self._precompute_inverses()

        # Solve for nu using Brent's method
        self._nu = optimize.brentq(self._power_residual, 1e-10, 1e3)
        self._mu = self._compute_mu(self._nu)  # pyright:ignore

    @property
    def nu(self) -> float:
        return self._nu  # pyright:ignore

    @property
    def mu(self) -> float:
        return self._mu

    def _precompute_inverses(self) -> np.ndarray:
        """Pre-compute (νE + k*I@I*)^{-1} structure for vectorized operations.

        Returns:
            Array of shape (phase_len, dim, dim) containing I@I* for each phase.
        """
        # isf shape: (phase_len, dim)
        # We need I @ I* for each phase point
        isf_outer = np.einsum("ij,ik->ijk", self.isf, self.isf.conj()).real
        return isf_outer

    def _compute_inverse(self, nu: float, phase_idx: int) -> np.ndarray:
        """Compute (νE + k*I@I*)^{-1} for a specific phase point."""
        dim = self.isf.shape[1]
        matrix = nu * np.eye(dim) + self.optimization_weight * self._inverses[phase_idx]
        return np.linalg.inv(matrix)

    def _compute_mu(self, nu: float) -> float:
        """Compute the μ Lagrange multiplier for a given ν.

        μ = ([Z @ inv @ Z'/ω]_θ - 2δ) / [Z @ inv @ Z]_θ
        """
        numerator = 0.0
        denominator = 0.0

        for i in range(self.phase_len):
            inv = self._compute_inverse(nu, i)
            psf = self.psf[i]
            psf_diff = self.psf_diff[i]

            # Z @ inv @ Z'/ω
            numerator += psf @ inv @ (psf_diff / self.omega)
            # Z @ inv @ Z
            denominator += psf @ inv @ psf

        numerator = numerator / self.phase_len - 2 * self.delta
        denominator = denominator / self.phase_len

        return numerator / denominator

    def _power_residual(self, nu: float) -> float:
        """Compute residual: [|q|²]_θ - P for root finding."""
        mu = self._compute_mu(nu)
        q_squared_sum = 0.0

        for i in range(self.phase_len):
            q = self._compute_input_at_phase(nu, mu, i)
            q_squared_sum += np.dot(q, q)

        return q_squared_sum / self.phase_len - self.power

    def _compute_input_at_phase(
        self, nu: float, mu: float, phase_idx: int
    ) -> np.ndarray:
        """Compute optimal input at a specific phase point."""
        inv = self._compute_inverse(nu, phase_idx)
        psf = self.psf[phase_idx]
        psf_diff = self.psf_diff[phase_idx]
        return 0.5 * inv @ (-psf_diff / self.omega + mu * psf)

    def calculate_optimal_input(
        self, psf: np.ndarray, psf_diff: np.ndarray, isf: np.ndarray
    ) -> np.ndarray:
        """Calculate optimal input for given PSF, PSF', and ISF values.

        Args:
            psf: Phase sensitivity at current phase.
            psf_diff: PSF derivative at current phase.
            isf: Isostable sensitivity at current phase.

        Returns:
            Optimal input vector.
        """
        dim = isf.shape[0]
        isf = isf.reshape(-1)
        isf_outer = np.outer(isf, isf.conj()).real
        inv = np.linalg.inv(
            self.nu * np.eye(dim) + self.optimization_weight * isf_outer
        )
        return 0.5 * inv @ (-psf_diff / self.omega + self.mu * psf)

    def get_input_and_phase(self) -> tuple[np.ndarray, np.ndarray]:
        """Compute optimal input signal and external force phase.

        Returns:
            Tuple of (optimal_input, external_force_phase) where:
                - optimal_input has shape (num_steps, dim)
                - external_force_phase has shape (num_steps,)
        """
        efp, idp, second_points = self.phase_and_interpolation_parameters()
        efp_indexes = efp.astype(int)
        next_indexes = (efp_indexes + 1) % self.phase_len

        # Interpolate PSF, PSF', and ISF
        psf = (
            idp[:, np.newaxis] * self.psf[efp_indexes]
            + second_points[:, np.newaxis] * self.psf[next_indexes]
        )
        psf_diff = (
            idp[:, np.newaxis] * self.psf_diff[efp_indexes]
            + second_points[:, np.newaxis] * self.psf_diff[next_indexes]
        )
        isf = (
            idp[:, np.newaxis] * self.isf[efp_indexes]
            + second_points[:, np.newaxis] * self.isf[next_indexes]
        )

        # Compute optimal input for each time step
        input_vec = np.zeros_like(psf)
        for i in range(psf.shape[0]):
            input_vec[i] = self.calculate_optimal_input(psf[i], psf_diff[i], isf[i])

        return input_vec, efp
