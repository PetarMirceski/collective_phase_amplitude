import numpy as np

from solvers.control_algorithms.optimal_entrainment_abstractor import (
    OptimalEntrainmentAbstractor,
)
from utils.math_utils import inner_product_of_vector_arrays


class OptimalEntrainmentSolver(OptimalEntrainmentAbstractor):
    def __init__(
        self,
        psf: np.ndarray,
        psf_diff: np.ndarray,
        power: float,
        delta: float,
        omega: float,
        input_freq: float,
        simulation_time: float,
        dt: float,
        phase_len: int,
    ) -> None:
        """
        Class that represents the Optimal Entrainment Solver
        Parameters:
            psf: Represents the PSF(Z) of an oscillatory system.
            psf_diff: Represents the dPSF/dt(dZ/dt) of an oscillatory system.
            power: The power constraint of the optimization problem [<q,q>]_t = power.
            delta: The phase constraint of the optimization problem delta + F(psi) = 0.
            omega: The natural frequency of the oscillatory system.
        Properties:
            nu: The nu lagrangian parameter of the lagrangian optimization problem.
            mu: The mu lagrangian parameter of the lagrangian optimization problem.
        """
        self.psf: np.ndarray = psf
        self.psf_diff: np.ndarray = psf_diff
        self._power = power
        self._delta = delta
        num_steps = int(simulation_time / dt)

        self._nu: float | None = None
        self._mu: float | None = None
        super().__init__(dt, phase_len, omega, input_freq, num_steps)

        self._z_inner_prod: float = np.mean(
            inner_product_of_vector_arrays(psf, psf), axis=0
        )
        self._z_diff_inner_prod: float = (
            np.mean(inner_product_of_vector_arrays(psf_diff, psf_diff), axis=0)
            / omega**2
        )

        # We only do these ones for static caching
        # self._z_inner_prod: float = z_mod  # Z'(theta) = 1/omega dZ/dt
        # This represents [|Z'|]_t => [<Z(theta)', Z(theta)'>]_t = [<Z(t)'/dt 1/w, Z(t)'/dt 1/w> ]_t
        # self._z_diff_inner_prod: float = z_diff_mod

    @property
    def nu(self) -> float:
        """
        Returns the nu lagrangian multiplier as:
        nu = (1/2) * sqrt([<Z',Z'>]_t / (P - (delta / [<Z,Z>]_t)))
        """
        inner_square_product: float = self._z_diff_inner_prod / (
            self._power - (self._delta**2 / self._z_inner_prod)
        )
        if inner_square_product < 0:
            raise Exception("Can't take square root of a negative number")
        nu: float = (1 / 2) * (inner_square_product) ** (1 / 2)
        return nu

    @property
    def mu(self) -> float:
        """
        Returns the nu lagrangian multiplier as:
        mu = -(2*nu*delta) / [<Z, Z>]_t
        """
        mu = (-2 * self.nu * self._delta) / self._z_inner_prod
        return mu

    def calculate_optimal_input(
        self,
        psf_diff_point: np.ndarray,
        psf_point: np.ndarray,
    ) -> np.ndarray:
        """
        TODO: write better documentation because this can be used in vectorized calculations
        NOTE: The inputs can be both [2, N] and [2, 1]
        Returns the optimal input as:
        (1/(2* nu)) * ( -Z' + mu * Z )
        """
        q: np.ndarray = (1 / (2 * self.nu)) * (
            -psf_diff_point / self.omega + self.mu * psf_point
        )
        return q

    def get_input_and_phase(
        self,
    ) -> tuple[np.ndarray, np.ndarray]:
        efp, idp, second_points = self.phase_and_interpolation_parameters()
        efp_indexes = efp.astype(int)
        v0 = np.multiply(idp[:, np.newaxis], self.psf[efp_indexes]) + np.multiply(
            second_points[:, np.newaxis],
            self.psf[(efp_indexes + 1) % self.phase_len],
        )
        v0diff = np.multiply(
            idp[:, np.newaxis], self.psf_diff[efp_indexes]
        ) + np.multiply(
            second_points[:, np.newaxis],
            self.psf_diff[(efp_indexes + 1) % self.phase_len],
        )
        optimal_input = self.calculate_optimal_input(v0diff, v0)
        return optimal_input, efp
