from abc import ABC, abstractmethod
from collections.abc import Callable

import numpy as np
from scipy.integrate import solve_ivp
from scipy.interpolate import interp1d


class OptimalEntrainmentAbstractor(ABC):
    def __init__(
        self,
        dt: float,
        phase_len: int,
        omega: float,
        external_freq: float,
        num_steps: int,
    ):
        self.dt = dt
        self.phase_len = phase_len
        self.omega = omega
        self.inp_freq = external_freq
        self.num_steps = num_steps

    def phase_and_interpolation_parameters(
        self,
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        indexes = np.arange(0, self.num_steps, 1)
        efp = self.inp_freq * indexes / self.omega
        efp = efp - (efp / self.phase_len).astype(int) * self.phase_len
        idp = efp - efp.astype(int)  # internally dividing point
        second_points = 1 - idp
        return efp, idp, second_points

    def apply_input(
        self,
        perturbated_oscillator: Callable[[np.ndarray, np.ndarray], np.ndarray],
        point: np.ndarray,
        input_func: np.ndarray,
    ) -> np.ndarray:
        """Apply the optimal input to the oscillator using solve_ivp.

        Args:
            perturbated_oscillator: Function F(state, perturbation) -> dstate/dt.
            point: Initial state of the oscillator.
            input_func: Pre-computed optimal input signal, shape (num_steps, dim).

        Returns:
            Trajectory of the oscillator under the optimal input, shape (num_steps, dim).
        """
        t_final = self.num_steps * self.dt
        t_eval = np.arange(1, self.num_steps + 1) * self.dt
        t_input = np.arange(self.num_steps) * self.dt

        input_interp = interp1d(
            t_input,
            input_func,
            axis=0,
            kind="previous",  # Use previous value (step function) like original
            bounds_error=False,
            fill_value=(input_func[0], input_func[-1]),  # type:ignore[arg-type],
        )

        def rhs(t: float, y: np.ndarray) -> np.ndarray:
            perturbation = input_interp(t)
            return perturbated_oscillator(y, perturbation)

        sol = solve_ivp(
            rhs,
            (0, t_final),
            point,
            method="RK45",
            t_eval=t_eval,
            rtol=1e-6,
            atol=1e-9,
        )

        return sol.y.T

    @property
    @abstractmethod
    def nu(self) -> float:
        pass

    @property
    @abstractmethod
    def mu(self) -> float:
        pass
