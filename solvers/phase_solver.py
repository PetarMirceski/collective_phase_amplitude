from collections.abc import Callable

import numpy as np
import numpy.typing as npt
from joblib import Parallel, delayed
from scipy.integrate import solve_ivp
from scipy.interpolate import InterpolatedUnivariateSpline, interp1d
from tqdm import tqdm


def _relax_single_point(
    point: np.ndarray,
    system: Callable[[np.ndarray], np.ndarray],
    period: float,
    rotations: int,
) -> np.ndarray:
    """Relax a single point back to the limit cycle by integrating for multiple periods.

    Args:
        point: Initial state to relax.
        system: Vector field function F(x) -> dx/dt.
        period: Period of one oscillation.
        rotations: Number of periods to integrate.

    Returns:
        Final state after relaxation.
    """
    t_final = period * rotations

    def rhs(t: float, y: np.ndarray) -> np.ndarray:
        return system(y)

    sol = solve_ivp(
        rhs,
        (0, t_final),
        point,
        method="RK45",
        rtol=1e-7,
        atol=1e-10,
    )
    return sol.y[:, -1]


class PhaseSolver:
    def __init__(
        self,
        unperturbated_oscillator: Callable[[np.ndarray], np.ndarray],
        phase_len: int,
        step: float,
    ):
        self.unperturbated_oscillator = unperturbated_oscillator
        self.phase_len = phase_len
        self.step = step
        self._period = phase_len * step

    def relax_state(self, limit_cycle: np.ndarray, rotations: int = 7) -> np.ndarray:
        """Relax states back to the limit cycle by integrating for multiple periods.

        Args:
            limit_cycle: Array of states to relax, shape (num_points, dim).
            rotations: Number of periods to integrate for relaxation.

        Returns:
            Array of relaxed states, shape (num_points, dim).
        """
        relaxed_points = Parallel(n_jobs=4)(
            delayed(_relax_single_point)(
                limit_cycle[i],
                self.unperturbated_oscillator,
                self._period,
                rotations,
            )
            for i in tqdm(range(limit_cycle.shape[0]), leave=False)
        )
        relaxed_points_array: np.ndarray = np.array(relaxed_points)
        return relaxed_points_array

    def calculate_phase(
        self, relaxed_state: np.ndarray, phase_points: np.ndarray
    ) -> np.ndarray:
        """Calculate phase by finding closest point on the limit cycle.

        Args:
            relaxed_state: Array of relaxed states, shape (num_points, dim).
            phase_points: Reference limit cycle points, shape (phase_len, dim).

        Returns:
            Array of phase indices for each state.
        """
        phase = np.zeros(relaxed_state.shape[0])
        for i, point in tqdm(
            enumerate(relaxed_state), total=relaxed_state.shape[0], leave=False
        ):
            norm = np.linalg.norm(phase_points - point, axis=1)
            closest_point_index = np.argmin(norm)
            phase[i] = closest_point_index
        return phase

    def phase_origin_finder(
        self, point: npt.NDArray[np.float64], zero_phase_point: npt.NDArray[np.float64]
    ) -> tuple[int, np.ndarray]:
        """Find the phase of a point by integrating until it reaches zero phase.

        Args:
            point: State to find phase for.
            zero_phase_point: Reference zero-phase state on the limit cycle.

        Returns:
            Tuple of (phase, trajectory_array).
        """
        threshold = 4e-3
        max_iterations = self.phase_len * 15
        t_final = max_iterations * self.step

        def rhs(t: float, y: np.ndarray) -> np.ndarray:
            return self.unperturbated_oscillator(y)

        # Event to detect when we're close to zero phase point
        def reached_zero_phase(t: float, y: np.ndarray) -> float:
            return float(np.linalg.norm(y - zero_phase_point) - threshold)

        reached_zero_phase.terminal = True  # type: ignore
        reached_zero_phase.direction = -1  # type: ignore

        sol = solve_ivp(
            rhs,
            (0, t_final),
            point,
            method="RK45",
            events=reached_zero_phase,
            dense_output=True,
            rtol=1e-7,
            atol=1e-10,
        )

        if sol.t_events[0].size == 0:
            raise Exception("Failed to reach the zero phase state after max iterations")

        # Calculate iterations from event time
        event_time = sol.t_events[0][0]
        iterations = int(event_time / self.step)

        # Build debugging array from the trajectory
        t_debug = np.linspace(0, event_time, min(iterations + 1, max_iterations + 1))
        debugging_array = sol.sol(t_debug).T

        return self.phase_len - iterations, debugging_array

    def calculate_phase_from_zero_phase_point(
        self, states: np.ndarray, zero_phase_point: np.ndarray
    ) -> np.ndarray:
        """Calculate phase for multiple states relative to a zero-phase point.

        Args:
            states: Array of states, shape (num_points, dim).
            zero_phase_point: Reference zero-phase state.

        Returns:
            Array of phase values for each state.
        """
        iterations = np.zeros((states.shape[0]), dtype=np.int64)
        for idx in range(states.shape[0]):
            c_point = states[idx]
            itter, _ = self.phase_origin_finder(c_point, zero_phase_point)
            iterations[idx] = itter

        return iterations

    def minus_pi_pi_range_transform(self, phase: np.ndarray) -> np.ndarray:
        """Transform phase to [-pi, pi] range.

        Args:
            phase: Phase values in index units.

        Returns:
            Phase values in radians, range [0, 2*pi].
        """
        theta = (phase / self.phase_len) * 2 * np.pi + 2 * np.pi
        scaled_theta: np.ndarray = np.where(theta > 2 * np.pi, theta - 2 * np.pi, theta)
        return scaled_theta

    @staticmethod
    def calculate_phase_coupling_function_fast(
        delta: float,
        psf: np.ndarray,
        optimal_input: np.ndarray,
        dt: float,
        ext_period: float,
    ) -> np.ndarray:
        """Calculate phase coupling function using FFT-based cyclic correlation."""
        n = psf.shape[0]
        opt_inp = optimal_input[:n]

        # Cyclic correlation via FFT along axis 0, then sum over dimensions
        fft_psf = np.fft.fft(psf, axis=0)
        fft_inp = np.fft.fft(opt_inp, axis=0)
        corr = np.fft.ifft(fft_psf * np.conj(fft_inp), axis=0).real
        gamma = corr.sum(axis=1) * dt / ext_period

        gamma_pi_pi = delta + np.roll(gamma, -n // 2)
        phi = np.linspace(-np.pi, np.pi, n)
        return np.vstack((phi, gamma_pi_pi)).T

    def calculate_phase_coupling_function(
        self,
        delta: float,
        psf: np.ndarray,
        optimal_input: np.ndarray,
        divisions: int = 400,
    ) -> np.ndarray:
        """Calculate phase coupling function with explicit loop.

        Args:
            delta: Frequency detuning parameter.
            psf: Phase sensitivity function, shape (phase_len, dim).
            optimal_input: Optimal input signal, shape (num_steps, dim).
            divisions: Number of phase divisions to compute.

        Returns:
            Phase coupling function as array of (phi, gamma) pairs.
        """
        gamma_func = np.empty((divisions + 1, 2))
        for tp in tqdm(range(divisions + 1), leave=True):
            pnum = tp * int(self.phase_len / divisions) - int(self.phase_len / 2)
            tt = np.arange(self.phase_len)
            indexes = (pnum + tt) % self.phase_len
            gamma = np.mean(psf[indexes] * optimal_input[tt])

            gamma_func[tp, 0] = pnum / self.phase_len * 2 * np.pi
            gamma_func[tp, 1] = delta + gamma
        return gamma_func

    def linear_interpolation(
        self,
        x: npt.NDArray[np.float64],
        v0a: npt.NDArray[np.float64],
        v0b: npt.NDArray[np.float64],
        X0a: npt.NDArray[np.float64],
        X0b: npt.NDArray[np.float64],
    ) -> tuple[float, bool]:
        """Linear interpolation for phase approximation.

        Solves the equation: a*beta^2 + b*beta + c = 0

        Args:
            x: Point to find phase for.
            v0a, v0b: Left Floquet vectors at adjacent points.
            X0a, X0b: Limit cycle points at adjacent phases.

        Returns:
            Tuple of (beta interpolation parameter, success flag).
        """
        a = np.dot((X0b - X0a), (v0a - v0b))
        b = np.dot(x, (v0a - v0b)) + np.dot(v0b, (X0b - X0a)) - np.dot(X0b, (v0a - v0b))
        c = np.dot((x - X0b), v0b)
        discriminant = b * b - 4 * a * c
        if discriminant >= 0:
            return float((-b - np.sqrt(discriminant)) / (2 * a)), True
        return 0.0, False

    def phase_approximation_via_floquet(
        self, point: np.ndarray, x0: np.ndarray, v0: np.ndarray, x_phase: int
    ) -> tuple[np.ndarray, int, bool]:
        """Approximate phase using Floquet vectors.

        Args:
            point: State to find phase for.
            x0: Limit cycle trajectory.
            v0: Left Floquet vector (phase sensitivity function).
            x_phase: Initial phase guess.

        Returns:
            Tuple of (deviation from limit cycle, phase, success flag).
        """

        def smaller(r_: np.ndarray) -> np.ndarray:
            return r_ < 0

        def bigger(r_: np.ndarray) -> np.ndarray:
            return r_ >= 0

        interval = self.phase_len // 2
        ts = np.arange(interval)
        y: npt.NDArray[np.float64] = point - x0[x_phase]
        r_ = np.dot(y, v0[x_phase])
        thresh_func = smaller

        if r_ < 0:
            ts = -ts
            thresh_func = bigger

        phase_temp = (x_phase + ts) % self.phase_len
        y = point - x0[phase_temp]
        r_ = np.sum(y * v0[phase_temp], axis=1)
        smaller_idxs, *_ = np.where(thresh_func(r_))
        if smaller_idxs.shape[0] == 0:
            return y, -1, False

        phase_temp = phase_temp[smaller_idxs[0]]
        beta, flag = self.linear_interpolation(
            point,
            v0[phase_temp],
            v0[(phase_temp - 1) % self.phase_len],
            x0[phase_temp],
            x0[(phase_temp - 1) % self.phase_len],
        )
        if not flag:
            return y, -1, False

        x0_interpolation = (1 - beta) * x0[
            (phase_temp - 1) % self.phase_len
        ] + beta * x0[phase_temp]
        y = point - x0_interpolation

        return y, int(phase_temp), True

    def distance(self, list_of_points: np.ndarray, target: float) -> float:
        """Find the point closest to target using circular distance.

        Args:
            list_of_points: Array of candidate points.
            target: Target value to find closest point to.

        Returns:
            The point closest to target.
        """
        options = {}
        for point in list_of_points:
            dist = np.sqrt(2 - 2 * np.cos(np.deg2rad(target - point)))
            if dist not in options:
                options[dist] = point

        return float(options[min(options)])

    def phase_approximation_via_floquet_improved_(
        self, point: np.ndarray, x0: np.ndarray, v0: np.ndarray, curr_phase: int
    ) -> tuple[np.ndarray, float]:
        """Improved phase approximation using Floquet vectors with spline interpolation.

        Finds theta which minimizes <Z(theta), [X - X0(theta)]>.

        Args:
            point: State to find phase for.
            x0: Limit cycle trajectory.
            v0: Left Floquet vector (phase sensitivity function).
            curr_phase: Current phase estimate.

        Returns:
            Tuple of (deviation from limit cycle, phase).
        """
        y = point - x0
        prod = np.sum(y * v0, axis=1)
        product_interpolation = InterpolatedUnivariateSpline(
            np.arange(prod.shape[0]), prod
        )
        roots_array = product_interpolation.roots()

        if len(roots_array) == 0:
            # No roots found - state likely diverged, use current phase as fallback
            min_distance_theta = float(curr_phase % x0.shape[0])
        else:
            min_distance_theta = self.distance(roots_array, curr_phase) % x0.shape[0]  # pyright:ignore

        f = interp1d(np.arange(x0.shape[0]), x0, axis=0)
        x0_actual = f(min_distance_theta)

        # Deviation from limit cycle
        y = point - x0_actual

        return y, min_distance_theta
