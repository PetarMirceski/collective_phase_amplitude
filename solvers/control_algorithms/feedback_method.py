from collections.abc import Callable

import numpy as np
from scipy.integrate import solve_ivp
from scipy.interpolate import interp1d
from tqdm import tqdm

from solvers.phase_solver import PhaseSolver


def resample_to_step(
    data: np.ndarray,
    original_dt: float,
    target_dt: float,
) -> tuple[np.ndarray, int]:
    """Resample data array to a coarser time step.

    Args:
        data: Array of shape (num_steps, dim) sampled at original_dt.
        original_dt: Original time step.
        target_dt: Target (coarser) time step.

    Returns:
        Tuple of (resampled_data, stride) where stride is the resampling factor.
    """
    stride = max(1, int(round(target_dt / original_dt)))
    return data[::stride], stride


class OptimalEntrainmentSolverFeedback:
    def __init__(
        self,
        oscillator: Callable[[np.ndarray], np.ndarray],
        perturbated_oscillator: Callable[[np.ndarray, np.ndarray], np.ndarray],
        phase_points: np.ndarray,
        psf: np.ndarray,
        simulation_time: float,
        dt: float,
        phase_len: int,
        alpha: float = 50,
        simulation_dt: float | None = None,
    ) -> None:
        """Initialize the feedback solver.

        Args:
            oscillator: Unperturbed oscillator dynamics.
            perturbated_oscillator: Perturbed oscillator dynamics F(x, u).
            phase_points: Limit cycle points, shape (phase_len, dim).
            psf: Phase sensitivity function, shape (phase_len, dim).
            simulation_time: Total simulation time.
            dt: Original time step (for phase_points resolution).
            phase_len: Number of points on the limit cycle.
            alpha: Feedback gain.
            simulation_dt: Time step for simulation (default: dt).
                Use a larger value (e.g., 0.1) for faster simulation.
        """
        self.oscillator = oscillator
        self.perturbated_oscillator = perturbated_oscillator
        self.simulation_time = simulation_time
        self.original_dt = dt
        self.dt = simulation_dt if simulation_dt is not None else dt
        self.num_steps = int(simulation_time / self.dt)
        self.phase_solver = PhaseSolver(oscillator, phase_len, dt)
        self.phase_len = phase_len
        self.alpha = alpha

        # Resample phase points and PSF if using coarser simulation step
        if simulation_dt is not None and simulation_dt > dt:
            self.phase_points, self._stride = resample_to_step(
                phase_points, dt, simulation_dt
            )
            self.psf, _ = resample_to_step(psf, dt, simulation_dt)
            self.resampled_phase_len = self.phase_points.shape[0]
        else:
            self.phase_points = phase_points
            self.psf = psf
            self._stride = 1
            self.resampled_phase_len = phase_len

    def _fast_phase_approximation(
        self,
        point: np.ndarray,
        x_phase: int,
        search_radius: int = 50,
    ) -> tuple[np.ndarray, int, bool]:
        """Fast phase approximation with limited search radius.

        Args:
            point: State to find phase for.
            x_phase: Current phase estimate (in resampled indices).
            search_radius: Number of phase points to search in each direction.

        Returns:
            Tuple of (deviation, phase, success_flag).
        """
        x0 = self.phase_points
        v0 = self.psf
        phase_len = self.resampled_phase_len

        # Clamp search radius to half the cycle
        search_radius = min(search_radius, phase_len // 2)

        # Check current phase first
        y = point - x0[x_phase]
        r = np.dot(y, v0[x_phase])

        # Determine search direction
        if r >= 0:
            ts = np.arange(1, search_radius + 1)
        else:
            ts = np.arange(-1, -search_radius - 1, -1)

        phase_candidates = (x_phase + ts) % phase_len
        y_candidates = point - x0[phase_candidates]
        r_candidates = np.sum(y_candidates * v0[phase_candidates], axis=1)

        # Find sign change
        if r >= 0:
            sign_change_idx = np.where(r_candidates < 0)[0]
        else:
            sign_change_idx = np.where(r_candidates >= 0)[0]

        if sign_change_idx.size == 0:
            return y, x_phase, False

        idx = sign_change_idx[0]
        phase_found = phase_candidates[idx]

        # Get the residual values at the sign change boundary
        # r_prev is the residual before the sign change, r_curr is after
        r_curr = r_candidates[idx]
        r_prev = r_candidates[idx - 1] if idx > 0 else r

        # Get the corresponding phase indices
        # phase_prev is where r_prev was computed, phase_found is where r_curr was computed
        phase_prev = phase_candidates[idx - 1] if idx > 0 else x_phase

        denom = r_prev - r_curr
        if abs(denom) < 1e-12:
            beta = 0.5
        else:
            beta = r_prev / denom

        beta = np.clip(beta, 0.0, 1.0)

        # Interpolate between the two phase points where sign change occurred
        x0_interp = (1 - beta) * x0[phase_prev] + beta * x0[phase_found]
        deviation = point - x0_interp

        return deviation, int(phase_found), True

    def apply_input(
        self,
        optimal_input: np.ndarray,
        state: np.ndarray,
        x_phase: int,
    ) -> tuple[np.ndarray, np.ndarray]:
        """Apply optimal input with Floquet-based feedback correction.

        Args:
            optimal_input: Pre-computed optimal input signal, shape (num_steps, dim).
            state: Initial state of the oscillator.
            x_phase: Initial phase index on the limit cycle (original resolution).

        Returns:
            Tuple of (trajectory, applied_input) arrays.
        """
        # Resample input to simulation step
        if self._stride > 1:
            optimal_input_resampled, _ = resample_to_step(
                optimal_input, self.original_dt, self.dt
            )
            x_phase_resampled = x_phase // self._stride
        else:
            optimal_input_resampled = optimal_input
            x_phase_resampled = x_phase

        t_final = self.num_steps * self.dt
        t_eval = np.arange(1, self.num_steps + 1) * self.dt
        t_input = np.arange(self.num_steps) * self.dt

        input_interp = interp1d(
            t_input,
            optimal_input_resampled,
            axis=0,
            kind="previous",
            bounds_error=False,
            fill_value=(
                optimal_input_resampled[0],
                optimal_input_resampled[-1],
            ),  # type:ignore[arg-type]
        )

        input_saving_array = np.zeros((self.num_steps, state.shape[0]))
        current_phase = [x_phase_resampled]
        last_step_idx = [-1]

        def rhs(t: float, y: np.ndarray) -> np.ndarray:
            base_input = input_interp(t)

            deviation, new_phase, flag = self._fast_phase_approximation(
                y, current_phase[0], search_radius=30
            )

            if not flag:
                deviation, new_phase = (
                    self.phase_solver.phase_approximation_via_floquet_improved_(
                        y, self.phase_points, self.psf, current_phase[0]
                    )
                )
                new_phase = int(new_phase)

            current_phase[0] = new_phase
            feedback_input = base_input - self.alpha * deviation

            step_idx = min(int(t / self.dt), self.num_steps - 1)
            if step_idx != last_step_idx[0]:
                input_saving_array[step_idx] = feedback_input
                last_step_idx[0] = step_idx

            return self.perturbated_oscillator(y, feedback_input)

        sol = solve_ivp(
            rhs,
            (0, t_final),
            state,
            method="RK45",
            t_eval=t_eval,
            rtol=1e-6,
            atol=1e-9,
        )

        return sol.y.T, input_saving_array

    def apply_input_fixed_step(
        self,
        optimal_input: np.ndarray,
        state: np.ndarray,
        x_phase: int,
    ) -> tuple[np.ndarray, np.ndarray]:
        """Apply optimal input with fixed-step RK4 integration.

        Args:
            optimal_input: Pre-computed optimal input signal, shape (num_steps, dim).
            state: Initial state of the oscillator.
            x_phase: Initial phase index on the limit cycle (original resolution).

        Returns:
            Tuple of (trajectory, applied_input) arrays.
        """
        # Create interpolator for optimal input based on original time grid
        t_input_original = np.arange(optimal_input.shape[0]) * self.original_dt
        input_interp = interp1d(
            t_input_original,
            optimal_input,
            axis=0,
            kind="linear",
            bounds_error=False,
            fill_value=(optimal_input[0], optimal_input[-1]),  # pyright: ignore
        )

        # Resample phase points for feedback calculation
        if self._stride > 1:
            x_phase_resampled = x_phase // self._stride
        else:
            x_phase_resampled = x_phase

        num_steps = self.num_steps
        trajectory = np.zeros((num_steps, state.shape[0]))
        input_saving_array = np.zeros((num_steps, state.shape[0]))

        current_state = state.copy()
        current_phase = x_phase_resampled

        for i in tqdm(range(num_steps), desc="Feedback simulation"):
            # Get optimal input at current time via interpolation
            current_time = i * self.dt
            base_input = input_interp(current_time)

            deviation, new_phase, flag = self._fast_phase_approximation(
                current_state, current_phase, search_radius=30
            )

            if not flag:
                deviation, new_phase = (
                    self.phase_solver.phase_approximation_via_floquet_improved_(
                        current_state, self.phase_points, self.psf, current_phase
                    )
                )
                new_phase = int(new_phase)

            current_phase = new_phase
            feedback_input = base_input - self.alpha * deviation

            # Check for divergence
            if not np.all(np.isfinite(current_state)):
                raise RuntimeError(
                    f"Simulation diverged at step {i}. State contains NaN/inf values."
                )

            # RK4 step
            k1 = self.perturbated_oscillator(current_state, feedback_input)
            k2 = self.perturbated_oscillator(
                current_state + 0.5 * self.dt * k1, feedback_input
            )
            k3 = self.perturbated_oscillator(
                current_state + 0.5 * self.dt * k2, feedback_input
            )
            k4 = self.perturbated_oscillator(
                current_state + self.dt * k3, feedback_input
            )
            current_state = current_state + (self.dt / 6.0) * (
                k1 + 2 * k2 + 2 * k3 + k4
            )

            trajectory[i] = current_state
            input_saving_array[i] = feedback_input

        return trajectory, input_saving_array

    def apply_input_slow(
        self,
        optimal_input: np.ndarray,
        state: np.ndarray,
    ) -> tuple[np.ndarray, np.ndarray]:
        """Apply optimal input with slow phase relaxation feedback.

        Args:
            optimal_input: Pre-computed optimal input signal, shape (num_steps, dim).
            state: Initial state of the oscillator.

        Returns:
            Tuple of (trajectory, applied_input) arrays.
        """
        t_final = self.num_steps * self.dt
        t_eval = np.arange(1, self.num_steps + 1) * self.dt
        t_input = np.arange(self.num_steps) * self.dt

        input_interp = interp1d(
            t_input,
            optimal_input,
            axis=0,
            kind="previous",
            bounds_error=False,
            fill_value=(optimal_input[0], optimal_input[-1]),  # type:ignore[arg-type]
        )

        input_saving_array = np.zeros((self.num_steps, state.shape[0]))

        def rhs(t: float, y: np.ndarray) -> np.ndarray:
            base_input = input_interp(t)

            relaxed_phase = self.phase_solver.relax_state(np.array([y]), 5)
            phase = self.phase_solver.calculate_phase(relaxed_phase, self.phase_points)
            deviation = y - self.phase_points[int(phase.item())]

            feedback_input = base_input - self.alpha * deviation

            step_idx = min(int(t / self.dt), self.num_steps - 1)
            input_saving_array[step_idx] = feedback_input

            return self.perturbated_oscillator(y, feedback_input)

        sol = solve_ivp(
            rhs,
            (0, t_final),
            state,
            method="RK45",
            t_eval=t_eval,
            rtol=1e-6,
            atol=1e-9,
        )

        return sol.y.T, input_saving_array

    def apply_input_pid(
        self,
        optimal_input: np.ndarray,
        state: np.ndarray,
        kp: float | None = None,
        ki: float = 0.5,
    ) -> tuple[np.ndarray, np.ndarray]:
        """Apply optimal input with PI feedback control.

        Args:
            optimal_input: Pre-computed optimal input signal, shape (num_steps, dim).
            state: Initial state of the oscillator.
            kp: Proportional gain (defaults to self.alpha).
            ki: Integral gain (default 0.5).

        Returns:
            Tuple of (trajectory, applied_input) arrays.
        """
        if kp is None:
            kp = self.alpha

        phase_saving_array = np.zeros((self.num_steps, state.shape[0]))
        input_saving_array = np.zeros((self.num_steps, state.shape[0]))

        integral_sum = np.zeros_like(state)
        current_state = state.copy()
        current_phase = 0

        for i in tqdm(range(self.num_steps)):
            base_input = optimal_input[i]

            deviation, new_phase, flag = self._fast_phase_approximation(
                current_state, current_phase, search_radius=30
            )

            if not flag:
                relaxed_phase = self.phase_solver.relax_state(
                    np.array([current_state]), 5
                )
                phase = self.phase_solver.calculate_phase(
                    relaxed_phase, self.phase_points
                )
                new_phase = int(phase.item())
                deviation = current_state - self.phase_points[new_phase]

            current_phase = new_phase

            integral_sum += deviation * self.dt
            feedback_input = base_input - (kp * deviation + ki * integral_sum)

            k1 = self.perturbated_oscillator(current_state, feedback_input)
            k2 = self.perturbated_oscillator(
                current_state + 0.5 * self.dt * k1, feedback_input
            )
            k3 = self.perturbated_oscillator(
                current_state + 0.5 * self.dt * k2, feedback_input
            )
            k4 = self.perturbated_oscillator(
                current_state + self.dt * k3, feedback_input
            )
            current_state = current_state + (self.dt / 6.0) * (
                k1 + 2 * k2 + 2 * k3 + k4
            )

            phase_saving_array[i] = current_state
            input_saving_array[i] = feedback_input

        return phase_saving_array, input_saving_array

    def phase_diff_control(
        self,
        target_phase_diff: float,
        state: np.ndarray,
        kp: float | None = None,
        ki: float = 0.5,
    ) -> tuple[np.ndarray, np.ndarray]:
        """Apply PI control to maintain a target phase difference.

        Args:
            target_phase_diff: Target phase difference to maintain (in phase units).
            state: Initial state of the oscillator.
            kp: Proportional gain (defaults to self.alpha).
            ki: Integral gain (default 0.5).

        Returns:
            Tuple of (trajectory, applied_input) arrays.
        """
        if kp is None:
            kp = self.alpha

        phase_saving_array = np.zeros((self.num_steps, state.shape[0]))
        input_saving_array = np.zeros((self.num_steps, state.shape[0]))

        integral_sum = np.zeros_like(state)
        current_state = state.copy()
        current_phase = 0

        for i in tqdm(range(self.num_steps)):
            deviation, new_phase, flag = self._fast_phase_approximation(
                current_state, current_phase, search_radius=30
            )

            if not flag:
                relaxed_phase = self.phase_solver.relax_state(
                    np.array([current_state]), 5
                )
                phase_sol = self.phase_solver.calculate_phase(
                    relaxed_phase, self.phase_points
                )
                new_phase = int(phase_sol.item())

            current_phase = new_phase

            target_phase = (
                int(current_phase + target_phase_diff) % self.resampled_phase_len
            )
            state_error = self.phase_points[target_phase] - current_state

            integral_sum += state_error * self.dt
            input_sig = kp * state_error + ki * integral_sum

            k1 = self.perturbated_oscillator(current_state, input_sig)
            k2 = self.perturbated_oscillator(
                current_state + 0.5 * self.dt * k1, input_sig
            )
            k3 = self.perturbated_oscillator(
                current_state + 0.5 * self.dt * k2, input_sig
            )
            k4 = self.perturbated_oscillator(current_state + self.dt * k3, input_sig)
            current_state = current_state + (self.dt / 6.0) * (
                k1 + 2 * k2 + 2 * k3 + k4
            )

            phase_saving_array[i] = current_state
            input_saving_array[i] = input_sig

        return phase_saving_array, input_saving_array
