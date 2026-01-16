from collections.abc import Callable

import numpy as np
from scipy.integrate import solve_ivp


def create_crossing_event(
    threshold_value: float = 0.0,
    state_index: int = 1,
    direction: int = -1,
) -> Callable[[float, np.ndarray], float]:
    """Creates an event function for solve_ivp to detect threshold crossings.

    Args:
        threshold_value: The threshold value for the crossing detection.
        state_index: The index of the state variable to monitor.
        direction: -1 for crossing from above, +1 for crossing from below,
            0 for both directions.

    Returns:
        An event function compatible with solve_ivp.
    """

    def event(t: float, y: np.ndarray) -> float:
        return y[state_index] - threshold_value

    event.terminal = True  # pyright: ignore
    event.direction = direction  # pyright: ignore
    return event


def find_period(
    system: Callable[[float, np.ndarray], np.ndarray],
    initial_state: np.ndarray,
    threshold_value: float = 0.0,
    state_index: int = 1,
    direction: int = -1,
    max_time: float = 1e4,
    rtol: float = 1e-9,
    atol: float = 1e-12,
) -> tuple[float, np.ndarray]:
    """Finds the natural period of a limit cycle oscillator using solve_ivp events.

    The function integrates the system until it detects two consecutive threshold
    crossings (in the specified direction), returning the time between them as
    the period.

    Args:
        system: The ODE system function with signature (t, y) -> dy/dt.
        initial_state: Initial state vector on or near the limit cycle.
        threshold_value: The threshold value for crossing detection.
        state_index: Index of the state variable to monitor for crossings.
        direction: Crossing direction: -1 (from above), +1 (from below), 0 (both).
        max_time: Maximum integration time before raising an error.
        rtol: Relative tolerance for the integrator.
        atol: Absolute tolerance for the integrator.

    Returns:
        A tuple of (period, zero_phase_state) where period is the oscillation
        period and zero_phase_state is the state at the threshold crossing.

    Raises:
        RuntimeError: If no crossing is detected within max_time.
    """
    event = create_crossing_event(threshold_value, state_index, direction)

    # First integration: find the first crossing to establish the zero-phase point
    sol1 = solve_ivp(
        system,
        (0, max_time),
        initial_state,
        method="RK45",
        events=event,
        rtol=rtol,
        atol=atol,
        dense_output=True,
    )

    if len(sol1.t_events[0]) == 0:
        raise RuntimeError(
            f"No threshold crossing detected within max_time={max_time}. "
            "Check that the system has a limit cycle and the threshold is appropriate."
        )

    first_crossing_state = sol1.y_events[0][0]

    # Step slightly past the crossing to avoid immediate re-trigger at t=0
    # Use a small fraction of the elapsed time as an estimate for a safe step
    t_skip = max(1e-6, sol1.t_events[0][0] * 1e-4)
    sol_skip = solve_ivp(
        system,
        (0, t_skip),
        first_crossing_state,
        method="RK45",
        rtol=rtol,
        atol=atol,
    )
    stepped_state = sol_skip.y[:, -1]

    # Second integration: find the next crossing to measure the period
    sol2 = solve_ivp(
        system,
        (0, max_time),
        stepped_state,
        method="RK45",
        events=event,
        rtol=rtol,
        atol=atol,
        dense_output=True,
    )

    if len(sol2.t_events[0]) == 0:
        raise RuntimeError(
            f"Second threshold crossing not detected within max_time={max_time}. "
            "The system may not have a stable limit cycle."
        )

    # Total period = skip time + time to next crossing
    period = t_skip + sol2.t_events[0][0]
    zero_phase_state = sol2.y_events[0][0]

    return period, zero_phase_state


def get_limit_cycle(
    system: Callable[[float, np.ndarray], np.ndarray],
    zero_phase_state: np.ndarray,
    period: float,
    num_points: int,
    rtol: float = 1e-9,
    atol: float = 1e-12,
) -> tuple[np.ndarray, np.ndarray]:
    """Integrates the system over one period to get the limit cycle trajectory.

    Args:
        system: The ODE system function with signature (t, y) -> dy/dt.
        zero_phase_state: The state at the zero-phase point (threshold crossing).
        period: The oscillation period.
        num_points: Number of points to sample along the limit cycle.
        rtol: Relative tolerance for the integrator.
        atol: Absolute tolerance for the integrator.

    Returns:
        A tuple of (t_eval, limit_cycle) where t_eval is the time array and
        limit_cycle is the state trajectory of shape (num_points, state_dim).
    """
    t_eval = np.linspace(0, period, num_points)

    sol = solve_ivp(
        system,
        (0, period),
        zero_phase_state,
        method="RK45",
        t_eval=t_eval,
        rtol=rtol,
        atol=atol,
    )

    return sol.t, sol.y.T
