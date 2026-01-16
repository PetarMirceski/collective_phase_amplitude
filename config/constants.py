from __future__ import annotations

import copy
import pickle
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np

from config.paths import FLOQUET_PATH, SIMULATION_OUTPUT

STEP = 1e-3
STEP_FITZ = 1e-3  # with:step 1e-3 around 2 min for fitz model
NUM_PHASE_POINTS: int = 1000

# TODO: Think about whether the dataclasses should be moved
# TODO: Think about Class inheritance of the Oscillator Parameters and the oscillator classes


@dataclass
class OscillatorParameters:
    """Class for storing and persisting Floquet analysis results."""

    oscillator_name: str
    natural_freq: float
    natural_period: float
    number_of_itters: int
    limit_cycle: np.ndarray
    left_floquet_vectors: np.ndarray
    right_floquet_vectors: np.ndarray
    floquet_exponents: np.ndarray
    v0_diff: np.ndarray

    def dump(self) -> None:
        """Save the oscillator parameters to a pickle file."""
        path = SIMULATION_OUTPUT / self.oscillator_name / FLOQUET_PATH
        path.mkdir(exist_ok=True, parents=True)
        save_path = (path / "data").with_suffix(".pkl")
        with open(str(save_path), "wb") as handle:
            pickle.dump(self, handle, protocol=pickle.HIGHEST_PROTOCOL)

    @staticmethod
    def load(name: str | Path) -> OscillatorParameters:
        path = SIMULATION_OUTPUT / name / FLOQUET_PATH / "data"
        load_path = Path(path).with_suffix(".pkl")
        with open(load_path, "rb") as infile:
            oscillator: OscillatorParameters = pickle.load(infile, encoding="utf-8")
        return oscillator


@dataclass
class OptimizationParameters:
    """Class for storing and persisting optimization results."""

    nu: float
    mu: float
    input_index: list[int]
    gamma: np.ndarray
    input: np.ndarray
    limit_cycle_input: np.ndarray
    phase_diff: np.ndarray
    simulation_time: float
    delta: float
    init_phase: float
    power: float
    name: str
    type: str

    def dump(self) -> None:
        """Save the optimization parameters to a pickle file."""
        power_string = f"power_{str(self.power).replace('.', ',')}"
        delta_string = f"delta_{str(self.delta).replace('.', ',')}"
        save_name = f"{power_string}_{delta_string}_element_{self.input_index}.pkl"

        path = SIMULATION_OUTPUT / self.name / self.type
        path.mkdir(exist_ok=True, parents=True)
        save_path = path / save_name
        with open(str(save_path), "wb") as handle:
            pickle.dump(self, handle, protocol=pickle.HIGHEST_PROTOCOL)

    @staticmethod
    def load(
        name: str | Path, method: str | Path, file: str | Path
    ) -> OptimizationParameters:
        load_path = SIMULATION_OUTPUT / name / method / file
        with open(load_path, "rb") as infile:
            data: OptimizationParameters = pickle.load(infile, encoding="utf-8")
        return data


@dataclass
class NetworkConfig:
    states: list = field(default_factory=lambda: [])

    @property
    def simple_states(cls) -> list[int]:
        simple_states = [state[0] for state in cls.states if len(state) == 1]
        return simple_states

    @property
    def complex_states(cls) -> list[list[int]]:
        complex_states = [state for state in cls.states if len(state) != 1]
        return complex_states


@dataclass
class OptimizationConfig:
    states: list[int]
    delta: float = 0.0
    simulation_time: float = 100.0
    initial_phase: float = 1 / 4
    k: float = 10.0
    power: float = 0.0
    alpha: float = 100.0


@dataclass
class OptimizationConfigVan(NetworkConfig):
    delta: float = 0.0
    simulation_time: float = 100.0
    initial_phase: float = 1 / 4
    k: float = 10.0
    alpha = 100
    states: list[list[int]] = field(default_factory=lambda: copy.copy([[0, 1]]))
    power_list: list[float] = field(default_factory=lambda: [1.0])


VanConfig = OptimizationConfigVan()
