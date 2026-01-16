from collections.abc import Callable
from pathlib import Path
from typing import Any

import matplotlib.pyplot as plt
import numpy as np

from config.constants import STEP_FITZ, OscillatorParameters
from config.oscillator_constants import FitzNagumo

BASE_PATH = Path("presentation_plots/")
plt.rcParams["figure.figsize"] = [12, 8]
plt.rcParams["figure.autolayout"] = True
plt.rcParams["font.size"] = 20
plt.rcParams["legend.fontsize"] = 18
major = 5.0
minor = 3.0
plt.rcParams["xtick.major.size"] = major
plt.rcParams["xtick.minor.size"] = minor
plt.rcParams["ytick.major.size"] = major
plt.rcParams["ytick.minor.size"] = minor


def euler_step(ode: Any, state: complex, dt: float) -> complex:
    return complex(state + dt * ode(state))


# SIMULATE WITHOUT INPUT
def isf_no_input(
    floquet_exp: complex,
) -> Callable[[complex], complex]:
    def model(
        r: complex,
    ) -> complex:
        dr = r * floquet_exp
        return dr

    return model


# SIMULATE WITH INPUT
def isf_input(
    floquet_exp: complex,
) -> Callable[[complex, np.ndarray, np.ndarray], complex]:
    def model(r: complex, isf: np.ndarray, input: np.ndarray) -> complex:
        dr = r * floquet_exp + np.dot(isf, input)
        return complex(dr)

    return model


def r_decay_simu() -> None:
    time = 150
    dt = STEP_FITZ
    r_value = complex(0.1, 0.0)
    fitz_data = OscillatorParameters.load(FitzNagumo.name)
    floquet_exponent = fitz_data.floquet_exponents[1]
    amp_no_input = isf_no_input(floquet_exponent)
    time_vector = np.arange(0, time, dt)
    r_no_input = np.empty_like(time_vector, dtype=complex)
    for i in range(time_vector.shape[0]):
        r_value = euler_step(amp_no_input, r_value, dt)
        r_no_input[i] = r_value

    # plt.title("Evolution of the Amplitude Function\n for the Fitzhugn-Nagumo network")
    plt.grid()
    plt.axhline(color="black", linestyle="--")
    plt.xlabel("time(s)")
    plt.ylabel(r"$r(t)$")
    plt.plot(time_vector, r_no_input)
    plt.savefig(BASE_PATH / "amp_no_input.png")
    plt.show()


def r_input_simu() -> None:
    dt = STEP_FITZ
    r_value = complex(0.01, 0.0)

    fitz_data = OscillatorParameters.load(FitzNagumo.name)
    floquet_exponent = fitz_data.floquet_exponents[1]
    isf = fitz_data.left_floquet_vectors[1]

    amp_input = isf_input(floquet_exponent)

    time_vector = np.arange(isf.shape[0]) * STEP_FITZ
    r_input = np.empty_like(time_vector, dtype=complex)

    sin_arg = np.arange(isf.shape[0]) * dt
    sin_wave = 1e-5 * np.sin(fitz_data.natural_freq * sin_arg)
    input_sig = np.zeros_like(isf)
    for i in range(input_sig.shape[1]):
        input_sig[:, i] = sin_wave

    for i in range(isf.shape[0]):
        r_value = r_value + dt * amp_input(r_value, isf[i], input_sig[i])
        r_input[i] = r_value

    # plt.title(
    #     "Amplitude decay of the Fitzhugn-Nagumo network\nunder weak sinusoidal forcing"
    # )
    plt.plot(time_vector, r_input)
    plt.xlabel("time(s)")
    plt.ylabel(r"$r(t)$")
    plt.savefig(BASE_PATH / "amp_input.png")
    plt.show()


def main() -> None:
    r_input_simu()
    r_decay_simu()


if __name__ == "__main__":
    main()
