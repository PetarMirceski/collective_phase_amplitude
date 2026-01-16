from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

BASE_PATH = Path("plots/presentation_plots/")
BASE_PATH.mkdir(exist_ok=True, parents=True)

plt.rcParams["figure.figsize"] = [14, 10]


def sawtooth_wave(t: np.ndarray, T: float, num_terms: int) -> float:
    result = 0.0
    for n in range(1, num_terms + 1):
        result += (2 / np.pi) * ((-1) ** (n + 1) / n) * np.sin(2 * np.pi * n * t / T)
    return result


def main() -> None:
    # Parameters
    T = 1.0  # Period of the sawtooth wave
    duration = 1 * T  # Duration of the plot

    # Generate time values
    t_values = np.linspace(0, duration, 1000)

    # Plot the original sawtooth wave and its Fourier series approximation
    plt.figure()
    for num_terms in range(1, 5):
        plt.plot(t_values, sawtooth_wave(t_values, T, num_terms), zorder=0)
    # plt.title("Fourier Series Approximation of a Sawtooth Wave")
    plt.xlabel("$\\phi$")
    plt.ylabel("$\\Gamma(\\phi)$")
    plt.xticks(
        [0.0, 0.25, 0.5, 0.75, 1.0],
        ["$-\\pi$", "$-\\frac{\\pi}{2}$", "$0$", "$-\\frac{\\pi}{2}$", "$\\pi$"],
    )
    plt.yticks([-1, 1], [r"$min \Gamma$", r"$max\Gamma$"])
    plt.grid(True)

    plt.axhline(color="black", linewidth=1)
    plt.axvline(0.5, color="black", linewidth=1)

    plt.scatter([0.5, 1, 0], [0, 0, 0], color="tab:blue", zorder=2, s=50)
    plt.scatter([1, 0], [0, 0], color="tab:red", zorder=2, s=50)

    # Left arrow
    plt.annotate(
        "",
        xytext=(0.2, 0.25),
        xycoords="data",
        xy=(0.4, 0.2),
        textcoords="data",
        arrowprops={
            "arrowstyle": "->",
            "connectionstyle": "angle,angleA=-90,angleB=180,rad=0",
        },
    )

    # Right arrow
    plt.annotate(
        "",
        xytext=(0.9, -0.25),
        xycoords="data",
        xy=(0.7, -0.2),
        textcoords="data",
        arrowprops={
            "arrowstyle": "->",
            "connectionstyle": "angle,angleA=-90,angleB=180,rad=0",
        },
    )

    # Gradient arrow
    plt.text(
        0.6,
        0,
        r"$-\Gamma^{'}(\phi^{*})$",
        rotation=-65,
        c="tab:red",
        ha="right",
    )

    line_x = [0.5, 0.55]
    line_y = [0.2, -0.05]
    plt.plot(line_x, line_y)

    plt.savefig(BASE_PATH / "steep_gamma.png")
    plt.close()


main()
