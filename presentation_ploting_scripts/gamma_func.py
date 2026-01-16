from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

STEP = 0.01
BASE_PATH = Path("plots/presentation_plots/")
BASE_PATH.mkdir(exist_ok=True, parents=True)
plt.rcParams["figure.figsize"] = [14, 10]


def main() -> None:
    sin_arg = np.arange(-np.pi - 0.1, np.pi + 0.1, STEP)
    mid_idx = int(sin_arg.shape[0] / 2)
    sin = -np.sin(sin_arg)

    plt.axhline(color="black", linewidth=1)
    plt.axvline(color="black", linewidth=1)
    plt.plot(sin_arg, sin)
    plt.xlabel(r"$\phi$")
    plt.ylabel(r"$\Gamma(\phi)$")
    plt.yticks([-1, 1], [r"$min \Gamma$", r"$max\Gamma$"])
    plt.xticks(
        [-np.pi, 0, np.pi],
        [r"$-\pi$", "0", r"$\pi$"],
    )

    # Unstable equilibrium points
    plt.text(-np.pi + 1, 0 + 0.05, r"unstable", ha="right")
    plt.text(np.pi - 1, 0 + 0.05, r"unstable", ha="left")

    # Stable equilibrium points
    plt.scatter(0, 0, facecolor="none", edgecolor="tab:blue", s=130)
    plt.text(0 - 0.1, 0 + 0.05, r"$\phi^{*} = 0$", ha="right")
    plt.text(0 - 0.05, 0 - 0.1, r"stable", ha="right")

    plt.axhline(linewidth=1, linestyle="dashed", color="black")
    # phi < zero equilibrium state

    plt.scatter(sin_arg[100], sin[100], color="tab:blue", s=130)
    plt.annotate(
        "",
        xytext=(sin_arg[100], sin[100] - 0.05),
        xycoords="data",
        xy=(sin_arg[100] + 1, sin[100] - 0.1),
        textcoords="data",
        arrowprops={
            "arrowstyle": "->",
            "connectionstyle": "angle,angleA=-90,angleB=180,rad=0",
        },
    )
    # phi > zero equilibrium state
    plt.scatter(sin_arg[600], sin[600], color="tab:blue", s=130)

    plt.annotate(
        "",
        xytext=(sin_arg[600], sin[600] + 0.05),
        xycoords="data",
        xy=(sin_arg[600] - 1, sin[600] + 0.1),
        textcoords="data",
        arrowprops={
            "arrowstyle": "->",
            "connectionstyle": "angle,angleA=-90,angleB=180,rad=0",
        },
    )

    # STABILITY LINE
    line_x = [sin_arg[mid_idx - 20] + 0.1, sin_arg[mid_idx + 35] + 0.1]
    line_y = [sin[mid_idx - 20] + 0.1, sin[mid_idx + 35] + 0.1]

    plt.plot(line_x, line_y, c="tab:red")
    plt.text(
        sin_arg[mid_idx] + 0.6,
        sin[mid_idx] - 0.05,
        r"$-\Gamma^{'}(\phi^{*})$",
        rotation=-65,
        c="tab:red",
        ha="right",
    )

    plt.scatter(
        [-np.pi, np.pi], [0, 0], facecolor="red", edgecolor="red", s=130, zorder=10
    )

    plt.axhline(-1, linestyle="dashed", color="black", linewidth=1)
    plt.axhline(1, linestyle="dashed", color="black", linewidth=1)

    plt.savefig(BASE_PATH / "gamma.png")
    plt.close()


if __name__ == "__main__":
    main()
