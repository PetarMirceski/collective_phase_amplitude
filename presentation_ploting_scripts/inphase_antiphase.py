from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

STEP = 0.01
BASE_PATH = Path("plots/presentation_plots/")
BASE_PATH.mkdir(exist_ok=True, parents=True)
plt.rcParams["figure.figsize"] = [14, 10]
plt.rcParams["figure.autolayout"] = True
plt.rcParams["font.size"] = 20
plt.rcParams["legend.fontsize"] = 18
major = 5.0
minor = 3.0
plt.rcParams["xtick.major.size"] = major
plt.rcParams["xtick.minor.size"] = minor
plt.rcParams["ytick.major.size"] = major
plt.rcParams["ytick.minor.size"] = minor


def main() -> None:
    sin_arg = np.arange(0, 20, STEP)
    sin = -np.sin(sin_arg)

    fig, (ax1, ax2) = plt.subplots(2, 1)
    fig.subplots_adjust(hspace=0.5)

    ax1.set_title("in-phase entrainment")
    ax1.set_yticks([])
    ax1.plot(sin_arg, sin)
    ax1.plot(sin_arg + 0.1, sin, "--")
    ax1.set_xlabel("time")

    ax2.set_title("anti-phase entrainment")
    ax2.set_yticks([])
    ax2.plot(sin_arg, sin)
    ax2.plot(sin_arg, -sin, "--")
    ax2.set_xlabel("time")

    plt.savefig(BASE_PATH / "in_anti_phase.png")
    plt.close()


if __name__ == "__main__":
    main()
