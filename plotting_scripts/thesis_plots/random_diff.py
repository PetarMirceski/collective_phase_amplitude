from config.oscillator_constants import FitzNagumo
from config.paths import FIGURE_PATH
from config.simulation_parameters.fitz_random import optimization_configuration
from plotting_scripts.paper_plots.utils.diff_plot import plot_diff

BASE_DIR = FIGURE_PATH / "thesis" / "diff_plots/"
BASE_DIR.mkdir(exist_ok=True, parents=True)


def main() -> None:
    first_states = optimization_configuration[: len(optimization_configuration) // 2]
    second_states = optimization_configuration[len(optimization_configuration) // 2 :]
    name = FitzNagumo.name

    plot_diff(
        first_states,
        name,
        BASE_DIR / "random_three.pdf",
        2 / 3,
        right_x_lim=3000,
        strong=True,
        bbox_inches="tight",
    )
    plot_diff(
        second_states,
        name,
        BASE_DIR / "random_one.pdf",
        2 / 3,
        right_x_lim=3000,
        bbox_inches="tight",
    )


if __name__ == "__main__":
    main()
