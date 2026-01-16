from config.oscillator_constants import FitzNagumoRing
from config.paths import FIGURE_PATH
from config.simulation_parameters.fitz_ring import optimization_configurations
from plotting_scripts.paper_plots.utils.diff_plot import plot_diff

BASE_DIR = FIGURE_PATH / "thesis" / "diff_plots/"
BASE_DIR.mkdir(exist_ok=True, parents=True)


def main() -> None:
    first_states = optimization_configurations[: len(optimization_configurations) // 2]
    second_states = optimization_configurations[len(optimization_configurations) // 2 :]
    name = FitzNagumoRing.name
    plot_diff(
        first_states,
        name,
        BASE_DIR / "ring_three.pdf",
        top_y_lim=0.4,
        right_x_lim=1500,
        bbox_inches="tight",
    )

    plot_diff(
        second_states,
        name,
        BASE_DIR / "ring_one.pdf",
        top_y_lim=0.4,
        right_x_lim=3000,
        bbox_inches="tight",
    )


if __name__ == "__main__":
    main()
