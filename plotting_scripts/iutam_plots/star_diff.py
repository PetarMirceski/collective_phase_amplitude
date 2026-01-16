from config.oscillator_constants import FitzNagumoStar
from config.paths import FIGURE_PATH
from config.simulation_parameters.fitz_star import optimization_configurations
from plotting_scripts.paper_plots.utils.diff_plot_iutam import plot_diff

BASE_DIR = FIGURE_PATH / "paper" / "diff_plots/"
BASE_DIR.mkdir(exist_ok=True, parents=True)


def main() -> None:
    first_states = optimization_configurations[: len(optimization_configurations)]
    name = FitzNagumoStar.name

    plot_diff(
        first_states,
        name,
        BASE_DIR / "star_one.pdf",
        scale_multiplier=0.7,
        multiplier=2.0,
        top_y_lim=0.8,
        right_x_lim=1200,
    )


if __name__ == "__main__":
    main()
