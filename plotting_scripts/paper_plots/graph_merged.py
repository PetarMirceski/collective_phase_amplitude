import matplotlib.pyplot as plt
import networkx as nx
import numpy as np
from mpl_toolkits.axes_grid1 import make_axes_locatable

from config.constants import OscillatorParameters
from config.oscillator_constants import FitzNagumo, FitzNagumoRing
from config.paths import FIGURE_PATH

BASE_DIR = FIGURE_PATH / "paper" / "graph_plots_merged/"
BASE_DIR.mkdir(exist_ok=True, parents=True)

COLORS = [
    "tab:blue",
    "tab:orange",
    "tab:green",
    "tab:red",
    "tab:purple",
    "tab:brown",
    "tab:pink",
    "tab:gray",
    "tab:olive",
    "tab:cyan",
]

LABELS = {i: rf"\# ${i + 1}$" for i in range(10)}
NODE_SIZE = 50
MIN_RANGE = 0.0
MAX_RANGE = 1.0


def construct_colors(limit_cycle: np.ndarray) -> list[float]:
    v_states = limit_cycle[limit_cycle.shape[0] // 2, 1::2]
    scaled_range_v_states = (v_states - np.min(v_states)) / (
        np.max(v_states) - np.min(v_states)
    ) * (MAX_RANGE - MIN_RANGE) + MIN_RANGE
    color_rgb: list[float] = []
    for idx in range(scaled_range_v_states.shape[0]):
        color = scaled_range_v_states[idx]
        color_rgb.append(color)

    return color_rgb


def main() -> None:
    fitz_random = nx.from_numpy_array(FitzNagumo.coupling_parameters)
    fitz_ring = nx.from_numpy_array(FitzNagumoRing.coupling_parameters)
    fitz_random_data = OscillatorParameters.load(FitzNagumo.name)
    fitz_ring_data = OscillatorParameters.load(FitzNagumoRing.name)

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(24, 12))

    cycle_colors = construct_colors(fitz_ring_data.limit_cycle)
    nx.draw(
        fitz_ring,
        pos=nx.circular_layout(fitz_ring),
        ax=ax1,
        node_color=cycle_colors,
        node_size=NODE_SIZE * 200,
        edge_color="black",
        cmap="Blues",
        font_size=35,
        labels=LABELS,
        with_labels=True,
    )
    ax1.text(
        0.05,
        0.95,
        "a)",
        transform=ax1.transAxes,
        fontweight="bold",
        va="top",
    )

    cycle_colors = construct_colors(fitz_random_data.limit_cycle)
    nx.draw(
        fitz_random,
        pos=nx.circular_layout(fitz_random),
        ax=ax2,
        node_color=cycle_colors,
        node_size=NODE_SIZE * 200,
        edge_color="black",
        cmap="Blues",
        font_size=35,
        labels=LABELS,
        with_labels=True,
    )
    ax2.text(
        0.05,
        0.95,
        "b)",
        transform=ax2.transAxes,
        fontweight="bold",
        va="top",
    )

    sm = plt.cm.ScalarMappable(
        cmap=plt.cm.Blues,
        norm=plt.Normalize(vmin=np.min(cycle_colors), vmax=np.max(cycle_colors)),
    )
    divider = make_axes_locatable(ax2)
    cax = divider.append_axes("right", size="5%", pad=0.05)
    fig.colorbar(sm, cax=cax)

    fig.tight_layout()
    plt.savefig(BASE_DIR / "graph.pdf", transparent=True)
    plt.close()


if __name__ == "__main__":
    main()
