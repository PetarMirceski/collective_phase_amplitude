import tempfile
from pathlib import Path

import imageio
import matplotlib
import matplotlib.pyplot as plt
import networkx as nx
import numpy as np
from tqdm import tqdm

from config.constants import OscillatorParameters
from config.oscillator_constants import FitzNagumo, FitzNagumoRing
from config.paths import FIGURE_PATH

matplotlib.use("Agg")

BASE_DIR = FIGURE_PATH / "graph_plots/"
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


def construct_colors(limit_cycle: np.ndarray, t: int) -> list[float]:
    v_states = limit_cycle[t, 1::2]
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

    def create_frame_fitz_nagumo(t: int, tmpdirname: str) -> None:
        cycle_colors = construct_colors(fitz_random_data.limit_cycle, t)
        plt.figure()
        nx.draw(
            fitz_random,
            pos=nx.circular_layout(fitz_random),
            node_color=cycle_colors,
            node_size=NODE_SIZE * 300,
            edge_color="black",
            cmap="Blues",
            font_size=24,
            labels=LABELS,
            with_labels=True,
        )
        plt.savefig(Path(tmpdirname) / f"random_network{t}.png", transparent=True)
        plt.close(plt.gcf())

    time_steps = np.linspace(
        0, fitz_random_data.limit_cycle.shape[0] - 1, 400, dtype=int
    )
    with tempfile.TemporaryDirectory() as tmpdirname:
        for t in tqdm(time_steps):
            create_frame_fitz_nagumo(t, tmpdirname)
        frames: list[imageio.core.util.Array] = []
        for t in tqdm(time_steps):
            image = imageio.v2.imread(Path(tmpdirname) / f"random_network{t}.png")
            frames.append(image)

        imageio.mimsave(
            BASE_DIR / "random_network.gif",
            frames,
            fps=30,
            loop=0,  # type:ignore
        )

    def create_frame_fitz_ring(t: int, temporarydir: str) -> None:
        cycle_colors = construct_colors(fitz_ring_data.limit_cycle, t)
        plt.figure()
        nx.draw(
            fitz_ring,
            pos=nx.circular_layout(fitz_ring),
            node_color=cycle_colors,
            node_size=NODE_SIZE * 300,
            edge_color="black",
            cmap="Blues",
            font_size=24,
            labels=LABELS,
            with_labels=True,
        )
        plt.savefig(Path(tmpdirname) / f"ring_network{t}.png", transparent=True)
        plt.close()

    time_steps = np.linspace(0, fitz_ring_data.limit_cycle.shape[0] - 1, 150, dtype=int)
    with tempfile.TemporaryDirectory() as tmpdirname:
        for t in tqdm(time_steps):
            create_frame_fitz_ring(t, tmpdirname)
        frames = []
        for t in tqdm(time_steps):
            image = imageio.v2.imread(Path(tmpdirname) / f"ring_network{t}.png")
            frames.append(image)

        imageio.mimsave(
            BASE_DIR / "ring_network.gif",
            frames,
            fps=30,
            loop=0,  # type:ignore
        )


if __name__ == "__main__":
    main()
