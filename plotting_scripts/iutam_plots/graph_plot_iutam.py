import matplotlib.pyplot as plt
import networkx as nx
import numpy as np

from config.constants import OscillatorParameters
from config.oscillator_constants import FitzNagumoStar
from config.paths import FIGURE_PATH

BASE_DIR = FIGURE_PATH / "paper" / "graph_plots/"
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

LABELS = {i: rf"\# ${i + 1}$" for i in range(4)}
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
    star_graph = nx.star_graph(3)
    fitz_star_data = OscillatorParameters.load(FitzNagumoStar.name)

    G = star_graph.to_directed()

    adjacency_matrix = FitzNagumoStar.coupling_parameters_k
    num_nodes = len(adjacency_matrix)

    # Create nodes and add them to the graph
    nodes = [0, 1, 2, 3]

    # Add weighted edges from the adjacency matrix
    for i in range(num_nodes):
        for j in range(num_nodes):
            weight = adjacency_matrix[i][j]
            if weight != 0:
                G.add_edge(nodes[i], nodes[j], weight=weight)

    center_node = 0  # Or any other node to be in the center
    edge_nodes = set(G) - {center_node}
    cycle_colors = construct_colors(fitz_star_data.limit_cycle)
    # Ensures the nodes around the circle are evenly distributed
    pos = nx.circular_layout(G.subgraph(edge_nodes))
    pos[center_node] = np.array([0, 0])  # manually specify node position

    fig, ax = plt.subplots()

    nx.draw(
        G,
        pos,
        with_labels=True,
        node_size=5000,
        edge_color="black",
        node_color=cycle_colors,
        font_size=35,
        font_color="black",
        cmap="Blues",
        connectionstyle="arc3, rad = 0.1",
        labels=LABELS,
    )

    # Add edge labels with weights
    edge_labels = {edge: G[edge[0]][edge[1]]["weight"] for edge in G.edges()}

    edge_labels = nx.get_edge_attributes(G, "weight")
    label_pos = 0.3
    rotate = True
    rad = 0.25

    # Adjust the positions of edge labels along curved edges
    for (n1, n2), label in edge_labels.items():
        (x1, y1) = pos[n1]
        (x2, y2) = pos[n2]
        (x, y) = (
            x1 * label_pos + x2 * (1.0 - label_pos),
            y1 * label_pos + y2 * (1.0 - label_pos),
        )
        pos_1 = ax.transData.transform(np.array(pos[n1]))
        pos_2 = ax.transData.transform(np.array(pos[n2]))
        linear_mid = 0.5 * pos_1 + 0.5 * pos_2
        d_pos = pos_2 - pos_1
        rotation_matrix = np.array([(0, 1), (-1, 0)])
        ctrl_1 = linear_mid + rad * rotation_matrix @ d_pos
        ctrl_mid_1 = 0.5 * pos_1 + 0.5 * ctrl_1
        ctrl_mid_2 = 0.5 * pos_2 + 0.5 * ctrl_1
        bezier_mid = 0.5 * ctrl_mid_1 + 0.5 * ctrl_mid_2
        (x, y) = ax.transData.inverted().transform(bezier_mid)

        if rotate:
            # in degrees
            angle = np.arctan2(y2 - y1, x2 - x1) / (2.0 * np.pi) * 360
            # make label orientation "right-side-up"
            if angle > 90:
                angle -= 180
            if angle < -90:
                angle += 180
            # transform data coordinate angle to screen coordinate angle
            xy = np.array((x, y))
            trans_angle = ax.transData.transform_angles(
                np.array((angle,)), xy.reshape((1, 2))
            )[0]
        else:
            trans_angle = 0.0

        ax.text(
            x,
            y,
            label,
            ha="center",
            size=20,
            color="red",
            rotation=trans_angle,
            rotation_mode="anchor",
            transform_rotates_text=True,
            transform=ax.transData,
        )

    # Show the plot
    plt.savefig(BASE_DIR / "star_network_adj.pdf")
    plt.close()


if __name__ == "__main__":
    main()
