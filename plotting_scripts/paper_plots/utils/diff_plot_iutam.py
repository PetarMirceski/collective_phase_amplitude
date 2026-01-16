from pathlib import Path

# from solvers.average_phase_diff import simulate
import matplotlib.pyplot as plt
import numpy as np
from mpl_toolkits.axes_grid1.inset_locator import inset_axes
from tqdm import tqdm

from config.constants import OptimizationConfig, OptimizationParameters


def plot_diff(
    total_states: list[OptimizationConfig],
    name: str,
    save_path: Path,
    scale_multiplier: float = 1,
    multiplier: float = 2,
    top_y_lim: float = -1,
    right_x_lim: float = -1,
    strong: bool = False,
) -> None:
    fig, axs = plt.subplots(
        len(total_states),
        2,
        figsize=(
            9.5 * multiplier,
            12 * multiplier * scale_multiplier,
        ),
    )
    for i, config in tqdm(enumerate(total_states), total=len(total_states)):
        states = config.states
        power_str = str(config.power).replace(".", ",")

        fitz_simple_data = OptimizationParameters.load(
            name,
            "simple",
            f"power_{power_str}_delta_0_element_{states}.pkl",
        )

        fitz_amp_data = OptimizationParameters.load(
            name,
            "amplitude",
            f"power_{power_str}_delta_0_element_{states}.pkl",
        )

        fitz_feed_data = OptimizationParameters.load(
            name,
            "feedback",
            f"power_{power_str}_delta_0_element_{states}.pkl",
        )

        fitz_sine_data = OptimizationParameters.load(
            name,
            "sine",
            f"power_{power_str}_delta_0_element_{states}.pkl",
        )

        # phase diff plotting
        diff_simple = fitz_simple_data.phase_diff
        diff_amp = fitz_amp_data.phase_diff
        diff_feed = fitz_feed_data.phase_diff
        diff_sine = fitz_sine_data.phase_diff

        diff_time_vector = np.linspace(0, config.simulation_time, diff_simple.shape[0])
        diff_time_vector_sine = np.linspace(
            0, config.simulation_time, diff_sine.shape[0]
        )

        gamma_simple = fitz_simple_data.gamma
        gamma_amp = fitz_amp_data.gamma
        gamma_sine = fitz_sine_data.gamma

        l1 = axs[i, 1].scatter(
            diff_time_vector,
            diff_simple,
            marker="x",
            label="phase optimization",
        )
        l2 = axs[i, 1].scatter(
            diff_time_vector,
            diff_amp,
            marker=".",
            label="amplitude suppression",
        )
        l3 = axs[i, 1].scatter(
            diff_time_vector,
            diff_feed,
            marker="o",
            label="amplitude feedback",
        )

        # location = (
        #     "center left" if strong and (i == len(total_states) - 1) else "center right"
        # )
        axins = inset_axes(axs[i, 1], width="40%", height="40%", loc="center right")
        axins.scatter(diff_time_vector, diff_simple, marker="x", s=1)
        axins.scatter(diff_time_vector, diff_amp, marker=".", s=1)
        axins.scatter(diff_time_vector, diff_feed, marker="o", s=1)
        l4 = axins.scatter(
            diff_time_vector_sine, diff_sine, marker="8", s=1, label="sine wave"
        )
        # num_ticks_x = 3
        # num_ticks_y = 3

        # axins.set_xticks(
        #     axins.get_xticks()[:: len(axins.get_xticks()) // num_ticks_x]
        # )  # Set only 3 ticks on the x-axis
        # axins.set_yticks(
        #     axins.get_yticks()[:: len(axins.get_yticks()) // num_ticks_y]
        # )  # Set only 4 ticks on the y-axis
        axins.tick_params(labelsize=16)  # axis="both", which="both",

        axins.axhline(linewidth=2, linestyle="dashed", color="black", alpha=0.4)

        axs[i, 0].plot(
            gamma_simple[:, 0], gamma_simple[:, 1], label="phase optimization"
        )
        axs[i, 0].plot(gamma_amp[:, 0], gamma_amp[:, 1], label="amplitude suppression")
        axs[i, 0].plot(
            gamma_sine[:, 0],
            gamma_sine[:, 1] * 5,
            label=r"sinusoidal control $\times 5$",
        )

        # GAMMA INSET
        # insetPosition = [-0.17, -0.17, 0.7, 0.7]  # [left, bottom, width, height]
        # axins = inset_axes(
        #     axs[i, 0],
        #     width="50%",
        #     height="50%",
        #     bbox_to_anchor=insetPosition,
        #     bbox_transform=axs[i, 0].transAxes,
        # )
        # axins.plot(
        #     gamma_sine[:, 0],
        #     gamma_sine[:, 1],
        #     c="tab:green",
        #     label="sinusoidal control",
        # )

        # axins.axhline(linewidth=1, linestyle="dashed", color="black")
        # axins.axvline(linewidth=1, linestyle="dashed", color="black")
        # axins.set_xticks(
        #     [-np.pi, -np.pi / 2, 0, np.pi / 2, np.pi],
        #     [
        #         r"$-\pi$",
        #         r"$-\frac{\pi}{2}$",
        #         "0",
        #         r"$\frac{\pi}{2}$",
        #         r"$\pi$",
        #     ],
        # )
        # axins.tick_params(
        #     axis="both", which="major", labelsize=plt.rcParams["xtick.labelsize"] // 2
        # )

        axs[i, 1].axhline(linewidth=9, linestyle="dashed", color="black", alpha=0.4)
        axs[i, 0].axhline(linewidth=2, linestyle="dashed", color="black")
        axs[i, 0].axvline(linewidth=2, linestyle="dashed", color="black")
        axs[i, 0].set_xticks(
            [-np.pi, -np.pi / 2, 0, np.pi / 2, np.pi],
            [
                r"$-\pi$",
                r"$-\frac{\pi}{2}$",
                "0",
                r"$\frac{\pi}{2}$",
                r"$\pi$",
            ],
        )
        axs[i, 1].set_xlabel(r"$time(s)$")
        axs[i, 1].set_ylabel(r"$\phi$")
        axs[i, 0].set_xlabel(r"$\phi$")
        axs[i, 0].set_ylabel(r"$\Delta + \Gamma(\phi)$")
        # plt.ylim([-0.05, 0.3])
        axs[i, 1].grid()

        handles, labels = axs[i, 0].get_legend_handles_labels()

        axs[i, 1].legend(handles=[l1, l2, l3, l4], prop={"size": 15}, loc="upper right")
        axs[i, 0].legend(prop={"size": 15}, loc="upper right")

        if top_y_lim != -1:
            axs[i, 1].set_ylim(top=top_y_lim)

        if right_x_lim != -1:
            axs[i, 1].set_xlim(right=right_x_lim)

    fig_label = "a"
    for ax in axs.flat:
        ax.text(
            -0.1,
            1.1,
            f"{fig_label})",
            transform=ax.transAxes,
            fontweight="bold",
            va="top",
        )
        fig_label = chr(ord(fig_label) + 1)  # Increment label

    plt.subplots_adjust(wspace=0.6, hspace=0.2)  # Adjust the spacing as needed
    plt.savefig(save_path, bbox_inches="tight")
    # plt.show()
