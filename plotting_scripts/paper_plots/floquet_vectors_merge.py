import io

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.figure import Figure
from PIL import Image

from config.constants import STEP_FITZ, OscillatorParameters
from config.oscillator_constants import FitzNagumo, FitzNagumoRing
from config.paths import FIGURE_PATH

BASE_DIR = FIGURE_PATH / "paper" / "floquets_and_limit_cycles_merge/"
BASE_DIR.mkdir(exist_ok=True, parents=True)


# plt.matplotlib.use("Agg")
Y_OFFSET = 0.06
# plt.rcParams.update(
#     {
#         "figure.labelsize": 50,
#     }
# )


def fig2img(fig: Figure) -> Image:
    buf = io.BytesIO()
    fig.savefig(buf)
    buf.seek(0)
    img = Image.open(buf)
    return img


def plot_states(l_cycle: np.ndarray, time_array: np.ndarray) -> Figure:
    fig, axs = plt.subplots(l_cycle.shape[1] // 4, 2, sharex=True, layout="tight")
    first_half_states = l_cycle[:, : l_cycle.shape[1] // 2].real
    next_half_states = l_cycle[:, l_cycle.shape[1] // 2 :].real

    for i in range(first_half_states.shape[1] // 2):
        axs[i, 0].plot(time_array, first_half_states[:, 2 * i])
        axs[i, 0].plot(time_array, first_half_states[:, 2 * i + 1])
        axs[i, 0].text(
            1.00,
            0.99,
            rf"\#{i + 1}",
            ha="right",
            va="top",
            transform=axs[i, 0].transAxes,
        )
        # axs[i, 0].set_ylabel(rf"\#{i + 1}")
        axs[i, 0].grid(True)

    for i in range(next_half_states.shape[1] // 2):
        index = i + first_half_states.shape[1] // 2 + 1
        axs[i, 1].plot(time_array, next_half_states[:, 2 * i])
        axs[i, 1].plot(time_array, next_half_states[:, 2 * i + 1])
        axs[i, 1].text(
            1.00,
            0.99,
            rf"\#{index}",
            ha="right",
            va="top",
            transform=axs[i, 1].transAxes,
        )
        # axs[i, 1].set_ylabel(rf"\#{index}")
        axs[i, 1].grid(True)

    fig.supxlabel(r"$time(s)$", y=Y_OFFSET)

    fig.legend(
        (r"$u_{{i}}(t)$", r"$v_{{i}}(t)$"),
        loc=3,
        ncol=2,
        bbox_to_anchor=(0, 0),
        handlelength=1,
    )
    return fig


def plot_real_psf_isf(
    floq_vector: np.ndarray, phase: np.ndarray, label: str = ""
) -> Figure:
    fig, axs = plt.subplots(floq_vector.shape[1] // 4, 2, sharex=True, layout="tight")
    fig.subplots_adjust(hspace=0)
    first_half_states = floq_vector[:, : floq_vector.shape[1] // 2].real
    next_half_states = floq_vector[:, floq_vector.shape[1] // 2 :].real

    for i in range(first_half_states.shape[1] // 2):
        axs[i, 0].plot(phase, first_half_states[:, 2 * i])
        axs[i, 0].plot(phase, first_half_states[:, 2 * i + 1])
        axs[i, 0].text(
            1.00,
            0.99,
            rf"\#{i + 1}",
            ha="right",
            va="top",
            transform=axs[i, 0].transAxes,
        )
        axs[i, 0].grid(True)

    for i in range(next_half_states.shape[1] // 2):
        index = i + first_half_states.shape[1] // 2 + 1
        axs[i, 1].plot(phase, next_half_states[:, 2 * i])
        axs[i, 1].plot(phase, next_half_states[:, 2 * i + 1])
        axs[i, 1].text(
            1.00,
            0.99,
            rf"\#{index}",
            ha="right",
            va="top",
            transform=axs[i, 1].transAxes,
        )
        axs[i, 1].grid(True)

    labels = [
        "$0$",
        r"$\pi/2$",
        r"$\pi$",
        r"$3\pi/2$",
        r"$2\pi$",
    ]

    axs[-1, 0].set_xticks(np.arange(0, 2 * np.pi + 0.01, np.pi / 2))
    axs[-1, 1].set_xticklabels(labels)

    fig.supxlabel(r"$\theta$", y=Y_OFFSET)

    # for ax_col in range(axs.shape[1]):
    #     axs[-1, ax_col].set_xlabel()

    fig.legend(
        (
            rf"${label}_{{u_i}}(\theta)$",
            rf"${label}_{{v_i}}(\theta)$",
        ),
        loc=3,
        ncol=2,
        bbox_to_anchor=(0, 0),
        handlelength=1,
    )
    return fig


def plot_comp_isf(isf: np.ndarray, phase: np.ndarray, label: str = "") -> Figure:
    fig, axs = plt.subplots(isf.shape[1] // 4, 2, sharex=True, layout="tight")
    first_half_states = isf[:, : isf.shape[1] // 2]
    next_half_states = isf[:, isf.shape[1] // 2 :]

    for i in range(first_half_states.shape[1] // 2):
        axs[i, 0].plot(phase, first_half_states[:, 2 * i].real, c="tab:blue")
        axs[i, 0].plot(
            phase, first_half_states[:, 2 * i].imag, c="tab:blue", linestyle="--"
        )
        axs[i, 0].plot(phase, first_half_states[:, 2 * i + 1].real, c="tab:orange")
        axs[i, 0].plot(
            phase, first_half_states[:, 2 * i + 1].imag, c="tab:orange", linestyle="--"
        )
        axs[i, 0].text(
            1.00,
            0.99,
            rf"\#{i + 1}",
            ha="right",
            va="top",
            transform=axs[i, 0].transAxes,
        )
        axs[i, 0].grid(True)

    for i in range(next_half_states.shape[1] // 2):
        index = i + first_half_states.shape[1] // 2 + 1
        axs[i, 1].plot(phase, next_half_states[:, 2 * i].real, c="tab:blue")
        axs[i, 1].plot(
            phase, next_half_states[:, 2 * i].imag, c="tab:blue", linestyle="--"
        )
        axs[i, 1].plot(phase, next_half_states[:, 2 * i + 1].real, c="tab:orange")
        axs[i, 1].plot(
            phase, next_half_states[:, 2 * i + 1].imag, c="tab:orange", linestyle="--"
        )
        axs[i, 1].text(
            1.00,
            0.99,
            rf"\#{index}",
            ha="right",
            va="top",
            transform=axs[i, 1].transAxes,
        )
        axs[i, 1].grid(True)

    labels = [
        "$0$",
        r"$\pi/2$",
        r"$\pi$",
        r"$3\pi/2$",
        r"$2\pi$",
    ]
    for ax in axs.flat:
        ax.set_xticks(np.arange(0, 2 * np.pi + 0.01, np.pi / 2))
        ax.set_xticklabels(labels)

    fig.supxlabel(r"$\theta$", y=Y_OFFSET)

    fig.legend(
        (
            r"$\Re(I_{u_i}(\theta))$",
            r"$\Im(I_{u_i}(\theta))$",
            r"$\Re(I_{v_i}(\theta))$",
            r"$\Im(I_{v_i}(\theta))$",
        ),
        loc=3,
        ncol=4,
        bbox_to_anchor=(0, -0.02),
        framealpha=0,
        handlelength=1,
    )
    return fig


def main() -> None:
    # Loading the Random Network saved parameters
    fitz_random = OscillatorParameters.load(FitzNagumo.name)
    length_of_data_random = fitz_random.limit_cycle.shape[0]
    step_fitz_phase_random = 2 * np.pi / length_of_data_random

    # NOTE: Be carefull of the proper paper definitons:
    # NOTE: Maybe smth like this for the PSF zv_i zv_i or
    # NOTE: Maybe smth like this for the ASF Iv_{m,1} or Iu_{m,1}...
    # PSF and ASF plotting of Random network
    phase_vector_random = np.arange(0, length_of_data_random) * step_fitz_phase_random
    random_psf_fig = plot_real_psf_isf(
        fitz_random.left_floquet_vectors[0].real, phase_vector_random, label="Z"
    )

    random_isf_fig = plot_real_psf_isf(
        fitz_random.left_floquet_vectors[1].real, phase_vector_random, label="I"
    )
    time_vector = np.arange(0, length_of_data_random) * STEP_FITZ
    random_states_fig = plot_states(fitz_random.limit_cycle, time_vector)

    # Loading the Ring Network saved parameters
    fitz_ring = OscillatorParameters.load(FitzNagumoRing.name)
    length_of_data_ring = fitz_ring.limit_cycle.shape[0]
    step_fitz_phase_ring = 2 * np.pi / length_of_data_ring

    # Plottin the PSF and ISF of the Ring network
    phase_vector_ring = np.arange(0, length_of_data_ring) * step_fitz_phase_ring
    ring_psf_fig = plot_real_psf_isf(
        fitz_ring.left_floquet_vectors[0].real, phase_vector_ring, label="Z"
    )
    ring_isf_fig = plot_comp_isf(
        fitz_ring.left_floquet_vectors[1], phase_vector_ring, label="I"
    )
    time_vector = np.arange(0, length_of_data_ring) * STEP_FITZ
    ring_states_fig = plot_states(fitz_ring.limit_cycle, time_vector)

    ## COMBINED PSF
    combined_fig = plt.figure(figsize=(24, 12))
    ax1 = combined_fig.add_subplot(1, 2, 1)
    ax1.axis("off")
    ax1.text(0.0, 0.95, "a)", transform=ax1.transAxes, fontweight="bold", va="top")
    ax1.imshow(fig2img(random_psf_fig))

    ax2 = combined_fig.add_subplot(1, 2, 2)
    ax2.axis("off")
    ax2.text(0.0, 0.95, "b)", transform=ax2.transAxes, fontweight="bold", va="top")
    ax2.imshow(fig2img(random_isf_fig))
    combined_fig.savefig(BASE_DIR / "random_psf_isf.pdf", bbox_inches="tight")

    ## COMBINED ISF
    combined_fig = plt.figure(figsize=(24, 12))
    ax1 = combined_fig.add_subplot(1, 2, 1)
    ax1.axis("off")
    ax1.text(0.0, 0.95, "a)", transform=ax1.transAxes, fontweight="bold", va="top")
    ax1.imshow(fig2img(ring_psf_fig))

    ax2 = combined_fig.add_subplot(1, 2, 2)
    ax2.axis("off")
    ax2.text(0.0, 0.95, "b)", transform=ax2.transAxes, fontweight="bold", va="top")
    ax2.imshow(fig2img(ring_isf_fig))
    combined_fig.savefig(BASE_DIR / "ring_psf_isf.pdf", bbox_inches="tight")

    ## COMBINED STATES
    combined_fig = plt.figure(figsize=(24, 12))
    ax1 = combined_fig.add_subplot(1, 2, 1)
    ax1.axis("off")
    ax1.text(0.0, 0.95, "a)", transform=ax1.transAxes, fontweight="bold", va="top")
    ax1.imshow(fig2img(ring_states_fig))

    ax2 = combined_fig.add_subplot(1, 2, 2)
    ax2.axis("off")
    ax2.text(0.0, 0.95, "b)", transform=ax2.transAxes, fontweight="bold", va="top")
    ax2.imshow(fig2img(random_states_fig))
    combined_fig.savefig(BASE_DIR / "combined_states.pdf", bbox_inches="tight")


if __name__ == "__main__":
    main()
