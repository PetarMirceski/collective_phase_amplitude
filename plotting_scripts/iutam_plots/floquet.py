import matplotlib.pyplot as plt
import numpy as np
from matplotlib.figure import Figure

from config.constants import STEP_FITZ, OscillatorParameters
from config.oscillator_constants import FitzNagumoStar
from config.paths import FIGURE_PATH
from solvers.floquet import check_orthogonality

BASE_DIR = FIGURE_PATH / "paper" / "floquets_and_limit_cycles/"
BASE_DIR.mkdir(exist_ok=True, parents=True)


# plt.matplotlib.use("Agg")
Y_OFFSET = 0.06
plt.rcParams.update(
    {
        "figure.labelsize": 50,
    }
)


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
        (r"$\bm{u_{{i}}(t)}$", r"$\bm{v_{{i}}(t)}$"),
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

    fig.supxlabel(r"$\bm \theta$", y=Y_OFFSET)

    # for ax_col in range(axs.shape[1]):
    #     axs[-1, ax_col].set_xlabel()

    fig.legend(
        (
            rf"$\bm{{ {label}_{{u_i}}(\theta)}}$",
            rf"$\bm{{ {label}_{{v_i}}(\theta)}}$",
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

    fig.supxlabel(r"$\bm \theta$", y=Y_OFFSET)

    fig.legend(
        (
            r"$\bm{\Re(I_{u_i}(\theta))}$",
            r"$\bm{\Im(I_{u_i}(\theta))}$",
            r"$\bm{\Re(I_{v_i}(\theta))}$",
            r"$\bm{\Im(I_{v_i}(\theta))}$",
        ),
        loc=3,
        ncol=4,
        bbox_to_anchor=(0, -0.02),
        framealpha=0,
        handlelength=1,
    )
    return fig


def main() -> None:
    # Loading the Star Network saved parameters
    fitz_data = OscillatorParameters.load(FitzNagumoStar.name)
    length_of_data = fitz_data.limit_cycle.shape[0]
    step_fitz_phase = 2 * np.pi / length_of_data

    # Plottin the PSF and ISF of the Ring network
    phase_vector = np.arange(0, length_of_data) * step_fitz_phase
    plot_real_psf_isf(fitz_data.left_floquet_vectors[0].real, phase_vector, label="Z")
    plt.savefig(BASE_DIR / "star_net_psf.pdf", bbox_inches="tight")
    plt.close()

    plot_real_psf_isf(fitz_data.left_floquet_vectors[1], phase_vector, label="I")
    plt.savefig(BASE_DIR / "star_net_isf.pdf", bbox_inches="tight")
    plt.close()

    # Limit Cycle plotting of Random network
    time_vector = np.arange(0, length_of_data) * STEP_FITZ
    plot_states(fitz_data.limit_cycle, time_vector)
    plt.savefig(BASE_DIR / "star_net_limit_cycle.pdf", bbox_inches="tight")
    plt.close()

    print("Report for the Star Network")
    print(f"The time period T is: {fitz_data.natural_period}")
    print(f"The freq W is: {fitz_data.natural_freq}")

    for idx, exponent in enumerate(fitz_data.floquet_exponents):
        print(f"The {idx}-th  floquet exponent has value: {exponent}")

    is_orthogonal = check_orthogonality(
        fitz_data.left_floquet_vectors,
        fitz_data.right_floquet_vectors,
    )
    print(f"Orthogonality check: {'PASSED' if is_orthogonal else 'FAILED'}")


if __name__ == "__main__":
    main()
