# Phase-Amplitude Reduction and Optimal Phase Locking of Collectively Oscillating Networks

This repository contains the code for the paper:

> **Mircheski, P., Zhu, J., & Nakao, H.** (2023). Phase-amplitude reduction and optimal phase locking of collectively oscillating networks. *Chaos: An Interdisciplinary Journal of Nonlinear Science*, 33(10). [DOI: 10.1063/5.0161119](https://doi.org/10.1063/5.0161119)

## Overview

This codebase implements methods for:

- **Floquet analysis** of limit cycle oscillators (computing Floquet exponents and vectors)
- **Phase sensitivity functions (PSF)** and **isostable sensitivity functions (ISF)**
- **Optimal phase-locking** of oscillator networks with minimal input power
- **Phase-amplitude optimal control** that minimizes amplitude deviations
- **Feedback control** for robust phase synchronization

## Oscillator Networks

The code includes several pre-configured oscillator networks:

| Network | Description | Dimension |
|---------|-------------|-----------|
| `VanDerPolScaled` | Single Van der Pol oscillator | 2 |
| `FitzNagumo` | 10-oscillator random coupling network | 20 |
| `FitzNagumoRing` | 10-oscillator ring topology | 20 |
| `FitzNagumoStar` | 4-oscillator star topology | 8 |

## Installation

This project uses [Poetry](https://python-poetry.org/) for dependency management.

### Prerequisites

- Python 3.12 or higher
- Poetry

### Install Poetry

```bash
# Install Poetry (recommended method)
curl -sSL https://install.python-poetry.org | python3 -

# Or using pipx
pipx install poetry
```

### Install Dependencies

```bash
# Clone the repository
git clone https://github.com/yourusername/collective_phase_amplitude.git
cd collective_phase_amplitude

# Install all dependencies
poetry install
```

### Activate Virtual Environment

```bash
# Option 1: Spawn a shell within the virtual environment
poetry shell

# Option 2: Run commands directly with poetry run
poetry run python script.py
```

## Usage

### Running Floquet Analysis

Compute Floquet vectors and exponents for an oscillator network:

```bash
# Run Floquet analysis for the star network
poetry run python simulation_scripts/floquet_scripts/fitz_star.py

# Run all Floquet analyses
make run_floquet
```

### Running Optimal Entrainment Simulations

```bash
# Phase-only optimization
make run_simple

# Phase + amplitude optimization
make run_amplitude

# Feedback control
make run_feedback

# Sinusoidal perturbations
make run_sine
```

### Generating Plots

```bash
# Generate paper figures
make run_paper_plots

# Generate all plots
make run_all_plots
```

## Makefile Targets

Run `make help` to see all available targets:

| Target | Description |
|--------|-------------|
| `make install` | Install dependencies using Poetry |
| `make update` | Update dependencies to latest versions |
| `make run_checks` | Run code quality checks (ruff) |
| `make run_floquet` | Run all Floquet analysis scripts |
| `make run_simple` | Run phase-only optimization |
| `make run_amplitude` | Run phase + amplitude optimization |
| `make run_feedback` | Run feedback control simulations |
| `make run_sine` | Run sinusoidal perturbation simulations |
| `make run_paper_plots` | Generate publication figures |
| `make run_all_plots` | Generate all plots |

## Citation

If you use this code in your research, please cite:

```bibtex
@article{mircheski2023phase,
  title={Phase-amplitude reduction and optimal phase locking of collectively oscillating networks},
  author={Mircheski, Petar and Zhu, Jinjie and Nakao, Hiroya},
  journal={Chaos: An Interdisciplinary Journal of Nonlinear Science},
  volume={33},
  number={10},
  year={2023},
  publisher={AIP Publishing},
  doi={10.1063/5.0161119}
}
```

## License

MIT License - see [LICENSE](LICENSE) for details.
