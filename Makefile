.PHONY: help
help:
	@grep -E '^[a-zA-Z_-]+:.*?## .*$$' 'Makefile' | sort | awk 'BEGIN {FS = ":.*?## "}; {printf "\033[36m%-30s\033[0m %s\n", $$1, $$2}'
.DEFAULT_GOAL := help

.PHONY: run_checks
run_checks:  ## Run checks on all files
	poetry run ruff check .
	poetry run ruff format .

.PHONY: install
install:  ## Install dependencies using Poetry
	poetry install

.PHONY: update
update:  ## Update dependencies to latest compatible versions
	poetry update

.PHONY: lock
lock:  ## Update poetry.lock without installing
	poetry lock

.PHONY: run_simple
run_simple: export MPLBACKEND=agg
run_simple: ## Run simple simulations
	poetry run python simulation_scripts/simple_optimization/van_scaled.py
	poetry run python simulation_scripts/simple_optimization/fitz_ring.py
	poetry run python simulation_scripts/simple_optimization/fitz_random.py
	poetry run python simulation_scripts/simple_optimization/fitz_star.py

.PHONY: run_amplitude
run_amplitude: export MPLBACKEND=agg
run_amplitude: ## Run amplitude simulations
	poetry run python simulation_scripts/amplitude_optimization/van_scaled.py
	poetry run python simulation_scripts/amplitude_optimization/fitz_ring.py
	poetry run python simulation_scripts/amplitude_optimization/fitz_random.py
	poetry run python simulation_scripts/amplitude_optimization/fitz_star.py

.PHONY: run_feedback
run_feedback: export MPLBACKEND=agg
run_feedback: ## Run feedback simulations
	poetry run python simulation_scripts/feedback/van_scaled.py
	poetry run python simulation_scripts/feedback/fitz_ring.py
	poetry run python simulation_scripts/feedback/fitz_random.py
	poetry run python simulation_scripts/feedback/fitz_star.py

.PHONY: run_sine
run_sine: export MPLBACKEND=agg
run_sine: ## Run feedback simulations
	poetry run python simulation_scripts/sinusoid/fitz_ring.py
	poetry run python simulation_scripts/sinusoid/fitz_random.py
	poetry run python simulation_scripts/sinusoid/fitz_star.py

.PHONY: run_floquet
run_floquet: export MPLBACKEND=agg
run_floquet:  ## Run all of the floquet scripts
	poetry run python simulation_scripts/floquet_scripts/willamowski.py
	poetry run python simulation_scripts/floquet_scripts/fitz_star.py
	poetry run python simulation_scripts/floquet_scripts/fitz_random.py
	poetry run python simulation_scripts/floquet_scripts/fitz_ring.py
	poetry run python simulation_scripts/floquet_scripts/van_scaled.py
	poetry run python simulation_scripts/floquet_scripts/fitz_element.py

.PHONY: run_plotting
run_plotting:
	poetry run python plotting_scripts/fitz_nets_floquets.py
	poetry run python plotting_scripts/same_plot_ring.py
	poetry run python plotting_scripts/same_plot_nagumo.py

.PHONY: run_paper_plots
run_paper_plots: export MPLBACKEND=agg
run_paper_plots:  ## Generate paper plots
	poetry run python plotting_scripts/paper_plots/random_diff.py
	poetry run python plotting_scripts/paper_plots/ring_diff.py
	poetry run python plotting_scripts/paper_plots/graph.py
	poetry run python plotting_scripts/paper_plots/graph_merged.py
	poetry run python plotting_scripts/paper_plots/floquet_vectors.py
	poetry run python plotting_scripts/paper_plots/floquet_vectors_merge.py

.PHONY: run_iutam_plots
run_iutam_plots: export MPLBACKEND=agg
run_iutam_plots:  ## Generate IUTAM plots
	poetry run python plotting_scripts/iutam_plots/star_diff.py
	poetry run python plotting_scripts/iutam_plots/floquet.py
	poetry run python plotting_scripts/iutam_plots/graph_plot_iutam.py

.PHONY: run_thesis_plots
run_thesis_plots: export MPLBACKEND=agg
run_thesis_plots:  ## Generate thesis plots
	poetry run python plotting_scripts/thesis_plots/random_diff.py
	poetry run python plotting_scripts/thesis_plots/ring_diff.py

.PHONY: run_presentation_plots
run_presentation_plots: export MPLBACKEND=agg
run_presentation_plots:  ## Generate presentation plots
	poetry run python presentation_ploting_scripts/inphase_antiphase.py
	poetry run python presentation_ploting_scripts/steep_gamma.py
	poetry run python presentation_ploting_scripts/gamma_func.py

.PHONY: run_all_plots
run_all_plots: run_paper_plots run_iutam_plots run_thesis_plots run_presentation_plots  ## Generate all plots (paper, IUTAM, thesis, presentation)
