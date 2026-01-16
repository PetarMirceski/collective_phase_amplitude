from config.constants import OptimizationConfig

DELTA = 0
DIFF = 1 / 8
SIMULATION_TIME = 1200.0
# SIMULATION_TIME = 300.0

optimization_configurations = [
    OptimizationConfig(
        delta=DELTA,
        simulation_time=SIMULATION_TIME,
        initial_phase=DIFF,
        power=0.001,
        k=10.0,
        alpha=0.1,
        states=[1],
    ),
    OptimizationConfig(
        delta=DELTA,
        simulation_time=SIMULATION_TIME,
        initial_phase=DIFF,
        power=0.01,
        k=10.0,
        alpha=10.0,
        states=[1],
    ),
]
