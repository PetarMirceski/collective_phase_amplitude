from config.constants import OptimizationConfig

DELTA = 0
DIFF = 1 / 8
SIMULATION_TIME = 3000.0

optimization_configurations = [
    OptimizationConfig(
        delta=DELTA,
        simulation_time=SIMULATION_TIME,
        initial_phase=DIFF,
        power=0.001,
        k=30.0,
        alpha=30.0,
        states=[1, 3, 5],
    ),
    OptimizationConfig(
        delta=DELTA,
        simulation_time=SIMULATION_TIME,
        initial_phase=DIFF,
        power=0.01,
        k=30.0,
        alpha=30.0,
        states=[1, 3, 5],
    ),
    OptimizationConfig(
        delta=DELTA,
        simulation_time=SIMULATION_TIME,
        initial_phase=DIFF,
        power=0.05,
        k=30.0,
        alpha=30.0,
        states=[1, 3, 5],
    ),
    OptimizationConfig(
        delta=DELTA,
        simulation_time=SIMULATION_TIME,
        initial_phase=DIFF,
        power=0.0005,
        k=30.0,
        alpha=30.0,
        states=[1],
    ),
    OptimizationConfig(
        delta=DELTA,
        simulation_time=SIMULATION_TIME,
        initial_phase=DIFF,
        power=0.01,
        k=30.0,
        alpha=30.0,
        states=[1],
    ),
    OptimizationConfig(
        delta=DELTA,
        simulation_time=SIMULATION_TIME,
        initial_phase=DIFF,
        power=0.05,
        k=30.0,
        alpha=30.0,
        states=[1],
    ),
]
