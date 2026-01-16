from config.constants import OptimizationConfig

DELTA = 0
DIFF = 1 / 30
SIMULATION_TIME = 4000.0
SIMULATION_TIME_1 = 4000.0

optimization_configuration = [
    OptimizationConfig(
        delta=DELTA,
        simulation_time=SIMULATION_TIME,
        initial_phase=DIFF,
        power=1e-5,
        k=20.0,
        alpha=200.0,
        states=[1, 3, 5],
    ),
    # OptimizationConfig(
    #     delta=DELTA,
    #     simulation_time=SIMULATION_TIME,
    #     initial_phase=DIFF,
    #     power=1e-4,
    #     k=20.0,
    #     alpha=500.0,
    #     states=[1, 3, 5],
    # ),
    OptimizationConfig(
        delta=DELTA,
        simulation_time=SIMULATION_TIME,
        initial_phase=DIFF,
        power=5e-4,
        k=20.0,
        alpha=200.0,
        states=[1, 3, 5],
    ),
    OptimizationConfig(
        delta=DELTA,
        simulation_time=SIMULATION_TIME_1,
        initial_phase=DIFF,
        power=5e-5,
        k=20.0,
        alpha=200.0,
        states=[1],
    ),
    OptimizationConfig(
        delta=DELTA,
        simulation_time=SIMULATION_TIME_1,
        initial_phase=DIFF,
        power=5e-4,
        k=20.0,
        alpha=200.0,
        states=[1],
    ),
]
