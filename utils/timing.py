from dataclasses import dataclass
from time import perf_counter
from typing import Any


@dataclass
class TermColors:
    HEADER = "\033[95m"
    OKBLUE = "\033[94m"
    OKCYAN = "\033[96m"
    OKGREEN = "\033[92m"
    WARNING = "\033[93m"
    FAIL = "\033[91m"
    ENDC = "\033[0m"
    BOLD = "\033[1m"
    UNDERLINE = "\033[4m"


class ExecutionTimer:
    def __init__(self, func_name: str) -> None:
        print(f"{TermColors.HEADER}{TermColors.BOLD}{func_name}{TermColors.ENDC}")
        self.func_name = func_name

    def __enter__(self) -> Any:
        self.time = perf_counter()
        return self

    def __exit__(self, *_: Any) -> None:
        self.time = perf_counter() - self.time
        self.readout = f"Time: {self.time:.3f} seconds \n"
        print(self.readout)
