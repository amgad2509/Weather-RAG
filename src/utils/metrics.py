import time
from typing import Dict

class Timer:
    def __init__(self):
        self.t0 = time.perf_counter()
        self.steps: Dict[str, float] = {}

    def mark(self, name: str):
        self.steps[name] = time.perf_counter()

    def total_ms(self) -> int:
        return int((time.perf_counter() - self.t0) * 1000)

    def by_step_ms(self) -> Dict[str, int]:
        # returns deltas between marks in insertion order
        keys = list(self.steps.keys())
        if not keys:
            return {}
        out = {}
        prev = self.t0
        for k in keys:
            out[k] = int((self.steps[k] - prev) * 1000)
            prev = self.steps[k]
        return out
