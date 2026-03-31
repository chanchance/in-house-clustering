"""optuna_compat.py — Optuna drop-in replacement (random search, zero external dependencies).

Implements the subset of optuna API used across all optimization scripts:
  - create_study(direction)
  - study.optimize(objective, n_trials)
  - study.best_trial  (.params, .value, .number)
  - trial.suggest_int(name, low, high)
  - trial.suggest_float(name, low, high, log=False)
  - trial.suggest_categorical(name, choices)
  - trial.number
  - logging.set_verbosity / logging.WARNING  (no-op)

Strategy: random search with numpy RNG.
"""

from __future__ import annotations
import math
import numpy as np


# ---------------------------------------------------------------------------
# Logging stub (no-op)
# ---------------------------------------------------------------------------
class _Logging:
    WARNING = 30

    def set_verbosity(self, level: int) -> None:
        pass


logging = _Logging()


# ---------------------------------------------------------------------------
# Trial
# ---------------------------------------------------------------------------
class Trial:
    """Single trial — lazily samples parameters on first suggest call."""

    def __init__(self, rng: np.random.Generator, trial_number: int) -> None:
        self._rng = rng
        self.number = trial_number
        self.params: dict = {}

    def suggest_int(self, name: str, low: int, high: int, **kwargs) -> int:
        if name not in self.params:
            self.params[name] = int(self._rng.integers(low, high + 1))
        return self.params[name]

    def suggest_float(
        self, name: str, low: float, high: float, *, log: bool = False, **kwargs
    ) -> float:
        if name not in self.params:
            if log:
                log_val = self._rng.uniform(math.log(low), math.log(high))
                self.params[name] = float(math.exp(log_val))
            else:
                self.params[name] = float(self._rng.uniform(low, high))
        return self.params[name]

    def suggest_categorical(self, name: str, choices: list, **kwargs):
        if name not in self.params:
            idx = int(self._rng.integers(len(choices)))
            self.params[name] = choices[idx]
        return self.params[name]


# ---------------------------------------------------------------------------
# FrozenTrial (read-only result)
# ---------------------------------------------------------------------------
class FrozenTrial:
    def __init__(self, number: int, params: dict, value: float) -> None:
        self.number = number
        self.params = params
        self.value = value


# ---------------------------------------------------------------------------
# Study
# ---------------------------------------------------------------------------
class Study:
    def __init__(self, direction: str = "minimize", seed: int = 42) -> None:
        self._direction = direction
        self._seed = seed
        self.trials: list[FrozenTrial] = []
        self.best_trial: FrozenTrial | None = None
        self._best_value = float("inf") if direction == "minimize" else float("-inf")

    def _is_better(self, value: float) -> bool:
        if value != value:  # NaN
            return False
        if self._direction == "minimize":
            return value < self._best_value
        return value > self._best_value

    def optimize(self, objective, n_trials: int, n_jobs: int = 1, **kwargs) -> None:
        rng = np.random.default_rng(self._seed)
        for i in range(n_trials):
            trial = Trial(rng, i)
            try:
                value = objective(trial)
            except Exception as exc:
                print(f"[optuna_compat] Trial {i + 1} failed: {exc}")
                continue

            frozen = FrozenTrial(i, dict(trial.params), float(value))
            self.trials.append(frozen)

            if self._is_better(value):
                self._best_value = value
                self.best_trial = frozen


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------
def create_study(direction: str = "minimize", seed: int = 42, **kwargs) -> Study:
    return Study(direction=direction, seed=seed)
