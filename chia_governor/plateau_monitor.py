"""
PlateauMonitor - stop a loop once it stops paying for itself.

The CHIA paper's critical-path study ran 14 iterations at about $14.43 each
but had already plateaued after the seventh, so roughly half of its $202 was
spent after the result stopped moving. Nothing in the framework noticed.

This block watches how much the objective improves per dollar spent and stops
the loop when that stays below a caller-set threshold for several iterations
in a row. Patience matters: a single flat round is noise, three in a row is a
plateau.

    mon = PlateauMonitor(min_gain_per_usd=0.05, patience=3, mode="maximize")
    for i in range(max_iters):
        score = run_iteration(i)
        mon.record(i, score, cost_usd=ledger.iteration_cost(i))
        if mon.should_stop():
            print(mon.explain()); break
"""
from __future__ import annotations

from dataclasses import dataclass


@dataclass
class Point:
    iteration: int
    score: float
    cost_usd: float
    best_so_far: float
    gain: float             # improvement in the best score this iteration
    gain_per_usd: float


class PlateauMonitor:
    def __init__(self, min_gain_per_usd: float = 0.0, patience: int = 3,
                 mode: str = "maximize", min_iterations: int = 3,
                 max_spend_usd: float | None = None):
        """
        min_gain_per_usd : improvement per dollar below which an iteration
                           counts as unproductive. Units are the caller's
                           objective, so set it from what a point is worth.
        patience         : consecutive unproductive iterations before stopping.
        mode             : "maximize" or "minimize" the objective.
        min_iterations   : never stop before this many, so a slow start is not
                           mistaken for a plateau.
        max_spend_usd    : optional hard ceiling; stop regardless once passed.
        """
        if mode not in ("maximize", "minimize"):
            raise ValueError("mode must be 'maximize' or 'minimize'")
        self.min_gain_per_usd = min_gain_per_usd
        self.patience = patience
        self.mode = mode
        self.min_iterations = min_iterations
        self.max_spend_usd = max_spend_usd

        self.history: list[Point] = []
        self._best: float | None = None
        self._flat_run = 0
        self._stop_reason: str | None = None

    # ---------------------------------------------------------------- record

    def _is_better(self, score: float, best: float) -> bool:
        return score > best if self.mode == "maximize" else score < best

    def record(self, iteration: int, score: float, cost_usd: float) -> Point:
        if self._best is None:
            gain = 0.0
            self._best = score
        elif self._is_better(score, self._best):
            gain = abs(score - self._best)
            self._best = score
        else:
            gain = 0.0

        gpu = gain / cost_usd if cost_usd > 0 else (gain if gain else 0.0)
        p = Point(iteration=iteration, score=score, cost_usd=cost_usd,
                  best_so_far=self._best, gain=gain, gain_per_usd=gpu)
        self.history.append(p)

        if gpu <= self.min_gain_per_usd:
            self._flat_run += 1
        else:
            self._flat_run = 0
        return p

    # ---------------------------------------------------------------- decide

    def should_stop(self) -> bool:
        if self.max_spend_usd is not None and self.total_spend >= self.max_spend_usd:
            self._stop_reason = (
                f"spend ${self.total_spend:.2f} reached the ceiling of "
                f"${self.max_spend_usd:.2f}")
            return True
        if len(self.history) < self.min_iterations:
            return False
        if self._flat_run >= self.patience:
            self._stop_reason = (
                f"{self._flat_run} consecutive iterations returned at most "
                f"{self.min_gain_per_usd} improvement per dollar")
            return True
        return False

    @property
    def total_spend(self) -> float:
        return sum(p.cost_usd for p in self.history)

    @property
    def stop_reason(self) -> str | None:
        return self._stop_reason

    def wasted_since_plateau(self) -> float:
        """Dollars spent after the last iteration that actually improved."""
        last_gain = None
        for p in self.history:
            if p.gain > 0:
                last_gain = p.iteration
        if last_gain is None:
            return self.total_spend
        return sum(p.cost_usd for p in self.history if p.iteration > last_gain)

    def explain(self) -> str:
        lines = [f"stopping after {len(self.history)} iterations"]
        if self._stop_reason:
            lines.append(f"reason: {self._stop_reason}")
        lines.append(f"best score: {self._best}")
        lines.append(f"total spend: ${self.total_spend:.2f}")
        w = self.wasted_since_plateau()
        if w > 0:
            lines.append(f"spent after the last real improvement: ${w:.2f}")
        lines.append("")
        lines.append(f"{'iter':>5} {'score':>10} {'best':>10} {'cost':>9} {'gain/$':>10}")
        for p in self.history:
            lines.append(f"{p.iteration:5d} {p.score:10.4f} {p.best_so_far:10.4f} "
                         f"{p.cost_usd:9.3f} {p.gain_per_usd:10.4f}")
        return "\n".join(lines)

    def counterfactual(self) -> str:
        """
        What this monitor would have saved on a completed run. Feed it a full
        history and it reports where it would have stopped.
        """
        sim = PlateauMonitor(self.min_gain_per_usd, self.patience, self.mode,
                             self.min_iterations, self.max_spend_usd)
        stop_at = None
        for p in self.history:
            sim.record(p.iteration, p.score, p.cost_usd)
            if stop_at is None and sim.should_stop():
                stop_at = p.iteration
        if stop_at is None:
            return "would not have stopped early on this history"
        spent = sum(p.cost_usd for p in self.history if p.iteration <= stop_at)
        saved = self.total_spend - spent
        pct = 100 * saved / self.total_spend if self.total_spend else 0
        final_best = self.history[-1].best_so_far
        best_at_stop = next(p.best_so_far for p in self.history
                            if p.iteration == stop_at)
        return (f"would have stopped at iteration {stop_at} of "
                f"{self.history[-1].iteration}\n"
                f"  spend:  ${spent:.2f} instead of ${self.total_spend:.2f} "
                f"(saved ${saved:.2f}, {pct:.1f}%)\n"
                f"  result: {best_at_stop} instead of {final_best}")
