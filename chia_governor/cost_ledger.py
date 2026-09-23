"""
CostLedger - per-node, per-iteration accounting of what a CHIA loop spends.

This is the measurement substrate the other two blocks depend on, and it is
useful on its own to anyone running a loop against a fixed budget. It records
every charge with the node that caused it and the iteration it belongs to, so
you can answer "where did the money actually go" instead of guessing.

Usage:

    ledger = CostLedger(budget_usd=300.0)

    with ledger.track("llm_propose", iteration=3) as t:
        resp = call_model(...)
        t.llm(in_tok=resp.in_tok, out_tok=resp.out_tok,
              price_in=0.30, price_out=2.50)

    with ledger.track("rtl_sim", iteration=3) as t:
        run_simulation(...)
        t.compute(hourly_usd=0.70)      # billed from elapsed wall-clock

    print(ledger.summary())
    ledger.save("ledger.jsonl")
"""
from __future__ import annotations

import json
import time
from collections import defaultdict
from contextlib import contextmanager
from dataclasses import dataclass, field, asdict
from typing import Optional


class BudgetExceeded(RuntimeError):
    """Raised when a charge would push total spend past the ledger's budget."""


@dataclass
class Charge:
    node: str
    iteration: int
    kind: str                 # "llm" | "compute" | "other"
    usd: float
    seconds: float = 0.0
    in_tok: int = 0
    out_tok: int = 0
    note: str = ""
    at: float = field(default_factory=time.time)


class _Tracker:
    """Handed to the caller inside `with ledger.track(...)`."""

    def __init__(self, ledger: "CostLedger", node: str, iteration: int):
        self._ledger = ledger
        self._node = node
        self._iteration = iteration
        self._t0 = time.time()
        self._charged = False

    @property
    def elapsed(self) -> float:
        return time.time() - self._t0

    def llm(self, in_tok: int, out_tok: int,
            price_in: float, price_out: float, note: str = "") -> float:
        """Charge model usage. Prices are USD per 1M tokens."""
        usd = in_tok / 1e6 * price_in + out_tok / 1e6 * price_out
        self._ledger._add(Charge(
            node=self._node, iteration=self._iteration, kind="llm", usd=usd,
            seconds=self.elapsed, in_tok=in_tok, out_tok=out_tok, note=note))
        self._charged = True
        return usd

    def compute(self, hourly_usd: float, seconds: Optional[float] = None,
                note: str = "") -> float:
        """Charge wall-clock time on a machine billed by the hour."""
        secs = self.elapsed if seconds is None else seconds
        usd = hourly_usd * secs / 3600.0
        self._ledger._add(Charge(
            node=self._node, iteration=self._iteration, kind="compute",
            usd=usd, seconds=secs, note=note))
        self._charged = True
        return usd

    def flat(self, usd: float, note: str = "") -> float:
        """Charge a fixed amount (a licence, a fixed-price API call)."""
        self._ledger._add(Charge(
            node=self._node, iteration=self._iteration, kind="other",
            usd=usd, seconds=self.elapsed, note=note))
        self._charged = True
        return usd


class CostLedger:
    def __init__(self, budget_usd: Optional[float] = None,
                 enforce: bool = False):
        """
        budget_usd : optional ceiling for the whole run.
        enforce    : if True, a charge that would exceed the budget raises
                     BudgetExceeded instead of merely being recorded.
        """
        self.budget_usd = budget_usd
        self.enforce = enforce
        self.charges: list[Charge] = []

    # ---------------------------------------------------------------- record

    def _add(self, c: Charge) -> None:
        if (self.enforce and self.budget_usd is not None
                and self.total + c.usd > self.budget_usd):
            raise BudgetExceeded(
                f"{c.node} would spend ${c.usd:.4f}, taking the run to "
                f"${self.total + c.usd:.2f} against a budget of "
                f"${self.budget_usd:.2f}")
        self.charges.append(c)

    @contextmanager
    def track(self, node: str, iteration: int = 0):
        t = _Tracker(self, node, iteration)
        try:
            yield t
        finally:
            if not t._charged:
                # the caller forgot to charge; record the time so the
                # iteration's wall-clock is still complete
                self._add(Charge(node=node, iteration=iteration, kind="other",
                                 usd=0.0, seconds=t.elapsed,
                                 note="untracked (no charge recorded)"))

    # ---------------------------------------------------------------- query

    @property
    def total(self) -> float:
        return sum(c.usd for c in self.charges)

    @property
    def remaining(self) -> Optional[float]:
        return None if self.budget_usd is None else self.budget_usd - self.total

    def by_node(self) -> dict[str, float]:
        out: dict[str, float] = defaultdict(float)
        for c in self.charges:
            out[c.node] += c.usd
        return dict(sorted(out.items(), key=lambda kv: -kv[1]))

    def by_iteration(self) -> dict[int, float]:
        out: dict[int, float] = defaultdict(float)
        for c in self.charges:
            out[c.iteration] += c.usd
        return dict(sorted(out.items()))

    def iteration_cost(self, iteration: int) -> float:
        return sum(c.usd for c in self.charges if c.iteration == iteration)

    def summary(self) -> str:
        lines = [f"total spend  ${self.total:.4f}"]
        if self.budget_usd is not None:
            pct = 100 * self.total / self.budget_usd if self.budget_usd else 0
            lines.append(f"budget       ${self.budget_usd:.2f} "
                         f"({pct:.1f}% used, ${self.remaining:.2f} left)")
        lines.append("")
        lines.append("by node:")
        for node, usd in self.by_node().items():
            share = 100 * usd / self.total if self.total else 0
            lines.append(f"  {node:24s} ${usd:9.4f}  ({share:5.1f}%)")
        lines.append("")
        lines.append("by iteration:")
        for it, usd in self.by_iteration().items():
            lines.append(f"  {it:4d}  ${usd:9.4f}")
        return "\n".join(lines)

    # ---------------------------------------------------------------- io

    def save(self, path: str) -> None:
        with open(path, "w") as fh:
            for c in self.charges:
                fh.write(json.dumps(asdict(c)) + "\n")

    @classmethod
    def load(cls, path: str, **kw) -> "CostLedger":
        led = cls(**kw)
        with open(path) as fh:
            for line in fh:
                led.charges.append(Charge(**json.loads(line)))
        return led
