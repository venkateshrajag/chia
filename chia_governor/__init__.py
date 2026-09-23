"""
chia-governor: reusable cost-governance blocks for agentic design loops.

Three blocks that clip onto an existing CHIA loop and make it cheaper to run
without changing the result it produces:

    CostLedger      records what every node spends, per iteration
    TriageGate      blocks doomed candidates before they reach simulation
    PlateauMonitor  stops the loop once it stops paying for itself

Each is useful on its own. CostLedger is the measurement substrate the other
two use, so it is the natural first thing to adopt.
"""
from .cost_ledger import CostLedger, BudgetExceeded, Charge
from .plateau_monitor import PlateauMonitor, Point
from .triage_gate import (TriageGate, GateDecision, LocalScorer,
                          CallableScorer, SYSTEM, TMPL)

__version__ = "0.1.0"

__all__ = [
    "CostLedger", "BudgetExceeded", "Charge",
    "PlateauMonitor", "Point",
    "TriageGate", "GateDecision", "LocalScorer", "CallableScorer",
    "SYSTEM", "TMPL",
]
