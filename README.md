# chia-governor

Reusable cost-governance blocks for agentic hardware design loops, built for the
CHIA framework.

A CHIA loop repeats a simple cycle: an AI agent proposes a change to a design,
tools evaluate that change, and the loop decides whether to keep going. The
published CHIA results show that much of the money goes to rounds that add
nothing. The critical-path study ran 14 iterations at about $14.43 each but had
already stopped improving after the eighth, so roughly half of its $202 bought
nothing. The gem5-alignment study ran 202 iterations for $2,355.98.

CHIA is excellent at letting you *express* these loops. It has nothing reusable
for *governing* what they cost. These three blocks fill that gap, and none of
them changes the result a loop produces.

| Block | What it does |
|---|---|
| `CostLedger` | Records what every node spends, per iteration, so you can see where the money went |
| `TriageGate` | Blocks candidates that will fail before they reach the simulator |
| `PlateauMonitor` | Stops the loop once improvement per dollar flatlines |

`CostLedger` is the measurement substrate the other two rely on, and it is
useful on its own to anyone working against a fixed budget.

## Install

```bash
pip install -e .
```

Only `TriageGate`'s `LocalScorer` needs `torch`, `transformers` and `peft`. The
ledger and the monitor are pure Python with no dependencies.

## Quick start

```python
from chia_governor import CostLedger, TriageGate, PlateauMonitor, LocalScorer

ledger = CostLedger(budget_usd=300.0)

gate = TriageGate(
    LocalScorer("Qwen/Qwen2.5-Coder-7B-Instruct",
                adapter="./adapter_7b", load_4bit=True),
    threshold=0.9,               # block only when quite sure it will fail
    ledger=ledger, hourly_usd=0.70,
)

monitor = PlateauMonitor(min_gain_per_usd=0.05, patience=3, mode="maximize")

for i in range(max_iterations):
    spec, code = agent.propose()

    if not gate.allow(spec, code, iteration=i):
        continue                 # skipped a simulation we would have wasted

    with ledger.track("rtl_sim", iteration=i) as t:
        score = run_simulation(code)
        t.compute(hourly_usd=0.70)

    monitor.record(i, score, cost_usd=ledger.iteration_cost(i))
    if monitor.should_stop():
        print(monitor.explain())
        break

print(ledger.summary())
print(gate.stats())
```

## The gate is tunable, and that matters

`TriageGate` returns a probability rather than a verdict, because the right
operating point depends on what a wasted simulation costs you relative to a
discarded good design. Measured on 630 held-out examples with the shipped
fine-tuned model:

| Threshold | Accuracy | Precision (FAIL) | Recall (FAIL) | Good designs lost |
|---|---|---|---|---|
| 0.50 | 85.2% | 83.3% | 84.8% | 49 of 341 |
| 0.70 | 83.7% | 86.9% | 75.8% | 33 of 341 |
| 0.90 | 80.2% | 92.7% | 61.6% | 14 of 341 |
| 0.95 | 77.0% | 95.0% | 52.6% | 8 of 341 |

**Precision on FAIL is the measure to watch.** A wrong FAIL silently discards a
working design, which is worse for a design loop than paying for one more
simulation. Recall tells you how many simulations you save. The default
threshold of 0.9 is deliberately conservative.

## Results

The gate ships with a LoRA adapter for `Qwen2.5-Coder-7B-Instruct`, trained on
2,571 Verilog candidates whose PASS/FAIL labels come from actually running
VerilogEval test benches. Evaluated on 630 examples drawn from problems held
out entirely from training:

| Model | Accuracy | Precision (FAIL) | Recall (FAIL) | Cost / 1k | Median latency |
|---|---|---|---|---|---|
| Qwen2.5-Coder-7B, untrained | 61.4% | 56.1% | 73.4% | $0.049 | 0.25s |
| **Qwen2.5-Coder-7B, fine-tuned** | 85.2% | **83.3%** | 84.8% | $0.063 | 0.32s |
| Gemini 2.5 Flash | 86.2% | 77.9% | 97.6% | $0.163 | 6.5s |
| Gemini 2.5 Pro | 81.5% | 71.5% | 99.7% | $0.676 | 14.4s |
| Gemini 2.5 Flash-Lite | 68.1% | 59.3% | 96.9% | $0.054 | 0.4s |

The fine-tuned 7B has the best precision of any model tested, at roughly a
quarter of Flash's cost and a twentieth of its latency. Flash retains clearly
better recall, and slightly better raw accuracy.

### Why the fine-tune helped

Accuracy split by the kind of edit made to the design:

| | Untrained | Fine-tuned |
|---|---|---|
| **benign** edits (behaviour preserved) | 55.6% | **94.8%** |
| **faulty** edits (behaviour changed) | 65.6% | 79.4% |

The untrained model was at chance on harmless edits. It assumed any change to
working code was a bug. Every frontier model showed the same bias, which is why
their recall is near-perfect while their precision is poor. Training on data
that contains *verified-harmless* edits, alongside real faults, closes that gap.

## Building your own training data

The dataset is generated rather than collected, with labels correct by
construction. Take working RTL, apply a scripted edit, then run the real test
bench to find out what actually happened. Two families of edits:

- **faulty**: flipped comparisons, swapped operators, off-by-one constants,
  altered bit ranges, changed reset values, non-blocking to blocking, dropped
  assignments, swapped signals
- **benign**: commutative operand swaps, consistent internal renames, redundant
  parentheses, reformatted literals, reordered independent assignments, dead
  wire declarations

Both families are needed. A first pass using only faulty edits produced a set
that was 79.7% FAIL, on which a model can score 80% by always answering FAIL.
Adding the benign family brought it to 44.8% FAIL and made the task meaningful.

## Reproducing the tables

```bash
python3 make_results.py --dir /path/to/predictions > results.md
```

Reads `preds_base.jsonl`, `preds_tuned.jsonl` and `preds_gemini-*.jsonl` and
emits the model comparison, gate economics, threshold sweeps, per-family
accuracy, and a counterfactual showing what `PlateauMonitor` would have saved
on CHIA's own published critical-path run.

## Licence

BSD-3-Clause, matching CHIA.
