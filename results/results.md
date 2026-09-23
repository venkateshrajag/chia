## Model comparison

Positive class is FAIL. Precision is the measure that matters for a gate: a wrong FAIL discards a working design.

| Model | n | Accuracy | Precision (FAIL) | Recall (FAIL) | F1 | Cost / 1k | Median latency |
|---|---|---|---|---|---|---|---|
| Qwen2.5-Coder-7B (untrained) | 630 | 61.4% | 56.1% | 73.4% | 63.6% | $0.049 | 0.25s |
| Qwen2.5-Coder-7B (fine-tuned) | 630 | 85.4% | 83.6% | 84.8% | 84.2% | $0.063 | 0.32s |
| gemini-2.5-flash-lite | 630 | 68.1% | 59.3% | 96.9% | 73.6% | $0.054 | 0.42s |
| gemini-2.5-flash | 629 | 86.2% | 77.9% | 97.6% | 86.6% | $0.163 | 6.47s |
| gemini-2.5-pro | 626 | 81.5% | 71.5% | 99.7% | 83.2% | $0.676 | 14.40s |
| gemini-3.1-pro-preview | 0 | *all calls errored* | | | | | |
| gemini-3.5-flash | 0 | *all calls errored* | | | | | |
| gemini-3.8-flash | 0 | *all calls errored* | | | | | |

## Gate economics

Of the candidates that would really pass, how many does each gate throw away, and how many simulations does it save?

| Model | Simulations skipped | Good designs discarded | Share of good designs lost |
|---|---|---|---|
| Qwen2.5-Coder-7B (untrained) | 378 of 630 | 166 of 341 | 48.7% |
| Qwen2.5-Coder-7B (fine-tuned) | 293 of 630 | 48 of 341 | 14.1% |
| gemini-2.5-flash-lite | 472 of 630 | 192 of 341 | 56.3% |
| gemini-2.5-flash | 362 of 629 | 80 of 340 | 23.5% |
| gemini-2.5-pro | 403 of 626 | 115 of 337 | 34.1% |

## Threshold sweep (local models)

A gate is tunable. Raising the bar before it blocks discards fewer good designs but saves fewer simulations.


**Qwen2.5-Coder-7B (untrained)**

| Threshold | Accuracy | Precision (FAIL) | Recall (FAIL) | Blocked | Good lost |
|---|---|---|---|---|---|
| 0.30 | 60.2% | 54.4% | 81.7% | 434 | 198 |
| 0.50 | 61.4% | 56.1% | 73.4% | 378 | 166 |
| 0.70 | 63.5% | 58.9% | 67.5% | 331 | 136 |
| 0.80 | 63.0% | 59.2% | 62.3% | 304 | 124 |
| 0.90 | 63.7% | 61.4% | 56.1% | 264 | 102 |
| 0.95 | 64.8% | 69.1% | 41.9% | 175 | 54 |

**Qwen2.5-Coder-7B (fine-tuned)**

| Threshold | Accuracy | Precision (FAIL) | Recall (FAIL) | Blocked | Good lost |
|---|---|---|---|---|---|
| 0.30 | 82.5% | 75.4% | 92.0% | 353 | 87 |
| 0.50 | 85.2% | 83.3% | 84.8% | 294 | 49 |
| 0.70 | 83.7% | 86.9% | 75.8% | 252 | 33 |
| 0.80 | 81.6% | 87.8% | 69.6% | 229 | 28 |
| 0.90 | 80.2% | 92.7% | 61.6% | 192 | 14 |
| 0.95 | 77.0% | 95.0% | 52.6% | 160 | 8 |

## Accuracy by edit family

`benign` edits preserve behaviour (renames, reorderings, redundant parentheses, reformatted literals). `faulty` edits change it. This is where the fine-tune earned its keep.

| Model | benign | faulty | none |
|---|---|---|---|
| Qwen2.5-Coder-7B (untrained) | 55.6% | 65.6% | 63.2% |
| Qwen2.5-Coder-7B (fine-tuned) | 94.8% | 79.4% | 76.3% |
| gemini-2.5-flash-lite | 49.2% | 83.5% | 55.3% |
| gemini-2.5-flash | 83.7% | 88.5% | 81.6% |
| gemini-2.5-pro | 74.5% | 86.9% | 78.9% |

## PlateauMonitor on the published CHIA critical-path run

The published run took 14 iterations at $14.43 each ($202.03) and lifted frequency from 47 MHz to 95 MHz, but stopped improving after iteration 8. `patience` is how many flat iterations to tolerate before stopping; lower is more aggressive.

| Patience | Would stop at | Spend | Saved | Final result |
|---|---|---|---|---|
| 1 | iteration 9 of 14 | $129.87 | $72.15 (35.7%) | 95 MHz |
| 2 | iteration 10 of 14 | $144.30 | $57.72 (28.6%) | 95 MHz |
| 3 | iteration 11 of 14 | $158.73 | $43.29 (21.4%) | 95 MHz |

In every case the final result is unchanged at 95 MHz, because the iterations it skips are the ones that produced nothing.
