# Dataset

Verilog candidates whose PASS/FAIL labels come from actually running
VerilogEval test benches under Icarus Verilog. Nothing here is a guess.

| File | Contents |
|---|---|
| `labeled_v3.jsonl` | 3,171 candidates, 2,571 of which compile. Final dataset, 44.8% FAIL. |
| `train.jsonl` | 1,941 examples, spec attached |
| `test.jsonl` | 630 examples, from problems held out entirely from training |
| `labeled.jsonl` | First attempt, faulty edits only. 79.7% FAIL, unusable. |
| `labeled_v2.jsonl` | Intermediate, after adding benign edits |

The train/test split is **by problem**, so no VerilogEval problem appears on
both sides. `labeled.jsonl` is kept deliberately: it shows why the benign
operators were necessary. A set that is 79.7% FAIL can be scored at 80% by a
model that always answers FAIL.

Fields: `id`, `problem`, `family` (faulty/benign/none), `op`, `detail`,
`code`, `label`, `mismatches`, `samples`, `spec`.

Regenerate from scratch:

```bash
git clone https://github.com/NVlabs/verilog-eval.git
python3 ../pipeline/mutate2.py --dataset verilog-eval/dataset_spec-to-rtl \
  --out mutants.jsonl --per-problem 40 --benign-frac 0.5 --max-per-op 4 --seed 1
python3 ../pipeline/run_tests.py --dataset verilog-eval/dataset_spec-to-rtl \
  --mutants mutants.jsonl --out labeled_v3.jsonl --workers 32
```
