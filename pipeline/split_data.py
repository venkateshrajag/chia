#!/usr/bin/env python3
"""
split_data.py - Turn labeled mutants into train/test sets, split BY PROBLEM.

Splitting by problem (not by example) is the thing that makes the headline
result honest: the model is always evaluated on VerilogEval problems it has
never seen in any form, so a good score cannot come from memorising a module.

Drops COMPILE_ERROR and TIMEOUT: the compiler already catches those for free,
so they are not what the gate exists to predict.
"""
import argparse, json, random
from collections import Counter
from pathlib import Path


def load_prompt(dataset, rec):
    if not rec.get("prompt_file"):
        return ""
    p = Path(dataset) / rec["prompt_file"]
    return p.read_text().strip() if p.exists() else ""


def stats(name, rows):
    c = Counter(r["label"] for r in rows)
    n = len(rows)
    if not n:
        print(f"{name}: EMPTY")
        return
    fail = c.get("FAIL", 0)
    majority = max(c.values()) / n
    print(f"{name:6s} n={n:5d}  PASS={c.get('PASS',0):5d}  FAIL={fail:5d}"
          f"  FAIL share={100*fail/n:5.1f}%  majority-class baseline={100*majority:5.1f}%")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--labeled", default="labeled_v3.jsonl")
    ap.add_argument("--dataset", required=True)
    ap.add_argument("--train", default="train.jsonl")
    ap.add_argument("--test", default="test.jsonl")
    ap.add_argument("--test-frac", type=float, default=0.25)
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()

    rows = [json.loads(l) for l in open(args.labeled)]
    usable = [r for r in rows if r["label"] in ("PASS", "FAIL")]
    print(f"read {len(rows)} rows, {len(usable)} usable (PASS/FAIL)")

    problems = sorted({r["problem"] for r in usable})
    rng = random.Random(args.seed)
    rng.shuffle(problems)
    n_test = max(1, int(len(problems) * args.test_frac))
    test_probs = set(problems[:n_test])
    print(f"problems: {len(problems)} total -> {len(test_probs)} held out for test")

    # attach the natural-language spec each model will be shown
    cache = {}
    for r in usable:
        pf = r.get("prompt_file")
        if pf not in cache:
            cache[pf] = load_prompt(args.dataset, r)
        r["spec"] = cache[pf]

    train = [r for r in usable if r["problem"] not in test_probs]
    test = [r for r in usable if r["problem"] in test_probs]

    for path, rows_ in ((args.train, train), (args.test, test)):
        with open(path, "w") as fh:
            for r in rows_:
                fh.write(json.dumps(r) + "\n")

    print()
    stats("TRAIN", train)
    stats("TEST", test)

    overlap = {r["problem"] for r in train} & {r["problem"] for r in test}
    print(f"\nproblem overlap between train and test: {len(overlap)}  "
          f"({'OK' if not overlap else 'LEAK - INVESTIGATE'})")

    print("\nper-family in test:")
    for fam, n in Counter(r.get("family", "?") for r in test).most_common():
        sub = [r for r in test if r.get("family") == fam]
        fr = 100 * sum(1 for r in sub if r["label"] == "FAIL") / len(sub)
        print(f"  {fam:8s} n={len(sub):5d}  FAIL={fr:5.1f}%")

    print(f"\nwrote {args.train} and {args.test}")


if __name__ == "__main__":
    main()
