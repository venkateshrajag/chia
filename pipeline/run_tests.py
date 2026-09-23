#!/usr/bin/env python3
"""
run_tests.py - Compile and simulate every mutant, label it from the real result.

Label per mutant:
  PASS          compiles, "Mismatches: 0 in N samples"
  FAIL          compiles, but one or more mismatches
  COMPILE_ERROR iverilog refused it (dropped later: the compiler catches these free)
  TIMEOUT       simulation did not finish in time

Reads  mutants.jsonl
Writes labeled.jsonl
"""
import argparse, json, os, re, shutil, subprocess, sys, tempfile
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path

MISMATCH_RE = re.compile(r"Mismatches:\s*(\d+)\s+in\s+(\d+)\s+samples")

DATASET = None
TIMEOUT = 30


def init(dataset, timeout):
    global DATASET, TIMEOUT
    DATASET = Path(dataset)
    TIMEOUT = timeout


def evaluate(rec):
    """Compile ref + candidate + testbench, run, parse the mismatch line."""
    work = tempfile.mkdtemp(prefix="mut_")
    try:
        cand = Path(work) / "cand.sv"
        cand.write_text(rec["code"])
        ref = DATASET / rec["ref_file"]
        tb = DATASET / rec["test_file"]
        sim = Path(work) / "sim"

        try:
            cp = subprocess.run(
                ["iverilog", "-g2012", "-o", str(sim), str(ref), str(cand), str(tb)],
                capture_output=True, text=True, timeout=TIMEOUT)
        except subprocess.TimeoutExpired:
            return dict(id=rec["id"], label="TIMEOUT", stage="compile",
                        mismatches=None, samples=None, msg="compile timeout")

        if cp.returncode != 0:
            msg = (cp.stderr or cp.stdout).strip().split("\n")[0][:300]
            return dict(id=rec["id"], label="COMPILE_ERROR", stage="compile",
                        mismatches=None, samples=None, msg=msg)

        try:
            rp = subprocess.run([str(sim)], capture_output=True, text=True,
                                timeout=TIMEOUT, cwd=work)
        except subprocess.TimeoutExpired:
            return dict(id=rec["id"], label="TIMEOUT", stage="sim",
                        mismatches=None, samples=None, msg="sim timeout")

        out = (rp.stdout or "") + (rp.stderr or "")
        m = MISMATCH_RE.search(out)
        if not m:
            return dict(id=rec["id"], label="FAIL", stage="sim",
                        mismatches=None, samples=None,
                        msg="no mismatch line: " + out.strip()[-200:])

        mism, samples = int(m.group(1)), int(m.group(2))
        return dict(id=rec["id"], label="PASS" if mism == 0 else "FAIL", stage="sim",
                    mismatches=mism, samples=samples, msg="")
    finally:
        shutil.rmtree(work, ignore_errors=True)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dataset", required=True)
    ap.add_argument("--mutants", default="mutants.jsonl")
    ap.add_argument("--out", default="labeled.jsonl")
    ap.add_argument("--workers", type=int, default=os.cpu_count())
    ap.add_argument("--timeout", type=int, default=30)
    ap.add_argument("--limit", type=int, default=0, help="only first N (for smoke tests)")
    args = ap.parse_args()

    recs = [json.loads(l) for l in open(args.mutants)]
    if args.limit:
        recs = recs[:args.limit]
    print(f"evaluating {len(recs)} mutants on {args.workers} workers ...", flush=True)

    counts = {}
    done = 0
    with open(args.out, "w") as fh, ProcessPoolExecutor(
            max_workers=args.workers, initializer=init,
            initargs=(args.dataset, args.timeout)) as ex:
        futs = {ex.submit(evaluate, r): r for r in recs}
        for fut in as_completed(futs):
            rec = futs[fut]
            try:
                res = fut.result()
            except Exception as e:
                res = dict(id=rec["id"], label="COMPILE_ERROR", stage="harness",
                           mismatches=None, samples=None, msg=f"{type(e).__name__}: {e}")
            merged = {**rec, **res}
            fh.write(json.dumps(merged) + "\n")
            counts[res["label"]] = counts.get(res["label"], 0) + 1
            done += 1
            if done % 250 == 0:
                print(f"  {done}/{len(recs)}  {counts}", flush=True)

    print("\n=== RESULTS ===")
    total = sum(counts.values())
    for k in sorted(counts):
        print(f"{k:14s} {counts[k]:6d}  ({100*counts[k]/total:.1f}%)")

    usable = counts.get("PASS", 0) + counts.get("FAIL", 0)
    if usable:
        fr = 100 * counts.get("FAIL", 0) / usable
        print(f"\nusable (compiles): {usable}   FAIL share: {fr:.1f}%")
        if fr > 80 or fr < 20:
            print("WARNING: dataset is lopsided; adjust the operator mix.")
    print(f"output: {args.out}")


if __name__ == "__main__":
    main()
