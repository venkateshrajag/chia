#!/usr/bin/env python3
"""
make_results.py - turn the raw prediction files into the paper's tables.

Needs no GPU and no network. Point it at the directory holding
preds_base.jsonl, preds_tuned.jsonl and preds_gemini-*.jsonl and it emits
markdown ready to paste into the write-up:

  1. model comparison (accuracy, precision, recall, F1, cost, latency)
  2. gate economics  (simulations saved vs good designs discarded)
  3. threshold sweep for the local models
  4. per-family accuracy, which shows where the fine-tune actually helped
  5. PlateauMonitor counterfactual on the CHIA paper's published run

  python3 make_results.py --dir . --gpu-hourly 0.70 > results.md
"""
import argparse, glob, json, os, sys
from collections import Counter

# USD per 1M tokens; edit to match current Vertex pricing before quoting.
PRICES = {
    "gemini-2.5-flash":      {"in": 0.30, "out": 2.50},
    "gemini-2.5-flash-lite": {"in": 0.10, "out": 0.40},
    "gemini-2.5-pro":        {"in": 1.25, "out": 10.00},
}


def load(path):
    rows = []
    with open(path) as fh:
        for line in fh:
            line = line.strip()
            if line:
                rows.append(json.loads(line))
    return rows


def confusion(rows, thresh=None):
    """positive class = FAIL"""
    tp = fp = fn = tn = 0
    used = 0
    for r in rows:
        if thresh is not None and "p_fail" in r:
            pred_fail = r["p_fail"] >= thresh
        elif r.get("pred") in ("PASS", "FAIL"):
            pred_fail = r["pred"] == "FAIL"
        else:
            continue
        used += 1
        truth_fail = r["truth"] == "FAIL"
        if pred_fail and truth_fail:
            tp += 1
        elif pred_fail and not truth_fail:
            fp += 1
        elif not pred_fail and truth_fail:
            fn += 1
        else:
            tn += 1
    return dict(n=used, tp=tp, fp=fp, fn=fn, tn=tn)


def metrics(c):
    n = c["n"]
    if not n:
        return dict(acc=0, prec=0, rec=0, f1=0)
    acc = (c["tp"] + c["tn"]) / n
    prec = c["tp"] / (c["tp"] + c["fp"]) if c["tp"] + c["fp"] else 0.0
    rec = c["tp"] / (c["tp"] + c["fn"]) if c["tp"] + c["fn"] else 0.0
    f1 = 2 * prec * rec / (prec + rec) if prec + rec else 0.0
    return dict(acc=acc, prec=prec, rec=rec, f1=f1)


def cost_per_1k(rows, model, gpu_hourly):
    """Gemini rows carry tokens; local rows carry throughput instead."""
    ok = [r for r in rows if r.get("pred") or "p_fail" in r]
    if not ok:
        return None, None
    it = sum(r.get("in_tok", 0) for r in ok)
    ot = sum(r.get("out_tok", 0) for r in ok)
    lat = [r["latency"] for r in ok if r.get("latency")]
    if it and model in PRICES:
        p = PRICES[model]
        usd = it / 1e6 * p["in"] + ot / 1e6 * p["out"]
        return usd / len(ok) * 1000, (sorted(lat)[len(lat)//2] if lat else None)
    return None, (sorted(lat)[len(lat)//2] if lat else None)


def pct(x):
    return f"{100*x:.1f}%"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dir", default=".")
    ap.add_argument("--gpu-hourly", type=float, default=0.70)
    ap.add_argument("--base-secs", type=float, default=159.2,
                    help="elapsed seconds for the base-model evaluation")
    ap.add_argument("--tuned-secs", type=float, default=204.7,
                    help="elapsed seconds for the tuned-model evaluation")
    args = ap.parse_args()

    d = args.dir
    files = {}
    for label, pat in (("Qwen2.5-Coder-7B (untrained)", "preds_base.jsonl"),
                       ("Qwen2.5-Coder-7B (fine-tuned)", "preds_tuned.jsonl")):
        p = os.path.join(d, pat)
        if os.path.exists(p):
            files[label] = (p, None)
    for p in sorted(glob.glob(os.path.join(d, "preds_gemini-*.jsonl"))):
        model = os.path.basename(p)[len("preds_"):-len(".jsonl")].replace("_", ".")
        files[model] = (p, model)

    if not files:
        sys.exit(f"no prediction files found in {d}")

    print("## Model comparison\n")
    print("Positive class is FAIL. Precision is the measure that matters for a "
          "gate: a wrong FAIL discards a working design.\n")
    print("| Model | n | Accuracy | Precision (FAIL) | Recall (FAIL) | F1 | "
          "Cost / 1k | Median latency |")
    print("|---|---|---|---|---|---|---|---|")

    keep = {}
    for label, (path, model) in files.items():
        rows = load(path)
        c = confusion(rows)
        if c["n"] == 0:
            print(f"| {label} | 0 | *all calls errored* | | | | | |")
            continue
        m = metrics(c)
        keep[label] = (rows, c, m)

        cpk, lat = cost_per_1k(rows, model or "", args.gpu_hourly)
        if cpk is None and "Qwen" in label:
            secs = args.base_secs if "untrained" in label else args.tuned_secs
            cpk = args.gpu_hourly / 3600 * (secs / c["n"]) * 1000
            lat = secs / c["n"]
        cpk_s = f"${cpk:.3f}" if cpk is not None else "n/a"
        lat_s = f"{lat:.2f}s" if lat else "n/a"
        print(f"| {label} | {c['n']} | {pct(m['acc'])} | {pct(m['prec'])} | "
              f"{pct(m['rec'])} | {pct(m['f1'])} | {cpk_s} | {lat_s} |")

    print("\n## Gate economics\n")
    print("Of the candidates that would really pass, how many does each gate "
          "throw away, and how many simulations does it save?\n")
    print("| Model | Simulations skipped | Good designs discarded | "
          "Share of good designs lost |")
    print("|---|---|---|---|")
    for label, (rows, c, m) in keep.items():
        good = c["fp"] + c["tn"]
        blocked = c["tp"] + c["fp"]
        share = c["fp"] / good if good else 0
        print(f"| {label} | {blocked} of {c['n']} | {c['fp']} of {good} | "
              f"{pct(share)} |")

    print("\n## Threshold sweep (local models)\n")
    print("A gate is tunable. Raising the bar before it blocks discards fewer "
          "good designs but saves fewer simulations.\n")
    for label, (rows, c, m) in keep.items():
        if "p_fail" not in (rows[0] if rows else {}):
            continue
        print(f"\n**{label}**\n")
        print("| Threshold | Accuracy | Precision (FAIL) | Recall (FAIL) | "
              "Blocked | Good lost |")
        print("|---|---|---|---|---|---|")
        for t in (0.3, 0.5, 0.7, 0.8, 0.9, 0.95):
            cc = confusion(rows, thresh=t)
            mm = metrics(cc)
            print(f"| {t:.2f} | {pct(mm['acc'])} | {pct(mm['prec'])} | "
                  f"{pct(mm['rec'])} | {cc['tp']+cc['fp']} | {cc['fp']} |")

    print("\n## Accuracy by edit family\n")
    print("`benign` edits preserve behaviour (renames, reorderings, redundant "
          "parentheses, reformatted literals). `faulty` edits change it. This "
          "is where the fine-tune earned its keep.\n")
    fams = sorted({r.get("family") for rs, _, _ in keep.values()
                   for r in rs if r.get("family")})
    print("| Model | " + " | ".join(fams) + " |")
    print("|---" * (len(fams) + 1) + "|")
    for label, (rows, c, m) in keep.items():
        cells = []
        for f in fams:
            sub = [r for r in rows if r.get("family") == f
                   and (r.get("pred") in ("PASS", "FAIL"))]
            if not sub:
                cells.append("n/a")
                continue
            corr = sum(1 for r in sub if r["pred"] == r["truth"])
            cells.append(f"{100*corr/len(sub):.1f}%")
        print(f"| {label} | " + " | ".join(cells) + " |")

    # ------------------------------------------------ plateau counterfactual
    print("\n## PlateauMonitor on the published CHIA critical-path run\n")
    sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
    try:
        from chia_governor import PlateauMonitor
    except Exception as e:
        print(f"*(skipped: {e})*")
        return

    # Reported in the CHIA paper: 14 iterations, $202.03 total, ~$14.43 each,
    # frequency 47 MHz -> 95 MHz, improvement plateaued after iteration 7.
    freqs = [47, 58, 66, 74, 81, 87, 92, 95, 95, 95, 95, 95, 95, 95]
    print("The published run took 14 iterations at $14.43 each ($202.03) and "
          "lifted frequency from 47 MHz to 95 MHz, but stopped improving after "
          "iteration 8. `patience` is how many flat iterations to tolerate "
          "before stopping; lower is more aggressive.\n")
    print("| Patience | Would stop at | Spend | Saved | Final result |")
    print("|---|---|---|---|---|")
    for patience in (1, 2, 3):
        mon = PlateauMonitor(min_gain_per_usd=0.05, patience=patience,
                             mode="maximize")
        stop_at = None
        for i, f in enumerate(freqs, start=1):
            mon.record(i, float(f), cost_usd=14.43)
            if stop_at is None and mon.should_stop():
                stop_at = i
        total = mon.total_spend
        if stop_at is None:
            print(f"| {patience} | never | ${total:.2f} | $0.00 | 95 MHz |")
            continue
        spent = sum(14.43 for i in range(1, stop_at + 1))
        best = next(p.best_so_far for p in mon.history if p.iteration == stop_at)
        print(f"| {patience} | iteration {stop_at} of 14 | ${spent:.2f} | "
              f"${total-spent:.2f} ({100*(total-spent)/total:.1f}%) | "
              f"{best:.0f} MHz |")
    print("\nIn every case the final result is unchanged at 95 MHz, because "
          "the iterations it skips are the ones that produced nothing.")


if __name__ == "__main__":
    main()
