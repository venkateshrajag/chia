#!/usr/bin/env python3
"""
eval_gemini.py - Baseline: how well does Gemini predict simulation outcome?

Each test example is shown the natural-language spec plus the candidate code,
and asked one question: will this pass its testbench? We record the answer,
the token counts and the latency, then report the measures that matter for a
gate placed in front of simulation.

The key measure is PRECISION ON FAIL. When the gate says "this will fail" the
loop skips the simulation, so a wrong FAIL silently discards a good design.
Recall on FAIL tells you how many simulations you save.

Usage:
  python3 eval_gemini.py --test test.jsonl --model gemini-2.5-flash \
      --project a3-chia-hack26ath-7722 --workers 16 --out preds_flash.jsonl
"""
import argparse, json, os, re, sys, threading, time
from concurrent.futures import ThreadPoolExecutor, as_completed

# USD per 1M tokens. Check current Vertex pricing and edit before quoting costs.
PRICES = {
    "gemini-2.5-flash":      {"in": 0.30, "out": 2.50},
    "gemini-2.5-flash-lite": {"in": 0.10, "out": 0.40},
    "gemini-2.5-pro":        {"in": 1.25, "out": 10.00},
}

SYSTEM = (
    "You are a hardware verification assistant. You are shown a specification "
    "for a Verilog module and a candidate implementation. Decide whether the "
    "candidate would PASS or FAIL a thorough simulation testbench checking it "
    "against a correct reference.\n"
    "Some candidates contain real bugs. Others contain harmless edits such as "
    "renamed internal signals, reordered independent statements, redundant "
    "parentheses, or reformatted literals; those still PASS.\n"
    "Answer with exactly one word: PASS or FAIL."
)

TMPL = """Specification:
{spec}

Candidate implementation:
```verilog
{code}
```

Will this candidate PASS or FAIL the testbench? Answer with one word."""

_lock = threading.Lock()


def parse(text):
    if not text:
        return None
    t = text.strip().upper()
    m = re.search(r"\b(PASS|FAIL)\b", t)
    return m.group(1) if m else None


def make_client(project, location):
    from google import genai
    return genai.Client(vertexai=True, project=project, location=location)


def ask(client, model, rec, retries=4):
    from google.genai import types
    prompt = TMPL.format(spec=rec.get("spec", ""), code=rec["code"])
    cfg = types.GenerateContentConfig(
        system_instruction=SYSTEM, temperature=0, max_output_tokens=2048)
    last = None
    for attempt in range(retries):
        try:
            t0 = time.time()
            r = client.models.generate_content(
                model=model, contents=prompt, config=cfg)
            dt = time.time() - t0
            um = getattr(r, "usage_metadata", None)
            return dict(
                id=rec["id"], problem=rec["problem"], truth=rec["label"],
                family=rec.get("family"), op=rec.get("op"),
                pred=parse(getattr(r, "text", None)),
                latency=round(dt, 3),
                in_tok=getattr(um, "prompt_token_count", 0) or 0,
                out_tok=getattr(um, "candidates_token_count", 0) or 0,
                error=None)
        except Exception as e:
            last = f"{type(e).__name__}: {e}"
            time.sleep(min(2 ** attempt, 16))
    return dict(id=rec["id"], problem=rec["problem"], truth=rec["label"],
                family=rec.get("family"), op=rec.get("op"), pred=None,
                latency=None, in_tok=0, out_tok=0, error=last)


def report(rows, model):
    ok = [r for r in rows if r["pred"] in ("PASS", "FAIL")]
    print(f"\n===== {model} =====")
    print(f"answered {len(ok)} / {len(rows)}"
          f"   unparsed/errored: {len(rows) - len(ok)}")
    if not ok:
        return None

    tp = sum(1 for r in ok if r["pred"] == "FAIL" and r["truth"] == "FAIL")
    fp = sum(1 for r in ok if r["pred"] == "FAIL" and r["truth"] == "PASS")
    fn = sum(1 for r in ok if r["pred"] == "PASS" and r["truth"] == "FAIL")
    tn = sum(1 for r in ok if r["pred"] == "PASS" and r["truth"] == "PASS")

    acc = (tp + tn) / len(ok)
    prec = tp / (tp + fp) if tp + fp else 0.0
    rec_ = tp / (tp + fn) if tp + fn else 0.0
    f1 = 2 * prec * rec_ / (prec + rec_) if prec + rec_ else 0.0
    majority = max(sum(1 for r in ok if r["truth"] == "FAIL"),
                   sum(1 for r in ok if r["truth"] == "PASS")) / len(ok)

    it = sum(r["in_tok"] for r in ok)
    ot = sum(r["out_tok"] for r in ok)
    p = PRICES.get(model)
    cost = (it / 1e6 * p["in"] + ot / 1e6 * p["out"]) if p else None
    lat = [r["latency"] for r in ok if r["latency"]]

    print(f"\nconfusion (positive = FAIL)")
    print(f"              truth FAIL   truth PASS")
    print(f"  pred FAIL   {tp:10d}   {fp:10d}")
    print(f"  pred PASS   {fn:10d}   {tn:10d}")
    print(f"\naccuracy            {acc*100:6.2f}%   (majority baseline {majority*100:.2f}%)")
    print(f"precision on FAIL   {prec*100:6.2f}%   <- wrong FAIL discards a good design")
    print(f"recall on FAIL      {rec_*100:6.2f}%   <- share of simulations saved")
    print(f"F1 on FAIL          {f1*100:6.2f}%")
    if lat:
        lat.sort()
        print(f"latency median      {lat[len(lat)//2]:.2f}s   p95 {lat[int(len(lat)*0.95)]:.2f}s")
    print(f"tokens              in {it:,}  out {ot:,}")
    if cost is not None:
        print(f"cost this run       ${cost:.4f}")
        print(f"cost per 1k calls   ${cost/len(ok)*1000:.3f}")
    else:
        print("cost                no price entry for this model; add one to PRICES")

    return dict(model=model, n=len(ok), acc=acc, precision_fail=prec,
                recall_fail=rec_, f1_fail=f1, in_tok=it, out_tok=ot, cost=cost,
                majority=majority)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--test", default="test.jsonl")
    ap.add_argument("--model", default="gemini-2.5-flash")
    ap.add_argument("--project", required=True)
    ap.add_argument("--location", default="us-central1")
    ap.add_argument("--workers", type=int, default=16)
    ap.add_argument("--limit", type=int, default=0)
    ap.add_argument("--out", default=None)
    ap.add_argument("--summary", default="baseline_summary.jsonl")
    args = ap.parse_args()

    out = args.out or f"preds_{args.model.replace('.', '_')}.jsonl"
    recs = [json.loads(l) for l in open(args.test)]
    if args.limit:
        recs = recs[:args.limit]

    done = set()
    if os.path.exists(out):
        for l in open(out):
            try:
                done.add(json.loads(l)["id"])
            except Exception:
                pass
        if done:
            print(f"resuming: {len(done)} already done in {out}")
    todo = [r for r in recs if r["id"] not in done]
    print(f"{args.model}: {len(todo)} to evaluate on {args.workers} threads")

    client = make_client(args.project, args.location)
    fh = open(out, "a")
    n = 0
    with ThreadPoolExecutor(max_workers=args.workers) as ex:
        futs = [ex.submit(ask, client, args.model, r) for r in todo]
        for fut in as_completed(futs):
            row = fut.result()
            with _lock:
                fh.write(json.dumps(row) + "\n")
                fh.flush()
            n += 1
            if n % 100 == 0:
                print(f"  {n}/{len(todo)}", flush=True)
    fh.close()

    rows = [json.loads(l) for l in open(out)]
    s = report(rows, args.model)
    if s:
        with open(args.summary, "a") as sf:
            sf.write(json.dumps(s) + "\n")
        print(f"\nsummary appended to {args.summary}")


if __name__ == "__main__":
    main()
