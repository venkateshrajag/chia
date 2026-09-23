#!/usr/bin/env python3
"""
eval_local.py - Score a local model (base or LoRA-tuned) on the same test set.

Scores by comparing the logits of the PASS and FAIL tokens at the answer
position rather than generating text. That is faster, fully deterministic,
never produces an unparseable answer, and gives a confidence score for free.

The confidence matters: a gate in front of simulation can be tuned. Raising
the threshold before it blocks a candidate trades away saved simulations in
exchange for not discarding good designs. This script reports that whole
tradeoff curve, not just the single default operating point.

Prompt format is IDENTICAL to eval_gemini.py and train_lora.py.

  python3 eval_local.py --test test.jsonl \
      --model Qwen/Qwen2.5-Coder-7B-Instruct --adapter ./adapter_7b \
      --load-4bit --out preds_tuned.jsonl
"""
import argparse, json, time
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

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


def first_token_ids(tok, word):
    """Token ids that could start the given answer word."""
    ids = set()
    for variant in (word, " " + word, word.capitalize(), word.lower()):
        t = tok(variant, add_special_tokens=False)["input_ids"]
        if t:
            ids.add(t[0])
    return sorted(ids)


def metrics(rows, thresh=0.5):
    """Positive class = FAIL. p_fail >= thresh means the gate blocks it."""
    tp = fp = fn = tn = 0
    for r in rows:
        pred_fail = r["p_fail"] >= thresh
        truth_fail = r["truth"] == "FAIL"
        if pred_fail and truth_fail:
            tp += 1
        elif pred_fail and not truth_fail:
            fp += 1
        elif not pred_fail and truth_fail:
            fn += 1
        else:
            tn += 1
    n = tp + fp + fn + tn
    acc = (tp + tn) / n if n else 0
    prec = tp / (tp + fp) if tp + fp else 0.0
    rec = tp / (tp + fn) if tp + fn else 0.0
    f1 = 2 * prec * rec / (prec + rec) if prec + rec else 0.0
    return dict(thresh=thresh, tp=tp, fp=fp, fn=fn, tn=tn,
                acc=acc, precision_fail=prec, recall_fail=rec, f1_fail=f1)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--test", default="test.jsonl")
    ap.add_argument("--model", default="Qwen/Qwen2.5-Coder-7B-Instruct")
    ap.add_argument("--adapter", default=None,
                    help="path to LoRA adapter; omit to score the untrained base")
    ap.add_argument("--load-4bit", action="store_true")
    ap.add_argument("--max-len", type=int, default=3072)
    ap.add_argument("--limit", type=int, default=0)
    ap.add_argument("--out", default="preds_local.jsonl")
    ap.add_argument("--gpu-hourly", type=float, default=0.70,
                    help="USD/hour for this GPU, used for cost per 1k decisions")
    args = ap.parse_args()

    tok = AutoTokenizer.from_pretrained(args.adapter or args.model,
                                        trust_remote_code=True)
    if tok.pad_token is None:
        tok.pad_token = tok.eos_token

    kw = dict(torch_dtype=torch.bfloat16, device_map="auto", trust_remote_code=True)
    if args.load_4bit:
        from transformers import BitsAndBytesConfig
        kw["quantization_config"] = BitsAndBytesConfig(
            load_in_4bit=True, bnb_4bit_compute_dtype=torch.bfloat16,
            bnb_4bit_quant_type="nf4", bnb_4bit_use_double_quant=True)

    model = AutoModelForCausalLM.from_pretrained(args.model, **kw)
    if args.adapter:
        from peft import PeftModel
        model = PeftModel.from_pretrained(model, args.adapter)
        print(f"loaded adapter: {args.adapter}")
    else:
        print("scoring UNTRAINED base model")
    model.eval()

    pass_ids = first_token_ids(tok, "PASS")
    fail_ids = first_token_ids(tok, "FAIL")
    print(f"PASS token ids {pass_ids}   FAIL token ids {fail_ids}")

    recs = [json.loads(l) for l in open(args.test)]
    if args.limit:
        recs = recs[:args.limit]

    rows = []
    t0 = time.time()
    with torch.no_grad():
        for i, r in enumerate(recs):
            msgs = [{"role": "system", "content": SYSTEM},
                    {"role": "user",
                     "content": TMPL.format(spec=r.get("spec", ""), code=r["code"])}]
            prompt = tok.apply_chat_template(msgs, tokenize=False,
                                             add_generation_prompt=True)
            ids = tok(prompt, add_special_tokens=False)["input_ids"]
            if len(ids) > args.max_len:
                ids = ids[:40] + ids[-(args.max_len - 40):]
            inp = torch.tensor([ids], device=model.device)
            logits = model(inp).logits[0, -1].float()
            lp = torch.logsumexp(logits[pass_ids], 0)
            lf = torch.logsumexp(logits[fail_ids], 0)
            p_fail = torch.softmax(torch.stack([lp, lf]), 0)[1].item()
            rows.append(dict(id=r["id"], problem=r["problem"], truth=r["label"],
                             family=r.get("family"), op=r.get("op"),
                             p_fail=round(p_fail, 5),
                             pred="FAIL" if p_fail >= 0.5 else "PASS"))
            if (i + 1) % 100 == 0:
                print(f"  {i+1}/{len(recs)}", flush=True)
    elapsed = time.time() - t0

    with open(args.out, "w") as fh:
        for row in rows:
            fh.write(json.dumps(row) + "\n")

    tag = "TUNED" if args.adapter else "BASE"
    m = metrics(rows, 0.5)
    print(f"\n===== {tag}: {args.model} =====")
    print(f"examples {len(rows)}   elapsed {elapsed:.1f}s"
          f"   {len(rows)/elapsed:.2f}/s")
    print(f"\nconfusion at threshold 0.5 (positive = FAIL)")
    print(f"              truth FAIL   truth PASS")
    print(f"  pred FAIL   {m['tp']:10d}   {m['fp']:10d}")
    print(f"  pred PASS   {m['fn']:10d}   {m['tn']:10d}")
    nf = sum(1 for r in rows if r["truth"] == "FAIL")
    majority = max(nf, len(rows) - nf) / len(rows)
    print(f"\naccuracy            {m['acc']*100:6.2f}%   (majority baseline {majority*100:.2f}%)")
    print(f"precision on FAIL   {m['precision_fail']*100:6.2f}%")
    print(f"recall on FAIL      {m['recall_fail']*100:6.2f}%")
    print(f"F1 on FAIL          {m['f1_fail']*100:6.2f}%")

    cost_1k = args.gpu_hourly / 3600 * (elapsed / len(rows)) * 1000
    print(f"\nthroughput          {len(rows)/elapsed:.2f} decisions/s")
    print(f"cost per 1k calls   ${cost_1k:.4f}   (at ${args.gpu_hourly}/hr)")

    print(f"\nthreshold sweep - how the gate can be tuned")
    print(f"{'thresh':>7} {'acc':>7} {'prec_FAIL':>10} {'rec_FAIL':>9} "
          f"{'blocked':>8} {'good lost':>10}")
    for t in [0.3, 0.5, 0.7, 0.8, 0.9, 0.95]:
        mm = metrics(rows, t)
        blocked = mm["tp"] + mm["fp"]
        print(f"{t:7.2f} {mm['acc']*100:6.1f}% {mm['precision_fail']*100:9.1f}% "
              f"{mm['recall_fail']*100:8.1f}% {blocked:8d} {mm['fp']:10d}")

    print(f"\nby family")
    for fam in sorted({r["family"] for r in rows if r["family"]}):
        sub = [r for r in rows if r["family"] == fam]
        corr = sum(1 for r in sub if r["pred"] == r["truth"])
        print(f"  {fam:8s} n={len(sub):5d}  accuracy {100*corr/len(sub):5.1f}%")

    print(f"\nwrote {args.out}")


if __name__ == "__main__":
    main()
