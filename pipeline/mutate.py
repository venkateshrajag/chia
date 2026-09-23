#!/usr/bin/env python3
"""
mutate.py - Generate realistic design faults in VerilogEval reference solutions.

Each VerilogEval problem ships:
  ProbNNN_name_prompt.txt   natural-language spec
  ProbNNN_name_ref.sv       correct solution, module RefModule
  ProbNNN_name_test.sv      testbench: instantiates BOTH RefModule and TopModule
                            and reports "Mismatches: X in Y samples"

The candidate under test must be named TopModule. We therefore rename
RefModule -> TopModule to form the baseline candidate, then inject faults.

Output: mutants.jsonl, one JSON record per mutant, code inlined.
"""
import argparse, json, random, re, sys
from pathlib import Path

# ---------------------------------------------------------------- operators

def _ints(code):
    """Integer literals that are safe to perturb (skip sized-literal widths)."""
    out = []
    for m in re.finditer(r"(?<![\w'])(\d+)(?![\w'])", code):
        # skip the width part of a sized literal like 8'b0 / 4'd3
        tail = code[m.end():m.end() + 2]
        if tail[:1] == "'":
            continue
        out.append(m)
    return out


def op_flip_compare(code, rng):
    pairs = [(">=", "<"), ("<=", ">"), ("==", "!="), ("!=", "=="), (">", "<="), ("<", ">=")]
    cands = []
    for a, b in pairs:
        for m in re.finditer(re.escape(a), code):
            # '<=' may be a non-blocking assignment; only treat as comparison
            # when it appears inside parentheses (a condition)
            if a == "<=" and not _inside_parens(code, m.start()):
                continue
            cands.append((m.start(), m.end(), b, f"compare {a}->{b}"))
    return _pick(code, cands, rng)


def _inside_parens(code, idx):
    depth = 0
    for ch in code[:idx]:
        if ch == "(":
            depth += 1
        elif ch == ")":
            depth -= 1
    return depth > 0


def op_swap_binop(code, rng):
    swaps = {"+": "-", "-": "+", "&": "|", "|": "&", "^": "&"}
    cands = []
    for m in re.finditer(r"(?<=[\w\)\]\s])([+\-&|^])(?=[\s\w\(])", code):
        ch = m.group(1)
        if ch not in swaps:
            continue
        # avoid &&, ||, unary contexts, and reduction ops
        nxt = code[m.end():m.end() + 1]
        prv = code[m.start() - 1:m.start()]
        if nxt == ch or prv == ch:
            continue
        cands.append((m.start(), m.end(), swaps[ch], f"binop {ch}->{swaps[ch]}"))
    return _pick(code, cands, rng)


def op_off_by_one(code, rng):
    cands = []
    for m in _ints(code):
        v = int(m.group(1))
        nv = v + rng.choice([-1, 1])
        if nv < 0:
            nv = v + 1
        cands.append((m.start(), m.end(), str(nv), f"const {v}->{nv}"))
    return _pick(code, cands, rng)


def op_bit_range(code, rng):
    cands = []
    for m in re.finditer(r"\[\s*(\d+)\s*:\s*(\d+)\s*\]", code):
        hi, lo = int(m.group(1)), int(m.group(2))
        if hi > lo + 0:
            nhi = hi - 1 if hi - 1 >= lo else hi + 1
            cands.append((m.start(), m.end(), f"[{nhi}:{lo}]", f"range [{hi}:{lo}]->[{nhi}:{lo}]"))
    for m in re.finditer(r"\[\s*(\d+)\s*\]", code):
        i = int(m.group(1))
        ni = i + rng.choice([-1, 1])
        if ni < 0:
            ni = i + 1
        cands.append((m.start(), m.end(), f"[{ni}]", f"index [{i}]->[{ni}]"))
    return _pick(code, cands, rng)


def op_reset_value(code, rng):
    cands = []
    for m in re.finditer(r"(\d+)'([bdh])0\b", code):
        cands.append((m.start(), m.end(), f"{m.group(1)}'{m.group(2)}1", "reset 0->1"))
    for m in re.finditer(r"(\d+)'([bdh])1\b", code):
        cands.append((m.start(), m.end(), f"{m.group(1)}'{m.group(2)}0", "reset 1->0"))
    return _pick(code, cands, rng)


def op_blocking(code, rng):
    """Turn a non-blocking assignment into a blocking one (classic RTL bug)."""
    cands = []
    for m in re.finditer(r"^(\s*[\w\[\]\.:]+\s*)<=(\s*)", code, re.M):
        if _inside_parens(code, m.start()):
            continue
        cands.append((m.start(), m.end(), f"{m.group(1)}={m.group(2)}", "nonblocking <= -> ="))
    return _pick(code, cands, rng)


def op_drop_assign(code, rng):
    """Delete one assignment statement."""
    lines = code.split("\n")
    idxs = [i for i, l in enumerate(lines)
            if re.match(r"^\s*(assign\s+)?[\w\[\]\.:]+\s*(<=|=)[^=]", l)
            and not l.strip().startswith("//")]
    if not idxs:
        return None
    i = rng.choice(idxs)
    removed = lines[i].strip()
    del lines[i]
    return "\n".join(lines), f"dropped: {removed[:50]}"


def op_swap_identifier(code, rng):
    """Swap two occurrences of different signal names in an expression."""
    names = set(re.findall(r"\b([a-z_]\w{1,})\b", code))
    kw = {"module", "endmodule", "input", "output", "inout", "wire", "reg", "logic",
          "always", "always_ff", "always_comb", "assign", "begin", "end", "if", "else",
          "case", "endcase", "posedge", "negedge", "default", "parameter", "localparam",
          "signed", "unsigned", "integer", "genvar", "generate", "endgenerate", "for",
          "TopModule", "RefModule"}
    names = sorted(n for n in names if n not in kw and len(n) > 1)
    if len(names) < 2:
        return None
    a, b = rng.sample(names, 2)
    occ = [m for m in re.finditer(rf"\b{re.escape(a)}\b", code)]
    if not occ:
        return None
    m = rng.choice(occ)
    return code[:m.start()] + b + code[m.end():], f"signal {a}->{b}"


def _pick(code, cands, rng):
    if not cands:
        return None
    s, e, rep, desc = rng.choice(cands)
    return code[:s] + rep + code[e:], desc


OPERATORS = [
    ("flip_compare", op_flip_compare),
    ("swap_binop", op_swap_binop),
    ("off_by_one", op_off_by_one),
    ("bit_range", op_bit_range),
    ("reset_value", op_reset_value),
    ("blocking", op_blocking),
    ("drop_assign", op_drop_assign),
    ("swap_identifier", op_swap_identifier),
]

# ---------------------------------------------------------------- driver

def to_candidate(ref_src):
    return re.sub(r"\bRefModule\b", "TopModule", ref_src)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dataset", required=True, help="path to dataset_spec-to-rtl")
    ap.add_argument("--out", default="mutants.jsonl")
    ap.add_argument("--per-problem", type=int, default=40)
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()

    ds = Path(args.dataset)
    refs = sorted(ds.glob("*_ref.sv"))
    if not refs:
        sys.exit(f"no *_ref.sv found in {ds}")

    rng = random.Random(args.seed)
    n_written = 0
    n_dup = 0
    per_op = {}

    with open(args.out, "w") as fh:
        for ref in refs:
            stem = ref.name[:-len("_ref.sv")]
            test = ds / f"{stem}_test.sv"
            prompt = ds / f"{stem}_prompt.txt"
            if not test.exists():
                continue

            ref_src = ref.read_text()
            base = to_candidate(ref_src)

            # the unmutated candidate: a guaranteed-correct PASS example
            rec = dict(id=f"{stem}#base", problem=stem, op="none", detail="unmutated",
                       code=base, ref_file=ref.name, test_file=test.name,
                       prompt_file=prompt.name if prompt.exists() else None)
            fh.write(json.dumps(rec) + "\n")
            n_written += 1

            seen = {base}
            tries = 0
            made = 0
            while made < args.per_problem and tries < args.per_problem * 8:
                tries += 1
                op_name, op_fn = rng.choice(OPERATORS)
                try:
                    res = op_fn(base, rng)
                except Exception:
                    res = None
                if not res:
                    continue
                code, detail = res
                if code in seen:
                    n_dup += 1
                    continue
                seen.add(code)
                rec = dict(id=f"{stem}#{made:03d}", problem=stem, op=op_name,
                           detail=detail, code=code, ref_file=ref.name,
                           test_file=test.name,
                           prompt_file=prompt.name if prompt.exists() else None)
                fh.write(json.dumps(rec) + "\n")
                per_op[op_name] = per_op.get(op_name, 0) + 1
                made += 1
                n_written += 1

    print(f"problems      : {len(refs)}")
    print(f"mutants written: {n_written}  (dupes skipped: {n_dup})")
    for k in sorted(per_op):
        print(f"  {k:16s} {per_op[k]}")
    print(f"output        : {args.out}")


if __name__ == "__main__":
    main()
