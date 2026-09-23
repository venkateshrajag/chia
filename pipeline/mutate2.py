#!/usr/bin/env python3
"""
mutate2.py - Balanced fault injection for VerilogEval.

v1 produced ~80% FAIL because every operator broke something. A gate that
only ever sees broken code learns nothing useful: the realistic question an
agentic loop faces is "is this edit harmful or harmless?".

So this version has two families of operators:

  FAULTY  - change behaviour (expected label FAIL)
  BENIGN  - semantics-preserving refactors an agent plausibly makes
            (expected label PASS): commutative swaps, consistent internal
            renames, redundant parens, literal reformatting, reordering
            independent assigns, dead wire declarations.

Labels still come from actually running the testbench; the families are only
a generation-time prior, never a label.
"""
import argparse, json, random, re, sys
from pathlib import Path

# ============================================================ helpers

def _pick(code, cands, rng):
    if not cands:
        return None
    s, e, rep, desc = rng.choice(cands)
    return code[:s] + rep + code[e:], desc


def _inside_parens(code, idx):
    d = 0
    for ch in code[:idx]:
        if ch == "(":
            d += 1
        elif ch == ")":
            d -= 1
    return d > 0


def _body_start(code):
    """Index just after the module header's closing ');'."""
    m = re.search(r"\)\s*;", code)
    return m.end() if m else 0


def _internal_signals(code):
    """Names declared as wire/reg/logic inside the body (not ports)."""
    body = code[_body_start(code):]
    names = set()
    for m in re.finditer(
            r"^\s*(?:wire|reg|logic)\s*(?:signed|unsigned)?\s*(?:\[[^\]]*\]\s*)?"
            r"([A-Za-z_]\w*)\s*(?:[,;=]|\[)", body, re.M):
        names.add(m.group(1))
    return sorted(names)


KEYWORDS = {
    "module", "endmodule", "input", "output", "inout", "wire", "reg", "logic",
    "always", "always_ff", "always_comb", "always_latch", "assign", "begin",
    "end", "if", "else", "case", "casez", "casex", "endcase", "posedge",
    "negedge", "default", "parameter", "localparam", "signed", "unsigned",
    "integer", "genvar", "generate", "endgenerate", "for", "while", "function",
    "endfunction", "task", "endtask", "initial", "TopModule", "RefModule",
}

# ============================================================ FAULTY

def f_flip_compare(code, rng):
    pairs = [(">=", "<"), ("<=", ">"), ("==", "!="), ("!=", "=="), (">", "<="), ("<", ">=")]
    c = []
    for a, b in pairs:
        for m in re.finditer(re.escape(a), code):
            if a == "<=" and not _inside_parens(code, m.start()):
                continue
            c.append((m.start(), m.end(), b, f"compare {a}->{b}"))
    return _pick(code, c, rng)


def f_swap_binop(code, rng):
    sw = {"+": "-", "-": "+", "&": "|", "|": "&", "^": "&"}
    c = []
    for m in re.finditer(r"(?<=[\w\)\]\s])([+\-&|^])(?=[\s\w\(])", code):
        ch = m.group(1)
        if ch not in sw:
            continue
        if code[m.end():m.end()+1] == ch or code[m.start()-1:m.start()] == ch:
            continue
        c.append((m.start(), m.end(), sw[ch], f"binop {ch}->{sw[ch]}"))
    return _pick(code, c, rng)


def f_off_by_one(code, rng):
    c = []
    for m in re.finditer(r"(?<![\w'])(\d+)(?![\w'])", code):
        if code[m.end():m.end()+1] == "'":
            continue
        v = int(m.group(1))
        nv = v + rng.choice([-1, 1])
        if nv < 0:
            nv = v + 1
        c.append((m.start(), m.end(), str(nv), f"const {v}->{nv}"))
    return _pick(code, c, rng)


def f_bit_range(code, rng):
    c = []
    for m in re.finditer(r"\[\s*(\d+)\s*:\s*(\d+)\s*\]", code):
        hi, lo = int(m.group(1)), int(m.group(2))
        nhi = hi - 1 if hi - 1 >= lo else hi + 1
        c.append((m.start(), m.end(), f"[{nhi}:{lo}]", f"range [{hi}:{lo}]->[{nhi}:{lo}]"))
    for m in re.finditer(r"\[\s*(\d+)\s*\]", code):
        i = int(m.group(1))
        ni = i + rng.choice([-1, 1])
        if ni < 0:
            ni = i + 1
        c.append((m.start(), m.end(), f"[{ni}]", f"index [{i}]->[{ni}]"))
    return _pick(code, c, rng)


def f_reset_value(code, rng):
    c = []
    for m in re.finditer(r"(\d+)'([bdh])0\b", code):
        c.append((m.start(), m.end(), f"{m.group(1)}'{m.group(2)}1", "reset 0->1"))
    for m in re.finditer(r"(\d+)'([bdh])1\b", code):
        c.append((m.start(), m.end(), f"{m.group(1)}'{m.group(2)}0", "reset 1->0"))
    return _pick(code, c, rng)


def f_blocking(code, rng):
    c = []
    for m in re.finditer(r"^(\s*[\w\[\]\.:]+\s*)<=(\s*)", code, re.M):
        if _inside_parens(code, m.start()):
            continue
        c.append((m.start(), m.end(), f"{m.group(1)}={m.group(2)}", "nonblocking <= -> ="))
    return _pick(code, c, rng)


def f_drop_assign(code, rng):
    lines = code.split("\n")
    idx = [i for i, l in enumerate(lines)
           if re.match(r"^\s*(assign\s+)?[\w\[\]\.:]+\s*(<=|=)[^=]", l)
           and not l.strip().startswith("//")]
    if not idx:
        return None
    i = rng.choice(idx)
    removed = lines[i].strip()
    del lines[i]
    return "\n".join(lines), f"dropped: {removed[:50]}"


def f_swap_identifier(code, rng):
    names = [n for n in set(re.findall(r"\b([a-z_]\w+)\b", code)) if n not in KEYWORDS]
    if len(names) < 2:
        return None
    a, b = rng.sample(sorted(names), 2)
    occ = list(re.finditer(rf"\b{re.escape(a)}\b", code))
    if not occ:
        return None
    m = rng.choice(occ)
    return code[:m.start()] + b + code[m.end():], f"signal {a}->{b}"


# ============================================================ BENIGN

def b_swap_commutative(code, rng):
    """a OP b -> b OP a for commutative OP. Behaviour preserved."""
    c = []
    pat = r"\b([A-Za-z_]\w*)\s*([+&|^])\s*([A-Za-z_]\w*)\b"
    for m in re.finditer(pat, code):
        l, op, r = m.group(1), m.group(2), m.group(3)
        if l in KEYWORDS or r in KEYWORDS or l == r:
            continue
        if code[m.end():m.end()+1] == op or code[m.start()-1:m.start()] == op:
            continue
        c.append((m.start(), m.end(), f"{r} {op} {l}", f"commute {l}{op}{r} -> {r}{op}{l}"))
    return _pick(code, c, rng)


def b_rename_internal(code, rng):
    """Consistently rename an internal (non-port) signal everywhere."""
    names = _internal_signals(code)
    names = [n for n in names if n not in KEYWORDS]
    if not names:
        return None
    old = rng.choice(names)
    new = f"{old}_r{rng.randint(1, 999)}"
    if re.search(rf"\b{re.escape(new)}\b", code):
        return None
    return re.sub(rf"\b{re.escape(old)}\b", new, code), f"rename {old}->{new}"


def b_redundant_parens(code, rng):
    """assign x = EXPR;  ->  assign x = (EXPR);"""
    c = []
    for m in re.finditer(r"(assign\s+[\w\[\]\.:]+\s*=\s*)([^;]+)(;)", code):
        expr = m.group(2).strip()
        if expr.startswith("(") and expr.endswith(")"):
            continue
        c.append((m.start(), m.end(), f"{m.group(1)}({expr}){m.group(3)}",
                  "redundant parentheses"))
    return _pick(code, c, rng)


def b_literal_reformat(code, rng):
    """4'd10 -> 4'b1010 (same value, different notation)."""
    c = []
    for m in re.finditer(r"(\d+)'d(\d+)\b", code):
        w, v = int(m.group(1)), int(m.group(2))
        if w > 32 or v >= (1 << w):
            continue
        c.append((m.start(), m.end(), f"{w}'b{v:0{w}b}", f"{w}'d{v} -> binary"))
    for m in re.finditer(r"(\d+)'b([01]+)\b", code):
        w, bits = int(m.group(1)), m.group(2)
        if w > 32:
            continue
        c.append((m.start(), m.end(), f"{w}'d{int(bits, 2)}", f"{w}'b{bits} -> decimal"))
    return _pick(code, c, rng)


def b_reorder_assigns(code, rng):
    """Swap two independent continuous assigns (order is irrelevant in Verilog)."""
    lines = code.split("\n")
    idx = [i for i, l in enumerate(lines) if re.match(r"^\s*assign\s+", l)]
    if len(idx) < 2:
        return None
    i, j = rng.sample(idx, 2)
    # only safe if neither reads the other's LHS
    def lhs(l):
        m = re.match(r"^\s*assign\s+([\w\[\]\.:]+)", l)
        return m.group(1).split("[")[0] if m else None
    li, lj = lhs(lines[i]), lhs(lines[j])
    if not li or not lj:
        return None
    if re.search(rf"\b{re.escape(li)}\b", lines[j]) or re.search(rf"\b{re.escape(lj)}\b", lines[i]):
        return None
    lines[i], lines[j] = lines[j], lines[i]
    return "\n".join(lines), f"reordered assigns {li}/{lj}"


def b_dead_wire(code, rng):
    """Declare an unused internal wire. No behavioural effect."""
    pos = _body_start(code)
    if pos == 0:
        return None
    name = f"unused_{rng.randint(100, 999)}"
    w = rng.choice(["", "[3:0] ", "[7:0] "])
    ins = f"\n  wire {w}{name};"
    return code[:pos] + ins + code[pos:], f"dead wire {name}"


FAULTY = [
    ("flip_compare", f_flip_compare), ("swap_binop", f_swap_binop),
    ("off_by_one", f_off_by_one), ("bit_range", f_bit_range),
    ("reset_value", f_reset_value), ("blocking", f_blocking),
    ("drop_assign", f_drop_assign), ("swap_identifier", f_swap_identifier),
]
BENIGN = [
    ("commute", b_swap_commutative), ("rename_internal", b_rename_internal),
    ("parens", b_redundant_parens), ("literal_fmt", b_literal_reformat),
    ("reorder", b_reorder_assigns), ("dead_wire", b_dead_wire),
]

# ============================================================ driver

def to_candidate(src):
    return re.sub(r"\bRefModule\b", "TopModule", src)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dataset", required=True)
    ap.add_argument("--out", default="mutants.jsonl")
    ap.add_argument("--per-problem", type=int, default=40)
    ap.add_argument("--benign-frac", type=float, default=0.5,
                    help="fraction of generation attempts drawn from BENIGN ops")
    ap.add_argument("--max-per-op", type=int, default=3,
                    help="cap on how many times one operator may fire per problem; "
                         "stops any single easy-to-spot edit from dominating")
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()

    ds = Path(args.dataset)
    refs = sorted(ds.glob("*_ref.sv"))
    if not refs:
        sys.exit(f"no *_ref.sv in {ds}")

    rng = random.Random(args.seed)
    per_op, fam = {}, {"faulty": 0, "benign": 0, "none": 0}
    n = 0

    with open(args.out, "w") as fh:
        for ref in refs:
            stem = ref.name[:-len("_ref.sv")]
            test = ds / f"{stem}_test.sv"
            prompt = ds / f"{stem}_prompt.txt"
            if not test.exists():
                continue
            base = to_candidate(ref.read_text())

            fh.write(json.dumps(dict(
                id=f"{stem}#base", problem=stem, family="none", op="none",
                detail="unmutated", code=base, ref_file=ref.name,
                test_file=test.name,
                prompt_file=prompt.name if prompt.exists() else None)) + "\n")
            n += 1
            fam["none"] += 1

            seen = {base}
            used = {}          # per-problem per-operator counter
            made, tries = 0, 0
            while made < args.per_problem and tries < args.per_problem * 20:
                tries += 1
                if rng.random() < args.benign_frac:
                    family, pool = "benign", BENIGN
                else:
                    family, pool = "faulty", FAULTY
                op_name, fn = rng.choice(pool)
                if used.get(op_name, 0) >= args.max_per_op:
                    continue
                try:
                    res = fn(base, rng)
                except Exception:
                    res = None
                if not res:
                    continue
                code, detail = res
                if code in seen:
                    continue
                seen.add(code)
                used[op_name] = used.get(op_name, 0) + 1
                fh.write(json.dumps(dict(
                    id=f"{stem}#{made:03d}", problem=stem, family=family,
                    op=op_name, detail=detail, code=code, ref_file=ref.name,
                    test_file=test.name,
                    prompt_file=prompt.name if prompt.exists() else None)) + "\n")
                per_op[op_name] = per_op.get(op_name, 0) + 1
                fam[family] += 1
                made += 1
                n += 1

    print(f"problems       : {len(refs)}")
    print(f"records written: {n}")
    print(f"  faulty ops   : {fam['faulty']}")
    print(f"  benign ops   : {fam['benign']}")
    print(f"  unmutated    : {fam['none']}")
    for k in sorted(per_op):
        print(f"    {k:16s} {per_op[k]}")
    print(f"output: {args.out}")


if __name__ == "__main__":
    main()
