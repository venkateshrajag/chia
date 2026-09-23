"""
TriageGate - decide whether a candidate design is worth simulating.

An agentic design loop proposes many changes and sends every one to
simulation. Simulation is the expensive step, and many proposals are simply
wrong. This block asks a cheap model first: will this candidate pass its
tests? Candidates it is confident will fail are blocked before any simulator
runs.

The decision is a probability, not a verdict, because a gate has to be
tunable. Raising `threshold` makes the gate more reluctant to block, which
discards fewer good designs but saves fewer simulations. Measured behaviour
of the shipped model (630 held-out examples):

    threshold   accuracy   precision(FAIL)   recall(FAIL)   good designs lost
      0.50        85.2%         83.3%           84.8%            49 / 341
      0.70        83.7%         86.9%           75.8%            33 / 341
      0.90        80.2%         92.7%           61.6%            14 / 341

Precision is the measure that matters: a wrong FAIL silently throws away a
working design, which is worse for the loop than paying for one more run.

Scorers are pluggable. `LocalScorer` runs a fine-tuned open model on your own
GPU; `CallableScorer` wraps anything else (a frontier API, a heuristic, a
classifier). All of them return P(fail).
"""
from __future__ import annotations

import time
from dataclasses import dataclass
from typing import Callable, Optional, Protocol

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


# ---------------------------------------------------------------- scorers

class Scorer(Protocol):
    def p_fail(self, spec: str, code: str) -> float:
        """Return the probability in [0,1] that this candidate fails."""


class CallableScorer:
    """Wrap any function (spec, code) -> P(fail)."""

    def __init__(self, fn: Callable[[str, str], float], name: str = "callable"):
        self._fn = fn
        self.name = name

    def p_fail(self, spec: str, code: str) -> float:
        return float(self._fn(spec, code))


class LocalScorer:
    """
    Score with a local causal LM, optionally with a LoRA adapter.

    Compares the logits of the PASS and FAIL tokens at the answer position
    rather than generating text. That is faster, deterministic, can never
    produce an unparseable answer, and yields a calibrated-ish probability.
    """

    def __init__(self, model_name: str, adapter: Optional[str] = None,
                 load_4bit: bool = True, max_len: int = 3072, device_map="auto"):
        import torch
        from transformers import AutoModelForCausalLM, AutoTokenizer

        self._torch = torch
        self.name = adapter or model_name
        self.max_len = max_len

        self.tok = AutoTokenizer.from_pretrained(adapter or model_name,
                                                 trust_remote_code=True)
        if self.tok.pad_token is None:
            self.tok.pad_token = self.tok.eos_token

        kw = dict(dtype=torch.bfloat16, device_map=device_map,
                  trust_remote_code=True)
        if load_4bit:
            from transformers import BitsAndBytesConfig
            kw["quantization_config"] = BitsAndBytesConfig(
                load_in_4bit=True, bnb_4bit_compute_dtype=torch.bfloat16,
                bnb_4bit_quant_type="nf4", bnb_4bit_use_double_quant=True)

        self.model = AutoModelForCausalLM.from_pretrained(model_name, **kw)
        if adapter:
            from peft import PeftModel
            self.model = PeftModel.from_pretrained(self.model, adapter)
        self.model.eval()

        self._pass_ids = self._first_ids("PASS")
        self._fail_ids = self._first_ids("FAIL")

    def _first_ids(self, word: str) -> list[int]:
        ids = set()
        for v in (word, " " + word, word.capitalize(), word.lower()):
            t = self.tok(v, add_special_tokens=False)["input_ids"]
            if t:
                ids.add(t[0])
        return sorted(ids)

    def p_fail(self, spec: str, code: str) -> float:
        torch = self._torch
        msgs = [{"role": "system", "content": SYSTEM},
                {"role": "user", "content": TMPL.format(spec=spec, code=code)}]
        prompt = self.tok.apply_chat_template(msgs, tokenize=False,
                                              add_generation_prompt=True)
        ids = self.tok(prompt, add_special_tokens=False)["input_ids"]
        if len(ids) > self.max_len:
            ids = ids[:40] + ids[-(self.max_len - 40):]
        with torch.no_grad():
            logits = self.model(
                torch.tensor([ids], device=self.model.device)).logits[0, -1].float()
            lp = torch.logsumexp(logits[self._pass_ids], 0)
            lf = torch.logsumexp(logits[self._fail_ids], 0)
            return torch.softmax(torch.stack([lp, lf]), 0)[1].item()


# ---------------------------------------------------------------- the gate

@dataclass
class GateDecision:
    allow: bool           # True -> send to simulation
    p_fail: float
    threshold: float
    seconds: float
    reason: str

    def __bool__(self) -> bool:
        return self.allow


class TriageGate:
    """
    Sits in front of the expensive evaluation step of a CHIA loop.

    gate = TriageGate(scorer, threshold=0.9, ledger=ledger, hourly_usd=0.70)
    if gate.allow(spec, code, iteration=n):
        result = run_simulation(code)
    print(gate.stats())
    """

    def __init__(self, scorer: Scorer, threshold: float = 0.9,
                 ledger=None, hourly_usd: float = 0.0,
                 node_name: str = "triage_gate"):
        if not 0.0 < threshold < 1.0:
            raise ValueError("threshold must be strictly between 0 and 1")
        self.scorer = scorer
        self.threshold = threshold
        self.ledger = ledger
        self.hourly_usd = hourly_usd
        self.node_name = node_name
        self.decisions: list[GateDecision] = []

    def decide(self, spec: str, code: str, iteration: int = 0) -> GateDecision:
        t0 = time.time()
        if self.ledger is not None:
            with self.ledger.track(self.node_name, iteration=iteration) as t:
                p = self.scorer.p_fail(spec, code)
                if self.hourly_usd:
                    t.compute(hourly_usd=self.hourly_usd)
                else:
                    t.flat(0.0, note="gate scoring")
        else:
            p = self.scorer.p_fail(spec, code)

        secs = time.time() - t0
        allow = p < self.threshold
        d = GateDecision(
            allow=allow, p_fail=p, threshold=self.threshold, seconds=secs,
            reason=("below threshold, simulate" if allow
                    else f"P(fail)={p:.3f} >= {self.threshold}, skip"))
        self.decisions.append(d)
        return d

    def allow(self, spec: str, code: str, iteration: int = 0) -> bool:
        return self.decide(spec, code, iteration).allow

    # ---------------------------------------------------------------- stats

    def stats(self) -> str:
        n = len(self.decisions)
        if not n:
            return "no decisions yet"
        blocked = sum(1 for d in self.decisions if not d.allow)
        secs = sum(d.seconds for d in self.decisions)
        return (f"gate: {n} decisions, {blocked} blocked "
                f"({100*blocked/n:.1f}%), {n-blocked} sent to simulation\n"
                f"      {secs:.1f}s total, {secs/n:.3f}s per decision")

    @property
    def blocked(self) -> int:
        return sum(1 for d in self.decisions if not d.allow)
