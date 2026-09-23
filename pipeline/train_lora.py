#!/usr/bin/env python3
"""
train_lora.py - LoRA fine-tune a small code model to predict PASS/FAIL.

Deliberately uses plain transformers + peft (no trl), because trl's API moves
around between releases and we cannot afford a dependency surprise.

The prompt format is IDENTICAL to eval_gemini.py. That is what makes the
comparison fair: every model sees exactly the same spec and the same code.

Loss is computed on the answer token only. The model is not asked to
reproduce the prompt, just to decide PASS or FAIL.

  python3 train_lora.py --train train.jsonl --test test.jsonl \
      --model Qwen/Qwen2.5-Coder-7B-Instruct --out ./adapter
"""
import argparse, json, math, os, random
import torch
from torch.utils.data import Dataset
from transformers import (AutoModelForCausalLM, AutoTokenizer, Trainer,
                          TrainingArguments, DataCollatorForSeq2Seq)
from peft import LoraConfig, get_peft_model, prepare_model_for_kbit_training

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


class PassFailDataset(Dataset):
    def __init__(self, path, tok, max_len=3072, limit=0):
        self.rows = [json.loads(l) for l in open(path)]
        if limit:
            self.rows = self.rows[:limit]
        self.tok = tok
        self.max_len = max_len

    def __len__(self):
        return len(self.rows)

    def __getitem__(self, i):
        r = self.rows[i]
        user = TMPL.format(spec=r.get("spec", ""), code=r["code"])
        msgs = [{"role": "system", "content": SYSTEM},
                {"role": "user", "content": user}]
        prompt = self.tok.apply_chat_template(
            msgs, tokenize=False, add_generation_prompt=True)
        answer = r["label"]                      # "PASS" or "FAIL"

        p_ids = self.tok(prompt, add_special_tokens=False)["input_ids"]
        a_ids = self.tok(answer, add_special_tokens=False)["input_ids"]
        if self.tok.eos_token_id is not None:
            a_ids = a_ids + [self.tok.eos_token_id]

        # left-truncate the prompt if the code is very long; keep the answer
        room = self.max_len - len(a_ids)
        if len(p_ids) > room:
            p_ids = p_ids[:40] + p_ids[-(room - 40):]

        ids = p_ids + a_ids
        labels = [-100] * len(p_ids) + a_ids     # loss on the answer only
        return dict(input_ids=ids, attention_mask=[1] * len(ids), labels=labels)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--train", default="train.jsonl")
    ap.add_argument("--test", default="test.jsonl")
    ap.add_argument("--model", default="Qwen/Qwen2.5-Coder-7B-Instruct")
    ap.add_argument("--out", default="./adapter")
    ap.add_argument("--epochs", type=float, default=2.0)
    ap.add_argument("--lr", type=float, default=1e-4)
    ap.add_argument("--batch", type=int, default=1)
    ap.add_argument("--accum", type=int, default=16)
    ap.add_argument("--max-len", type=int, default=3072)
    ap.add_argument("--rank", type=int, default=16)
    ap.add_argument("--load-4bit", action="store_true",
                    help="use if the GPU has under 24GB (an L4 running a 7B)")
    ap.add_argument("--limit", type=int, default=0)
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()

    random.seed(args.seed); torch.manual_seed(args.seed)
    print(f"GPU: {torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'NONE'}")
    if torch.cuda.is_available():
        total = torch.cuda.get_device_properties(0).total_memory / 1e9
        print(f"VRAM: {total:.1f} GB")

    tok = AutoTokenizer.from_pretrained(args.model, trust_remote_code=True)
    if tok.pad_token is None:
        tok.pad_token = tok.eos_token
    tok.padding_side = "right"

    kw = dict(torch_dtype=torch.bfloat16, device_map="auto",
              trust_remote_code=True)
    if args.load_4bit:
        from transformers import BitsAndBytesConfig
        kw["quantization_config"] = BitsAndBytesConfig(
            load_in_4bit=True, bnb_4bit_compute_dtype=torch.bfloat16,
            bnb_4bit_quant_type="nf4", bnb_4bit_use_double_quant=True)

    model = AutoModelForCausalLM.from_pretrained(args.model, **kw)
    model.config.use_cache = False
    if args.load_4bit:
        model = prepare_model_for_kbit_training(model)

    lcfg = LoraConfig(
        r=args.rank, lora_alpha=args.rank * 2, lora_dropout=0.05,
        bias="none", task_type="CAUSAL_LM",
        target_modules=["q_proj", "k_proj", "v_proj", "o_proj",
                        "gate_proj", "up_proj", "down_proj"])
    model = get_peft_model(model, lcfg)
    model.print_trainable_parameters()

    train_ds = PassFailDataset(args.train, tok, args.max_len, args.limit)
    print(f"training examples: {len(train_ds)}")
    lab = [json.loads(l)["label"] for l in open(args.train)]
    print(f"  PASS={lab.count('PASS')}  FAIL={lab.count('FAIL')}")

    steps = max(1, math.ceil(len(train_ds) / (args.batch * args.accum) * args.epochs))
    print(f"optimizer steps: ~{steps}")

    targs = TrainingArguments(
        output_dir=args.out, num_train_epochs=args.epochs,
        per_device_train_batch_size=args.batch,
        gradient_accumulation_steps=args.accum,
        learning_rate=args.lr, lr_scheduler_type="cosine", warmup_ratio=0.03,
        logging_steps=5, save_strategy="epoch", save_total_limit=1,
        bf16=True, gradient_checkpointing=True, report_to=[],
        remove_unused_columns=False, seed=args.seed)

    collator = DataCollatorForSeq2Seq(tok, padding=True, label_pad_token_id=-100)
    trainer = Trainer(model=model, args=targs, train_dataset=train_ds,
                      data_collator=collator)
    trainer.train()

    model.save_pretrained(args.out)
    tok.save_pretrained(args.out)
    print(f"\nadapter saved to {args.out}")
    print("UPLOAD THIS NOW - the GCP account expires and cannot be recovered:")
    print(f"  gsutil -m cp -r {args.out} gs://chia-backup-7722/")


if __name__ == "__main__":
    main()
