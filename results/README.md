# Results

`results.md` holds every table in the paper. Regenerate it with:

```bash
python3 ../make_results.py --dir . > results.md
```

| File | Contents |
|---|---|
| `preds_base.jsonl` | Untrained Qwen2.5-Coder-7B on the test set |
| `preds_tuned.jsonl` | Fine-tuned model, same test set |
| `preds_gemini-2_5-*.jsonl` | Gemini Flash, Flash-Lite and Pro |
| `preds_gemini-3*.jsonl` | Empty. Listed by the SDK but not enabled for inference; all calls returned 404. |
| `baseline_summary.jsonl` | Aggregate metrics per model |
| `train_7b.log` | Training log, 244 steps over 72 minutes on one L4 |

Local rows carry `p_fail`, a probability, so the gate's operating point can be
swept after the fact. Gemini rows carry token counts and latency.
