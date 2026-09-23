# Pipeline

End to end, in order.

| Script | Does |
|---|---|
| `mutate.py` | First-generation generator, faulty edits only. Kept for the record. |
| `mutate2.py` | Generator used for the results: faulty + benign edits, per-operator cap |
| `run_tests.py` | Compiles and simulates every candidate; labels come from the test bench |
| `split_data.py` | Train/test split by problem, attaches the natural-language spec |
| `eval_gemini.py` | Scores Gemini models through Vertex AI; resumable |
| `train_lora.py` | LoRA fine-tune; loss on the answer token only |
| `eval_local.py` | Scores a local model by comparing PASS/FAIL logits; threshold sweep |

Every model sees an identical prompt, defined once and shared across
`eval_gemini.py`, `train_lora.py` and `eval_local.py`. That is what makes the
comparison fair.
