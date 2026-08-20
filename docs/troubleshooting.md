# Troubleshooting

- `Backend ... unavailable`: install `.[torch]`, `.[tensorflow]`, or select NumPy.
- unsafe batch rejection: remove `--batch-size`, lower it, or review the warning
  and pass `--accept-batch-risk` deliberately.
- external stream changed: restore the original file or import/register the new
  dataset as a new run; Kernelyra will not resume against different bytes.
- out of memory: use `--profile low-memory`, lower batch size, set `--prefetch 0`,
  reduce worker count, or choose NumPy.
- Ctrl+C: Kernelyra requests an emergency stop and keeps checkpoint state; inspect
  the run before resuming.
- training page not found: expected. Training is terminal/code only; there is no
  browser surface.
- machine integration: use `--json` for one-shot commands or JSONL RPC for a
  persistent process. Do not scrape human progress text.
