# CLI reference

All direct training commands work without a daemon:

```console
kernelyra [--workspace PATH] [--json] plan DATASET [OPTIONS]
kernelyra [--workspace PATH] [--json] train DATASET [OPTIONS]
kernelyra [--workspace PATH] [--json] finetune MODEL DATASET [OPTIONS]
kernelyra [--workspace PATH] rpc [--config FILE]
kernelyra [--workspace PATH] [--json] infer RUN_ID [--requests 200]
```

Training options are `--config`, `--target`, `--task`, `--backend`,
`--architecture`, `--model-format`, `--profile`,
`--batch-size`, `--accept-batch-risk`, `--max-steps`, `--target-metric`,
`--learning-rate`, `--weight-decay`, `--hidden-layers`, `--precision`, `--cpu`,
`--ram`, `--gpu`, `--data-workers`, `--prefetch`, `--seed`,
`--evaluation-interval`, `--min-improvement`, `--degradation-margin`,
`--degradation-patience`, `--early-stopping-patience`, and
`--target-patience`. Use `--profile custom` for complete manual control.

Resolution order is explicit CLI/Python value, environment, `[training]` TOML,
then automatic planning. Environment variables are:

`KERNELYRA_TARGET`, `KERNELYRA_TASK`, `KERNELYRA_BACKEND`,
`KERNELYRA_ARCHITECTURE`, `KERNELYRA_MODEL_FORMAT`,
`KERNELYRA_PROFILE`, `KERNELYRA_BATCH_SIZE`, `KERNELYRA_MAX_STEPS`,
`KERNELYRA_TARGET_METRIC`, `KERNELYRA_CPU_PERCENT`, `KERNELYRA_RAM_PERCENT`,
`KERNELYRA_GPU_PERCENT`, `KERNELYRA_SEED`, `KERNELYRA_LEARNING_RATE`,
`KERNELYRA_WEIGHT_DECAY`, `KERNELYRA_HIDDEN_LAYERS`, `KERNELYRA_PRECISION`,
`KERNELYRA_DATA_WORKERS`, and `KERNELYRA_PREFETCH`.

Inspection/maintenance commands remain available: `doctor`, `capabilities`,
`dataset`, `run`, `config`, `workspace`, `repair`, `cleanup`, and
`migrate`. `daemon`/`serve` expose the optional headless HTTP gateway; `mcp`
exposes the permission-scoped MCP gateway. No command opens a browser.

Expected failures return code 2, authorization failures 4, unavailable daemon 5,
and Ctrl+C 130. `--json` emits machine-readable results without tracebacks.
