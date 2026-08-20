# MCP 1.0

Kernelyra exposes stdio MCP tools for path/dataset inspection, import, dataset
listing, run creation/control, metrics, logs, hardware, capabilities, backend and
ingestor discovery, and secret-free run manifest export.

Resources are `kernelyra://system/health`, `kernelyra://system/capabilities`,
`kernelyra://datasets`, `kernelyra://datasets/{id}`, `kernelyra://runs`,
`kernelyra://runs/{id}`, `kernelyra://runs/{id}/metrics`,
`kernelyra://runs/{id}/logs`.

Configure roots/actions in `kernelyra.toml`. Relative roots resolve against the
config directory. Both the MCP wrapper and daemon resolve real paths; the daemon
is authoritative. Agent sessions expire, bind a client ID, are revocable and
cannot access user routes. A user approval is mandatory for import, start,
resume and export and cannot be replayed. Dataset content is never returned;
tools/resources return bounded metadata and schemas.
