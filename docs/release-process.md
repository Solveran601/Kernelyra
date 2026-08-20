# Release process

Use one entry point:

```console
python scripts/release.py check
python scripts/release.py build
python scripts/release.py verify
python scripts/release.py bundle
```

`check` validates source/version and runs lint, type checking and tests. It
enforces 85% line coverage on the explicitly listed Kernelyra core and 90% on
security plus runtime lifecycle/state-machine code through
`scripts/check_coverage.py`. Transport, optional-backend and platform adapters
retain dedicated integration and Windows gates.

`build` creates a Windows wheel, source distribution and allowlisted source ZIP.
`verify` scans and extracts every artifact, installs the wheel in a clean
environment, then exercises import, direct CLI training, JSONL RPC, the headless
daemon and MCP. `bundle` writes SHA-256 checksums, the dependency manifest/SBOM
and `RELEASE_MANIFEST.json`.

Stable 1.0 additionally requires green Windows x64 jobs on Python 3.11-3.13,
minimum/pinned/latest dependency profiles, a quick soak and an actually completed
60-minute full Windows soak. If any external gate is unavailable or red, the
version remains a prerelease. Artifacts are never uploaded to PyPI without owner
authorization.
