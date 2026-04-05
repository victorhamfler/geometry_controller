# LCM Geometry Module

Semantic memory overlay for OpenClaw LCM.

## Components

- `lcm_geometry_controller.py`: core geometry engine
- `lcm_geometry_backfill.py`: one-time/periodic import from LCM
- `GEOMETRY_CONTROLLER_MANUAL.md`: operational manual
- `scripts/smoke_test_geometry.py`: quick local verification

## Install

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

## Quick test

```bash
python3 scripts/smoke_test_geometry.py
```

## MCP integration

Use `extensions/geometry-mcp/server.py` as the MCP server entrypoint.

Current tool surface:
- `hybrid_search`
- `branch_report`
- `geometry_stats`
- `conversation_content`

## License

This repository is released under the MIT License. See `LICENSE`.

## Compatibility and attribution

This module is designed to interoperate with OpenClaw and the `@martian-engineering/lossless-claw` ecosystem by reading LCM data structures (for example `lcm.db` messages/summaries) and writing its own companion database.

No direct source-code copy from `lossless-claw` is required for this repository to function. If you later import or copy substantial code from that project, keep its original MIT copyright/license notice in the copied files.
