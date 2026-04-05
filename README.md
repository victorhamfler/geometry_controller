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
