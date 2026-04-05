# LCM Geometry Module - Project Summary

**Status:** Active (MCP runtime)  
**Updated:** 2026-04-05

## What this project does

The LCM Geometry module adds semantic retrieval and branch diagnostics on top of OpenClaw LCM.  
It is read-only with respect to `lcm.db` and stores geometry state in `lcm_geometry.db`.

Main capabilities:
- semantic branch ranking from embeddings
- branch-level geometry metrics (coherence, anisotropy, effective rank)
- branch lifecycle tracking for maintenance and retrieval trust
- MCP tools for hybrid search and content bridging

## Runtime architecture

- Native memory source: `lcm.db`
- Geometry companion database: `lcm_geometry.db`
- Engine module: `lcm_geometry_controller.py`
- Backfill/import utility: `lcm_geometry_backfill.py`
- MCP server: `extensions/geometry-mcp/server.py`

## MCP tools (current)

The `geometry-hybrid` MCP server exposes 4 tools:
- `hybrid_search`
- `branch_report`
- `geometry_stats`
- `conversation_content`

## Repository notes

- This repository includes only the module, docs, and smoke test.

## Quick runbook

1. Install dependencies from `requirements.txt`.
2. Run `python3 scripts/smoke_test_geometry.py`.
3. Configure OpenClaw MCP to run `extensions/geometry-mcp/server.py`.
4. Restart gateway and validate `geometry-hybrid` appears in `openclaw mcp list`.
