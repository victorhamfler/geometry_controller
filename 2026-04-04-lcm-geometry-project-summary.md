# LCM Geometry Module - Project Summary

**Status:** Active (MCP v1.1 runtime)
**Updated:** 2026-04-05

## What this project does

The LCM Geometry module adds semantic retrieval and branch diagnostics on top of OpenClaw LCM.
It is read-only with respect to lcm.db and stores geometry state in lcm_geometry.db.

Main capabilities:
- semantic branch ranking from embeddings
- branch-level geometry metrics (coherence, anisotropy, effective rank)
- branch lifecycle tracking for maintenance and retrieval trust
- MCP tools for hybrid search and content bridge

## Runtime architecture

- Native memory source: lcm.db
- Geometry companion database: lcm_geometry.db
- Engine module: lcm_geometry_controller.py
- Backfill/import utility: lcm_geometry_backfill.py
- MCP server: extensions/geometry-mcp/server.py

## MCP tools (current)

The geometry-hybrid MCP server exposes 4 tools:
- hybrid_search
- ranch_report
- geometry_stats
- conversation_content

## Notes for repository packaging

- Treat skills/geometry-hybrid as legacy tooling (not required for MCP runtime).
- Keep docs aligned to MCP v1.1 behavior (4-tool surface).
- Exclude local artifacts from repo (__pycache__, *.pyc, :Zone.Identifier, transient logs/progress files).

## Quick runbook

1. Install dependencies from equirements.txt.
2. Run python3 scripts/smoke_test_geometry.py.
3. Configure OpenClaw MCP server entry to point at extensions/geometry-mcp/server.py.
4. Restart gateway and validate geometry-hybrid appears in openclaw mcp list.
