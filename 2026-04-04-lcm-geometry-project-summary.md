# LCM Geometry Module - Project Summary

**Status:** Active (OpenClaw + MCP runtime)
**Last Updated:** 2026-04-06

## Purpose

This project adds a semantic geometry layer on top of OpenClaw LCM memory.

- Source of truth remains `lcm.db` (messages/summaries)
- Geometry signals and retrieval metadata are stored in `lcm_geometry.db`
- MCP server exposes the geometry layer as tools for agents

## Current runtime architecture

- `lcm_geometry_controller.py`: branch geometry engine
- `lcm_geometry_backfill.py`: creates/refreshes geometry DB from LCM
- `extensions/geometry-mcp/server.py`: MCP tool server
- OpenClaw runtime data:
  - `<openclaw_home>/lcm.db`
  - `<openclaw_home>/lcm_geometry.db`

## Current capabilities (v1.2 line)

- Real-time compatible ingest API via `poll_lcm_for_new_items(...)`
- Parent-aware node insertion (`parent_lcm_id`) + `TEMPORAL_NEXT` link creation
- Contradiction refresh pipeline:
  - contradiction density recomputation per branch
  - bounded `CONTRADICTS` edge regeneration
- Merge scoring with runtime evidence:
  - graph overlap (memory edges)
  - retrieval co-use signal
- Split execution pipeline:
  - pending split jobs executed automatically
  - internal k-means(2) branch partitioning
  - `REFINES` edges from source to child branches
- Operational observability APIs:
  - `health_report()`
  - cross-agent links (`mark_branch_agent_interest`, `add_cross_agent_shared_edge`, `list_cross_agent_links`)

## What is required for a working setup

1. Python dependencies installed from `requirements.txt`
2. Engine files deployed to OpenClaw module path:
   - `<openclaw_home>/workspace/module/lcm_geometry_controller.py`
   - `<openclaw_home>/workspace/module/lcm_geometry_backfill.py`
3. MCP entrypoint deployed:
   - `<openclaw_home>/extensions/geometry-mcp/server.py`
4. Geometry DB generated (or refreshed) by running backfill
5. MCP server configured in OpenClaw as `geometry-hybrid`

## Installation and deployment checklist

```bash
# 1) install deps
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt

# 2) deploy files
export OPENCLAW_HOME="$HOME/.openclaw"
mkdir -p "$OPENCLAW_HOME/workspace/module"
mkdir -p "$OPENCLAW_HOME/extensions/geometry-mcp"
cp lcm_geometry_controller.py "$OPENCLAW_HOME/workspace/module/"
cp lcm_geometry_backfill.py "$OPENCLAW_HOME/workspace/module/"
cp extensions/geometry-mcp/server.py "$OPENCLAW_HOME/extensions/geometry-mcp/"

# 3) build geometry db
OPENCLAW_HOME="$OPENCLAW_HOME" \
GEOMETRY_MODULE_HOME="$OPENCLAW_HOME/workspace/module" \
python3 lcm_geometry_backfill.py

# 4) register MCP server + restart
openclaw mcp set geometry-hybrid '{"command":"python3","args":["<openclaw_home>/extensions/geometry-mcp/server.py"]}'
openclaw gateway restart
openclaw mcp list
```

## MCP tool surface (current)

- `hybrid_search`
- `branch_report`
- `geometry_stats`
- `conversation_content`

## Operational notes

- Backfill is the key synchronization step between LCM and geometry.
- For incremental ingest, use `poll_lcm_for_new_items(...)` with a persisted rowid cursor.
- If retrieval quality drops or results look stale, run backfill again.
- `GEOMETRY_CONTROLLER_MANUAL.md` is the full reference.
- `GEOMETRY_MODULE_TUTORIAL.md` is the practical user/agent guide.