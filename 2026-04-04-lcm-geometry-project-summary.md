# LCM Geometry Module - Project Summary

**Status:** Active (OpenClaw + MCP runtime)
**Last Updated:** 2026-04-08

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
- Split gate hardening and throttling:
  - score gate uses `split_score_threshold`
  - readiness uses `max(split_min_nodes, min_branch_size)`
  - real embedded-node gate required before split hysteresis can accumulate
  - per-cycle split enqueue cap with score-priority selection
- Split observability:
  - per-branch split decisions persisted in `maintenance_split_observations`
  - maintenance output includes `split_trace_run_id` and `split_observations`
- Backfill reliability:
  - deterministic branch-lock insertion keeps `conv_<conversation_id>` mapping stable
  - structured backfill error logging and `errors_logged` metric
- Operational observability APIs:
  - `health_report()`
  - cross-agent links (`mark_branch_agent_interest`, `add_cross_agent_shared_edge`, `list_cross_agent_links`)
- Performance/runtime hardening:
  - scalar/lazy branch loading for retrieval and allocation prefilter
  - scalar-first maintenance persistence (full blobs loaded only when needed)
  - bounded contradiction compute for large branches (temporal stratified sampling)
  - merge execution pipeline (`soft` mode) with queue drain
  - dormancy policy based on inactivity + usefulness
  - split child prior propagation (usefulness/retrieval_error + centroid anchor seed)
- Allocation policy hardening:
  - branch-type aware CSD thresholds (`attach_threshold_by_type`, `tension_threshold_by_type`)
  - fallback to global thresholds (`attach_threshold`, `tension_threshold`)
- Retrieval policy hardening:
  - retrieval modes (`balanced`, `factual`, `exploratory`)
  - regime/state-aware ranking multipliers per mode
- Branch-type metric profiles:
  - optional per-branch class weight profiles for CSD, retrieval trust/react, split, and merge
  - default path remains unchanged without profile config
- Protected memory safety gates:
  - protected branch types can force fork on risky attach
  - protected merge queueing can be blocked by policy
- Safe reactivation:
  - wake transitions gated by contradiction density, retrieval error, and optional similarity checks
- Schema compatibility:
  - connect-time migration keeps legacy DBs compatible by auto-adding `reactivation_score`
- Update-mode observability:
  - each inserted node is labeled with `update_mode` (`fork`, `attach`, `refine`, `contradict`, `supersede`)
  - branch diagnostics include per-mode counters (`update_mode_counts`)
  - runtime thresholds configurable via `update_mode_*` keys
- Versioned correction flow:
  - nodes persist correction lineage (`correction_kind`, `correction_prev_id`, `correction_root_id`, `correction_version`)
  - explicit relation edges now include `SUPERSEDES` in addition to `REFINES` / `CONTRADICTS`
- MCP real-time ingest wiring:
  - automatic incremental poll from `lcm.db` on tool calls (cooldown + persistent rowid cursor)
  - manual force tool: `sync_lcm_ingest`
  - duplicate-safe polling path (skips existing message ids)
  - polling lag telemetry (`lag_rows`, `cursor_rowid`, `lcm_max_rowid`)
- Runtime config support in MCP server:
  - optional `extensions/geometry-mcp/runtime_config.json`
  - env override `GEOMETRY_RUNTIME_CONFIG_JSON`

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
cp extensions/geometry-mcp/runtime_config.example.json "$OPENCLAW_HOME/extensions/geometry-mcp/"

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
