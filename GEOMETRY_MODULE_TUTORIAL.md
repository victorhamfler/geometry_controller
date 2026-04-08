# LCM Geometry Module - User and Agent Tutorial

**Version:** 1.3
**Module:** `lcm_geometry_controller.py`
**Last Updated:** 2026-04-08

This tutorial explains how to use the geometry module as a semantic companion to OpenClaw LCM.

---

## 1. Core idea

LCM keyword search and geometry semantic search are complementary:

1. `lcm.db` stores raw messages/summaries (source of truth).
2. `lcm_geometry.db` stores branch geometry, lifecycle/regime signals, and retrieval metadata.

The geometry layer uses embeddings (default 384-dim, `all-MiniLM-L6-v2`) to find related context even when exact keywords differ.

---

## 2. Branch states (practical use)

The engine has multiple states (`FORMING`, `ACTIVE`, `STABLE`, `TENSIONED`, `DORMANT`, `REACTIVATING`, `SPLIT_PENDING`, `MERGE_CANDIDATE`, `COLLAPSING`).

For retrieval decisions, the most useful simplification is:

| State | Practical meaning | Recall value |
|---|---|---|
| `ACTIVE` | Ongoing, evolving useful work context | High |
| `STABLE` | Settled reference context | Medium |
| `DORMANT` | Quiet/low-usefulness branch (kept for possible reactivation) | Contextual |
| `FORMING` | Early/small branches with weaker signal | Low |

Current runtime uses inactivity + usefulness policy to move ACTIVE/STABLE/TENSIONED branches into `DORMANT`, and fresh activity moves them through `REACTIVATING` back to `ACTIVE`.

---

## 3. MCP tools

Use these 4 tools exposed by `geometry-hybrid`:

- `geometry-hybrid__hybrid_search`: combined semantic + keyword ranking.
- `geometry-hybrid__conversation_content`: read summaries/messages for a branch (`conv_*`).
- `geometry-hybrid__branch_report`: inspect one branch (state/regime/metrics).
- `geometry-hybrid__geometry_stats`: global geometry DB health metrics.

Recommended flow: `hybrid_search` -> `branch_report` (if needed) -> `conversation_content`.

---

## 4. Agent recall loop

1. Search first with `geometry-hybrid__hybrid_search`.
2. Prioritize strong semantic candidates and healthy branch signals.
3. Pull summaries first via `conversation_content`.
4. Expand to messages only when needed for evidence.

---

## 5. Important config behavior

| Field | Default | Meaning |
|---|---|---|
| `embedding_dim` | `384` | Embedding vector size, must match model output |
| `min_branch_size` | `8` | Minimum rows for full geometry recompute |
| `attach_threshold` | `0.50` | CSD below this -> `attach` |
| `tension_threshold` | `0.70` | CSD in `[attach_threshold, tension_threshold)` -> `attach_tension`; above -> `fork` |
| `alpha_sem` | `0.60` | Semantic weight in retrieval ranking |
| `beta_trust` | `0.25` | Trust/quality weight in retrieval ranking |
| `split_score_threshold` | `0.075` | Split score gate threshold |
| `split_min_nodes` | `6` | Baseline split gate; effective readiness uses `max(split_min_nodes, min_branch_size)` |
| `max_split_enqueues_per_cycle` | `5` | Maximum split jobs queued in one maintenance cycle (highest scores first) |
| `merge_signal_lookback` | `5000` | Retrieval co-use lookback rows for merge scoring |
| `merge_execution_mode` | `"soft"` | Execute pending merge jobs by writing affinity edges and clearing queue |
| `merge_max_jobs_per_cycle` | `5` | Max merge jobs executed per maintenance cycle |
| `contradiction_sim_threshold` | `-0.30` | Cosine threshold used to detect contradiction pairs |
| `contradiction_sample_max_nodes` | `192` | Cap contradiction matrix size for large branches (`0` disables cap) |
| `dormant_after_days` | `14.0` | Inactivity threshold for dormancy |
| `dormant_usefulness_max` | `0.20` | Branch usefulness must be below this to become dormant |
| `split_child_copy_usefulness` | `true` | Split children inherit parent usefulness |
| `split_child_anchor_from_centroid` | `true` | Split children seed anchor from cluster centroid |

---

## 6. Quick start commands

### Optional MCP runtime config

You can tune geometry behavior at runtime without editing Python code:

```bash
cp <module_repo_root>/extensions/geometry-mcp/runtime_config.example.json \
   <openclaw_home>/extensions/geometry-mcp/runtime_config.json
```

Then edit `runtime_config.json` and restart gateway:

```bash
openclaw gateway restart
```

The MCP server loads:
- `<openclaw_home>/extensions/geometry-mcp/runtime_config.json`
- optional env override `GEOMETRY_RUNTIME_CONFIG_JSON` (JSON object)

### Verify module

```bash
cd <module_repo_root>
python3 scripts/smoke_test_geometry.py
# expected: SMOKE_OK ...
```

### Run maintenance manually

```bash
python3 - <<'PYCODE'
import sys
sys.path.insert(0, '<module_repo_root>')
from lcm_geometry_controller import GeometryController

gc = GeometryController('<openclaw_home>/lcm_geometry.db')
print(gc.run_maintenance_cycle())
PYCODE
```

If you choose to use `~` in Python paths, expand it explicitly:

```python
import os
db_path = os.path.expanduser('~/.openclaw/lcm_geometry.db')
```

---

## 7. Incremental ingest loop

Use `poll_lcm_for_new_items(...)` for real-time style ingestion without full backfill:

```python
cursor = 0
while True:
    r = gc.poll_lcm_for_new_items('<openclaw_home>/lcm.db', since_rowid=cursor, limit=200)
    cursor = r['next_rowid']
    # sleep or schedule externally
```

This preserves a rowid cursor and processes only new messages.

---

## 8. Operating tips

- Run `run_maintenance_cycle()` periodically (for example every 20-30 minutes).
- Use `resume=True` for incremental backfill runs.
- Check `split_trace_run_id` and `split_observations` from maintenance output when validating split behavior.
- Use `gc.health_report()` for quick state/regime and pending-job visibility.
- Keep this tutorial aligned with `GEOMETRY_CONTROLLER_MANUAL.md` when behavior changes.

---

## 9. Summary

The geometry module improves memory retrieval quality by adding semantic structure on top of LCM history. It complements LCM; it does not replace it.
