# LCM Geometry Module - User and Agent Tutorial

**Version:** 1.1  
**Module:** `lcm_geometry_controller.py`  
**Last Updated:** 2026-04-05

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
| `FORMING` | Early/small branches with weaker signal | Low |

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

---

## 6. Quick start commands

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

## 7. Operating tips

- Run `run_maintenance_cycle()` periodically (for example every 20-30 minutes).
- Use `resume=True` for incremental backfill runs.
- Keep this tutorial aligned with `GEOMETRY_CONTROLLER_MANUAL.md` when behavior changes.

---

## 8. Summary

The geometry module improves memory retrieval quality by adding semantic structure on top of LCM history. It complements LCM; it does not replace it.
