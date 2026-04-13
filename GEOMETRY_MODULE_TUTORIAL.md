# LCM Geometry Module - User and Agent Tutorial

**Version:** 1.5
**Module:** `lcm_geometry_controller.py`
**Last Updated:** 2026-04-13

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

Use these 11 tools exposed by `geometry-hybrid`:

- `geometry-hybrid__hybrid_search`: combined semantic + keyword ranking.
- `geometry-hybrid__retrieval_feedback`: explicit feedback for a `hybrid_search` result (`query_id`, `branch_id`).
- `geometry-hybrid__conversation_content`: read summaries/messages for a branch (`conv_*`).
- `geometry-hybrid__branch_report`: inspect one branch (state/regime/metrics + `update_mode_counts`).
- `geometry-hybrid__geometry_stats`: global geometry DB health metrics.
- `geometry-hybrid__maintenance_cycle`: run one maintenance cycle (supports chunking via `max_branches`) and reports retrieval-feedback pruning counters.
- `geometry-hybrid__geometry_snapshot`: export compact branch metrics (`state`, `branch_ids`, `limit`, optional `include_means`).
- `geometry-hybrid__latest_correction`: resolve correction chain and return latest correction for any seed `node_id`.
- `geometry-hybrid__sync_lcm_ingest`: force one incremental LCM->geometry ingest poll.
- `geometry-hybrid__sync_lcm_dag_edges`: rebuild imported DAG edges and return orphan-validation counters.
- `geometry-hybrid__backfill_lcm_conversations`: targeted backfill for specific LCM conversation IDs.

Recommended flow: `hybrid_search` -> `branch_report` (if needed) -> `conversation_content`.

`sync_lcm_ingest` output now includes:
- `skipped_duplicates` (already-ingested messages in requested row window)
- `lag_rows` + rowid cursor status (`next_rowid` vs `lcm_max_rowid`)

`conversation_content` output now includes:
- `resolution_mode` (`branch_lineage`, `suffix_fallback`, `daily_log`)
- `resolved_conversation_ids` (which LCM conversations were actually used)
- optional warnings for mapping/mixing:
  - `branch_suffix_mismatch:conv_X->conv_Y`
  - `mixed_branch_content:<ids>`
  - `lineage_empty_used_suffix_fallback`

For `hybrid_search`, use:
- `retrieval_mode="factual"` when precision/reliability matters.
- `retrieval_mode="exploratory"` when broad discovery matters.
- `retrieval_mode="balanced"` for normal usage.

For `backfill_lcm_conversations`:
- Use `dry_run=true` for preview mode without embeddings.
- Real mode (`dry_run=false`) needs an embedding provider.
- If provider is missing, output shows `provider_ready=false`, `aborted=true`, and `preflight_error`.

`maintenance_cycle` output includes:
- `retrieval_feedback_pruned`
- `retrieval_feedback_pruned_age`
- `retrieval_feedback_pruned_cap`

`latest_correction` supports:
- `include_chain=true` for ordered chain entries
- `chain_limit=<N>` to cap returned chain payload

---

## 4. Agent recall loop

1. Search first with `geometry-hybrid__hybrid_search`.
2. Prioritize strong semantic candidates and healthy branch signals.
3. Pull summaries first via `conversation_content`.
4. Expand to messages only when needed for evidence.
5. Run `sync_lcm_dag_edges` after major backfill/import refreshes.

---

## 5. Important config behavior

| Field | Default | Meaning |
|---|---|---|
| `embedding_dim` | `384` | Embedding vector size, must match model output |
| `min_branch_size` | `8` | Minimum rows for full geometry recompute |
| `attach_threshold` | `0.50` | Global fallback attach threshold |
| `tension_threshold` | `0.70` | Global fallback tension threshold |
| `attach_threshold_by_type` | `{"default":0.5}` | Optional per-branch-type attach threshold map |
| `tension_threshold_by_type` | `{"default":0.7}` | Optional per-branch-type tension threshold map (must be >= attach threshold) |
| `branch_type_profiles` | `{...}` | Optional per-branch metric profile overrides (`csd_gamma`, `retrieval_kappa`, `split_zeta`, `split_policy`, `merge_eta`, `merge_policy`) |
| `alpha_sem` | `0.60` | Semantic weight in retrieval ranking |
| `beta_trust` | `0.25` | Trust/quality weight in retrieval ranking |
| `retrieval_mode_default` | `"balanced"` | Default retrieval routing mode (`balanced`, `factual`, `exploratory`) |
| `retrieval_mode_factors` | `{...}` | Regime/state multipliers per retrieval mode |
| `split_score_threshold` | `0.075` | Split score gate threshold |
| `split_min_nodes` | `6` | Baseline split gate; effective readiness uses `max(split_min_nodes, min_branch_size)` |
| `max_split_enqueues_per_cycle` | `5` | Maximum split jobs queued in one maintenance cycle (highest scores first) |
| `merge_signal_lookback` | `5000` | Retrieval co-use lookback rows for merge scoring |
| `merge_execution_mode` | `"soft"` | Execute pending merge jobs by writing affinity edges and clearing queue |
| `merge_max_jobs_per_cycle` | `5` | Max merge jobs executed per maintenance cycle |
| `topic_drift_sim_threshold` | `0.00` | Cosine threshold for subtopic drift pairs (`sim < threshold`) |
| `topic_drift_min_temporal_gap` | `48` | Minimum sequence distance between compared messages |
| `topic_drift_sample_max_nodes` | `192` | Cap topic-drift matrix size for large branches (`0` disables cap) |
| `topic_drift_min_content_chars` | `30` | Require both paired messages to have at least this many non-whitespace chars |
| `topic_drift_require_content_nonempty` | `true` | Enforce content-based filtering using `lcm.db` message text |
| `dormant_after_days` | `14.0` | Inactivity threshold for dormancy |
| `dormant_usefulness_max` | `0.20` | Branch usefulness must be below this to become dormant |
| `protected_branch_types` | `["identity","user_fact",...]` | Branch types with hard attach/merge protection |
| `protected_attach_conflict_threshold` | `0.35` | Conflict gate for protected attach |
| `protected_attach_topic_drift_threshold` | `0.20` | Topic-drift density gate for protected attach |
| `reactivation_guard_enabled` | `true` | Enable safe reactivation checks |
| `reactivation_min_score` | `0.60` | Minimum reactivation score before wake |
| `reactivation_max_topic_drift` | `0.35` | Max topic drift allowed for wake |
| `reactivation_max_retrieval_error` | `0.60` | Max retrieval error allowed for wake |
| `reactivation_min_similarity` | `0.15` | Min semantic similarity for relevance-triggered wake |
| `update_mode_refine_similarity_min` | `0.92` | Similarity floor to classify node insertions as `refine` |
| `update_mode_contradict_conflict_min` | `0.25` | Conflict floor to classify insertions as `contradict` |
| `update_mode_supersede_similarity_min` | `0.78` | Similarity floor for `supersede` classification |
| `update_mode_supersede_conflict_min` | `0.70` | Conflict floor for `supersede` classification |
| `update_mode_supersede_branch_types` | `["identity","user_fact",...]` | Branch types eligible for `supersede` |
| `polling.enabled` *(top-level)* | `true` | Enable automatic incremental ingest on tool calls |
| `polling.interval_seconds` *(top-level)* | `8` | Cooldown between automatic ingest polls |
| `polling.limit` *(top-level)* | `200` | Max messages ingested per poll |
| `polling.cursor_path` *(top-level)* | `<openclaw_home>/.../poll_cursor.json` | Persistent rowid cursor path |
| `split_child_copy_usefulness` | `true` | Split children inherit parent usefulness |
| `split_child_anchor_from_centroid` | `true` | Split children seed anchor from cluster centroid |

Compatibility note:
- `topic_drift_density` is the primary stored field.
- `contradiction_density` is kept as a synced legacy mirror, and report aliases `topic_drift_density` and `subtopic_diversity_density` resolve to the same signal.

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

## 8. DAG edge integrity sync (admin)

When you need to refresh imported summary DAG links (`summarizes`, `derived_from`) from LCM:

```
geometry-hybrid__sync_lcm_dag_edges(backup=true)
```

Expected healthy validation:
- `orphan_by_type.summarizes = 0`
- `orphan_by_type.derived_from = 0`

Use `backup=false` only when you explicitly do not want a pre-sync DB backup.

### 8.1 One-command cleanup + sync

If you want a standard maintenance pass (Lossless doctor-clean + geometry DAG sync/validation):

```bash
/home/victo/.openclaw/workspace/scripts/lossless_geometry_maintenance.sh --text
```

Apply cleanup + sync:

```bash
/home/victo/.openclaw/workspace/scripts/lossless_geometry_maintenance.sh --apply --text
```

This avoids dependency on root `/lcm` shell command routing and gives deterministic JSON/text outputs for operator logs.

---

## 9. Operating tips

- Run `run_maintenance_cycle()` periodically (for example every 20-30 minutes).
- Maintenance is scalar-first: branch blobs are only loaded when full recompute/merge vectors are needed.
- Use `resume=True` for incremental backfill runs.
- Check `split_trace_run_id` and `split_observations` from maintenance output when validating split behavior.
- Use `gc.health_report()` for quick state/regime and pending-job visibility.
- Keep this tutorial aligned with `GEOMETRY_CONTROLLER_MANUAL.md` when behavior changes.

---

## 10. Summary

The geometry module improves memory retrieval quality by adding semantic structure on top of LCM history. It complements LCM; it does not replace it.
