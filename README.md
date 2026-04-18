# LCM Geometry Controller + MCP Server

Semantic memory overlay for OpenClaw LCM, with an MCP server that exposes geometry-aware tools.

## Latest Update (2026-04-18)

- MCP timeout/stability hardening for GGUF backends:
  - auto-poll is now tool-scoped (`polling.auto_tools`, default `["hybrid_search"]`)
  - auto-poll uses independent low cap (`polling.auto_limit`, default `3` for `llama_cpp`)
  - auto-poll skips while startup warmup is running (`skipped=warming_up`)
  - `geometry_stats` avoids blocking on GC init during warmup and returns DB scalars fast
  - startup warmup probe defaults to off for `llama_cpp` (`startup.warmup_probe_embed=false`)
  - per-call timing diagnostics added in server stderr:
    - `[geometry-mcp] tool=<name> dur_ms=<ms> poll=<state> warmup_running=<bool>`

## Previous Update (2026-04-12)

- Added scalar/lazy branch loading paths:
  - retrieval prefilter by branch scalars (`retrieval_prefilter_limit`)
  - candidate prefilter for `on_new_item(...)` (`candidate_prefilter_limit`, `candidate_branch_cap`)
  - `health_report()` now uses scalar scan instead of full geometry blob load
  - `run_maintenance_cycle()` now starts from scalar scans and only full-loads branch blobs when required
- Added merge execution pipeline (safe mode):
  - `execute_pending_merges(...)` wired into maintenance
  - `merge_execution_mode="soft"` creates `SAME_TOPIC` affinity edges and drains merge queue
- Reworked contradiction signal into **Topic Drift / Subtopic Diversity** semantics:
  - detector now measures temporally distant low-similarity pairs inside a branch
  - empty/blank content is filtered using LCM message text length
  - trust polarity fixed: higher drift reduces retrieval trust
  - split polarity fixed: higher drift increases split pressure
  - merge polarity fixed: higher drift penalizes merge score
  - controlled by topic_drift_* keys (legacy contradiction_* keys still supported)
  - Step 2 physical rename complete: primary storage is topic_drift_density, drift edges use topic_drift, and contradiction_density remains a compatibility mirror
- Added real dormancy policy:
  - inactivity + low-usefulness -> `DORMANT`
  - activity-based wake path `DORMANT -> REACTIVATING -> ACTIVE`
  - controlled by `dormant_after_days`, `dormant_usefulness_max`, `dormant_min_nodes`
- Added split child prior propagation:
  - child branches inherit parent `usefulness` and `retrieval_error`
  - child anchor can be seeded from split cluster centroid
  - controlled by `split_child_copy_*` options
- Added branch-type aware allocation thresholds:
  - `attach_threshold_by_type` and `tension_threshold_by_type`
  - falls back to global `attach_threshold` / `tension_threshold`
- Added per-node update-mode classification metadata:
  - `update_mode` persisted on `memory_nodes` (`fork` / `attach` / `refine` / `contradict` / `supersede`)
  - branch report now includes `update_mode_counts`
  - configurable via `update_mode_*` runtime keys
- Added versioned correction flow (conflicting-fact lineage):
  - `memory_nodes` now persists `correction_kind`, `correction_prev_id`, `correction_root_id`, `correction_version`
  - explicit correction edges are written: `refines`, `contradicts`, `supersedes`
  - branch report now includes `correction_counts` and `recent_corrections`
- Added real-time incremental LCM polling in MCP server:
  - cursor-based ingest using `poll_lcm_for_new_items(...)` on tool calls (cooldown + lock)
  - duplicate-safe ingest path skips already-seen message `lcm_id` values (`skipped_duplicates`)
  - polling status now exposes lag telemetry (`lag_rows`, `lcm_max_rowid`, `cursor_rowid`)
  - manual force-sync MCP tool: `sync_lcm_ingest`
  - polling behavior configurable via top-level `polling` config block
- Fixed `conversation_content` branch mapping:
  - content resolution now uses actual branch lineage from geometry nodes (`memory_nodes.lcm_id`)
  - no longer assumes `conv_<suffix>` equals LCM `conversation_id`
  - per-branch output now includes `resolution_mode`, `resolved_conversation_ids`, and warnings:
    - `branch_suffix_mismatch:conv_X->conv_Y`
    - `mixed_branch_content:<ids>`
    - `lineage_empty_used_suffix_fallback`
- Fixed DAG edge import integrity:
  - `import_dag_edges_from_lcm(...)` now maps LCM IDs to real geometry node IDs before writing edges
  - imported `summarizes`/`derived_from` edges are rebuilt cleanly (old imported edges purged first)
  - added MCP admin tool `sync_lcm_dag_edges` for one-command rebuild + validation counters
- Added safe schema migration for lifecycle persistence:
  - auto-adds `reactivation_score` column on existing DBs when missing
- Added protected-memory hard gates:
  - protected branch types can force `fork` on high conflict/topic-drift
  - protected merges can be blocked by policy
- Added safe reactivation guard:
  - `DORMANT -> REACTIVATING` now checks topic-drift/error/similarity gates
- Added regime-aware retrieval routing:
  - retrieval modes `balanced` / `factual` / `exploratory`
  - mode profiles weight branch `state` + `regime` during ranking
- Added branch-type metric profiles:
  - optional `branch_type_profiles` lets branch classes override CSD/retrieval/split/merge weights
  - default behavior is unchanged when no profile is configured
- Hardened targeted backfill preflight:
  - `backfill_selected_conversations_from_lcm(...)` now reports `provider_ready`, `aborted`, and `preflight_error`
  - real mode (`dry_run=false`) now fails clearly with `failed_preflight` details when no embedding provider is configured
- Improved MCP targeted-backfill visibility:
  - `backfill_lcm_conversations` now surfaces provider readiness and preflight abort reason in tool output
- MCP server now supports runtime config file/env overrides:
  - `extensions/geometry-mcp/runtime_config.json`
  - `GEOMETRY_RUNTIME_CONFIG_JSON`

## What is included

- `lcm_geometry_controller.py` - core geometry engine (branch metrics, lifecycle/regime, retrieval ranking)
- `lcm_geometry_backfill.py` - builds/refreshes `lcm_geometry.db` from `lcm.db`
- `extensions/geometry-mcp/server.py` - MCP server exposing geometry tools to OpenClaw agents
- `scripts/smoke_test_geometry.py` - fast local smoke test
- `scripts/run_update_mode_regression.py` - deterministic update-mode regression (`fork/refine/contradict/supersede`)
- `scripts/run_polling_regression.py` - deterministic incremental polling regression (cursor + conv mapping)
- `scripts/run_ml_split_regression.py` - end-to-end backfill + split-maintenance regression on real `lcm.db`
- `scripts/cleanup_geometry_duplicates.py` - optional one-time DB repair utility for historical duplicate message ingest rows
- `GEOMETRY_CONTROLLER_MANUAL.md` - full technical manual
- `GEOMETRY_MODULE_TUTORIAL.md` - practical usage tutorial

## Prerequisites

- Linux environment (OpenClaw runtime host)
- Python 3.10+
- OpenClaw installed and running
- LCM database present at `<openclaw_home>/lcm.db` (typically generated by Lossless Claw)

## Install dependencies

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

## Quick local validation

```bash
python3 scripts/smoke_test_geometry.py
# expected: SMOKE_OK ...
python3 scripts/run_update_mode_regression.py
# expected: UPDATE_MODE_REGRESSION_OK
python3 scripts/run_polling_regression.py
# expected: POLLING_REGRESSION_OK ...
# optional (heavier, real lcm.db regression):
# python3 scripts/run_ml_split_regression.py --max-per-conv 40 --cycles 1
# optional (recovery-only): duplicate cleanup dry-run
# python3 scripts/cleanup_geometry_duplicates.py --db <openclaw_home>/lcm_geometry.db
```

`cleanup_geometry_duplicates.py` is not part of normal operation. Use it only if a legacy DB shows inflated duplicate message rows from older polling runs.

## Deploy files to OpenClaw

Set your OpenClaw home (example default):

```bash
export OPENCLAW_HOME="$HOME/.openclaw"
```

Copy module + MCP server files:

```bash
mkdir -p "$OPENCLAW_HOME/workspace/module"
mkdir -p "$OPENCLAW_HOME/extensions/geometry-mcp"

cp lcm_geometry_controller.py "$OPENCLAW_HOME/workspace/module/"
cp lcm_geometry_backfill.py "$OPENCLAW_HOME/workspace/module/"
cp extensions/geometry-mcp/server.py "$OPENCLAW_HOME/extensions/geometry-mcp/"
cp extensions/geometry-mcp/runtime_config.example.json "$OPENCLAW_HOME/extensions/geometry-mcp/"
```

Optional: create runtime config for MCP server:

```bash
cp "$OPENCLAW_HOME/extensions/geometry-mcp/runtime_config.example.json" \
   "$OPENCLAW_HOME/extensions/geometry-mcp/runtime_config.json"
```

Then edit `runtime_config.json` to tune `geometry_config` fields.

## Build/refresh geometry database

Run backfill from repo root or deployed module path:

```bash
OPENCLAW_HOME="$OPENCLAW_HOME" \
GEOMETRY_MODULE_HOME="$OPENCLAW_HOME/workspace/module" \
python3 lcm_geometry_backfill.py
```

What it writes:

- `<openclaw_home>/lcm_geometry.db`
- `<openclaw_home>/workspace/module/backfill_progress.json`
- `<openclaw_home>/workspace/module/backfill.log`

## Register MCP server in OpenClaw

```bash
openclaw mcp set geometry-hybrid '{"command":"python3","args":["<openclaw_home>/extensions/geometry-mcp/server.py"]}'
openclaw gateway restart
openclaw mcp list
```

Expected server name: `geometry-hybrid`.

## Runtime tuning (new)

The MCP server automatically loads runtime overrides from:

- `<openclaw_home>/extensions/geometry-mcp/runtime_config.json` (if present)
- env var `GEOMETRY_RUNTIME_CONFIG_JSON` (JSON object; merged on top)

Top-level runtime block (outside `geometry_config`):

- `polling.enabled`
- `polling.interval_seconds`
- `polling.limit`
- `polling.auto_limit` (lightweight auto-poll cap used for auto-poll tools)
- `polling.auto_tools` (list of tool names that trigger auto-poll; default `["hybrid_search"]`)
- `polling.conversation_id`
- `polling.cursor_path`
- `polling.show_status`
- `polling.debug_log`
- `startup.warmup_gc`
- `startup.warmup_probe_embed`
- `startup.warmup_query`

Embedding runtime block:

- `embedding.backend` = `sentence_transformers | llama_cpp | http`
- `embedding.model`
- `embedding.device` (for sentence-transformers)
- `embedding.dim` (default `384`)
- `embedding.gguf_path`, `embedding.gguf_n_ctx`, `embedding.gguf_n_threads` (for GGUF)
- `embedding.http_url`, `embedding.http_timeout_sec` (for HTTP backend)

Gemma 300M GGUF example:

```json
{
  "embedding": {
    "backend": "llama_cpp",
    "model": "embedding-gemma-300M-Q8_0.gguf",
    "gguf_path": "/home/victo/models/embedding-gemma-300M-Q8_0.gguf",
    "dim": 768
  }
}
```

Production cutover runbook:

- [`GEOMETRY_GGUF_MIGRATION_RUNBOOK.md`](./GEOMETRY_GGUF_MIGRATION_RUNBOOK.md)
- ready config template:
  `extensions/geometry-mcp/runtime_config.embeddinggemma_gguf.example.json`

Safety guard:

- Controller persists an embedding runtime signature in `maintenance_state`.
- If backend/model/dimension changes on an existing `lcm_geometry.db`, startup fails by default to prevent mixed-vector corruption.
- To intentionally migrate in-place, set `GEOMETRY_ALLOW_EMBEDDING_SIGNATURE_CHANGE=1` (recommended only after controlled migration/rebuild).

Useful keys in `geometry_config`:

- Retrieval/lazy loading:
  - `retrieval_prefilter_limit`
  - `retrieval_result_limit`
  - `candidate_prefilter_limit`
  - `candidate_branch_cap`
- Retrieval routing:
  - `retrieval_mode_default`
  - `retrieval_mode_factors`
- Allocation thresholds (branch-type aware):
  - `attach_threshold`
  - `tension_threshold`
  - `attach_threshold_by_type`
  - `tension_threshold_by_type`
- Branch-type metric profiles:
  - `branch_type_profiles`
  - groups supported: `csd_gamma`, `retrieval_kappa`, `split_zeta`, `split_policy`, `merge_eta`, `merge_policy`
- Protected memory + safe reactivation:
  - `protected_branch_types`
  - `protected_attach_conflict_threshold`
  - `protected_attach_topic_drift_threshold`
  - `protected_merge_block`
  - `protected_merge_topic_drift_threshold`
  - `reactivation_min_score`
  - `reactivation_guard_enabled`
  - `reactivation_max_topic_drift`
  - `reactivation_max_retrieval_error`
  - `reactivation_min_similarity`
- Update-mode metadata classification:
  - `update_mode_refine_similarity_min`
  - `update_mode_contradict_conflict_min`
  - `update_mode_supersede_similarity_min`
  - `update_mode_supersede_conflict_min`
  - `update_mode_supersede_branch_types`
- Merge execution:
  - `merge_execution_mode` (`soft` or `off`)
  - `merge_max_jobs_per_cycle`
  - `merge_soft_edge_weight`
- Topic drift bounded compute:
  - `topic_drift_sim_threshold`
  - `topic_drift_min_temporal_gap`
  - `topic_drift_max_temporal_gap`
  - `topic_drift_allowed_roles`
  - `topic_drift_min_token_count`
  - `topic_drift_min_content_chars`
  - `topic_drift_require_content_nonempty`
  - `topic_drift_sample_min_nodes`
  - `topic_drift_sample_max_nodes`
  - `topic_drift_edge_max_pairs`
- Dormancy policy:
  - `dormant_after_days`
  - `dormant_usefulness_max`
  - `dormant_min_nodes`
- Split child priors:
  - `split_child_copy_usefulness`
  - `split_child_copy_retrieval_error`
  - `split_child_anchor_from_centroid`

## MCP tools provided

- `hybrid_search` - combined semantic + keyword retrieval with recommendation (`geometry` / `lcm` / `both`), now with optional `retrieval_mode` (`balanced` / `factual` / `exploratory`)
- `retrieval_feedback` - record explicit feedback for `hybrid_search` results using `query_id` and `branch_id`
- `branch_report` - branch diagnostics (state, regime, rank/coherence/anisotropy, etc.)
- `geometry_stats` - global DB health and distribution stats
- `maintenance_cycle` - run one maintenance cycle, optionally chunked (`max_branches`) with optional cursor reset; output includes retrieval-feedback pruning counters (`retrieval_feedback_pruned`, `_age`, `_cap`)
- `geometry_snapshot` - export compact branch metrics for ops/debug (supports `state`, `branch_ids`, `limit`, optional `include_means`)
- `latest_correction` - resolve correction lineage and return latest correction node/version for any seed `node_id` (optional chain expansion)
- `sync_lcm_ingest` - force one incremental ingest poll from `lcm.db` into geometry DB
- `sync_lcm_dag_edges` - rebuild imported DAG edges (`summarizes`, `derived_from`) and return orphan-validation counters
- `backfill_lcm_conversations` - targeted backfill for specific LCM conversation IDs with dry-run and resume options
- `conversation_content` - geometry-to-LCM text bridge (summaries/messages by branch/state) with lineage-aware resolution metadata

## Typical usage flow

1. Use `hybrid_search` for first-pass recall.
2. Inspect top branch with `branch_report` when needed.
3. Pull text evidence with `conversation_content`.
4. Run `sync_lcm_dag_edges` after major backfill/import refreshes to keep DAG link integrity validated.
5. Re-run `lcm_geometry_backfill.py` periodically to keep geometry aligned with latest LCM data.

## One-command maintenance (Lossless cleanup + Geometry sync)

Use this wrapper when testing/maintaining Lossless-Claw cleanup together with Geometry DAG integrity:

```bash
/home/victo/.openclaw/workspace/scripts/lossless_geometry_maintenance.sh --text
```

Apply cleanup (with backup) + sync:

```bash
/home/victo/.openclaw/workspace/scripts/lossless_geometry_maintenance.sh --apply --text
```

Apply only one cleanup filter:

```bash
/home/victo/.openclaw/workspace/scripts/lossless_geometry_maintenance.sh --apply --filter null_subagent_context --text
```

Notes:

- Cleanup uses Lossless-Claw internal doctor-clean functions via headless runner (`lossless_doctor_clean_runner.ts`).
- After cleanup, the wrapper re-imports DAG edges (`derived_from`, `summarizes`) and validates orphan counts.
- This avoids relying on root CLI `/lcm` command routing.

## Troubleshooting

- `FileNotFoundError: ... lcm.db`: ensure `<openclaw_home>/lcm.db` exists.
- `ModuleNotFoundError: lcm_geometry_controller`: ensure file is present in `<openclaw_home>/workspace/module`.
- MCP server not visible: re-run `openclaw mcp set ...` and restart gateway.
- Empty semantic results: run backfill to populate/refresh `lcm_geometry.db`.
- Imported DAG links look inconsistent: run `sync_lcm_dag_edges` (default creates a DB backup and reports orphan counts).

## License

MIT. See `LICENSE`.
