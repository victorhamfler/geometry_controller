# CSD/G-CL Next Implementation Spec (2026-04-08)

## Scope agreed for this pass

1. Step 1: Versioned correction flow for conflicting facts (explicit supersession chain)
2. Step 3: Real-time LCM polling wiring in MCP runtime

Step 2 (summary anchor + drift guard) is intentionally skipped in this pass.

## Step 1 — Versioned Correction Flow (v1)

### Goals
- Preserve explicit correction lineage for factual updates.
- Avoid relying only on contradiction edges as conflict signal.
- Keep schema backward-compatible via additive migration.

### Data model changes
- `EdgeType.SUPERSEDES` is added.
- `memory_nodes` new columns:
  - `correction_kind` (`none|refine|contradict|supersede`)
  - `correction_prev_id` (points to prior node in correction chain)
  - `correction_root_id` (stable root of chain)
  - `correction_version` (monotonic integer version within chain)

### Write-path behavior
- On `on_new_item(...)`:
  - For `update_mode in {refine, contradict, supersede}` and if previous node exists in branch:
    - set `correction_prev_id` to previous node
    - inherit or initialize `correction_root_id`
    - increment `correction_version`
    - set `correction_kind=update_mode`
  - Else `correction_kind=none` and version defaults to 1.
- Add explicit relation edges from new node to previous version node:
  - `refine -> REFINES`
  - `contradict -> CONTRADICTS`
  - `supersede -> SUPERSEDES`

### Read/report behavior
- `branch_report(...)` includes:
  - `correction_counts` (`by_kind`, `chain_links`, `max_version`)
  - `recent_corrections` (latest chain-linked nodes)

### Compatibility and migration
- Existing DBs migrate in place (`ALTER TABLE ... ADD COLUMN` only).
- No destructive migration.

### Regression checks
- `scripts/run_update_mode_regression.py` must verify:
  - update-mode counts for fork/refine/contradict/supersede
  - correction counts for refine/contradict/supersede
  - explicit `refines/contradicts/supersedes` edges exist

## Step 3 — Real-time Polling Wiring (v1)

### Goals
- Keep geometry DB updated incrementally from live `lcm.db` messages.
- Use persistent rowid cursor and safe cooldown to avoid repeated heavy scans.

### Server integration
- MCP server initializes `GeometryController` with `EmbeddingProvider`.
- Add polling runtime config block:
  - `polling.enabled`
  - `polling.interval_seconds`
  - `polling.limit`
  - `polling.conversation_id` (optional filter)
  - `polling.cursor_path`
  - `polling.show_status`
  - `polling.debug_log`
- Automatic polling on tool calls (cooldown + mutex lock).
- Manual force tool: `sync_lcm_ingest`.

### Cursor and safety
- Cursor file stores `last_rowid` and timestamp.
- Polling is serialized with in-process lock.
- On error: tool call continues and returns ingest status error line.

### Observability
- Tool outputs include lightweight ingest status line when enabled.
- `sync_lcm_ingest` returns full poll status (processed, failed, next_rowid, has_more).

### Validation
- Compile checks pass.
- `scripts/run_update_mode_regression.py` passes.
- `scripts/smoke_test_geometry.py` passes.
- Optional heavier check: `scripts/run_ml_split_regression.py`.
