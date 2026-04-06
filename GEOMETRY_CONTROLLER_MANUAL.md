# LCM Geometry Controller — Manual

**Version:** 1.3
**Module:** `lcm_geometry_controller.py`
**Geometry DB:** `<openclaw_home>/lcm_geometry.db`
**LCM DB:** `<openclaw_home>/lcm.db`
**Last Updated:** 2026-04-06

---

## Table of Contents

1. [What Is the Geometry Controller?](#1-what-is-the-geometry-controller)
2. [Architecture](#2-architecture)
3. [Quick Start](#3-quick-start)
4. [Database Schema](#4-database-schema)
5. [Core API Reference](#5-core-api-reference)
6. [Configuration — GeometryConfig](#6-configuration--geometryconfig)
7. [EmbeddingProvider](#7-embeddingprovider)
8. [Retrieval & Ranking](#8-retrieval--ranking)
9. [Maintenance Cycle](#9-maintenance-cycle)
10. [Backfill — One-Time Setup](#10-backfill--one-time-setup)
11. [DAG Edge Import](#11-dag-edge-import)
12. [MCP Server Tools](#12-mcp-server-tools)
13. [Current Enhancements](#13-current-enhancements)
14. [Integration Status](#14-integration-status)
15. [Troubleshooting](#15-troubleshooting)

---

## 1. What Is the Geometry Controller?

The geometry controller is a semantic memory layer that sits alongside OpenClaw's LCM (Long-Term Context Memory). It annotates each conversation branch with **geometric state** — embedding centroids, anisotropy, coherence, effective rank — and uses these signals to:

- **Score** where a new message best belongs (which branch it fits)
- **Classify** branch health: PRODUCTIVE, RIGID, or UNSTABLE
- **Rank** retrieval candidates by semantic similarity + trust + reactivation

Think of it as the difference between a filing cabinet (LCM — stores messages) and a smart librarian (geometry controller — knows which drawer is healthiest for a new topic).

---

## 2. Architecture

### Two Databases

```
lcm.db                              lcm_geometry.db
─────────────────────────────       ───────────────────────────────────────
Messages (immutable)                Per-branch geometry (mean_vec, eff_rank,
Summaries (DAG nodes)               anisotropy, coherence, trace, anchor_drift)
Summary DAG edges                    Per-message/summary 384-dim embeddings
Context items (active ctx)           DAG edges (derived_from, summarizes)
                                    Branch lifecycle states + regimes
                                    Maintenance job queue
                                    Retrieval feedback signals
```

**LCM is the source of truth.** The geometry DB is a **read-heavy companion** — LCM never writes to it. The geometry controller only reads from LCM.

### Geometry Per Branch

Each conversation branch gets:
- `mean_vec` — 384-dim centroid (EMA-updated as messages are added)
- `cov_diagonal` — variance per dimension
- `eff_rank` — effective rank of the embedding cloud (how many orthogonal directions are actually used)
- `anisotropy` — concentration proxy computed from variance spectrum (`max_eigenvalue / sum_eigenvalues`)
- `coherence` — mean pairwise cosine similarity of messages in branch
- `trace` — sum of variances (total spread)
- `anchor_drift` — how much the centroid has shifted since the anchor was set
- `regime` — PRODUCTIVE (healthy), RIGID (over-consolidated), UNSTABLE (topic drift)

### Lifecycle States

```
FORMING → ACTIVE → STABLE
ACTIVE/STABLE → DORMANT → REACTIVATING → ACTIVE
STABLE → MERGE_CANDIDATE → (merged)
ACTIVE → SPLIT_PENDING → (forked)
```

---

## 3. Quick Start

### Prerequisites

```bash
# Activate ML venv
source ~/venvs/ml/bin/activate

# Check module works
python3 -c "
import sys; sys.path.insert(0, '<module_repo_root>')
from lcm_geometry_controller import create_geometry_controller, EmbeddingProvider
gc = create_geometry_controller(
    '<openclaw_home>/lcm_geometry.db',
    embedding_provider=EmbeddingProvider()
)
print('OK —', gc.db.conn.execute('SELECT COUNT(*) FROM branch_states').fetchone()[0], 'branches')
"
```

### Basic Usage — On New Message

```python
import sys
sys.path.insert(0, '<module_repo_root>')
from lcm_geometry_controller import create_geometry_controller, EmbeddingProvider, NodeType

gc = create_geometry_controller(
    '<openclaw_home>/lcm_geometry.db',
    embedding_provider=EmbeddingProvider()
)

# Auto-embed: pass text directly, EmbeddingProvider handles it
decision = gc.on_new_item(
    lcm_id="msg_abc123",
    node_type=NodeType.MESSAGE,
    text="What's the weather in Amposta?",  # ← auto-embeds
    role="user",
    token_count=6,
)
print(decision.action, decision.branch_id)  # e.g. "attach" "conv_42"
```

### Basic Usage — Retrieval

```python
import numpy as np

query_emb = gc.embedding_provider.embed(
    "CLGK dashboard integration"
)
ranked = gc.rank_retrieval(np.array(query_emb, dtype=np.float32))

for r in ranked[:5]:
    print(f"  {r.branch_id}  score={r.total_score:.3f}  sem={r.sem_score:.3f}")
```

### Branch Report

```python
r = gc.branch_report('conv_1')
print(r['state'], r['regime'], r['eff_rank'], r['coherence'])
```

---

## 4. Database Schema

### `branch_states` — One Row Per Conversation Branch

| Column | Type | Description |
|--------|------|-------------|
| `branch_id` | TEXT PK | Primary key (e.g. `conv_1`) |
| `state` | TEXT | BranchState enum: FORMING, ACTIVE, STABLE, TENSIONED, DORMANT, etc. |
| `regime` | TEXT | GeometricRegime: PRODUCTIVE, RIGID, UNSTABLE |
| `mean_vec` | TEXT (JSON) | 384-dim centroid as JSON list |
| `cov_diagonal` | TEXT (JSON) | Variance per dimension |
| `eff_rank` | REAL | Effective rank (1–384) |
| `anisotropy` | REAL | Variance concentration proxy (`max_eigenvalue / sum_eigenvalues`) |
| `coherence` | REAL | Mean pairwise cosine similarity |
| `trace` | REAL | Sum of variances |
| `compression_loss` | REAL | Compression error (reconstruction from low-rank) |
| `contradiction_density` | REAL | Density of contradictory message pairs |
| `retrieval_error` | REAL | EMA of retrieval mis-match scores |
| `anchor` | TEXT (JSON) | Stable reference centroid |
| `anchor_drift` | REAL | Distance current mean_vec has drifted from anchor |
| `node_count` | INT | Number of memory_nodes in this branch |
| `split_counter` | INT | Consecutive high split scores |
| `merge_counter` | INT | Consecutive merge candidates |
| `last_update_ts` | REAL | Unix timestamp of last update |
| `created_ts` | REAL | Unix timestamp of creation |

### `memory_nodes` — Per-Message/Per-Summary Embeddings

| Column | Type | Description |
|--------|------|-------------|
| `id` | TEXT PK | Node ID (e.g. `msg_abc123`) |
| `lcm_id` | TEXT | LCM ID for this item |
| `node_type` | TEXT | NodeType: MESSAGE, TOOL_RESULT, LEAF_SUMMARY, CONDENSED_SUMMARY, etc. |
| `parent_id` | TEXT | Parent node ID (for DAG) |
| `branch_id` | TEXT FK | Owning branch |
| `timestamp` | REAL | When added |
| `role` | TEXT | Message role: user, assistant, system |
| `embedding` | TEXT (JSON) | 384-dim vector as JSON list |

### `memory_edges` — DAG Edges

| Column | Type | Description |
|--------|------|-------------|
| `id` | TEXT PK | Edge ID |
| `source_id` | TEXT FK | Source node |
| `target_id` | TEXT FK | Target node |
| `edge_type` | TEXT | EdgeType: DERIVED_FROM, SUMMARIZES, REFINES, CONTRADICTS |
| `weight` | REAL | Edge weight (default 1.0) |

### `retrieval_feedback` — Feedback Signals (Future)

| Column | Type | Description |
|--------|------|-------------|
| `branch_id` | TEXT | Retrieved branch |
| `query_hash` | TEXT | Hashed query text |
| `usefulness` | REAL | User/agent signal: was this retrieval useful? |
| `ts` | REAL | Timestamp |

### `maintenance_split_observations` — Split Decision Trace

Stores per-branch split gate outcomes for each maintenance run.

| Column | Type | Description |
|--------|------|-------------|
| `run_id` | TEXT | Maintenance run correlation id |
| `branch_id` | TEXT | Branch evaluated in that run |
| `split_score` | REAL | Computed split score |
| `split_threshold` | REAL | Threshold used in that run |
| `gate_nodes` | INT | Node count gate (0/1) |
| `gate_score` | INT | Score threshold gate (0/1) |
| `gate_hysteresis` | INT | Hysteresis counter gate (0/1) |
| `should_split` | INT | Final split eligibility before cap (0/1) |
| `enqueued` | INT | Whether a split job was queued (0/1) |
| `reason` | TEXT | Semicolon-separated decision reasons (`eligible_candidate`, `eligible_throttled`, `eligible_enqueued_rank`, etc.) |

---

## 5. Core API Reference

### `GeometryController(db_path, cfg=None, embedding_provider=None)`

Main controller. All methods are on this class.

```python
gc = GeometryController('/path/to/lcm_geometry.db')
gc = GeometryController('/path/to/lcm_geometry.db', cfg=GeometryConfig())
gc = GeometryController('/path/to/lcm_geometry.db', embedding_provider=EmbeddingProvider())
```

### `gc.on_new_item(lcm_id, node_type, embedding=None, role='user', token_count=0, conflict_score=0.0, active_branch_id=None, force_branch_id=None, parent_lcm_id=None, text=None)`

Called after LCM persists a new message/summary. Computes CSD score against all active branches, decides where to attach.

- **`lcm_id`** — LCM ID string (e.g. `msg_abc123`)
- **`node_type`** — `NodeType` enum value
- **`text`** — raw text (auto-embeds via `embedding_provider` if provided)
- **`embedding`** — pre-computed 384-dim vector (if not using `text=`)
- **`role`** — message role: `user`, `assistant`, `system`
- **`token_count`** — approximate token count
- **`conflict_score`** - optional conflict severity signal for CSD scoring
- **`active_branch_id`** - preferred branch hint from caller runtime
- **`force_branch_id`** - bypass candidate selection and attach to a specific branch id (used by deterministic backfill mapping)
- **`parent_lcm_id`** - parent LCM id; controller resolves parent node and links DAG/temporal edges

**Returns:** `AllocationDecision` with fields:
- `action` — `"attach"`, `"attach_tension"`, or `"fork"`
- `branch_id` — target branch ID
- `csd_score` — CSD score for chosen branch
- `conflict_score` — conflict metric for chosen branch
- `rationale` — human-readable explanation

**Example:**
```python
decision = gc.on_new_item(
    lcm_id="msg_xyz",
    node_type=NodeType.MESSAGE,
    text="Tell me about the weather",
    role="user",
    token_count=5,
)
```

### `gc.rank_retrieval(query_emb, historical_use=None, same_project=None)`

Rank all branches by relevance to a query embedding.

- **`query_emb`** — numpy array, shape (384,)
- **`historical_use`** — dict `branch_id → float` of past retrieval frequency
- **`same_project`** — dict `branch_id → float` of same-project signal

**Returns:** list of `RetrievalCandidate` sorted by `total_score` descending:
- `branch_id`
- `sem_score` — cosine similarity to branch centroid
- `trust_score` — composite of coherence, compression_loss, contradiction_density
- `react_score` — reactivation signal
- `total_score` — weighted composite: `α·sem + β·trust + δ·react`

### `gc.branch_report(branch_id)`

Detailed geometry report for one branch.

```python
r = gc.branch_report('conv_1')
# r.keys(): branch_id, state, regime, node_count, eff_rank, anisotropy,
#           coherence, trace, anchor_drift, compression_loss,
#           contradiction_density, retrieval_error, mean_vec (truncated)
```

### `gc.run_maintenance_cycle()`

Full maintenance sweep over all branches. Should be run periodically (every 20–30 min).

**Operations per branch:**
1. **Recomputes geometry** for each branch from its `memory_nodes` rows
2. **Refreshes contradiction signals** (density + bounded `CONTRADICTS` edges)
3. **Reclassifies regime** - PRODUCTIVE / RIGID / UNSTABLE
4. **Scans for splits and prepares ranked candidates** when gate + hysteresis conditions are met
5. **Scores merges with runtime signals** (graph overlap + retrieval co-use)
6. **Executes pending split jobs** (k-means(2), child branches, `REFINES` edges)
7. **Runs reactivation scan**

**Returns:** `dict` with counts:
```python
{
    'recomputed': 437,
    'split_pending': 0,
    'split_executed': 0,
    'merge_candidates': 0,
    'reactivated': 0,
    'split_observations': 437,
    'split_trace_run_id': '...'
}
```

### `gc.backfill_from_lcm(lcm_db_path, resume=True, progress_cb=None, max_per_conv=200, error_log_path=None)`

One-time backfill from LCM. Creates `memory_nodes` rows for all messages and summaries.

- **`lcm_db_path`** — path to `lcm.db`
- **`resume`** — if True, skips branches that already have `node_count > 0`
- **`progress_cb`** — callback `(current, total) → None` for progress tracking
- **`max_per_conv`** — cap per-conversation sampled rows in large histories
- **`error_log_path`** — optional target file for structured per-conversation backfill errors

**Returns:** `dict` with `processed`, `sampled`, `skipped`, `failed`, `errors_logged`.

### `gc.import_dag_edges_from_lcm(lcm_db_path)`

Import summary DAG edges from LCM into `memory_edges`.

Reads:
- `summary_parents` → `DERIVED_FROM` edges (parent_summary → child_summary)
- `summary_messages` → `SUMMARIZES` edges (summary → message)

**Returns:** `dict` with `derived_from`, `summarizes`, `skipped` counts.

### `gc.audit_summary(summary_id, embedding=None, text=None)`

Score summary quality. Compares summary embedding to the mean of its constituent messages.

---

## 6. Configuration — GeometryConfig

All tunable hyperparameters in one place. Passed to `GeometryController.__init__`.

```python
from lcm_geometry_controller import GeometryConfig, GeometryController

cfg = GeometryConfig(
    embedding_dim=384,
    min_branch_size=8,
    stat_rho=0.05,
    anchor_alpha=0.01,
)

gc = GeometryController('/path/to/lcm_geometry.db', cfg=cfg)
```

### Key Fields

| Field | Default | Description |
|-------|---------|-------------|
| `embedding_dim` | 384 | Must match your embedding model |
| `min_branch_size` | 8 | Don't compute stats below this |
| `stat_rho` | 0.05 | EMA rate for geometry field updates |
| `anchor_alpha` | 0.01 | Anchor update rate |
| `attach_threshold` | 0.50 | CSD below this → attach; above → tension/fork |
| `alpha_sem` | 0.60 | Semantic similarity weight in retrieval |
| `beta_trust` | 0.25 | Trust score weight in retrieval |
| `delta_react` | 0.15 | Reactivation weight in retrieval |
| `split_score_threshold` | 0.075 | Split score threshold used by split eligibility |
| `split_min_nodes` | 6 | Baseline split gate; effective gate uses `max(split_min_nodes, min_branch_size)` |
| `split_hysteresis` | 3 | Consecutive high scores → SPLIT_PENDING |
| `max_split_enqueues_per_cycle` | 5 | Maximum split jobs enqueued per maintenance cycle (score-ranked) |
| `merge_hysteresis` | 3 | Consecutive merge candidates → MERGE_CANDIDATE |
| `usefulness_lambda` | 0.10 | EMA decay for usefulness signals |
| `rank_target` | dict | eff_rank target bands per branch type |
| `rigid_rank_ratio` | 0.15 | eff_rank below this → RIGID regime |
| `unstable_coh_floor` | 0.45 | Coherence below this → UNSTABLE |
| `unstable_comp_ceil` | 0.65 | Compression loss above this → UNSTABLE |

---

## 7. EmbeddingProvider

Pluggable embedding backend. Supports sentence-transformers (local) or any OpenAI-compatible API.

### Init

```python
EmbeddingProvider(
    model_name="all-MiniLM-L6-v2",  # HuggingFace model name
    device="cpu",                     # or "cuda"
    cache=None,                       # optional shared dict cache
)
```

### Methods

**`ep.embed(text)`** — Encode single text. Result cached by `text[:200]`.

**`ep.embed_batch(texts, batch_size=64)`** — Encode list of texts. No caching.

**`ep.embedding_dim`** — Lazy property returning model's embedding dimension.

### Integration with GeometryController

```python
from lcm_geometry_controller import create_geometry_controller, EmbeddingProvider

gc = create_geometry_controller(
    '/path/to/lcm_geometry.db',
    embedding_provider=EmbeddingProvider()
)

# Now on_new_item accepts text= directly
gc.on_new_item(lcm_id="msg_1", node_type=NodeType.MESSAGE,
               text="Hello world", role="user", token_count=2)
```

The `EmbeddingProvider` is lazy — the model is only loaded on first `embed()` call, not at init.

---

## 8. Retrieval & Ranking

The `RetrievalRanker` computes a composite score for each branch candidate:

```
total = α·sem + β·trust + δ·react
```

Where:
- **`sem`** — cosine similarity of query embedding to branch mean_vec
- **`trust`** — `κ_coherence·coherence + κ_comp_loss·compression_loss + κ_contradiction·contradiction_density + κ_ret_error·retrieval_error`
- **`react`** — recency/reactivation signal (half semantic + history + project signals)

### CSD Scoring (On New Message)

When a new message arrives, the `CSDScorer` scores it against each candidate branch:

```
csd = importance·novelty + conflict·resolution + coherence·alignment
```

Components:
- **`delta_mu`** — distance from branch centroid
- **`delta_r`** — change in effective rank
- **`delta_A`** — change in anisotropy
- **`delta_tau`** — change in trace

The scorer uses EMA-weighted residuals so one-off outliers don't destabilise stable branches.

---

## 9. Maintenance Cycle

Run `gc.run_maintenance_cycle()` every 20–30 minutes. It:

1. **Recomputes geometry** for each branch from its `memory_nodes` rows
2. **Updates scalars** (eff_rank, anisotropy, coherence, trace) via adiabatic EMA
3. **Reclassifies regime** — PRODUCTIVE / RIGID / UNSTABLE
4. **Scans for splits** — branch must pass:
   - score gate: `split_score > split_score_threshold`
   - node readiness gate: `node_count >= max(split_min_nodes, min_branch_size)`
   - real-node readiness gate: embedded rows also meet the same threshold
   - hysteresis gate: `split_counter >= split_hysteresis`
   Eligible branches are score-ranked and throttled by `max_split_enqueues_per_cycle`.
5. **Scans for merges** — stable branches with high topic overlap → `MERGE_CANDIDATE`

### Running as Cron

```bash
# Every 30 minutes
*/30 * * * * cd <user_home> && \
  <python_executable> -c "
import sys; sys.path.insert(0, '<module_repo_root>')
from lcm_geometry_controller import GeometryController
gc = GeometryController('<openclaw_home>/lcm_geometry.db')
r = gc.run_maintenance_cycle()
print(f'maint: recomputed={r[\"recomputed\"]} split={r[\"split_pending\"]} merge={r[\"merge_candidates\"]}')
" >> <openclaw_home>/logs/geometry_maint.log 2>&1
```

Or via OpenClaw heartbeat (edit `HEARTBEAT.md`):
```markdown
# HEARTBEAT.md
- Run maintenance every ~30 min:
  `python3 -c "import sys; sys.path.insert(0, '<module_repo_root>'); from lcm_geometry_controller import GeometryController; gc = GeometryController('<openclaw_home>/lcm_geometry.db'); print(gc.run_maintenance_cycle())"`
```

---

## 10. Backfill — One-Time Setup

If you're starting fresh (no `lcm_geometry.db` yet), run the backfill once to populate all historical data:

```python
import sys
sys.path.insert(0, '<module_repo_root>')
from lcm_geometry_controller import GeometryController, EmbeddingProvider
import time

gc = GeometryController('<openclaw_home>/lcm_geometry.db',
                        embedding_provider=EmbeddingProvider())

start = time.time()
r = gc.backfill_from_lcm('<openclaw_home>/lcm.db', resume=False)
elapsed = time.time() - start

print(f"Backfill done in {elapsed:.1f}s")
print(f"  processed={r['processed']} sampled={r.get('sampled',0)} "
      f"skipped={r['skipped']} failed={r['failed']} errors_logged={r.get('errors_logged',0)}")
```

Then import DAG edges:
```python
r2 = gc.import_dag_edges_from_lcm('<openclaw_home>/lcm.db')
print(f"Edges: {r2['derived_from']} derived_from + {r2['summarizes']} summarizes")
```

For incremental updates (new messages since last backfill):
```python
gc.backfill_from_lcm('<openclaw_home>/lcm.db', resume=True)
```

Backfill uses deterministic conversation mapping (`conv_<conversation_id>`) and branch-locked insertion, so branch IDs remain aligned with LCM conversation IDs.

---

## 11. DAG Edge Import

The `import_dag_edges_from_lcm()` method reads two LCM tables:

**`summary_parents`** → `DERIVED_FROM` edges
```
child_summary ────→ parent_summary
```

**`summary_messages`** → `SUMMARIZES` edges
```
summary ─────────→ message
```

These edges enable:
- **`topic_overlap`** in merge scoring — two branches share summaries → high topic overlap → merge candidate
- **`retrieval_co_use`** — branches that share message ancestors are related

**Results from last import:**
- 435 `DERIVED_FROM` edges
- 16,021 `SUMMARIZES` edges
- Total: 16,456 edges

---

## 12. MCP Server Tools

The OpenClaw MCP server (`<openclaw_home>/extensions/geometry-mcp/server.py`) exposes four tools:

### `geometry-hybrid__hybrid_search`

Combines LCM keyword search with geometry DB semantic search. Best for open-ended recall.

```
geometry-hybrid__hybrid_search(query="CLGK dashboard", top_n=5)
```

Returns combined results from both systems with recommendation.

### `geometry-hybrid__branch_report`

Detailed geometry metrics for one branch.

```
geometry-hybrid__branch_report(branch_id="conv_1")
```

Returns: state, regime, node_count, eff_rank, anisotropy, coherence, trace, anchor_drift, mean_vec (first 10 dims), compression_loss, contradiction_density, retrieval_error.

### `geometry-hybrid__geometry_stats`

Overall geometry DB statistics.

```
geometry-hybrid__geometry_stats()
```

Returns: branch count, state distribution, regime distribution, average metrics.

### `geometry-hybrid__conversation_content`

Bridge from geometry branch IDs to real LCM content. Use this after `hybrid_search` when you need summaries/messages from selected branches.

```
geometry-hybrid__conversation_content(
  branch_id="conv_385",
  content_type="both",
  max_entries=50
)
```

Supports:
- `branch_id="conv_N"` for one conversation
- `state="ACTIVE" | "STABLE" | "FORMING" | "ALL"` for batch mode
- `content_type="summaries" | "messages" | "both"`

### MCP Server Registration

Registered in `<openclaw_home>/openclaw.json` under `mcp.servers`:

```json
"geometry-hybrid": {
  "command": "<python_executable>",
  "args": ["<openclaw_home>/extensions/geometry-mcp/server.py"]
}
```

**CRITICAL:** Must use full path to ML venv Python. Restart gateway after config changes:
```bash
openclaw gateway restart
```

---

## 13. Current Enhancements

The current controller build includes these production features:

1. **Incremental ingest API** via `poll_lcm_for_new_items(...)` using rowid cursors.
2. **Parent-aware insertion** in `on_new_item(...)` with `parent_lcm_id` resolution.
3. **Automatic `TEMPORAL_NEXT` edge creation** between consecutive nodes in the same branch.
4. **Contradiction refresh** in maintenance:
   - contradiction density recomputation per branch
   - bounded `CONTRADICTS` edge regeneration
5. **Merge runtime signals** integrated into merge scoring:
   - graph overlap from memory edges
   - retrieval co-use from feedback history
6. **Split execution pipeline**:
   - pending split jobs consumed by `execute_pending_splits(...)`
   - internal k-means(2) partition
   - child branch creation + `REFINES` edges
7. **Split gating hardening and throttling**:
   - score threshold is config-driven (`split_score_threshold`)
   - split readiness uses `max(split_min_nodes, min_branch_size)`
   - real embedded-node gate prevents sparse/placeholder branches from accumulating split hysteresis
   - per-cycle queue cap (`max_split_enqueues_per_cycle`) with score-priority selection
8. **Split observability persistence**:
   - per-branch split gate traces in `maintenance_split_observations`
   - per-cycle trace id in maintenance return (`split_trace_run_id`)
9. **Operational observability APIs**:
   - `health_report()`
   - `mark_branch_agent_interest(...)`
   - `add_cross_agent_shared_edge(...)`
   - `list_cross_agent_links(...)`
10. **Backfill reliability improvements**:
    - deterministic branch-lock insertion via `force_branch_id`
    - structured backfill error log writing + `errors_logged` return metric
11. **Persisted CSD EMA state** (`csd_ema_state`) and retrieval history consistency updates.

---

## 14. Integration Status

### Implemented

- Incremental LCM polling API exists in controller (`poll_lcm_for_new_items`).
- Maintenance covers contradiction refresh, merge signal scoring, split queueing, and split execution.
- Split queueing is gate-hardened and per-cycle throttled with persisted decision traces.
- Backfill preserves conversation branch topology with deterministic branch-locked insertion.
- Cross-agent edge APIs are available and persisted in geometry DB.

### Optional runtime wiring (deployment choice)

- Schedule periodic `run_maintenance_cycle()` (cron/heartbeat).
- Run polling loop using `poll_lcm_for_new_items(...)` with a persisted cursor.
- Feed retrieval/usefulness feedback regularly for stronger merge co-use signal quality.

---

## 15. Troubleshooting
### Module won't import

```
ModuleNotFoundError: No module named 'lcm_geometry_controller'
```

**Fix:** Add module path:
```python
import sys
sys.path.insert(0, '<module_repo_root>')
from lcm_geometry_controller import ...
```

### Model download warning

```
Warning: You are sending unauthenticated requests to the HF Hub.
Please set a HF_TOKEN to enable higher rate limits.
```

**Fix:** Set HuggingFace token (optional, only for faster downloads):
```bash
export HF_TOKEN=your_token_here
```

### MCP tools not showing up

**Fix:** Restart the gateway:
```bash
openclaw gateway restart
```

Then check status:
```bash
openclaw gateway status
openclaw mcp list
```

### Maintenance cycle not recomputing all branches

If `recomputed` count is low despite Fix 5 being applied, check:
```python
gc = GeometryController('<openclaw_home>/lcm_geometry.db')
for s in gc.db.all_branches():
    if not s.mean_vec:
        print(f"{s.branch_id}: NO mean_vec — will use fallback")
```

### CSD score is inf or nan

Ensure `embedding_dim` in `GeometryConfig` matches your embedding model (default 384 for all-MiniLM-L6-v2).

### Backfill is slow

Use `resume=True` for incremental updates:
```python
gc.backfill_from_lcm('<openclaw_home>/lcm.db', resume=True)
```

For large conversations (>200 messages), the backfill stratifies to `max_per_conv` samples (default 200). To increase coverage, raise `max_per_conv`:
```python
gc.backfill_from_lcm('<openclaw_home>/lcm.db', resume=True, max_per_conv=1000)
```

### Gateway restart kills tmux sessions

Restart tandem after gateway restart:
```bash
<openclaw_home>/start-tandem.sh
```

---

## Quick Reference Card

```python
# Setup (one time)
import sys
sys.path.insert(0, '<module_repo_root>')
from lcm_geometry_controller import create_geometry_controller, EmbeddingProvider

gc = create_geometry_controller(
    '<openclaw_home>/lcm_geometry.db',
    embedding_provider=EmbeddingProvider()
)

# On new message
decision = gc.on_new_item(
    lcm_id="msg_abc", node_type=NodeType.MESSAGE,
    text="Hello world", role="user", token_count=2
)
# decision.action: "attach" | "attach_tension" | "fork"
# decision.branch_id: target branch

# Retrieve
ranked = gc.rank_retrieval(gc.embedding_provider.embed("what about X?"))
for r in ranked[:5]:
    print(r.branch_id, r.total_score)

# Branch report
r = gc.branch_report('conv_1')
print(r['state'], r['regime'], r['eff_rank'])

# Maintenance (every 20-30 min)
gc.run_maintenance_cycle()

# Incremental backfill
gc.backfill_from_lcm('<openclaw_home>/lcm.db', resume=True)
```

---

*Manual generated: 2026-04-06*
*Module: `<module_repo_root>/lcm_geometry_controller.py`*
*Companion backfill script: `<module_repo_root>/lcm_geometry_backfill.py`*
*MCP server: `<openclaw_home>/extensions/geometry-mcp/server.py`*
