# LCM Geometry Controller — Manual

**Version:** 1.0
**Module:** `lcm_geometry_controller.py`
**Geometry DB:** `~/.openclaw/lcm_geometry.db`
**LCM DB:** `~/.openclaw/lcm.db`
**Last Updated:** 2026-04-04

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
13. [All 5 Fixes Applied](#13-all-5-fixes-applied)
14. [Pending Integrations](#14-pending-integrations)
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
- `anisotropy` — ratio of smallest to largest eigenvalue (0 = isotropic, 1 = maximally anisotropic)
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
import sys; sys.path.insert(0, '/home/victo/.openclaw/workspace/module')
from lcm_geometry_controller import create_geometry_controller, EmbeddingProvider
gc = create_geometry_controller(
    '/home/victo/.openclaw/lcm_geometry.db',
    embedding_provider=EmbeddingProvider()
)
print('OK —', gc.db.conn.execute('SELECT COUNT(*) FROM branch_states').fetchone()[0], 'branches')
"
```

### Basic Usage — On New Message

```python
import sys
sys.path.insert(0, '/home/victo/.openclaw/workspace/module')
from lcm_geometry_controller import create_geometry_controller, EmbeddingProvider, NodeType

gc = create_geometry_controller(
    '/home/victo/.openclaw/lcm_geometry.db',
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
| `anisotropy` | REAL | Ratio smallest/largest eigenvalue (0–1) |
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

---

## 5. Core API Reference

### `GeometryController(db_path, cfg=None, embedding_provider=None)`

Main controller. All methods are on this class.

```python
gc = GeometryController('/path/to/lcm_geometry.db')
gc = GeometryController('/path/to/lcm_geometry.db', cfg=GeometryConfig())
gc = GeometryController('/path/to/lcm_geometry.db', embedding_provider=EmbeddingProvider())
```

### `gc.on_new_item(lcm_id, node_type, text=None, embedding=None, role='', token_count=0, parent_id=None)`

Called after LCM persists a new message/summary. Computes CSD score against all active branches, decides where to attach.

- **`lcm_id`** — LCM ID string (e.g. `msg_abc123`)
- **`node_type`** — `NodeType` enum value
- **`text`** — raw text (auto-embeds via `embedding_provider` if provided)
- **`embedding`** — pre-computed 384-dim vector (if not using `text=`)
- **`role`** — message role: `user`, `assistant`, `system`
- **`token_count`** — approximate token count
- **`parent_id`** — parent node ID for DAG construction

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
1. Recompute geometry from `memory_nodes` if `node_count >= min_branch_size`
2. Fallback: reclassify regime from stored scalar metrics if `mean_vec` exists
3. Scan for splits (high split score → increment `split_counter`)
4. Scan for merges (stable branches with high overlap → increment `merge_counter`)
5. Apply regime transitions

**Returns:** `dict` with counts:
```python
{
    'recomputed': 437,     # branches recomputed from memory_nodes
    'split_pending': 0,   # branches flagged for split
    'merge_candidates': 0 # branch pairs flagged for merge
}
```

### `gc.backfill_from_lcm(lcm_db_path, resume=True, progress_cb=None)`

One-time backfill from LCM. Creates `memory_nodes` rows for all messages and summaries.

- **`lcm_db_path`** — path to `lcm.db`
- **`resume`** — if True, skips branches that already have `node_count > 0`
- **`progress_cb`** — callback `(current, total) → None` for progress tracking

**Returns:** `dict` with `processed`, `sampled`, `skipped`, `failed` counts.

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
| `split_hysteresis` | 3 | Consecutive high scores → SPLIT_PENDING |
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
4. **Scans for splits** — if `split_score > 0.55` for `split_hysteresis` consecutive cycles → `SPLIT_PENDING`
5. **Scans for merges** — stable branches with high topic overlap → `MERGE_CANDIDATE`

### Running as Cron

```bash
# Every 30 minutes
*/30 * * * * cd /home/victo && \
  /home/victo/venvs/ml/bin/python3 -c "
import sys; sys.path.insert(0, '/home/victo/.openclaw/workspace/module')
from lcm_geometry_controller import GeometryController
gc = GeometryController('/home/victo/.openclaw/lcm_geometry.db')
r = gc.run_maintenance_cycle()
print(f'maint: recomputed={r[\"recomputed\"]} split={r[\"split_pending\"]} merge={r[\"merge_candidates\"]}')
" >> /home/victo/.openclaw/logs/geometry_maint.log 2>&1
```

Or via OpenClaw heartbeat (edit `HEARTBEAT.md`):
```markdown
# HEARTBEAT.md
- Run maintenance every ~30 min:
  `python3 -c "from lcm_geometry_controller import GeometryController; gc.run_maintenance_cycle()"`
```

---

## 10. Backfill — One-Time Setup

If you're starting fresh (no `lcm_geometry.db` yet), run the backfill once to populate all historical data:

```python
import sys
sys.path.insert(0, '/home/victo/.openclaw/workspace/module')
from lcm_geometry_controller import GeometryController, EmbeddingProvider
import time

gc = GeometryController('/home/victo/.openclaw/lcm_geometry.db',
                        embedding_provider=EmbeddingProvider())

start = time.time()
r = gc.backfill_from_lcm('/home/victo/.openclaw/lcm.db', resume=False)
elapsed = time.time() - start

print(f"Backfill done in {elapsed:.1f}s")
print(f"  processed={r['processed']} sampled={r.get('sampled',0)} "
      f"skipped={r['skipped']} failed={r['failed']}")
```

Then import DAG edges:
```python
r2 = gc.import_dag_edges_from_lcm('/home/victo/.openclaw/lcm.db')
print(f"Edges: {r2['derived_from']} derived_from + {r2['summarizes']} summarizes")
```

For incremental updates (new messages since last backfill):
```python
gc.backfill_from_lcm('/home/victo/.openclaw/lcm.db', resume=True)
```

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

The OpenClaw MCP server (`~/.openclaw/extensions/geometry-mcp/server.py`) exposes four tools:

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

Registered in `~/.openclaw/openclaw.json` under `mcp.servers`:

```json
"geometry-hybrid": {
  "command": "/home/victo/venvs/ml/bin/python3",
  "args": ["/home/victo/.openclaw/extensions/geometry-mcp/server.py"]
}
```

**CRITICAL:** Must use full path to ML venv Python. Restart gateway after config changes:
```bash
openclaw gateway restart
```

---

## 13. All 5 Fixes Applied

These fixes were applied on 2026-04-04 to the module at `~/.openclaw/workspace/module/lcm_geometry_controller.py` (1701 lines).

### Fix 1 — EmbeddingProvider (New Feature)

**What:** Added `EmbeddingProvider` class — pluggable embedding backend with lazy model loading, in-memory LRU cache, `embed()` and `embed_batch()` methods.

**Integration:** Passed to `GeometryController(embedding_provider=EmbeddingProvider())`. `on_new_item()` now accepts `text=` directly — auto-embeds via provider.

**Benefit:** No need for caller to pre-compute embeddings. First call to `embed()` loads the sentence-transformers model. Repeated calls use cache.

### Fix 2 — CSD `inf` on First Insert (Bug Fix)

**Bug:** `_residual()` used `hasattr(self.cfg, "EPS")` which was always `False` (GeometryConfig has no EPS attribute). First message in any branch got `csd=inf`.

**Fix:** Direct `if expected == 0.0` check with `expected = max(abs(delta), 1e-6)`.

**Result:** First-insert CSD = 0.0 (finite, correct — new branch is always a valid attachment target).

### Fix 3 — backfill_from_lcm() with Resume (New Feature)

**What:** Added `backfill_from_lcm()` method on `GeometryController` with resume support, stratified sampling for large conversations (>200 msgs), and progress callback.

**Benefit:** Full backfill takes ~25 min for 437 conversations. Incremental resume skips already-done branches and processes only new ones.

### Fix 4 — DAG Edge Import (New Feature)

**What:** Added `import_dag_edges_from_lcm()` — reads `summary_parents` and `summary_messages` from `lcm.db`, populates `memory_edges` table.

**Result:** 435 `DERIVED_FROM` edges + 16,021 `SUMMARIZES` edges imported. Merge/split scorers can now compute `topic_overlap` and `retrieval_co_use`.

### Fix 5 — Maintenance Fallback (Bug Fix)

**Bug:** `run_maintenance_cycle()` only recomputed branches with ≥8 `memory_nodes` rows. After backfill, most branches had 0–2 rows → silently skipped. Only 16/437 branches were recomputed.

**Fix:** Added `elif s.mean_vec:` fallback that recomputes regime from stored scalar metrics (eff_rank, anisotropy, coherence, compression_loss) even when no memory_nodes rows exist.

**Result:** All 437/437 branches recomputed on each maintenance cycle.

---

## 14. Pending Integrations

These items are designed but not yet wired into OpenClaw's runtime.

### 🔜 Real-Time LCM Polling

**Goal:** When Victor chats and new messages get added to `lcm.db`, the geometry controller should automatically process them via `gc.on_new_item()`.

**What's needed:** A background loop that polls `lcm.db` every ~30 seconds for new rows in `messages` table since last check, calls `gc.on_new_item()` for each, and commits to `lcm_geometry.db`.

**Integration point:** OpenClaw agent heartbeat or a dedicated cron process.

### 🔜 Maintenance Cron Job

**Goal:** Keep branch states current by running `gc.run_maintenance_cycle()` every 20–30 minutes.

**What's needed:** A cron entry or OpenClaw heartbeat that calls `gc.run_maintenance_cycle()` and logs results.

**Priority:** Medium — without it, regime classifications may drift stale.

### 🔜 Retrieval Feedback Loop

**Goal:** When Victor (or the agent) retrieves a conversation branch and acts on it, record that signal in `retrieval_feedback` table. The `RetrievalRanker` should then boost frequently-useful branches.

**What's needed:** After any `hybrid_search` call that results in Victor engaging with a conversation, write a feedback row. The `rank_retrieval()` method already accepts `historical_use=` and `same_project=` parameters.

### 🔜 on_new_item() → LCM Event Hook

**Goal:** `gc.on_new_item()` should be called by LCM's own persistence layer when it writes a new message, not just by a polling loop.

**What's needed:** Integration into `lossless-claw` plugin (LCM plugin) to call the geometry controller after persisting each message. This would require the LCM plugin to import the geometry module and call `on_new_item()` with the new message's data.

---

## 15. Troubleshooting

### Module won't import

```
ModuleNotFoundError: No module named 'lcm_geometry_controller'
```

**Fix:** Add module path:
```python
import sys
sys.path.insert(0, '/home/victo/.openclaw/workspace/module')
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
gc = GeometryController('/home/victo/.openclaw/lcm_geometry.db')
for s in gc.db.all_branches():
    if not s.mean_vec:
        print(f"{s.branch_id}: NO mean_vec — will use fallback")
```

### CSD score is inf or nan

Ensure `embedding_dim` in `GeometryConfig` matches your embedding model (default 384 for all-MiniLM-L6-v2).

### Backfill is slow

Use `resume=True` for incremental updates:
```python
gc.backfill_from_lcm('/home/victo/.openclaw/lcm.db', resume=True)
```

For large conversations (>200 messages), the backfill stratifies to 200 samples. To process all messages:
```python
gc.backfill_from_lcm('/home/victo/.openclaw/lcm.db', resume=True, force_full=True)
```

### Gateway restart kills tmux sessions

Restart tandem after gateway restart:
```bash
~/.openclaw/start-tandem.sh
```

---

## Quick Reference Card

```python
# Setup (one time)
import sys
sys.path.insert(0, '/home/victo/.openclaw/workspace/module')
from lcm_geometry_controller import create_geometry_controller, EmbeddingProvider

gc = create_geometry_controller(
    '/home/victo/.openclaw/lcm_geometry.db',
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
gc.backfill_from_lcm('/home/victo/.openclaw/lcm.db', resume=True)
```

---

*Manual generated: 2026-04-04*
*Module: `~/.openclaw/workspace/module/lcm_geometry_controller.py` (1701 lines)*
*Companion backfill script: `~/.openclaw/workspace/module/lcm_geometry_backfill.py`*
*MCP server: `~/.openclaw/extensions/geometry-mcp/server.py`*
