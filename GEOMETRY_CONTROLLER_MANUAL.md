# LCM Geometry Controller â€” Manual

**Version:** 1.6
**Module:** `lcm_geometry_controller.py`
**Geometry DB:** `<openclaw_home>/lcm_geometry.db`
**LCM DB:** `<openclaw_home>/lcm.db`
**Last Updated:** 2026-04-18

---

## Table of Contents

1. [What Is the Geometry Controller?](#1-what-is-the-geometry-controller)
2. [Architecture](#2-architecture)
3. [Quick Start](#3-quick-start)
4. [Database Schema](#4-database-schema)
5. [Core API Reference](#5-core-api-reference)
6. [Configuration â€” GeometryConfig](#6-configuration--geometryconfig)
7. [EmbeddingProvider](#7-embeddingprovider)
8. [Retrieval & Ranking](#8-retrieval--ranking)
9. [Maintenance Cycle](#9-maintenance-cycle)
10. [Backfill â€” One-Time Setup](#10-backfill--one-time-setup)
11. [DAG Edge Import](#11-dag-edge-import)
12. [MCP Server Tools](#12-mcp-server-tools)
13. [Current Enhancements](#13-current-enhancements)
14. [Integration Status](#14-integration-status)
15. [Troubleshooting](#15-troubleshooting)

---

## 1. What Is the Geometry Controller?

The geometry controller is a semantic memory layer that sits alongside OpenClaw's LCM (Lossless Context Management). It annotates each conversation branch with **geometric state** â€” embedding centroids, anisotropy, coherence, effective rank â€” and uses these signals to:

- **Score** where a new message best belongs (which branch it fits)
- **Classify** branch health: PRODUCTIVE, RIGID, or UNSTABLE
- **Rank** retrieval candidates by semantic similarity + trust + reactivation

Think of it as the difference between a filing cabinet (LCM â€” stores messages) and a smart librarian (geometry controller â€” knows which drawer is healthiest for a new topic).

---

## 2. Architecture

### Two Databases

```
lcm.db                              lcm_geometry.db
â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€       â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€
Messages (immutable)                Per-branch geometry (mean_vec, eff_rank,
Summaries (DAG nodes)               anisotropy, coherence, trace, anchor_drift)
Summary DAG edges                    Per-message/summary 384-dim embeddings
Context items (active ctx)           DAG edges (derived_from, summarizes)
                                    Branch lifecycle states + regimes
                                    Maintenance job queue
                                    Retrieval feedback signals
```

**LCM is the source of truth.** The geometry DB is a **read-heavy companion** â€” LCM never writes to it. The geometry controller only reads from LCM.

### Geometry Per Branch

Each conversation branch gets:
- `mean_vec` â€” 384-dim centroid (EMA-updated as messages are added)
- `cov_diagonal` â€” variance per dimension
- `eff_rank` â€” effective rank of the embedding cloud (how many orthogonal directions are actually used)
- `anisotropy` â€” concentration proxy computed from variance spectrum (`max_eigenvalue / sum_eigenvalues`)
- `coherence` â€” mean pairwise cosine similarity of messages in branch
- `trace` â€” sum of variances (total spread)
- `anchor_drift` â€” how much the centroid has shifted since the anchor was set
- `regime` â€” PRODUCTIVE (healthy), RIGID (over-consolidated), UNSTABLE (topic drift)

### Lifecycle States

```
FORMING â†’ ACTIVE â†’ STABLE
ACTIVE/STABLE/TENSIONED â†’ DORMANT â†’ REACTIVATING â†’ ACTIVE
STABLE â†’ MERGE_CANDIDATE â†’ (soft-merged affinity in current mode)
ACTIVE â†’ SPLIT_PENDING â†’ (forked)
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
print('OK â€”', gc.db.conn.execute('SELECT COUNT(*) FROM branch_states').fetchone()[0], 'branches')
"
```

### Basic Usage â€” On New Message

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
    text="What's the weather in Amposta?",  # â† auto-embeds
    role="user",
    token_count=6,
)
print(decision.action, decision.branch_id)  # e.g. "attach" "conv_42"
```

### Basic Usage â€” Retrieval

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

### `branch_states` â€” One Row Per Conversation Branch

| Column | Type | Description |
|--------|------|-------------|
| `branch_id` | TEXT PK | Primary key (e.g. `conv_1`) |
| `state` | TEXT | BranchState enum: FORMING, ACTIVE, STABLE, TENSIONED, DORMANT, etc. |
| `regime` | TEXT | GeometricRegime: PRODUCTIVE, RIGID, UNSTABLE |
| `mean_vec` | BLOB (JSON bytes) | 384-dim centroid as JSON list |
| `cov_diagonal` | BLOB (JSON bytes) | Variance per dimension |
| `eff_rank` | REAL | Effective rank (1â€“384) |
| `anisotropy` | REAL | Variance concentration proxy (`max_eigenvalue / sum_eigenvalues`) |
| `coherence` | REAL | Mean pairwise cosine similarity |
| `trace` | REAL | Sum of variances |
| `compression_loss` | REAL | Compression error (reconstruction from low-rank) |
| `topic_drift_density` | REAL | Primary topic drift / subtopic diversity density signal |
| `contradiction_density` | REAL | Legacy compatibility mirror of topic drift density |
| `retrieval_error` | REAL | EMA of retrieval mis-match scores |
| `usefulness` | REAL | EMA usefulness score from retrieval feedback |
| `reactivation_score` | REAL | Reactivation signal used for DORMANT -> REACTIVATING transitions |
| `anchor` | BLOB (JSON bytes) | Stable reference centroid |
| `anchor_drift` | REAL | Distance current mean_vec has drifted from anchor |
| `node_count` | INT | Number of memory_nodes in this branch |
| `split_counter` | INT | Consecutive high split scores |
| `merge_counter` | INT | Consecutive merge candidates |
| `last_update_ts` | REAL | Unix timestamp of last update |

### `memory_nodes` â€” Per-Message/Per-Summary Embeddings

| Column | Type | Description |
|--------|------|-------------|
| `id` | TEXT PK | Node ID (e.g. `msg_abc123`) |
| `lcm_id` | TEXT | LCM ID for this item |
| `node_type` | TEXT | NodeType: MESSAGE, TOOL_RESULT, LEAF_SUMMARY, CONDENSED_SUMMARY, etc. |
| `parent_id` | TEXT | Parent node ID (for DAG) |
| `branch_id` | TEXT FK | Owning branch |
| `timestamp` | REAL | When added |
| `role` | TEXT | Message role: user, assistant, system |
| `embedding` | BLOB (JSON bytes) | 384-dim vector as JSON list |
| `update_mode` | TEXT | Per-node update intent metadata: `fork`, `attach`, `refine`, `contradict`, `supersede` |
| `correction_kind` | TEXT | Version-flow label: `none`, `refine`, `contradict`, `supersede` |
| `correction_prev_id` | TEXT | Previous node in explicit correction chain |
| `correction_root_id` | TEXT | Stable root node ID for the correction chain |
| `correction_version` | INTEGER | Monotonic version index inside correction chain |

### `memory_edges` â€” DAG Edges

| Column | Type | Description |
|--------|------|-------------|
| `src_id` | TEXT | Source node |
| `dst_id` | TEXT | Target node |
| `edge_type` | TEXT | EdgeType: DERIVED_FROM, SUMMARIZES, TOPIC_DRIFT, REFINES, CONTRADICTS, SUPERSEDES |
| `weight` | REAL | Edge weight (default 1.0) |

### `retrieval_feedback` â€” Retrieval Feedback Signals

| Column | Type | Description |
|--------|------|-------------|
| `id` | TEXT PK | Feedback row ID |
| `branch_id` | TEXT | Retrieved branch |
| `query_id` | TEXT | Query correlation ID |
| `score` | REAL | Retrieval score at selection time |
| `used` | INTEGER | Whether result was used |
| `corrected` | INTEGER | Whether retrieval was corrected |
| `expanded` | INTEGER | Whether branch expansion was requested |
| `timestamp` | REAL | Timestamp |

### `maintenance_split_observations` â€” Split Decision Trace

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

- **`lcm_id`** â€” LCM ID string (e.g. `msg_abc123`)
- **`node_type`** â€” `NodeType` enum value
- **`text`** â€” raw text (auto-embeds via `embedding_provider` if provided)
- **`embedding`** â€” pre-computed 384-dim vector (if not using `text=`)
- **`role`** â€” message role: `user`, `assistant`, `system`
- **`token_count`** â€” approximate token count
- **`conflict_score`** - optional conflict severity signal for CSD scoring
- **`active_branch_id`** - preferred branch hint from caller runtime
- **`force_branch_id`** - bypass candidate selection and attach to a specific branch id (used by deterministic backfill mapping)
- **`parent_lcm_id`** - parent LCM id; controller resolves parent node and links DAG/temporal edges

**Returns:** `AllocationDecision` with fields:
- `action` â€” `"attach"`, `"attach_tension"`, or `"fork"`
- `branch_id` â€” target branch ID
- `csd_score` â€” CSD score for chosen branch
- `conflict_score` â€” conflict metric for chosen branch
- `update_mode` â€” metadata label for how the insert relates to branch history (`fork`, `attach`, `refine`, `contradict`, `supersede`)
- `correction_*` â€” explicit correction-chain metadata for versioned factual updates
- `rationale` â€” human-readable explanation

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

### `gc.rank_retrieval(query_emb, historical_use=None, same_project=None, retrieval_mode=None)`

Rank all branches by relevance to a query embedding.

- **`query_emb`** â€” numpy array, shape (384,)
- **`historical_use`** â€” dict `branch_id â†’ float` of past retrieval frequency
- **`same_project`** â€” dict `branch_id â†’ float` of same-project signal
- **`retrieval_mode`** â€” `balanced` (default) | `factual` | `exploratory`

**Returns:** list of `RetrievalCandidate` sorted by `total_score` descending:
- `branch_id`
- `sem_score` â€” cosine similarity to branch centroid
- `trust_score` â€” composite of coherence, compression_loss, topic drift density (penalty), retrieval_error
- `react_score` â€” reactivation signal
- `total_score` â€” weighted composite: `(Î±Â·sem + Î²Â·trust + Î´Â·react) * mode_multiplier(state, regime)`

### `gc.branch_report(branch_id)`

Detailed geometry report for one branch.

```python
r = gc.branch_report('conv_1')
# r.keys(): branch_id, state, regime, node_count, eff_rank, anisotropy,
#           coherence, trace, anchor_drift, compression_loss,
#           contradiction_density (legacy), topic_drift_density, subtopic_diversity_density,
#           retrieval_error, update_mode_counts,
#           correction_counts, recent_corrections, mean_vec (truncated)
```

### `gc.run_maintenance_cycle()`

Full maintenance sweep over all branches. Should be run periodically (every 20â€“30 min).

**Operations per branch:**
1. **Starts scalar-first scan** over `branch_states` and only full-loads branch blobs when needed
2. **Recomputes geometry** for eligible branches from `memory_nodes` rows
3. **Refreshes topic drift signals** (density + bounded `TOPIC_DRIFT` edges)
4. **Reclassifies regime** - PRODUCTIVE / RIGID / UNSTABLE
5. **Applies dormancy policy** based on real branch activity age + usefulness thresholds
6. **Scans for splits and prepares ranked candidates** when gate + hysteresis conditions are met
7. **Scores merges with runtime signals** (graph overlap + retrieval co-use)
8. **Executes pending split jobs** (k-means(2), child branches, `REFINES` edges + child priors)
9. **Executes pending merge jobs** (soft merge mode: affinity edges + queue drain)
10. **Runs reactivation scan**

**Returns:** `dict` with counts:
```python
{
    'recomputed': 437,
    'dormant_marked': 2,
    'split_pending': 0,
    'split_executed': 0,
    'merge_candidates': 0,
    'merge_executed': 1,
    'reactivated': 0,
    'retrieval_feedback_pruned': 0,
    'retrieval_feedback_pruned_age': 0,
    'retrieval_feedback_pruned_cap': 0,
    'split_observations': 437,
    'split_trace_run_id': '...',
    'maintenance_chunking': {
        'enabled': True,
        'chunk_size': 50,
        'selected_branches': 50,
        'wrapped': False,
        'cursor_before': 'conv_105',
        'cursor_after': 'conv_150'
    }
}
```

### `gc.backfill_from_lcm(lcm_db_path, resume=True, progress_cb=None, max_per_conv=200, error_log_path=None)`

One-time backfill from LCM. Creates `memory_nodes` rows for all messages and summaries.

- **`lcm_db_path`** â€” path to `lcm.db`
- **`resume`** â€” if True, skips branches that already have `node_count > 0`
- **`progress_cb`** â€” callback `(current, total) â†’ None` for progress tracking
- **`max_per_conv`** â€” cap per-conversation sampled rows in large histories
- **`error_log_path`** â€” optional target file for structured per-conversation backfill errors

**Returns:** `dict` with `processed`, `sampled`, `skipped`, `failed`, `errors_logged`.

### `gc.import_dag_edges_from_lcm(lcm_db_path)`

Import summary DAG edges from LCM into `memory_edges`.

Reads:
- `summary_parents` â†’ `DERIVED_FROM` edges (parent_summary â†’ child_summary)
- `summary_messages` â†’ `SUMMARIZES` edges (summary â†’ message)

Behavior:
- Purges previously imported `DERIVED_FROM` / `SUMMARIZES` edges before rebuild.
- Resolves LCM IDs to real geometry node IDs (`memory_nodes.id`) by `(lcm_id, node_type)`.
- Skips rows where required nodes are not present in geometry DB.

**Returns:** `dict` with:
- `derived_from`
- `summarizes`
- `skipped`
- `purged`
- `summary_nodes_indexed`
- `message_nodes_indexed`

### `gc.audit_summary(summary_id, embedding=None, text=None)`

Score summary quality. Compares summary embedding to the mean of its constituent messages.

---

## 6. Configuration â€” GeometryConfig

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
| `attach_threshold` | 0.50 | Global fallback attach threshold |
| `tension_threshold` | 0.70 | Global fallback tension threshold |
| `attach_threshold_by_type` | `{"default": 0.50}` | Optional per-branch-type attach threshold overrides |
| `tension_threshold_by_type` | `{"default": 0.70}` | Optional per-branch-type tension threshold overrides |
| `branch_type_profiles` | dict | Optional per-branch metric profiles (`csd_gamma`, `retrieval_kappa`, `split_zeta`, `split_policy`, `merge_eta`, `merge_policy`) |
| `alpha_sem` | 0.60 | Semantic similarity weight in retrieval |
| `beta_trust` | 0.25 | Trust score weight in retrieval |
| `delta_react` | 0.15 | Reactivation weight in retrieval |
| `split_score_threshold` | 0.075 | Split score threshold used by split eligibility |
| `split_min_nodes` | 6 | Baseline split gate; effective gate uses `max(split_min_nodes, min_branch_size)` |
| `split_hysteresis` | 3 | Consecutive high scores â†’ SPLIT_PENDING |
| `max_split_enqueues_per_cycle` | 5 | Maximum split jobs enqueued per maintenance cycle (score-ranked) |
| `split_child_copy_usefulness` | True | Copy parent usefulness to split children |
| `split_child_copy_retrieval_error` | True | Copy parent retrieval error to split children |
| `split_child_anchor_from_centroid` | True | Seed split child anchor from cluster centroid |
| `merge_hysteresis` | 3 | Consecutive merge candidates â†’ MERGE_CANDIDATE |
| `merge_execution_mode` | `soft` | Merge execution mode (`soft`/`off`) |
| `merge_max_jobs_per_cycle` | 5 | Max merge jobs executed per maintenance cycle |
| `merge_soft_edge_weight` | 1.0 | Fallback SAME_TOPIC edge weight in soft merges |
| `usefulness_lambda` | 0.10 | EMA decay for usefulness signals |
| `retrieval_prefilter_limit` | 256 | Max branches in scalar retrieval prefilter shortlist |
| `retrieval_result_limit` | 128 | Max retrieval candidates returned |
| `retrieval_mode_default` | `balanced` | Default retrieval routing mode (`balanced`/`factual`/`exploratory`) |
| `retrieval_mode_factors` | dict | Regime/state multipliers applied per retrieval mode |
| `candidate_prefilter_limit` | 128 | Max branches inspected by scalar prefilter for `on_new_item` |
| `candidate_branch_cap` | 10 | Max full branches loaded for final allocation scoring |
| `topic_drift_sim_threshold` | 0.00 | Topic drift gate (`sim < threshold`) on temporally distant pairs |
| `topic_drift_min_temporal_gap` | 48 | Minimum sequence distance between compared messages |
| `topic_drift_max_temporal_gap` | 0 | Optional upper sequence distance bound (`0` = no max) |
| `topic_drift_allowed_roles` | `["assistant","user"]` | Roles eligible for topic drift pairing |
| `topic_drift_min_token_count` | 8 | Minimum token count per side for eligible pair |
| `topic_drift_min_content_chars` | 30 | Minimum non-whitespace LCM content chars per side |
| `topic_drift_require_content_nonempty` | True | Enforces content-based filtering via `lcm.db` |
| `topic_drift_sample_min_nodes` | 64 | Minimum sample target for topic drift compute on large branches |
| `topic_drift_sample_max_nodes` | 192 | Sampling cap for topic drift compute (`0` disables cap) |
| `topic_drift_edge_max_pairs` | 256 | Max persisted topic drift edges per branch recompute |
| `lcm_db_path` | auto / explicit | Optional path override used for content-length filtering |
| `dormant_after_days` | 14.0 | Activity age threshold to mark dormant |
| `dormant_usefulness_max` | 0.20 | Dormancy requires usefulness at/below this value |
| `dormant_min_nodes` | 8 | Minimum branch size before dormancy policy applies |
| `protected_branch_types` | list[str] | Protected branch classes for hard attach/merge gates |
| `protected_attach_conflict_threshold` | 0.35 | Protected attach conflict gate |
| `protected_attach_topic_drift_threshold` | 0.20 | Protected attach topic-drift density gate |
| `protected_merge_block` | True | If true, blocks merge queueing when either branch is protected |
| `protected_merge_topic_drift_threshold` | 0.20 | Alternative topic-drift-based protected merge gate |
| `reactivation_min_score` | 0.60 | Minimum reactivation score before wake transitions |
| `reactivation_guard_enabled` | True | Enables topic-drift/error/similarity checks before wake |
| `reactivation_max_topic_drift` | 0.35 | Wake blocked above this topic drift density |
| `reactivation_max_retrieval_error` | 0.60 | Wake blocked above this retrieval error |
| `reactivation_min_similarity` | 0.15 | Query-to-branch similarity floor for relevance-triggered wake |
| `update_mode_refine_similarity_min` | 0.92 | Similarity floor for labeling node updates as `refine` |
| `update_mode_contradict_conflict_min` | 0.25 | Conflict floor for labeling updates as `contradict` |
| `update_mode_supersede_similarity_min` | 0.78 | Similarity floor for `supersede` classification |
| `update_mode_supersede_conflict_min` | 0.70 | Conflict floor for `supersede` classification |
| `update_mode_supersede_branch_types` | list[str] | Branch types eligible for `supersede` classification |
| `rank_target` | dict | eff_rank target bands per branch type |
| `rigid_rank_ratio` | 0.15 | eff_rank below this â†’ RIGID regime |
| `unstable_coh_floor` | 0.45 | Coherence below this â†’ UNSTABLE |
| `unstable_comp_ceil` | 0.65 | Compression loss above this â†’ UNSTABLE |

Compatibility notes:
- Legacy `contradiction_*` runtime keys are still accepted as fallbacks for topic drift settings.
- Primary stored field is `topic_drift_density`; `contradiction_density` is kept synced as a legacy compatibility mirror.
- Reports expose `contradiction_density`, `topic_drift_density`, and `subtopic_diversity_density` as equivalent aliases.

---

## 7. EmbeddingProvider

Pluggable embedding backend.

Supported backends:
- `sentence_transformers` (default)
- `llama_cpp` / GGUF (`llama-cpp-python`)
- `http` (OpenAI-compatible `/embeddings` endpoint)

### Init

```python
EmbeddingProvider(
    model_name="all-MiniLM-L6-v2",
    device="cpu",
    backend="sentence_transformers",
    gguf_model_path=None,      # required for llama_cpp unless model_name endswith .gguf
    gguf_n_ctx=2048,
    gguf_n_threads=None,
    http_url=None,             # required for http backend
    http_timeout_sec=30.0,
    cache=None,
)
```

### Methods

**`ep.embed(text)`** â€” Encode single text. Result cached by `text[:200]`.

**`ep.embed_batch(texts, batch_size=64)`** â€” Encode list of texts. No caching.

**`ep.embedding_dim`** â€” Lazy property returning model's embedding dimension.

**`ep.descriptor()`** â€” Runtime signature fields (`backend`, `model_name`, `embedding_dim`, backend-specific connection info). For `llama_cpp`, `model_name` is canonicalized to GGUF basename so signature identity is stable for filename/path input variants.

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

The `EmbeddingProvider` is lazy â€” the model is only loaded on first `embed()` call, not at init.

### Runtime signature guard

`GeometryController` enforces a persisted embedding runtime signature:

- compares backend/model/dimension against prior DB runtime signature
- blocks startup on mismatch to prevent mixed embedding spaces in one DB
- canonicalizes GGUF identity before comparison (backend aliases, model basename, normalized gguf path)
- auto-refreshes legacy raw signature variants when canonical identity matches
- override only when intentional: `GEOMETRY_ALLOW_EMBEDDING_SIGNATURE_CHANGE=1`

---

## 8. Retrieval & Ranking

The `RetrievalRanker` computes a composite score for each branch candidate:

```
total = Î±Â·sem + Î²Â·trust + Î´Â·react
```

Where:
- **`sem`** â€” cosine similarity of query embedding to branch mean_vec
- **`trust`** â€” `Îº_coherenceÂ·coherence + Îº_comp_lossÂ·compression_loss - |Îº_topic_drift|Â·topic_drift_density + Îº_ret_errorÂ·retrieval_error`
- **`react`** â€” recency/reactivation signal (half semantic + history + project signals)

### Source-Time Recency Reranking

The MCP `hybrid_search` tool can optionally blend source-time recency into the returned order. This is intentionally separate from geometry `last_update_ts`, which reflects ingestion, feedback, maintenance, or polling activity.

When `recency_boost > 0`, the MCP layer computes:

```
recency_score = 2 ** (-age_days / recency_half_life_days)
final_score   = (1 - recency_boost) * normalized_relevance
              + recency_boost * recency_score
```

`total_score` remains the original geometry relevance/trust score. `final_score` is the blended presentation score used for recency-aware ordering.

For geometry branches, source timestamps are resolved in this order:

1. LCM message `created_at` from numeric `memory_nodes.lcm_id`
2. LCM message min/max `created_at` for `conv_*` fallback when branch lineage is sparse
3. LCM conversation `created_at` for `conv_*` fallback when messages are unavailable
4. `daily_log_content.created_ts` for daily-log branches
5. Geometry `last_update_ts` only as a final fallback

The MCP layer now keeps two source-time axes:

- `source_timestamp`: the first/oldest canonical source time used for source-age recency.
- `last_source_timestamp`: the latest canonical source time used for workflow/activity filtering.

Supported hybrid-search filters:

| Parameter | Meaning |
|---|---|
| `recency_boost` | Blend weight in `[0,1]`; default `0` |
| `recency_half_life_days` | Freshness decay half-life; default `14` |
| `max_age_days` | Keep only items newer than this many days |
| `updated_within_days` | Alias for `max_age_days` |
| `min_age_days` | Keep only items at least this many days old |
| `updated_after` / `updated_before` | ISO date/time or epoch bounds |
| `date_from` / `date_to` | Date-range aliases for `YYYY-MM-DD` workflows |
| `state` | Optional geometry lifecycle filter, such as `FORMING`, `ACTIVE`, `STABLE`, or a list |
| `activity_state` | Optional latest-source-activity filter: `recent`, `stale`, or `dormant` |
| `activity_within_days` | Window for `activity_state`; default `14` |
| `state_group` | Convenience filter: `working` = recent `FORMING`/`ACTIVE`/`REACTIVATING`; `settled`/`dormant` = older `STABLE` |

Recency-aware result rows include `source_timestamp`, `timestamp_source`, `last_updated`, `age_days`, `recency_score`, and `recency_label`. Activity-aware rows also include `last_source_timestamp`, `last_source_updated`, `last_timestamp_source`, `activity_age_days`, `activity_score`, `activity_label`, and `activity_state`. `base_score` is the original semantic/trust score and `ranking_score` is the presented ordering score; the legacy `retrieval_kappa` field remains as a compatibility alias for `base_score`.

### Activity-Aware Content Retrieval

The MCP `conversation_content` tool supports the same branch-targeting vocabulary for multi-branch reads:

| Parameter | Meaning |
|---|---|
| `state` | Optional lifecycle state or list of states |
| `state_group` | `working`, `settled`, `dormant`, or `all` |
| `activity_state` | `recent`, `stale`, or `dormant` latest-source-activity filter |
| `activity_within_days` | Activity window; default `14` |
| `fallback_when_empty` | Default `true`; empty `ACTIVE` requests explicitly fall back to `state_group="working"` |

The fallback is never silent. Responses include a `fallback` object with the original filter, replacement filter, and reason. Branch rows include `last_source_timestamp`, `last_source_updated`, `activity_age_days`, `activity_label`, and `activity_state`. If a summaries-only branch has no summaries, `conversation_content` falls back to messages for that branch and adds `summary_empty_used_messages_fallback` to the branch warning.

### CSD Scoring (On New Message)

When a new message arrives, the `CSDScorer` scores it against each candidate branch:

```
csd = importanceÂ·novelty + conflictÂ·resolution + coherenceÂ·alignment
```

Components:
- **`delta_mu`** â€” distance from branch centroid
- **`delta_r`** â€” change in effective rank
- **`delta_A`** â€” change in anisotropy
- **`delta_tau`** â€” change in trace

The scorer uses EMA-weighted residuals so one-off outliers don't destabilise stable branches.

---

## 9. Maintenance Cycle

Run `gc.run_maintenance_cycle()` every 20â€“30 minutes. It:

1. **Starts scalar-first** from `branch_states` and updates scalar metrics without loading full blobs when possible
2. **Recomputes geometry** for branch subsets that pass recompute gates
3. **Updates scalars** (eff_rank, anisotropy, coherence, trace) via adiabatic EMA
4. **Reclassifies regime** â€” PRODUCTIVE / RIGID / UNSTABLE
5. **Applies dormancy policy** â€” inactive + low-usefulness branches move to `DORMANT`
6. **Scans for splits** â€” branch must pass:
   - score gate: `split_score > split_score_threshold`
   - node readiness gate: `node_count >= max(split_min_nodes, min_branch_size)`
   - real-node readiness gate: embedded rows also meet the same threshold
   - hysteresis gate: `split_counter >= split_hysteresis`
   Eligible branches are score-ranked and throttled by `max_split_enqueues_per_cycle`.
7. **Scans for merges** â€” stable/dormant branches with high topic overlap â†’ `MERGE_CANDIDATE`
8. **Executes pending split jobs** â€” children inherit key priors and centroid-seeded anchors
9. **Executes pending merge jobs** â€” in `soft` mode, writes SAME_TOPIC affinity edges and drains queue

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

### 9.1 Lossless cleanup + geometry sync wrapper (operations)

For maintenance runs that need Lossless-Claw doctor-clean plus immediate geometry DAG validation, use:

```bash
/home/victo/.openclaw/workspace/scripts/lossless_geometry_maintenance.sh --text
```

Apply cleanup (backup is created by Lossless-Claw):

```bash
/home/victo/.openclaw/workspace/scripts/lossless_geometry_maintenance.sh --apply --text
```

Single-filter apply:

```bash
/home/victo/.openclaw/workspace/scripts/lossless_geometry_maintenance.sh --apply --filter null_subagent_context --text
```

What the wrapper does:

1. Runs Lossless doctor-clean scan/apply via `lossless_doctor_clean_runner.ts`
2. Rebuilds imported DAG edges from `lcm.db` into `lcm_geometry.db`
3. Validates orphan counts for `derived_from` and `summarizes`

---

## 10. Backfill - One-Time Setup

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


### 10.1 Targeted backfill (zombie repair)

Use targeted backfill when you only want selected conversations:

```text
geometry-hybrid__backfill_lcm_conversations(
  conversation_ids=[57,365,403],
  max_per_conv=200,
  resume=true,
  dry_run=true
)
```

Behavior:
- `dry_run=true` works without embeddings and reports what would run.
- `dry_run=false` requires an embedding provider.
- If provider is missing, preflight abort is explicit:
  - `provider_ready=false`
  - `aborted=true`
  - `preflight_error=<reason>`
  - per-conversation detail status `failed_preflight`

This prevents partial/silent failures in direct Python usage and gives clear operator feedback in MCP output.
---

## 11. DAG Edge Import

The `import_dag_edges_from_lcm()` method reads two LCM tables:

**`summary_parents`** â†’ `DERIVED_FROM` edges
```
child_summary â”€â”€â”€â”€â†’ parent_summary
```

**`summary_messages`** â†’ `SUMMARIZES` edges
```
summary â”€â”€â”€â”€â”€â”€â”€â”€â”€â†’ message
```

These edges enable:
- **`topic_overlap`** in merge scoring â€” two branches share summaries â†’ high topic overlap â†’ merge candidate
- **`retrieval_co_use`** â€” branches that share message ancestors are related

Current importer guarantees node-level consistency:
- imported edges always target real `memory_nodes.id`
- stale imported edges are removed before rebuild
- integrity can be verified by orphan counters (`0` is expected)

Notes:
- This import currently uses `summary_parents` and `summary_messages`.
- Message reply-thread import (`REPLIES_TO` / `CHILD_OF`) is not available in the current `lcm.db` schema because `messages.parent_id` is not present.

**Results from latest live rebuild (2026-04-09):**
- 443 `DERIVED_FROM` edges
- 17,250 `SUMMARIZES` edges
- 847 skipped rows (missing summary/message nodes in geometry DB)
- orphan counts after rebuild: `0` for `derived_from` and `summarizes`

---

## 12. MCP Server Tools

The OpenClaw MCP server (`<openclaw_home>/extensions/geometry-mcp/server.py`) exposes eleven tools:

### `geometry-hybrid__hybrid_search`

Combines LCM keyword search with geometry DB semantic search. Best for open-ended recall.

```
geometry-hybrid__hybrid_search(query="CLGK dashboard", top_n=5, retrieval_mode="factual")
```

Recency-aware example:

```
geometry-hybrid__hybrid_search(
  query="CLGK dashboard",
  top_n=5,
  retrieval_mode="factual",
  recency_boost=0.35,
  updated_within_days=14
)
```

Returns combined results from both systems with recommendation. Result rows include original relevance (`total_score`), optional blended ordering score (`final_score`), and source-time recency metadata (`source_timestamp`, `timestamp_source`, `last_updated`, `age_days`, `recency_score`, `recency_label`).

LCM keyword results use keyword ranking v2:
- meaningful query terms are matched case-insensitively
- multi-term queries add an exact phrase pass
- all-term matches and phrase matches receive explicit boosts
- raw hit-count influence is capped at 40 matches
- conversations with more than 200 keyword hits receive a small volume penalty
- snippets are centered around the matched term or phrase

LCM result rows include `matched_keywords`, `phrase_matched`, and `keyword_debug` so an agent can see why a conversation ranked where it did.

### `geometry-hybrid__retrieval_feedback`

Record explicit feedback for a retrieval result from `hybrid_search`.

```
geometry-hybrid__retrieval_feedback(
  query_id="q_...",
  branch_id="conv_186",
  used=true
)
```

Used to update retrieval usefulness/error signals per branch.

### `geometry-hybrid__branch_report`

Detailed geometry metrics for one branch.

```
geometry-hybrid__branch_report(branch_id="conv_1")
```

Returns: state, regime, node_count, eff_rank, anisotropy, coherence, trace, anchor_drift, mean_vec (first 10 dims), compression_loss, contradiction_density/topic_drift_density/subtopic_diversity_density, retrieval_error, update/correction summaries.

### `geometry-hybrid__geometry_stats`

Overall geometry DB statistics.

```
geometry-hybrid__geometry_stats()
```

Returns: branch count, state distribution, regime distribution, average metrics.

### `geometry-hybrid__maintenance_cycle`

Run one maintenance cycle manually (supports low-RAM chunking):

```
geometry-hybrid__maintenance_cycle(max_branches=50, reset_chunk_cursor=false)
```

Returns recompute/split/merge counters, retrieval-feedback pruning counters, and chunk telemetry.

Pruning counters:
- `retrieval_feedback_pruned`
- `retrieval_feedback_pruned_age`
- `retrieval_feedback_pruned_cap`

### `geometry-hybrid__geometry_snapshot`

Export compact branch metrics for ops/debugging.

```
geometry-hybrid__geometry_snapshot(state="ACTIVE", limit=20, include_means=false)
```

Supports:
- `branch_ids=["conv_148","day_2026-04-07"]` for explicit selection
- `state="ACTIVE" | "STABLE" | "COLLAPSING" | "ALL"` filter
- `limit` row cap
- `include_means=true` to include full branch mean vectors

### `geometry-hybrid__latest_correction`

Resolve correction lineage and return the latest correction node/version for any seed node id.

```
geometry-hybrid__latest_correction(
  node_id="c5dff34c-61b3-4296-9af5-84c9a73af4e6",
  include_chain=true,
  chain_limit=10
)
```

Returns:
- `root_node_id`, `branch_id`
- latest correction metadata (`latest_node_id`, `latest_lcm_id`, `latest_update_mode`, `latest_correction_kind`, `latest_correction_version`)
- `chain_length`
- optional ordered `chain` payload when `include_chain=true`

### `geometry-hybrid__sync_lcm_ingest`

Force one incremental ingest poll from `lcm.db` into `lcm_geometry.db`.

```
geometry-hybrid__sync_lcm_ingest(limit=200)
```

Returns: processed/failed counters, rowid cursor movement, and `has_more`.

### `geometry-hybrid__sync_lcm_dag_edges`

Force DAG edge rebuild from LCM summary tables into geometry DB and return integrity counters.

```
geometry-hybrid__sync_lcm_dag_edges(backup=true)
```

Returns:
- backup path (or disabled)
- import stats (`summarizes`, `derived_from`, `skipped`, `purged`)
- indexed-node counts
- validation totals + orphan counters by edge type

### `geometry-hybrid__backfill_lcm_conversations`

Targeted LCM→geometry backfill for specific conversation IDs:

```
geometry-hybrid__backfill_lcm_conversations(
  conversation_ids=[57,365],
  max_per_conv=200,
  resume=true,
  dry_run=true
)
```

Returns requested/found/processed/skipped/failed counts plus details.
When run in real mode (`dry_run=false`), output includes:
- `provider_ready`
- `preflight_error` (when missing embeddings)
- `failed_preflight` per-conversation detail status

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
- `content_type="summaries" | "messages" | "both" | "logs"`

Resolution metadata:
- `resolution_mode`: `branch_lineage` | `suffix_fallback` | `daily_log`
- `resolved_conversation_ids`: LCM conversation IDs actually used
- `warning` when relevant:
  - `branch_suffix_mismatch:conv_X->conv_Y`
  - `mixed_branch_content:<ids>`
  - `lineage_empty_used_suffix_fallback`

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

Optional runtime tuning for MCP:

1. Copy:
```bash
cp <module_repo_root>/extensions/geometry-mcp/runtime_config.example.json \
   <openclaw_home>/extensions/geometry-mcp/runtime_config.json
```
2. Edit `runtime_config.json`:
   - top-level `polling` section for real-time ingest
     - `polling.auto_tools` defines which tools trigger auto-poll (default `["hybrid_search"]`)
     - `polling.auto_limit` caps auto-poll ingest size for low-latency tool calls (default `3` on GGUF)
   - top-level `startup` section for warmup behavior
     - `startup.warmup_gc` enables background warmup at server startup
     - `startup.warmup_probe_embed` defaults to disabled for GGUF to avoid startup contention
   - top-level `embedding` section for backend/model runtime (`sentence_transformers`, `llama_cpp`, `http`)
   - `geometry_config` section for controller policy tuning
3. Restart gateway.  
The server also supports env override `GEOMETRY_RUNTIME_CONFIG_JSON` (JSON object).

EmbeddingGemma GGUF cutover assets:

- Runbook: `GEOMETRY_GGUF_MIGRATION_RUNBOOK.md`
- Ready config template: `extensions/geometry-mcp/runtime_config.embeddinggemma_gguf.example.json`
- Use `dim=768` for `embeddinggemma-300m` variants.

---

## 13. Current Enhancements

The current controller build includes these production features:

1. **Incremental ingest API** via `poll_lcm_for_new_items(...)` using rowid cursors.
2. **Parent-aware insertion** in `on_new_item(...)` with `parent_lcm_id` resolution.
3. **Automatic `TEMPORAL_NEXT` edge creation** between consecutive nodes in the same branch.
4. **Topic drift refresh** in maintenance:
   - topic drift / subtopic diversity density recomputation per branch
   - bounded `TOPIC_DRIFT` edge regeneration (legacy `CONTRADICTS` reserved for correction lineage)
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
12. **Scalar/lazy loading paths** for retrieval/candidate selection to avoid full-blob scans.
13. **Soft merge execution** with pending merge queue draining.
14. **Bounded topic drift compute** via temporal stratified sampling on large branches.
15. **Dormancy lifecycle policy** based on real activity age + usefulness.
16. **Split child priors** (usefulness/retrieval_error inheritance + centroid anchor seeding).
17. **Branch-type aware allocation thresholds** with global fallback.
18. **Scalar-first maintenance persistence** to reduce blob I/O.
19. **Connect-time schema migration** for `reactivation_score`.
20. **Protected-memory hard gates** for attach/fork and merge queueing.
21. **Safe reactivation guard** using topic-drift/retrieval-error/similarity checks.
22. **Regime-aware retrieval routing** via `balanced`/`factual`/`exploratory` mode multipliers.
23. **Branch-type metric profiles** for CSD/retrieval/split/merge sensitivity tuning.
24. **Versioned correction flow** with explicit `REFINES`/`CONTRADICTS`/`SUPERSEDES` lineage metadata.
25. **Duplicate-safe polling ingest + lag telemetry**:
    - skips previously-ingested message `lcm_id` values in incremental polling
    - exposes `skipped_duplicates`, `lag_rows`, and cursor/max-rowid status via MCP ingest reporting
26. **Optional duplicate cleanup utility**:
    - `scripts/cleanup_geometry_duplicates.py` for one-time repair of legacy duplicate-ingest inflation
    - creates backup, dedupes message rows, rebuilds affected edges/metadata, and can run one maintenance cycle
27. **Lineage-aware content resolution**:
    - `conversation_content` resolves branch text from actual geometry node lineage instead of branch suffix only
    - exposes mismatch/mixed warnings and resolved conversation IDs
28. **DAG edge sync admin tool**:
    - `sync_lcm_dag_edges` rebuilds imported DAG edges and reports orphan validation counters in one command

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

### Historical duplicate message rows in geometry DB

If a legacy DB was built before duplicate-safe polling, you may see inflated message counts.

Run recovery dry-run:
```bash
python3 scripts/cleanup_geometry_duplicates.py --db <openclaw_home>/lcm_geometry.db
```

Apply (creates backup automatically) + refresh:
```bash
python3 scripts/cleanup_geometry_duplicates.py \
  --db <openclaw_home>/lcm_geometry.db \
  --apply \
  --run-maintenance
```

### `conversation_content` warns about branch suffix mismatch

If you see:
```
branch_suffix_mismatch:conv_X->conv_Y
```
that means branch content lineage points to a different LCM conversation than the branch suffix.
This is expected for split-derived branches and is now handled correctly by lineage resolution.

### Imported DAG edges need consistency refresh

Run MCP tool:
```
geometry-hybrid__sync_lcm_dag_edges(backup=true)
```

Use returned orphan counters to verify integrity (`derived_from=0`, `summarizes=0` expected).

### Maintenance cycle not recomputing all branches

If `recomputed` count is low despite Fix 5 being applied, check:
```python
gc = GeometryController('<openclaw_home>/lcm_geometry.db')
for s in gc.db.all_branches():
    if not s.mean_vec:
        print(f"{s.branch_id}: NO mean_vec â€” will use fallback")
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

*Manual generated: 2026-04-09*
*Module: `<module_repo_root>/lcm_geometry_controller.py`*
*Companion backfill script: `<module_repo_root>/lcm_geometry_backfill.py`*
*MCP server: `<openclaw_home>/extensions/geometry-mcp/server.py`*
