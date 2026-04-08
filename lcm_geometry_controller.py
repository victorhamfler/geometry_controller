"""
LCM Geometry Controller (LCM-GC)
=================================
A geometry-aware memory regulation layer that sits on top of LCM's
immutable-store + summary-DAG architecture.

Implements:
  - CSD (Constraint Sensitivity Diagnostic) residual scoring
  - Continuous-Learning Geometry (CLG) branch management
  - Allocation vs Stabilization separation
  - Adiabatic update law
  - Branch lifecycle state machine
  - Retrieval trust scoring
  - Summary health auditing
  - Background maintenance jobs

Design contract:
  - NEVER modifies the LCM immutable store
  - NEVER rewrites summary-node provenance
  - All compaction decisions remain with LCM
  - This module only annotates, scores, and advises
"""

from __future__ import annotations

import json
import math
import os
import sqlite3
import time
import threading
import traceback
import uuid
from dataclasses import dataclass, field, asdict
from enum import Enum
from typing import Any, Optional
import numpy as np



# ---------------------------------------------------------------------------
# Enums
# ---------------------------------------------------------------------------


class BranchState(str, Enum):
    FORMING         = "FORMING"
    ACTIVE          = "ACTIVE"
    STABLE          = "STABLE"
    TENSIONED       = "TENSIONED"
    DORMANT         = "DORMANT"
    REACTIVATING    = "REACTIVATING"
    SPLIT_PENDING   = "SPLIT_PENDING"
    MERGE_CANDIDATE = "MERGE_CANDIDATE"
    COLLAPSING      = "COLLAPSING"


class GeometricRegime(str, Enum):
    RIGID      = "RIGID"       # over-consolidated, too narrow
    PRODUCTIVE = "PRODUCTIVE"  # healthy, bounded novelty
    UNSTABLE   = "UNSTABLE"   # mixing incompatible content


class NodeType(str, Enum):
    MESSAGE           = "message"
    TOOL_RESULT       = "tool_result"
    LEAF_SUMMARY      = "leaf_summary"
    CONDENSED_SUMMARY = "condensed_summary"
    LARGE_FILE_REF    = "large_file_ref"
    EXPLORATION_SUMMARY = "exploration_summary"
    DAILY_LOG_ENTRY   = "daily_log_entry"


class EdgeType(str, Enum):
    SUMMARIZES       = "summarizes"
    DERIVED_FROM     = "derived_from"
    TEMPORAL_NEXT    = "temporal_next"
    SEMANTIC_NEIGHBOR = "semantic_neighbor"
    CONTRADICTS      = "contradicts"
    REFINES          = "refines"
    SUPERSEDES       = "supersedes"
    SAME_TOPIC       = "same_topic"
    SAME_TASK        = "same_task"
    SAME_USER_FACT   = "same_user_fact"
    CROSS_AGENT      = "cross_agent_shared"


# ---------------------------------------------------------------------------
# Data classes
# ---------------------------------------------------------------------------
@dataclass
class GeometryConfig:
    """All tunable hyperparameters. Passed to GeometryController and sub-components."""

    # ── Embedding ─────────────────────────────────────────────────────────
    embedding_dim: int = 384

    # ── Branch geometry ────────────────────────────────────────────────────
    min_branch_size: int = 8

    # ── EMA smoothing (adiabatic update) ──────────────────────────────────
    stat_rho: float = 0.05
    anchor_alpha: float = 0.01

    # ── CSD scoring weights ─────────────────────────────────────────────────
    importance_weight: float = 0.40
    novelty_weight:    float = 0.30
    conflict_weight:    float = 0.20
    coherence_weight:   float = 0.10

    # ── Allocation thresholds ───────────────────────────────────────────────
    attach_threshold: float = 0.50
    attach_threshold_by_type: dict = field(default_factory=lambda: {"default": 0.50})
    tension_threshold_by_type: dict = field(default_factory=lambda: {"default": 0.70})

    # ── Retrieval ranker ───────────────────────────────────────────────────
    alpha_sem:   float = 0.60
    beta_trust:  float = 0.25
    delta_react: float = 0.15
    kappa_coherence:     float = 0.30
    kappa_hist:          float = 0.20
    kappa_comp_loss:     float = 0.20
    kappa_contradiction: float = 0.25
    kappa_ret_error:     float = 0.25
    retrieval_prefilter_limit: int = 256
    retrieval_result_limit: int = 128
    candidate_branch_cap: int = 10
    candidate_prefilter_limit: int = 128
    retrieval_mode_default: str = "balanced"
    retrieval_mode_factors: dict = field(default_factory=lambda: {
        "balanced": {
            "regime": {"PRODUCTIVE": 1.05, "RIGID": 0.95, "UNSTABLE": 0.85},
            "state": {"REACTIVATING": 1.03, "DORMANT": 0.92, "TENSIONED": 0.90, "FORMING": 0.90},
        },
        "factual": {
            "regime": {"PRODUCTIVE": 1.10, "RIGID": 0.90, "UNSTABLE": 0.60},
            "state": {"STABLE": 1.05, "ACTIVE": 1.00, "REACTIVATING": 0.95, "DORMANT": 0.85, "TENSIONED": 0.75, "FORMING": 0.80},
        },
        "exploratory": {
            "regime": {"PRODUCTIVE": 1.00, "RIGID": 0.92, "UNSTABLE": 0.95},
            "state": {"REACTIVATING": 1.08, "ACTIVE": 1.02, "DORMANT": 0.96, "TENSIONED": 0.95, "STABLE": 0.98},
        },
    })
    branch_type_profiles: dict = field(default_factory=dict)

    # ── Split scorer ───────────────────────────────────────────────────────
    split_hysteresis: int = 3
    split_min_nodes: int = 6
    split_kmeans_max_iter: int = 20
    split_score_threshold: float = 0.075
    max_split_enqueues_per_cycle: int = 5
    split_observability_keep: int = 20000
    split_child_copy_usefulness: bool = True
    split_child_copy_retrieval_error: bool = True
    split_child_anchor_from_centroid: bool = True
    tension_threshold: float = 0.70  # CSD above this → TENSIONED state
    beta1_comp: float = 0.50
    beta2_comp: float = 0.30
    zeta_anisotropy:  float = 0.20
    zeta_comp_loss:   float = 0.20
    zeta_contradiction: float = 0.25
    zeta_incoherence: float = 0.15
    zeta_ret_error:   float = 0.20
    eta_topic:        float = 0.20
    eta_co_use:       float = 0.15
    eta_contradiction: float = 0.20
    eta_cos:          float = 0.15
    eta_drift:        float = 0.15
    eta_mu:           float = 0.10
    eta_r:            float = 0.05
    eta_tau:          float = 0.00

    # ── Merge scorer ───────────────────────────────────────────────────────
    merge_hysteresis: int = 3
    merge_max_jobs_per_cycle: int = 5
    merge_execution_mode: str = "soft"  # soft | off
    merge_soft_edge_weight: float = 1.0

    # ── Gamma weights ──────────────────────────────────────────────────────
    gamma_sem:      float = 0.15
    gamma_mu:       float = 0.15
    gamma_r:        float = 0.10
    gamma_A:        float = 0.15
    gamma_tau:      float = 0.10
    gamma_conflict: float = 0.20
    omega_sem:           float = 0.05
    omega_contradiction: float = 0.05
    omega_comp_loss:     float = 0.05
    omega_re_expand:     float = 0.00

    # ── Regime classifier ──────────────────────────────────────────────────
    rank_target: dict = field(default_factory=lambda: {
        "CONVERSATION": (50.0, 300.0),
        "default":       (20.0, 200.0),
    })
    rigid_rank_ratio: float = 0.15
    unstable_coh_floor: float = 0.45
    unstable_comp_ceil:  float = 0.65

    # ── Usefulness tracker ─────────────────────────────────────────────────
    usefulness_lambda: float = 0.10
    contradiction_sim_threshold: float = -0.30
    contradiction_edge_max_pairs: int = 256
    contradiction_sample_min_nodes: int = 64
    contradiction_sample_max_nodes: int = 192  # 0 disables sampling cap
    merge_signal_lookback: int = 5000
    dormant_after_days: float = 14.0
    dormant_usefulness_max: float = 0.20
    dormant_min_nodes: int = 8

    # ── Protected memory policy (hard gates) ───────────────────────────────
    protected_branch_types: list[str] = field(default_factory=lambda: [
        "identity",
        "user_fact",
        "user_preference",
        "preference",
        "daily_log",
    ])
    protected_attach_conflict_threshold: float = 0.35
    protected_attach_contradiction_threshold: float = 0.20
    protected_merge_block: bool = True
    protected_merge_contradiction_threshold: float = 0.20

    # ── Safe reactivation policy ────────────────────────────────────────────
    reactivation_min_score: float = 0.60
    reactivation_guard_enabled: bool = True
    reactivation_max_contradiction: float = 0.35
    reactivation_max_retrieval_error: float = 0.60
    reactivation_min_similarity: float = 0.15

    # ── Update-mode classification (metadata) ──────────────────────────────
    update_mode_refine_similarity_min: float = 0.92
    update_mode_contradict_conflict_min: float = 0.25
    update_mode_supersede_similarity_min: float = 0.78
    update_mode_supersede_conflict_min: float = 0.70
    update_mode_supersede_branch_types: list[str] = field(default_factory=lambda: [
        "identity",
        "user_fact",
        "user_preference",
        "preference",
    ])

    # ── Initial state ──────────────────────────────────────────────────────
    initial_state: BranchState = BranchState.FORMING



@dataclass
class BranchStats:
    """Geometric state of one branch B.

    All vector fields stored as flat list[float] for JSON/SQLite serialisation.
    """
    branch_id:           str
    branch_type:         str           = "default"
    state:               BranchState  = BranchState.FORMING
    regime:              GeometricRegime = GeometricRegime.PRODUCTIVE

    # Geometry (serialisable forms)
    mean_vec:            list[float]  = field(default_factory=list)
    anchor:              list[float]  = field(default_factory=list)
    cov_diagonal:        list[float]  = field(default_factory=list)  # store diag only

    # CSD observables
    eff_rank:            float = 0.0
    trace:               float = 0.0
    anisotropy:          float = 0.0
    anchor_drift:        float = 0.0
    coherence:           float = 0.0
    compression_loss:    float = 0.0

    # Counts & feedback
    node_count:          int   = 0
    contradiction_density: float = 0.0
    retrieval_error:     float = 0.0
    usefulness:          float = 0.0
    reactivation_score:  float = 0.0

    # Hysteresis counters
    split_counter:       int = 0
    merge_counter:       int = 0

    last_update_ts:      float = field(default_factory=time.time)


@dataclass
class MemoryNode:
    """Mirrors an LCM node with geometry annotations."""
    node_id:             str
    lcm_id:              str           # foreign key into LCM store
    node_type:           NodeType
    branch_id:           str
    parent_id:           Optional[str]
    timestamp:           float
    role:                str
    token_count:         int
    embedding:           list[float]

    importance_score:    float = 0.5
    novelty_score:       float = 0.0
    conflict_score:      float = 0.0
    coherence_score:     float = 0.0
    compression_loss:    float = 0.0
    reactivation_score:  float = 0.0
    stability_state:     str   = BranchState.ACTIVE
    update_mode:         str   = "attach"
    correction_kind:     str   = "none"
    correction_prev_id:  Optional[str] = None
    correction_root_id:  Optional[str] = None
    correction_version:  int   = 1


@dataclass
class AllocationDecision:
    action:         str          # "attach" | "attach_tension" | "fork"
    branch_id:      str
    csd_score:      float
    conflict_score: float
    rationale:      str
    update_mode:    str = "attach"


@dataclass
class RetrievalCandidate:
    branch_id:   str
    sem_score:   float
    trust_score: float
    react_score: float
    total_score: float


def _branch_type_profile(cfg: GeometryConfig, branch_type: str) -> dict[str, Any]:
    raw = getattr(cfg, "branch_type_profiles", None)
    if not isinstance(raw, dict) or not raw:
        return {}
    bt = str(branch_type or "").strip().lower()
    if bt and bt in raw and isinstance(raw.get(bt), dict):
        return dict(raw[bt])
    for k, v in raw.items():
        if str(k).strip().lower() == bt and isinstance(v, dict):
            return dict(v)
    default = raw.get("default")
    if isinstance(default, dict):
        return dict(default)
    for k, v in raw.items():
        if str(k).strip().lower() == "default" and isinstance(v, dict):
            return dict(v)
    return {}


def _profile_weight(
    profile: dict[str, Any],
    group: str,
    key: str,
    fallback: float,
    aliases: Optional[list[str]] = None,
) -> float:
    names = [str(key)]
    if aliases:
        names.extend(str(x) for x in aliases)

    g = profile.get(group)
    if isinstance(g, dict):
        for n in names:
            if n in g:
                try:
                    return float(g[n])
                except Exception:
                    pass
        for k, v in g.items():
            sk = str(k).strip().lower()
            for n in names:
                if sk == str(n).strip().lower():
                    try:
                        return float(v)
                    except Exception:
                        pass

    for n in names:
        if n in profile:
            try:
                return float(profile[n])
            except Exception:
                pass
    for k, v in profile.items():
        sk = str(k).strip().lower()
        for n in names:
            if sk == str(n).strip().lower():
                try:
                    return float(v)
                except Exception:
                    pass

    return float(fallback)


# ---------------------------------------------------------------------------
# Schema bootstrap
# ---------------------------------------------------------------------------

SCHEMA_SQL = """
CREATE TABLE IF NOT EXISTS memory_nodes (
    id                  TEXT PRIMARY KEY,
    lcm_id              TEXT NOT NULL,
    node_type           TEXT NOT NULL,
    parent_id           TEXT,
    branch_id           TEXT NOT NULL,
    timestamp           REAL NOT NULL,
    role                TEXT,
    token_count         INTEGER DEFAULT 0,
    embedding           BLOB,
    importance_score    REAL DEFAULT 0.5,
    novelty_score       REAL DEFAULT 0.0,
    conflict_score      REAL DEFAULT 0.0,
    coherence_score     REAL DEFAULT 0.0,
    compression_loss    REAL DEFAULT 0.0,
    reactivation_score  REAL DEFAULT 0.0,
    stability_state     TEXT DEFAULT 'ACTIVE',
    update_mode         TEXT DEFAULT 'attach',
    correction_kind     TEXT DEFAULT 'none',
    correction_prev_id  TEXT,
    correction_root_id  TEXT,
    correction_version  INTEGER DEFAULT 1,
    FOREIGN KEY(parent_id) REFERENCES memory_nodes(id)
);

CREATE TABLE IF NOT EXISTS memory_edges (
    src_id     TEXT NOT NULL,
    dst_id     TEXT NOT NULL,
    edge_type  TEXT NOT NULL,
    weight     REAL DEFAULT 1.0,
    PRIMARY KEY (src_id, dst_id, edge_type)
);

CREATE TABLE IF NOT EXISTS branch_states (
    branch_id            TEXT PRIMARY KEY,
    branch_type          TEXT DEFAULT 'default',
    state                TEXT NOT NULL DEFAULT 'FORMING',
    regime               TEXT NOT NULL DEFAULT 'PRODUCTIVE',
    mean_vec             BLOB,
    anchor               BLOB,
    cov_diagonal         BLOB,
    eff_rank             REAL DEFAULT 0.0,
    trace                REAL DEFAULT 0.0,
    anisotropy           REAL DEFAULT 0.0,
    anchor_drift         REAL DEFAULT 0.0,
    coherence            REAL DEFAULT 0.0,
    compression_loss     REAL DEFAULT 0.0,
    node_count           INTEGER DEFAULT 0,
    contradiction_density REAL DEFAULT 0.0,
    retrieval_error      REAL DEFAULT 0.0,
    usefulness           REAL DEFAULT 0.0,
    reactivation_score   REAL DEFAULT 0.0,
    split_counter        INTEGER DEFAULT 0,
    merge_counter        INTEGER DEFAULT 0,
    last_update_ts       REAL
);

CREATE TABLE IF NOT EXISTS retrieval_feedback (
    id          TEXT PRIMARY KEY,
    query_id    TEXT NOT NULL,
    branch_id   TEXT NOT NULL,
    score       REAL,
    used        INTEGER DEFAULT 0,
    corrected   INTEGER DEFAULT 0,
    expanded    INTEGER DEFAULT 0,
    timestamp   REAL
);

CREATE TABLE IF NOT EXISTS maintenance_jobs (
    id            TEXT PRIMARY KEY,
    job_type      TEXT NOT NULL,
    target_id     TEXT,
    status        TEXT NOT NULL DEFAULT 'pending',
    payload_json  TEXT,
    created_ts    REAL,
    completed_ts  REAL
);

CREATE TABLE IF NOT EXISTS maintenance_split_observations (
    id                    TEXT PRIMARY KEY,
    run_id                TEXT NOT NULL,
    branch_id             TEXT NOT NULL,
    state_before          TEXT,
    state_after           TEXT,
    regime                TEXT,
    node_count            INTEGER,
    split_score           REAL,
    split_threshold       REAL,
    split_counter_before  INTEGER,
    split_counter_after   INTEGER,
    split_hysteresis      INTEGER,
    split_min_nodes       INTEGER,
    gate_nodes            INTEGER,
    gate_score            INTEGER,
    gate_hysteresis       INTEGER,
    should_split          INTEGER,
    enqueued              INTEGER,
    reason                TEXT,
    created_ts            REAL NOT NULL
);

CREATE TABLE IF NOT EXISTS daily_log_content (
    node_id      TEXT PRIMARY KEY,
    text         TEXT NOT NULL,
    source       TEXT,
    created_ts   REAL NOT NULL
);

CREATE TABLE IF NOT EXISTS csd_ema_state (
    branch_id     TEXT NOT NULL,
    ema_key       TEXT NOT NULL,
    value         REAL NOT NULL,
    updated_ts    REAL NOT NULL,
    PRIMARY KEY (branch_id, ema_key)
);

CREATE INDEX IF NOT EXISTS idx_nodes_branch ON memory_nodes(branch_id);
CREATE INDEX IF NOT EXISTS idx_nodes_lcm    ON memory_nodes(lcm_id);
CREATE INDEX IF NOT EXISTS idx_edges_src    ON memory_edges(src_id);
CREATE INDEX IF NOT EXISTS idx_edges_dst    ON memory_edges(dst_id);
CREATE INDEX IF NOT EXISTS idx_edges_type   ON memory_edges(edge_type);
CREATE INDEX IF NOT EXISTS idx_feedback_branch ON retrieval_feedback(branch_id);
CREATE INDEX IF NOT EXISTS idx_csd_ema_branch ON csd_ema_state(branch_id);
CREATE INDEX IF NOT EXISTS idx_split_obs_run ON maintenance_split_observations(run_id);
CREATE INDEX IF NOT EXISTS idx_split_obs_branch ON maintenance_split_observations(branch_id);
CREATE INDEX IF NOT EXISTS idx_split_obs_ts ON maintenance_split_observations(created_ts);
CREATE INDEX IF NOT EXISTS idx_daily_log_created_ts ON daily_log_content(created_ts);
"""


# ---------------------------------------------------------------------------
# Geometry primitives
# ---------------------------------------------------------------------------
# ---------------------------------------------------------------------------
# Embedding Provider — pluggable backend for text → vector
# ---------------------------------------------------------------------------

class EmbeddingProvider:
    """
    Pluggable embedding backend. Supports sentence-transformers (local)
    or any OpenAI-compatible API.

    Thread-safe model loading via double-checked locking.
    In-memory LRU cache avoids re-encoding repeated texts.
    """

    def __init__(
        self,
        model_name: str = "all-MiniLM-L6-v2",
        device: str = "cpu",
        cache: Optional[dict] = None,
    ):
        self.model_name = model_name
        self.device = device
        self._model = None
        self._lock = threading.Lock()
        self._cache: dict[str, list[float]] = cache or {}

    # ---- public API ----

    def embed(self, text: str) -> list[float]:
        """Encode a single text. Result is cached keyed by text[:200]."""
        key = text[:200]
        if key in self._cache:
            return self._cache[key]
        self._load()
        # Truncate to 1000 chars to avoid embedding of very long texts
        vec = self._model.encode(
            [text[:1000]],
            normalize_embeddings=True,
            show_progress_bar=False,
            convert_to_numpy=True,
        )[0]
        result = vec.tolist()
        self._cache[key] = result
        return result

    def embed_batch(self, texts: list[str], batch_size: int = 64) -> list[list[float]]:
        """Encode a batch of texts. No caching (caller manages)."""
        if not texts:
            return []
        self._load()
        truncated = [t[:1000] for t in texts]
        vecs = self._model.encode(
            truncated,
            normalize_embeddings=True,
            show_progress_bar=False,
            batch_size=batch_size,
            convert_to_numpy=True,
        )
        return [v.tolist() for v in vecs]

    @property
    def embedding_dim(self) -> int:
        """Lazily return model embedding dimension."""
        self._load()
        return self._model.get_sentence_embedding_dimension()

    # ---- internal ----

    def _load(self):
        """Thread-safe lazy model loading (double-checked locking)."""
        if self._model is not None:
            return
        with self._lock:
            if self._model is None:
                from sentence_transformers import SentenceTransformer
                self._model = SentenceTransformer(self.model_name, device=self.device)




class GeometryMath:
    """Pure-function geometry operations. No DB, no side-effects."""

    EPS = 1e-9

    @staticmethod
    def effective_rank(eigenvalues: np.ndarray) -> float:
        """Entropy-based effective rank (Roy & Vetterli, 2007)."""
        lam = np.maximum(eigenvalues, 0.0)
        total = lam.sum() + GeometryMath.EPS
        p = lam / total
        p = p[p > GeometryMath.EPS]
        return float(np.exp(-np.sum(p * np.log(p + GeometryMath.EPS))))

    @staticmethod
    def anisotropy(eigenvalues: np.ndarray) -> float:
        total = eigenvalues.sum() + GeometryMath.EPS
        return float(eigenvalues.max() / total)

    @staticmethod
    def covariance_diagonal(embeddings: np.ndarray, weights: np.ndarray) -> np.ndarray:
        """Weighted diagonal covariance (memory-efficient: avoids d×d matrix)."""
        W = weights.sum() + GeometryMath.EPS
        mu = (weights[:, None] * embeddings).sum(axis=0) / W
        diff = embeddings - mu
        var = (weights[:, None] * diff**2).sum(axis=0) / W
        return var  # shape (d,)

    @staticmethod
    def weighted_mean(embeddings: np.ndarray, weights: np.ndarray) -> np.ndarray:
        W = weights.sum() + GeometryMath.EPS
        return (weights[:, None] * embeddings).sum(axis=0) / W

    @staticmethod
    def cosine_sim(a: np.ndarray, b: np.ndarray) -> float:
        na = np.linalg.norm(a) + GeometryMath.EPS
        nb = np.linalg.norm(b) + GeometryMath.EPS
        return float(np.dot(a, b) / (na * nb))

    @staticmethod
    def branch_coherence(embeddings: np.ndarray, weights: np.ndarray,
                         mean: np.ndarray) -> float:
        W = weights.sum() + GeometryMath.EPS
        sims = np.array([GeometryMath.cosine_sim(e, mean) for e in embeddings])
        return float((weights * sims).sum() / W)

    @staticmethod
    def compression_loss(
        summary_emb: np.ndarray,
        desc_embs: np.ndarray,
        desc_weights: np.ndarray,
        beta1: float = 0.6,
        beta2: float = 0.4,
    ) -> float:
        """Semantic + rank-mismatch compression loss."""
        mu_desc = GeometryMath.weighted_mean(desc_embs, desc_weights)
        sem_loss = 1.0 - GeometryMath.cosine_sim(summary_emb, mu_desc)

        # Rank mismatch (proxy: use scalar std-dev ratio)
        var_desc = ((desc_embs - mu_desc) ** 2).mean()
        var_summ = 0.0  # summary is one point → rank-1 proxy
        rank_mismatch = abs(
            math.log(max(var_desc, GeometryMath.EPS))
            - math.log(max(var_summ, GeometryMath.EPS) + GeometryMath.EPS)
        )
        # clamp to [0,1]
        rank_mismatch = min(rank_mismatch / 10.0, 1.0)

        return float(beta1 * sem_loss + beta2 * rank_mismatch)

    @staticmethod
    def compute_full_stats(
        embeddings: np.ndarray,
        weights: np.ndarray,
        mean: np.ndarray,
    ) -> dict[str, float]:
        """Return eff_rank, trace, anisotropy, coherence from diagonal cov."""
        var = GeometryMath.covariance_diagonal(embeddings, weights)
        eff_rank = GeometryMath.effective_rank(var)
        trace    = float(var.sum())
        aniso    = GeometryMath.anisotropy(var)
        coh      = GeometryMath.branch_coherence(embeddings, weights, mean)
        return {
            "eff_rank":   eff_rank,
            "trace":      trace,
            "anisotropy": aniso,
            "coherence":  coh,
        }


# ---------------------------------------------------------------------------
# CSD Scorer
# ---------------------------------------------------------------------------

class CSDScorer:
    """Computes the memory-CSD score for inserting item x into branch B.

    CSD(x→B) = γ1·Rμ + γ2·Rr + γ3·RA + γ4·Rτ + γ5·(1-cos(ex,μB)) + γ6·conflict
    """

    def __init__(self, cfg: GeometryConfig, db: Optional["GeometryDB"] = None):
        self.cfg = cfg
        self.db = db
        # Per-branch EMA cache; persisted via GeometryDB when available.
        self._history: dict[str, dict[str, float]] = {}

    def _get_ema(self, branch_id: str, key: str) -> float:
        cached = self._history.get(branch_id, {}).get(key)
        if cached is not None:
            return cached
        if self.db is not None:
            stored = self.db.get_csd_ema(branch_id, key)
            if stored is not None:
                self._history.setdefault(branch_id, {})[key] = stored
                return stored
        return 0.0

    def _update_ema(self, branch_id: str, key: str, value: float) -> None:
        prev = self._get_ema(branch_id, key)
        if prev <= 0.0:
            updated = float(value)
        else:
            updated = (1 - self.cfg.stat_rho) * prev + self.cfg.stat_rho * value
        self._history.setdefault(branch_id, {})[key] = updated
        if self.db is not None:
            self.db.set_csd_ema(branch_id, key, float(updated))

    def _residual(self, delta: float, branch_id: str, key: str) -> float:
        expected = max(self._get_ema(branch_id, key) + GeometryMath.EPS, GeometryMath.EPS)
        return delta / expected

    # Public surface
    def score(
        self,
        item_emb: np.ndarray,
        stats: BranchStats,
        conflict: float = 0.0,
    ) -> float:
        """Return CSD score ∈ [0, ∞). Higher = more incompatible."""
        if not stats.mean_vec:
            return 0.0   # FORMING branch: always attach

        mu_B   = np.array(stats.mean_vec,   dtype=np.float32)
        var_B  = np.array(stats.cov_diagonal, dtype=np.float32) if stats.cov_diagonal else np.zeros(len(mu_B))
        bid    = stats.branch_id

        # Hypothetical new mean after insertion (approx, weight=1)
        n      = max(stats.node_count, 1)
        mu_new = (n * mu_B + item_emb) / (n + 1)

        # Δμ
        delta_mu  = float(np.linalg.norm(mu_new - mu_B))
        R_mu      = self._residual(delta_mu, bid, "delta_mu")

        # Δr (effective rank proxy from diagonal variance)
        r_old     = GeometryMath.effective_rank(var_B)
        # hypothetical new variance row
        diff_new  = (item_emb - mu_new) ** 2
        var_new   = (n * var_B + diff_new) / (n + 1)
        r_new     = GeometryMath.effective_rank(var_new)
        delta_r   = abs(r_new - r_old)
        R_r       = self._residual(delta_r, bid, "delta_r")

        # ΔA (anisotropy)
        A_old     = GeometryMath.anisotropy(var_B)
        A_new     = GeometryMath.anisotropy(var_new)
        delta_A   = abs(A_new - A_old)
        R_A       = self._residual(delta_A, bid, "delta_A")

        # Δτ (trace)
        tau_old   = float(var_B.sum())
        tau_new   = float(var_new.sum())
        delta_tau = abs(tau_new - tau_old)
        R_tau     = self._residual(delta_tau, bid, "delta_tau")

        # Semantic distance
        sem_dist  = 1.0 - GeometryMath.cosine_sim(item_emb, mu_B)

        c = self.cfg
        p = _branch_type_profile(c, stats.branch_type)
        w_gamma_mu = _profile_weight(p, "csd_gamma", "mu", c.gamma_mu, aliases=["gamma_mu"])
        w_gamma_r = _profile_weight(p, "csd_gamma", "r", c.gamma_r, aliases=["gamma_r"])
        w_gamma_A = _profile_weight(p, "csd_gamma", "A", c.gamma_A, aliases=["a", "anisotropy", "gamma_A"])
        w_gamma_tau = _profile_weight(p, "csd_gamma", "tau", c.gamma_tau, aliases=["trace", "gamma_tau"])
        w_gamma_sem = _profile_weight(p, "csd_gamma", "sem", c.gamma_sem, aliases=["semantic", "gamma_sem"])
        w_gamma_conflict = _profile_weight(
            p,
            "csd_gamma",
            "conflict",
            c.gamma_conflict,
            aliases=["gamma_conflict"],
        )
        csd = (
            w_gamma_mu       * R_mu
            + w_gamma_r      * R_r
            + w_gamma_A      * R_A
            + w_gamma_tau    * R_tau
            + w_gamma_sem    * sem_dist
            + w_gamma_conflict * conflict
        )

        # Update EMA baselines
        self._update_ema(bid, "delta_mu",  delta_mu)
        self._update_ema(bid, "delta_r",   delta_r)
        self._update_ema(bid, "delta_A",   delta_A)
        self._update_ema(bid, "delta_tau", delta_tau)

        return float(csd)

    # Convenience: attach-or-fork decision
    def decide(
        self,
        csd_score: float,
        cfg: Optional[GeometryConfig] = None,
        branch_type: Optional[str] = None,
    ) -> str:
        """'attach' | 'attach_tension' | 'fork'"""
        c = cfg or self.cfg
        bt = str(branch_type or "").strip().lower()

        def _resolve(mapping: Any, fallback: float) -> float:
            try:
                if isinstance(mapping, dict) and mapping:
                    if bt and bt in mapping:
                        return float(mapping[bt])
                    for k, v in mapping.items():
                        if str(k).lower() == bt:
                            return float(v)
                    if "default" in mapping:
                        return float(mapping["default"])
                    for k, v in mapping.items():
                        if str(k).lower() == "default":
                            return float(v)
            except Exception:
                pass
            return float(fallback)

        attach_th = _resolve(getattr(c, "attach_threshold_by_type", None), c.attach_threshold)
        tension_th = _resolve(getattr(c, "tension_threshold_by_type", None), c.tension_threshold)
        tension_th = max(float(tension_th), float(attach_th))

        if csd_score < attach_th:
            return "attach"
        if csd_score < tension_th:
            return "attach_tension"
        return "fork"


# ---------------------------------------------------------------------------
# Merge / Split scorers
# ---------------------------------------------------------------------------

class MergeScorer:
    def __init__(self, cfg: GeometryConfig):
        self.cfg = cfg

    def score(
        self,
        s1: BranchStats,
        s2: BranchStats,
        topic_overlap: float = 0.0,
        retrieval_co_use: float = 0.0,
    ) -> float:
        if not s1.mean_vec or not s2.mean_vec:
            return 0.0
        mu1 = np.array(s1.mean_vec, dtype=np.float32)
        mu2 = np.array(s2.mean_vec, dtype=np.float32)
        c   = self.cfg
        p1 = _branch_type_profile(c, s1.branch_type)
        p2 = _branch_type_profile(c, s2.branch_type)

        def _avg(group: str, key: str, fallback: float, aliases: Optional[list[str]] = None) -> float:
            w1 = _profile_weight(p1, group, key, fallback, aliases=aliases)
            w2 = _profile_weight(p2, group, key, fallback, aliases=aliases)
            return 0.5 * (float(w1) + float(w2))

        w_eta_cos = _avg("merge_eta", "cos", c.eta_cos, aliases=["eta_cos"])
        w_eta_topic = _avg("merge_eta", "topic", c.eta_topic, aliases=["eta_topic"])
        w_eta_co_use = _avg("merge_eta", "co_use", c.eta_co_use, aliases=["co_use", "eta_co_use"])
        w_eta_contradiction = _avg(
            "merge_eta",
            "contradiction",
            c.eta_contradiction,
            aliases=["eta_contradiction"],
        )
        w_eta_drift = _avg("merge_eta", "drift", c.eta_drift, aliases=["eta_drift"])

        cos = GeometryMath.cosine_sim(mu1, mu2)
        drift_mismatch = abs(s1.anchor_drift - s2.anchor_drift)
        score = (
            w_eta_cos          * cos
            + w_eta_topic      * topic_overlap
            + w_eta_co_use     * retrieval_co_use
            + w_eta_contradiction * ((s1.contradiction_density + s2.contradiction_density) / 2)
            + w_eta_drift      * drift_mismatch
        )
        return float(score)

    def should_merge(self, s1: BranchStats, s2: BranchStats, score: float) -> bool:
        c = self.cfg
        p1 = _branch_type_profile(c, s1.branch_type)
        p2 = _branch_type_profile(c, s2.branch_type)
        th1 = _profile_weight(p1, "merge_policy", "threshold", 0.55, aliases=["merge_threshold"])
        th2 = _profile_weight(p2, "merge_policy", "threshold", 0.55, aliases=["merge_threshold"])
        merge_threshold = 0.5 * (float(th1) + float(th2))
        states_ok = s1.state in (BranchState.STABLE, BranchState.DORMANT)
        states_ok = states_ok and s2.state in (BranchState.STABLE, BranchState.DORMANT)
        return states_ok and score > merge_threshold


class SplitScorer:
    def __init__(self, cfg: GeometryConfig):
        self.cfg = cfg

    def score(self, s: BranchStats) -> float:
        c = self.cfg
        p = _branch_type_profile(c, s.branch_type)
        w_incoh = _profile_weight(
            p,
            "split_zeta",
            "incoherence",
            c.zeta_incoherence,
            aliases=["zeta_incoherence"],
        )
        w_aniso = _profile_weight(
            p,
            "split_zeta",
            "anisotropy",
            c.zeta_anisotropy,
            aliases=["zeta_anisotropy"],
        )
        w_comp = _profile_weight(
            p,
            "split_zeta",
            "comp_loss",
            c.zeta_comp_loss,
            aliases=["compression_loss", "zeta_comp_loss"],
        )
        w_ret = _profile_weight(
            p,
            "split_zeta",
            "ret_error",
            c.zeta_ret_error,
            aliases=["retrieval_error", "zeta_ret_error"],
        )
        w_contra = _profile_weight(
            p,
            "split_zeta",
            "contradiction",
            c.zeta_contradiction,
            aliases=["zeta_contradiction"],
        )
        return float(
            w_incoh   * (1.0 - s.coherence)
            + w_aniso  * s.anisotropy
            + w_comp   * s.compression_loss
            + w_ret   * s.retrieval_error
            + w_contra * s.contradiction_density
        )

    def should_split(self, s: BranchStats, score: float, cfg: GeometryConfig) -> bool:
        p = _branch_type_profile(cfg, s.branch_type)
        split_threshold = _profile_weight(
            p,
            "split_policy",
            "threshold",
            cfg.split_score_threshold,
            aliases=["split_threshold", "split_score_threshold"],
        )
        split_hysteresis = int(round(_profile_weight(
            p,
            "split_policy",
            "hysteresis",
            float(cfg.split_hysteresis),
            aliases=["split_hysteresis"],
        )))
        split_hysteresis = max(1, split_hysteresis)
        return (
            score > split_threshold
            and s.split_counter >= split_hysteresis
        )


# ---------------------------------------------------------------------------
# Retrieval Ranker
# ---------------------------------------------------------------------------

class RetrievalRanker:
    def __init__(self, cfg: GeometryConfig):
        self.cfg = cfg

    def _get_mode_profile(self, mode: str) -> dict[str, Any]:
        raw = getattr(self.cfg, "retrieval_mode_factors", None)
        if not isinstance(raw, dict):
            raw = {}
        m = str(mode or "").strip().lower()
        profile = raw.get(m)
        if not isinstance(profile, dict):
            profile = raw.get("balanced", {})
            if not isinstance(profile, dict):
                profile = {}
        return profile

    def _mode_multiplier(self, mode: str, s: BranchStats) -> float:
        profile = self._get_mode_profile(mode)
        regime_map = profile.get("regime") if isinstance(profile.get("regime"), dict) else {}
        state_map = profile.get("state") if isinstance(profile.get("state"), dict) else {}
        r_key = s.regime.value if isinstance(s.regime, GeometricRegime) else str(s.regime or "")
        s_key = s.state.value if isinstance(s.state, BranchState) else str(s.state or "")
        r_mul = float(regime_map.get(str(r_key), 1.0))
        s_mul = float(state_map.get(str(s_key), 1.0))
        return max(0.25, min(1.50, r_mul * s_mul))

    def rank(
        self,
        query_emb: np.ndarray,
        candidates: list[BranchStats],
        historical_use: Optional[dict[str, float]] = None,
        same_project: Optional[dict[str, float]] = None,
        retrieval_mode: Optional[str] = None,
    ) -> list[RetrievalCandidate]:
        results = []
        c = self.cfg
        mode = str(
            retrieval_mode
            or getattr(c, "retrieval_mode_default", "balanced")
            or "balanced"
        ).strip().lower()
        for s in candidates:
            if not s.mean_vec:
                continue
            mu = np.array(s.mean_vec, dtype=np.float32)
            p = _branch_type_profile(c, s.branch_type)
            w_kappa_coh = _profile_weight(
                p,
                "retrieval_kappa",
                "coherence",
                c.kappa_coherence,
                aliases=["kappa_coherence"],
            )
            w_kappa_comp = _profile_weight(
                p,
                "retrieval_kappa",
                "comp_loss",
                c.kappa_comp_loss,
                aliases=["compression_loss", "kappa_comp_loss"],
            )
            w_kappa_contra = _profile_weight(
                p,
                "retrieval_kappa",
                "contradiction",
                c.kappa_contradiction,
                aliases=["kappa_contradiction"],
            )
            w_kappa_ret = _profile_weight(
                p,
                "retrieval_kappa",
                "ret_error",
                c.kappa_ret_error,
                aliases=["retrieval_error", "kappa_ret_error"],
            )
            w_kappa_hist = _profile_weight(
                p,
                "retrieval_kappa",
                "hist",
                c.kappa_hist,
                aliases=["kappa_hist"],
            )

            sem   = GeometryMath.cosine_sim(query_emb, mu)
            trust = (
                w_kappa_coh     * s.coherence
                + w_kappa_comp   * s.compression_loss
                + w_kappa_contra * s.contradiction_density
                + w_kappa_ret   * s.retrieval_error
            )
            hist  = (historical_use or {}).get(s.branch_id, 0.0)
            proj  = (same_project  or {}).get(s.branch_id, 0.0)
            react = (
                0.5 * sem
                + w_kappa_hist * hist
                + 0.3 * proj
            )
            total_raw = c.alpha_sem * sem + c.beta_trust * trust + c.delta_react * react
            total = total_raw * self._mode_multiplier(mode, s)

            results.append(RetrievalCandidate(
                branch_id=s.branch_id,
                sem_score=sem,
                trust_score=trust,
                react_score=react,
                total_score=total,
            ))

        results.sort(key=lambda r: r.total_score, reverse=True)
        return results


# ---------------------------------------------------------------------------
# Summary Health Checker
# ---------------------------------------------------------------------------

class SummaryHealthChecker:
    def __init__(self, cfg: GeometryConfig):
        self.cfg = cfg

    def score(
        self,
        summary_emb:   np.ndarray,
        desc_embs:     np.ndarray,
        desc_weights:  np.ndarray,
        desc_mean:     np.ndarray,
        contradiction_rate: float = 0.0,
        re_expand_freq: float = 0.0,
    ) -> float:
        c = self.cfg
        comp_loss = GeometryMath.compression_loss(
            summary_emb, desc_embs, desc_weights,
            beta1=c.beta1_comp, beta2=c.beta2_comp,
        )
        cos_to_desc = GeometryMath.cosine_sim(summary_emb, desc_mean)
        health = (
            c.omega_sem           * cos_to_desc
            + c.omega_comp_loss   * comp_loss
            + c.omega_contradiction * contradiction_rate
            + c.omega_re_expand   * re_expand_freq
        )
        return float(health)

    def recommend(self, health: float) -> str:
        if health > 0.70:
            return "healthy"
        if health > 0.45:
            return "refine"
        return "split_or_prevent_condensation"


# ---------------------------------------------------------------------------
# Regime Classifier
# ---------------------------------------------------------------------------

class RegimeClassifier:
    def __init__(self, cfg: GeometryConfig):
        self.cfg = cfg

    def classify(self, s: BranchStats) -> GeometricRegime:
        c    = self.cfg
        band = c.rank_target.get(s.branch_type, c.rank_target["default"])
        target_mid = (band[0] + band[1]) / 2.0

        rigid_threshold    = c.rigid_rank_ratio * target_mid
        too_rigid          = s.eff_rank < rigid_threshold
        high_anisotropy    = s.anisotropy > 0.70

        low_coherence      = s.coherence < c.unstable_coh_floor
        high_comp_loss     = s.compression_loss > c.unstable_comp_ceil

        if (low_coherence or high_comp_loss) and not too_rigid:
            return GeometricRegime.UNSTABLE
        if too_rigid or high_anisotropy:
            return GeometricRegime.RIGID
        return GeometricRegime.PRODUCTIVE


# ---------------------------------------------------------------------------
# Database layer
# ---------------------------------------------------------------------------

class GeometryDB:
    """Thin SQLite wrapper for geometry-layer tables."""

    def __init__(self, db_path: str):
        self.db_path = db_path
        self._conn: Optional[sqlite3.Connection] = None

    def connect(self) -> sqlite3.Connection:
        if self._conn is None:
            self._conn = sqlite3.connect(self.db_path, check_same_thread=False)
            self._conn.row_factory = sqlite3.Row
            self._conn.execute("PRAGMA journal_mode=WAL")
            self._conn.execute("PRAGMA foreign_keys=ON")
            self._conn.executescript(SCHEMA_SQL)
            self._ensure_schema_migrations()
            self._conn.commit()
        return self._conn

    def _ensure_schema_migrations(self) -> None:
        branch_cols = {
            str(r["name"])
            for r in self._conn.execute("PRAGMA table_info(branch_states)").fetchall()
        }
        if "reactivation_score" not in branch_cols:
            self._conn.execute(
                "ALTER TABLE branch_states ADD COLUMN reactivation_score REAL DEFAULT 0.0"
            )
        node_cols = {
            str(r["name"])
            for r in self._conn.execute("PRAGMA table_info(memory_nodes)").fetchall()
        }
        if "update_mode" not in node_cols:
            self._conn.execute(
                "ALTER TABLE memory_nodes ADD COLUMN update_mode TEXT DEFAULT 'attach'"
            )
        if "correction_kind" not in node_cols:
            self._conn.execute(
                "ALTER TABLE memory_nodes ADD COLUMN correction_kind TEXT DEFAULT 'none'"
            )
        if "correction_prev_id" not in node_cols:
            self._conn.execute(
                "ALTER TABLE memory_nodes ADD COLUMN correction_prev_id TEXT"
            )
        if "correction_root_id" not in node_cols:
            self._conn.execute(
                "ALTER TABLE memory_nodes ADD COLUMN correction_root_id TEXT"
            )
        if "correction_version" not in node_cols:
            self._conn.execute(
                "ALTER TABLE memory_nodes ADD COLUMN correction_version INTEGER DEFAULT 1"
            )

    @property
    def conn(self) -> sqlite3.Connection:
        return self.connect()

    # ---- Branch stats ----

    def upsert_branch(self, s: BranchStats) -> None:
        d = asdict(s)
        # Serialise numpy-friendly lists to BLOB via JSON float list
        for field in ("mean_vec", "anchor", "cov_diagonal"):
            val = d[field]
            d[field] = json.dumps(val).encode() if val else None
        self.conn.execute("""
            INSERT OR REPLACE INTO branch_states
                (branch_id, branch_type, state, regime,
                 mean_vec, anchor, cov_diagonal,
                 eff_rank, trace, anisotropy, anchor_drift,
                 coherence, compression_loss,
                 node_count, contradiction_density,
                 retrieval_error, usefulness, reactivation_score,
                 split_counter, merge_counter, last_update_ts)
            VALUES
                (:branch_id, :branch_type, :state, :regime,
                 :mean_vec, :anchor, :cov_diagonal,
                 :eff_rank, :trace, :anisotropy, :anchor_drift,
                 :coherence, :compression_loss,
                 :node_count, :contradiction_density,
                 :retrieval_error, :usefulness, :reactivation_score,
                 :split_counter, :merge_counter, :last_update_ts)
        """, d)
        self.conn.commit()

    def update_branch_scalars(self, s: BranchStats) -> None:
        cur = self.conn.execute(
            """
            UPDATE branch_states
            SET
                branch_type=?,
                state=?,
                regime=?,
                eff_rank=?,
                trace=?,
                anisotropy=?,
                anchor_drift=?,
                coherence=?,
                compression_loss=?,
                node_count=?,
                contradiction_density=?,
                retrieval_error=?,
                usefulness=?,
                reactivation_score=?,
                split_counter=?,
                merge_counter=?,
                last_update_ts=?
            WHERE branch_id=?
            """,
            (
                str(s.branch_type),
                s.state.value if isinstance(s.state, BranchState) else str(s.state),
                s.regime.value if isinstance(s.regime, GeometricRegime) else str(s.regime),
                float(s.eff_rank),
                float(s.trace),
                float(s.anisotropy),
                float(s.anchor_drift),
                float(s.coherence),
                float(s.compression_loss),
                int(s.node_count),
                float(s.contradiction_density),
                float(s.retrieval_error),
                float(s.usefulness),
                float(getattr(s, "reactivation_score", 0.0)),
                int(s.split_counter),
                int(s.merge_counter),
                float(s.last_update_ts),
                str(s.branch_id),
            ),
        )
        if int(cur.rowcount or 0) <= 0:
            self.upsert_branch(s)
            return
        self.conn.commit()

    def load_branch(self, branch_id: str) -> Optional[BranchStats]:
        row = self.conn.execute(
            "SELECT * FROM branch_states WHERE branch_id=?", (branch_id,)
        ).fetchone()
        if row is None:
            return None
        d = dict(row)
        for f in ("mean_vec", "anchor", "cov_diagonal"):
            raw = d[f]
            d[f] = json.loads(raw.decode()) if raw else []
        d["state"]  = BranchState(d["state"])
        d["regime"] = GeometricRegime(d["regime"])
        return BranchStats(**d)

    def all_branches(self) -> list[BranchStats]:
        rows = self.conn.execute("SELECT branch_id FROM branch_states").fetchall()
        return [self.load_branch(r["branch_id"]) for r in rows]  # type: ignore

    def list_branch_scalars(
        self,
        include_states: Optional[set[BranchState | str]] = None,
        exclude_states: Optional[set[BranchState | str]] = None,
        limit: Optional[int] = None,
    ) -> list[dict[str, Any]]:
        q = (
            "SELECT "
            "branch_id, branch_type, state, regime, "
            "eff_rank, trace, anisotropy, anchor_drift, "
            "coherence, compression_loss, node_count, contradiction_density, "
            "retrieval_error, usefulness, reactivation_score, "
            "split_counter, merge_counter, last_update_ts, "
            "CASE WHEN mean_vec IS NOT NULL AND length(mean_vec) > 2 THEN 1 ELSE 0 END AS has_mean "
            "FROM branch_states"
        )
        clauses: list[str] = []
        params: list[Any] = []

        if include_states:
            vals = [s.value if isinstance(s, BranchState) else str(s) for s in include_states]
            ph = ",".join(["?"] * len(vals))
            clauses.append(f"state IN ({ph})")
            params.extend(vals)
        if exclude_states:
            vals = [s.value if isinstance(s, BranchState) else str(s) for s in exclude_states]
            ph = ",".join(["?"] * len(vals))
            clauses.append(f"state NOT IN ({ph})")
            params.extend(vals)
        if clauses:
            q += " WHERE " + " AND ".join(clauses)
        q += " ORDER BY COALESCE(last_update_ts, 0.0) DESC"
        if limit is not None:
            q += " LIMIT ?"
            params.append(max(1, int(limit)))

        rows = self.conn.execute(q, tuple(params)).fetchall()
        return [dict(r) for r in rows]

    def load_branch_mean_vectors(self, branch_ids: list[str]) -> dict[str, list[float]]:
        clean_ids = [str(x) for x in branch_ids if str(x).strip()]
        if not clean_ids:
            return {}
        ph = ",".join(["?"] * len(clean_ids))
        rows = self.conn.execute(
            f"SELECT branch_id, mean_vec FROM branch_states WHERE branch_id IN ({ph})",
            tuple(clean_ids),
        ).fetchall()
        out: dict[str, list[float]] = {}
        for r in rows:
            raw = r["mean_vec"]
            if not raw:
                continue
            try:
                out[str(r["branch_id"])] = json.loads(raw.decode())
            except Exception:
                continue
        return out

    def load_branches_by_ids(self, branch_ids: list[str]) -> list[BranchStats]:
        if not branch_ids:
            return []
        out: list[BranchStats] = []
        for bid in branch_ids:
            s = self.load_branch(bid)
            if s is not None:
                out.append(s)
        return out

    def get_csd_ema(self, branch_id: str, ema_key: str) -> Optional[float]:
        row = self.conn.execute(
            "SELECT value FROM csd_ema_state WHERE branch_id=? AND ema_key=?",
            (branch_id, ema_key),
        ).fetchone()
        if row is None:
            return None
        return float(row["value"])

    def set_csd_ema(self, branch_id: str, ema_key: str, value: float) -> None:
        self.conn.execute(
            "INSERT INTO csd_ema_state (branch_id, ema_key, value, updated_ts) VALUES (?, ?, ?, ?) "
            "ON CONFLICT(branch_id, ema_key) DO UPDATE SET value=excluded.value, updated_ts=excluded.updated_ts",
            (branch_id, ema_key, float(value), time.time()),
        )
        self.conn.commit()

    def next_conv_branch_id(self) -> str:
        row = self.conn.execute(
            "SELECT MAX(CAST(SUBSTR(branch_id, 6) AS INTEGER)) AS max_conv "
            "FROM branch_states WHERE branch_id GLOB 'conv_[0-9]*'"
        ).fetchone()
        max_conv = int(row["max_conv"] or 0) if row else 0
        return f"conv_{max_conv + 1}"

    # ---- Nodes ----

    def insert_node(self, n: MemoryNode) -> None:
        emb_blob = json.dumps(n.embedding).encode() if n.embedding else None
        self.conn.execute("""
            INSERT OR REPLACE INTO memory_nodes
            (id, lcm_id, node_type, parent_id, branch_id, timestamp,
             role, token_count, embedding,
             importance_score, novelty_score, conflict_score,
             coherence_score, compression_loss, reactivation_score,
             stability_state, update_mode,
             correction_kind, correction_prev_id, correction_root_id, correction_version)
            VALUES (?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?)
        """, (
            n.node_id, n.lcm_id, n.node_type, n.parent_id, n.branch_id,
            n.timestamp, n.role, n.token_count, emb_blob,
            n.importance_score, n.novelty_score, n.conflict_score,
            n.coherence_score, n.compression_loss, n.reactivation_score,
            n.stability_state, str(n.update_mode or "attach"),
            str(n.correction_kind or "none"),
            (str(n.correction_prev_id) if n.correction_prev_id else None),
            (str(n.correction_root_id) if n.correction_root_id else None),
            int(max(1, int(n.correction_version or 1))),
        ))
        self.conn.commit()

    def upsert_daily_log_content(
        self,
        node_id: str,
        text: str,
        source: str = "manual_log",
        created_ts: Optional[float] = None,
    ) -> None:
        ts = float(created_ts if created_ts is not None else time.time())
        self.conn.execute(
            """
            INSERT INTO daily_log_content (node_id, text, source, created_ts)
            VALUES (?, ?, ?, ?)
            ON CONFLICT(node_id) DO UPDATE SET
                text=excluded.text,
                source=excluded.source,
                created_ts=excluded.created_ts
            """,
            (str(node_id), str(text), str(source), ts),
        )
        self.conn.commit()

    def list_daily_log_content(
        self,
        branch_id: str,
        limit: int = 200,
        max_chars: int = 500,
    ) -> list[dict[str, Any]]:
        lim = max(1, int(limit))
        mchars = max(1, int(max_chars))
        rows = self.conn.execute(
            """
            SELECT mn.id AS node_id,
                   mn.branch_id,
                   mn.timestamp,
                   dl.source,
                   dl.created_ts,
                   SUBSTR(dl.text, 1, ?) AS text
            FROM memory_nodes mn
            JOIN daily_log_content dl ON dl.node_id = mn.id
            WHERE mn.branch_id=?
            ORDER BY mn.timestamp ASC, mn.rowid ASC
            LIMIT ?
            """,
            (mchars, branch_id, lim),
        ).fetchall()
        return [dict(r) for r in rows]

    def latest_daily_log_content(self, branch_id: str, max_chars: int = 500) -> Optional[dict[str, Any]]:
        mchars = max(1, int(max_chars))
        row = self.conn.execute(
            """
            SELECT mn.id AS node_id,
                   mn.branch_id,
                   mn.timestamp,
                   dl.source,
                   dl.created_ts,
                   SUBSTR(dl.text, 1, ?) AS text
            FROM memory_nodes mn
            JOIN daily_log_content dl ON dl.node_id = mn.id
            WHERE mn.branch_id=?
            ORDER BY mn.timestamp DESC, mn.rowid DESC
            LIMIT 1
            """,
            (mchars, branch_id),
        ).fetchone()
        return dict(row) if row else None

    def count_daily_log_content(self, branch_id: str) -> int:
        row = self.conn.execute(
            """
            SELECT COUNT(*) AS c
            FROM daily_log_content dl
            JOIN memory_nodes mn ON mn.id = dl.node_id
            WHERE mn.branch_id=?
            """,
            (branch_id,),
        ).fetchone()
        return int(row["c"] or 0) if row else 0

    def search_daily_log_keyword(self, query: str, limit: int = 50, max_chars: int = 250) -> list[dict[str, Any]]:
        q = str(query or "").strip()
        if not q:
            return []
        lim = max(1, int(limit))
        mchars = max(1, int(max_chars))
        rows = self.conn.execute(
            """
            SELECT mn.id AS node_id,
                   mn.branch_id,
                   mn.timestamp,
                   dl.source,
                   dl.created_ts,
                   SUBSTR(dl.text, 1, ?) AS text
            FROM daily_log_content dl
            JOIN memory_nodes mn ON mn.id = dl.node_id
            WHERE dl.text LIKE ?
            ORDER BY mn.timestamp DESC, mn.rowid DESC
            LIMIT ?
            """,
            (mchars, f"%{q}%", lim),
        ).fetchall()
        return [dict(r) for r in rows]

    def get_branch_embeddings(
        self, branch_id: str
    ) -> tuple[np.ndarray, np.ndarray]:
        """Return (embeddings, weights) for branch, weighted by importance."""
        rows = self.conn.execute(
            "SELECT embedding, importance_score FROM memory_nodes WHERE branch_id=?",
            (branch_id,),
        ).fetchall()
        if not rows:
            return np.zeros((0, 1)), np.zeros(0)
        embs, weights = [], []
        for r in rows:
            blob = r["embedding"]
            if blob:
                embs.append(json.loads(blob.decode()))
                weights.append(max(r["importance_score"], 1e-6))
        if not embs:
            return np.zeros((0, 1)), np.zeros(0)
        return np.array(embs, dtype=np.float32), np.array(weights, dtype=np.float32)

    def branch_node_count(self, branch_id: str, require_embedding: bool = True) -> int:
        if require_embedding:
            row = self.conn.execute(
                "SELECT COUNT(*) AS c FROM memory_nodes "
                "WHERE branch_id=? AND embedding IS NOT NULL AND length(embedding) > 2",
                (branch_id,),
            ).fetchone()
        else:
            row = self.conn.execute(
                "SELECT COUNT(*) AS c FROM memory_nodes WHERE branch_id=?",
                (branch_id,),
            ).fetchone()
        return int(row["c"] or 0) if row else 0

    def branch_last_activity_ts(self, branch_id: str) -> float:
        row_node = self.conn.execute(
            "SELECT MAX(timestamp) AS ts FROM memory_nodes WHERE branch_id=?",
            (branch_id,),
        ).fetchone()
        row_ret = self.conn.execute(
            "SELECT MAX(timestamp) AS ts FROM retrieval_feedback WHERE branch_id=?",
            (branch_id,),
        ).fetchone()
        n_ts = float(row_node["ts"] or 0.0) if row_node else 0.0
        r_ts = float(row_ret["ts"] or 0.0) if row_ret else 0.0
        return max(n_ts, r_ts, 0.0)

    # ---- Edges ----

    def add_edge(self, src: str, dst: str, etype: EdgeType, weight: float = 1.0) -> None:
        self.conn.execute("""
            INSERT OR REPLACE INTO memory_edges (src_id, dst_id, edge_type, weight)
            VALUES (?, ?, ?, ?)
        """, (src, dst, etype.value, weight))
        self.conn.commit()

    def add_edges_bulk(self, edges: list[tuple[str, str, EdgeType, float]]) -> None:
        if not edges:
            return
        rows = [(src, dst, etype.value if isinstance(etype, EdgeType) else str(etype), float(weight))
                for src, dst, etype, weight in edges]
        self.conn.executemany(
            "INSERT OR REPLACE INTO memory_edges (src_id, dst_id, edge_type, weight) VALUES (?, ?, ?, ?)",
            rows,
        )
        self.conn.commit()

    def find_node_id_by_lcm_id(self, lcm_id: str) -> Optional[str]:
        row = self.conn.execute(
            "SELECT id FROM memory_nodes WHERE lcm_id=? ORDER BY timestamp DESC LIMIT 1",
            (lcm_id,),
        ).fetchone()
        if row is None:
            return None
        return str(row["id"])

    def get_last_node_id(self, branch_id: str) -> Optional[str]:
        row = self.conn.execute(
            "SELECT id FROM memory_nodes WHERE branch_id=? ORDER BY timestamp DESC, rowid DESC LIMIT 1",
            (branch_id,),
        ).fetchone()
        if row is None:
            return None
        return str(row["id"])

    def existing_lcm_ids(self, lcm_ids: list[str], node_type: Optional[NodeType] = None) -> set[str]:
        ids = [str(x) for x in lcm_ids if x is not None and str(x) != ""]
        if not ids:
            return set()
        seen: set[str] = set()
        chunk_size = 800
        for i in range(0, len(ids), chunk_size):
            chunk = ids[i:i + chunk_size]
            ph = ",".join(["?"] * len(chunk))
            if node_type is None:
                q = f"SELECT DISTINCT lcm_id FROM memory_nodes WHERE lcm_id IN ({ph})"
                rows = self.conn.execute(q, tuple(chunk)).fetchall()
            else:
                node_type_val = str(getattr(node_type, "value", node_type))
                q = f"SELECT DISTINCT lcm_id FROM memory_nodes WHERE node_type=? AND lcm_id IN ({ph})"
                rows = self.conn.execute(q, tuple([node_type_val] + chunk)).fetchall()
            for r in rows:
                val = r["lcm_id"]
                if val is not None:
                    seen.add(str(val))
        return seen

    def get_node_correction_meta(self, node_id: str) -> dict[str, Any]:
        row = self.conn.execute(
            """
            SELECT id,
                   update_mode,
                   correction_kind,
                   correction_prev_id,
                   correction_root_id,
                   correction_version
            FROM memory_nodes
            WHERE id=?
            LIMIT 1
            """,
            (node_id,),
        ).fetchone()
        if row is None:
            return {
                "id": str(node_id),
                "update_mode": "attach",
                "correction_kind": "none",
                "correction_prev_id": None,
                "correction_root_id": None,
                "correction_version": 1,
            }
        return {
            "id": str(row["id"]),
            "update_mode": str(row["update_mode"] or "attach"),
            "correction_kind": str(row["correction_kind"] or "none"),
            "correction_prev_id": str(row["correction_prev_id"]) if row["correction_prev_id"] else None,
            "correction_root_id": str(row["correction_root_id"]) if row["correction_root_id"] else None,
            "correction_version": int(row["correction_version"] or 1),
        }

    def list_branch_nodes(self, branch_id: str, include_embeddings: bool = False) -> list[dict[str, Any]]:
        cols = (
            "id, lcm_id, branch_id, timestamp, token_count, role, update_mode, "
            "correction_kind, correction_prev_id, correction_root_id, correction_version"
        )
        if include_embeddings:
            cols += ", embedding"
        rows = self.conn.execute(
            f"SELECT {cols} FROM memory_nodes WHERE branch_id=? ORDER BY timestamp ASC, rowid ASC",
            (branch_id,),
        ).fetchall()
        out: list[dict[str, Any]] = []
        for r in rows:
            row = dict(r)
            if include_embeddings:
                blob = row.get("embedding")
                row["embedding"] = json.loads(blob.decode()) if blob else []
            out.append(row)
        return out

    def branch_update_mode_counts(self, branch_id: str) -> dict[str, int]:
        rows = self.conn.execute(
            """
            SELECT COALESCE(update_mode, 'attach') AS update_mode, COUNT(*) AS c
            FROM memory_nodes
            WHERE branch_id=?
            GROUP BY COALESCE(update_mode, 'attach')
            ORDER BY c DESC
            """,
            (branch_id,),
        ).fetchall()
        return {str(r["update_mode"]): int(r["c"] or 0) for r in rows}

    def branch_correction_counts(self, branch_id: str) -> dict[str, Any]:
        rows = self.conn.execute(
            """
            SELECT COALESCE(correction_kind, 'none') AS correction_kind, COUNT(*) AS c
            FROM memory_nodes
            WHERE branch_id=?
            GROUP BY COALESCE(correction_kind, 'none')
            ORDER BY c DESC
            """,
            (branch_id,),
        ).fetchall()
        by_kind = {str(r["correction_kind"]): int(r["c"] or 0) for r in rows}
        chain_links = int(
            (
                self.conn.execute(
                    "SELECT COUNT(*) AS c FROM memory_nodes WHERE branch_id=? AND correction_prev_id IS NOT NULL",
                    (branch_id,),
                ).fetchone()["c"]
            )
            or 0
        )
        max_version = int(
            (
                self.conn.execute(
                    "SELECT MAX(correction_version) AS v FROM memory_nodes WHERE branch_id=?",
                    (branch_id,),
                ).fetchone()["v"]
            )
            or 1
        )
        return {
            "by_kind": by_kind,
            "chain_links": chain_links,
            "max_version": max_version,
        }

    def recent_corrections(self, branch_id: str, limit: int = 8) -> list[dict[str, Any]]:
        lim = max(1, int(limit))
        rows = self.conn.execute(
            """
            SELECT id, lcm_id, timestamp, update_mode,
                   correction_kind, correction_prev_id, correction_root_id, correction_version
            FROM memory_nodes
            WHERE branch_id=?
              AND correction_prev_id IS NOT NULL
            ORDER BY timestamp DESC, rowid DESC
            LIMIT ?
            """,
            (branch_id, lim),
        ).fetchall()
        return [dict(r) for r in rows]

    def reassign_nodes_to_branch(self, node_ids: list[str], branch_id: str) -> int:
        if not node_ids:
            return 0
        self.conn.executemany(
            "UPDATE memory_nodes SET branch_id=? WHERE id=?",
            [(branch_id, nid) for nid in node_ids],
        )
        self.conn.commit()
        return len(node_ids)

    def remove_edges_for_nodes(self, node_ids: list[str], edge_type: Optional[EdgeType] = None) -> int:
        if not node_ids:
            return 0
        ph = ",".join(["?"] * len(node_ids))
        params: list[Any] = []
        q = f"DELETE FROM memory_edges WHERE src_id IN ({ph}) AND dst_id IN ({ph})"
        params.extend(node_ids)
        params.extend(node_ids)
        if edge_type is not None:
            q += " AND edge_type=?"
            params.append(edge_type.value)
        cur = self.conn.execute(q, tuple(params))
        self.conn.commit()
        return int(cur.rowcount or 0)

    def edge_counts_between_sets(self, a_ids: set[str], b_ids: set[str]) -> dict[str, int]:
        if not a_ids or not b_ids:
            return {}
        a = sorted(x for x in a_ids if x)
        b = sorted(x for x in b_ids if x)
        if not a or not b:
            return {}
        pha = ",".join(["?"] * len(a))
        phb = ",".join(["?"] * len(b))
        q = (
            f"SELECT edge_type FROM memory_edges "
            f"WHERE ((src_id IN ({pha}) AND dst_id IN ({phb})) "
            f"OR (src_id IN ({phb}) AND dst_id IN ({pha})))"
        )
        params = tuple(a + b + b + a)
        rows = self.conn.execute(q, params).fetchall()
        out: dict[str, int] = {}
        for r in rows:
            et = str(r["edge_type"])
            out[et] = out.get(et, 0) + 1
        return out

    def retrieval_co_use_score(self, branch_a: str, branch_b: str, lookback: int = 5000) -> float:
        lim = max(10, int(lookback))
        rows = self.conn.execute(
            "SELECT query_id, branch_id, used FROM retrieval_feedback "
            "WHERE branch_id IN (?, ?) ORDER BY timestamp DESC LIMIT ?",
            (branch_a, branch_b, lim),
        ).fetchall()
        if not rows:
            return 0.0
        grouped: dict[str, dict[str, int]] = {}
        for r in rows:
            qid = str(r["query_id"])
            bid = str(r["branch_id"])
            used = int(r["used"] or 0)
            slot = grouped.setdefault(qid, {})
            slot[bid] = max(slot.get(bid, 0), used)
        total = len(grouped)
        both = 0
        used_quality = 0.0
        for vals in grouped.values():
            if branch_a in vals and branch_b in vals:
                both += 1
                used_quality += 0.5 * (vals.get(branch_a, 0) + vals.get(branch_b, 0))
        if total <= 0 or both <= 0:
            return 0.0
        both_ratio = both / max(1, total)
        quality = used_quality / both
        return float(max(0.0, min(1.0, both_ratio * (0.5 + 0.5 * quality))))

    def list_cross_agent_edges(self, branch_id: Optional[str] = None, limit: int = 200) -> list[dict[str, Any]]:
        lim = max(1, int(limit))
        if branch_id:
            rows = self.conn.execute(
                "SELECT src_id, dst_id, edge_type, weight FROM memory_edges "
                "WHERE edge_type=? AND (src_id=? OR dst_id=?) ORDER BY weight DESC LIMIT ?",
                (EdgeType.CROSS_AGENT.value, branch_id, branch_id, lim),
            ).fetchall()
        else:
            rows = self.conn.execute(
                "SELECT src_id, dst_id, edge_type, weight FROM memory_edges "
                "WHERE edge_type=? ORDER BY weight DESC LIMIT ?",
                (EdgeType.CROSS_AGENT.value, lim),
            ).fetchall()
        return [dict(r) for r in rows]

    # ---- Retrieval feedback ----

    def log_retrieval(
        self,
        query_id: str,
        branch_id: str,
        score: float,
        used: bool = False,
        corrected: bool = False,
        expanded: bool = False,
    ) -> None:
        self.conn.execute("""
            INSERT INTO retrieval_feedback
            (id, query_id, branch_id, score, used, corrected, expanded, timestamp)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?)
        """, (
            str(uuid.uuid4()), query_id, branch_id, score,
            int(used), int(corrected), int(expanded), time.time(),
        ))
        self.conn.commit()

    def branch_retrieval_stats(self, branch_id: str) -> dict[str, float]:
        row = self.conn.execute("""
            SELECT
                AVG(used)      AS avg_used,
                AVG(corrected) AS avg_corrected,
                AVG(expanded)  AS avg_expanded,
                COUNT(*)       AS total
            FROM retrieval_feedback WHERE branch_id=?
        """, (branch_id,)).fetchone()
        if row is None or row["total"] == 0:
            return {"usefulness": 0.5, "error_rate": 0.0}
        return {
            "usefulness": float(row["avg_used"] or 0.5),
            "error_rate": float(row["avg_corrected"] or 0.0),
        }

    # ---- Maintenance jobs ----

    def enqueue_job(
        self,
        job_type: str,
        target_id: Optional[str] = None,
        payload: Optional[dict] = None,
    ) -> str:
        jid = str(uuid.uuid4())
        self.conn.execute("""
            INSERT INTO maintenance_jobs (id, job_type, target_id, status, payload_json, created_ts)
            VALUES (?, ?, ?, 'pending', ?, ?)
        """, (jid, job_type, target_id, json.dumps(payload or {}), time.time()))
        self.conn.commit()
        return jid

    def has_pending_job(self, job_type: str, target_id: Optional[str] = None) -> bool:
        if target_id is None:
            row = self.conn.execute(
                "SELECT 1 FROM maintenance_jobs WHERE status='pending' AND job_type=? LIMIT 1",
                (job_type,),
            ).fetchone()
        else:
            row = self.conn.execute(
                "SELECT 1 FROM maintenance_jobs WHERE status='pending' AND job_type=? AND target_id=? LIMIT 1",
                (job_type, target_id),
            ).fetchone()
        return row is not None

    def pending_jobs(self) -> list[dict]:
        rows = self.conn.execute(
            "SELECT * FROM maintenance_jobs WHERE status='pending' ORDER BY created_ts"
        ).fetchall()
        return [dict(r) for r in rows]

    def complete_job(self, jid: str) -> None:
        self.conn.execute(
            "UPDATE maintenance_jobs SET status='done', completed_ts=? WHERE id=?",
            (time.time(), jid),
        )
        self.conn.commit()

    # ---- Maintenance split observability ----

    def log_split_observation(
        self,
        *,
        run_id: str,
        branch_id: str,
        state_before: str,
        state_after: str,
        regime: str,
        node_count: int,
        split_score: float,
        split_threshold: float,
        split_counter_before: int,
        split_counter_after: int,
        split_hysteresis: int,
        split_min_nodes: int,
        gate_nodes: bool,
        gate_score: bool,
        gate_hysteresis: bool,
        should_split: bool,
        enqueued: bool,
        reason: str,
    ) -> None:
        self.conn.execute(
            """
            INSERT INTO maintenance_split_observations
                (id, run_id, branch_id,
                 state_before, state_after, regime, node_count,
                 split_score, split_threshold,
                 split_counter_before, split_counter_after,
                 split_hysteresis, split_min_nodes,
                 gate_nodes, gate_score, gate_hysteresis,
                 should_split, enqueued, reason, created_ts)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            """,
            (
                str(uuid.uuid4()),
                run_id,
                branch_id,
                state_before,
                state_after,
                regime,
                int(node_count),
                float(split_score),
                float(split_threshold),
                int(split_counter_before),
                int(split_counter_after),
                int(split_hysteresis),
                int(split_min_nodes),
                int(bool(gate_nodes)),
                int(bool(gate_score)),
                int(bool(gate_hysteresis)),
                int(bool(should_split)),
                int(bool(enqueued)),
                reason,
                time.time(),
            ),
        )
        self.conn.commit()

    def list_split_observations(
        self,
        limit: int = 200,
        run_id: Optional[str] = None,
        branch_id: Optional[str] = None,
    ) -> list[dict[str, Any]]:
        lim = max(1, int(limit))
        where: list[str] = []
        params: list[Any] = []
        if run_id:
            where.append("run_id=?")
            params.append(run_id)
        if branch_id:
            where.append("branch_id=?")
            params.append(branch_id)
        q = "SELECT * FROM maintenance_split_observations"
        if where:
            q += " WHERE " + " AND ".join(where)
        q += " ORDER BY created_ts DESC LIMIT ?"
        params.append(lim)
        rows = self.conn.execute(q, tuple(params)).fetchall()
        return [dict(r) for r in rows]

    def prune_split_observations(self, keep_max: int = 20000) -> int:
        safe_keep = max(100, int(keep_max))
        cur = self.conn.execute(
            """
            DELETE FROM maintenance_split_observations
            WHERE id IN (
                SELECT id FROM maintenance_split_observations
                ORDER BY created_ts DESC
                LIMIT -1 OFFSET ?
            )
            """,
            (safe_keep,),
        )
        self.conn.commit()
        return int(cur.rowcount or 0)


# ---------------------------------------------------------------------------
# Geometry Controller  (main public API)
# ---------------------------------------------------------------------------

class GeometryController:
    """
    The primary entry point for the geometry layer.

    Usage:
        gc = GeometryController("geometry.db")

        # On each new LCM message
        decision = gc.on_new_item(
            lcm_id="msg_123",
            node_type=NodeType.MESSAGE,
            embedding=my_embed_fn(text),
            role="user",
            token_count=42,
            conflict_score=0.1,
            active_branch_id="branch_current",
        )

        # On a retrieval query
        ranked = gc.rank_retrieval(query_embedding)

        # Periodically (e.g. every N turns or async)
        gc.run_maintenance_cycle()
    """

    def __init__(
        self,
        db_path:            str = "geometry.db",
        cfg:                 Optional[GeometryConfig] = None,
        embedding_provider:  Optional[EmbeddingProvider] = None,
    ):
        self.cfg               = cfg or GeometryConfig()
        self.db                = GeometryDB(db_path)
        self.csd               = CSDScorer(self.cfg, db=self.db)
        self.merger            = MergeScorer(self.cfg)
        self.splitter          = SplitScorer(self.cfg)
        self.ranker            = RetrievalRanker(self.cfg)
        self.health            = SummaryHealthChecker(self.cfg)
        self.regime            = RegimeClassifier(self.cfg)
        self.embedding_provider = embedding_provider  # may be None
        self.db.connect()
        self._poll_lock        = threading.Lock()

    def _branch_stats_from_scalar_row(self, row: dict[str, Any]) -> BranchStats:
        try:
            state = BranchState(str(row.get("state") or BranchState.FORMING.value))
        except Exception:
            state = BranchState.FORMING
        try:
            regime = GeometricRegime(str(row.get("regime") or GeometricRegime.PRODUCTIVE.value))
        except Exception:
            regime = GeometricRegime.PRODUCTIVE
        return BranchStats(
            branch_id=str(row.get("branch_id") or ""),
            branch_type=str(row.get("branch_type") or "default"),
            state=state,
            regime=regime,
            mean_vec=[],
            anchor=[],
            cov_diagonal=[],
            eff_rank=float(row.get("eff_rank") or 0.0),
            trace=float(row.get("trace") or 0.0),
            anisotropy=float(row.get("anisotropy") or 0.0),
            anchor_drift=float(row.get("anchor_drift") or 0.0),
            coherence=float(row.get("coherence") or 0.0),
            compression_loss=float(row.get("compression_loss") or 0.0),
            node_count=int(row.get("node_count") or 0),
            contradiction_density=float(row.get("contradiction_density") or 0.0),
            retrieval_error=float(row.get("retrieval_error") or 0.0),
            usefulness=float(row.get("usefulness") or 0.0),
            reactivation_score=float(row.get("reactivation_score") or 0.0),
            split_counter=int(row.get("split_counter") or 0),
            merge_counter=int(row.get("merge_counter") or 0),
            last_update_ts=float(row.get("last_update_ts") or 0.0),
        )

    # ------------------------------------------------------------------
    # 1. On-new-item pipeline (call after LCM persists to immutable store)
    # ------------------------------------------------------------------

    def on_new_item(
        self,
        lcm_id:           str,
        node_type:        NodeType,
        embedding:        Optional[list[float]] = None,
        role:             str            = "user",
        token_count:      int            = 0,
        conflict_score:   float          = 0.0,
        active_branch_id: Optional[str] = None,
        force_branch_id:  Optional[str] = None,
        parent_lcm_id:    Optional[str] = None,
        text:             Optional[str] = None,   # alternative to embedding
    ) -> AllocationDecision:
        """
        Full allocation pipeline:
          1. Find candidate branches
          2. Score CSD for each
          3. Decide attach / attach_tension / fork
          4. Adiabatically update branch geometry
          5. Return allocation decision

        Provide EITHER embedding (pre-computed vector) OR text (will embed via
        self.embedding_provider if available). If neither is provided, uses
        zero vector (not recommended — geometry will be meaningless).

        The caller should use the returned branch_id when registering
        the summary node in LCM's DAG metadata.
        """
        # Resolve embedding
        if embedding is not None:
            emb = np.array(embedding, dtype=np.float32)
        elif text is not None and self.embedding_provider is not None:
            vec = self.embedding_provider.embed(text)
            emb = np.array(vec, dtype=np.float32)
        elif text is not None:
            raise ValueError(
                "on_new_item(text=...) requires an EmbeddingProvider. "
                "Pass embedding_provider=EmbeddingProvider() to GeometryController."
            )
        else:
            emb = np.zeros(self.cfg.embedding_dim, dtype=np.float32)

        forced_branch = bool(force_branch_id)
        policy_note = ""
        best_branch_ref: Optional[BranchStats] = None
        if forced_branch:
            best_branch = self._ensure_branch(str(force_branch_id), node_type)
            best_csd = 0.0
            action = "attach"
            branch_id = best_branch.branch_id
            best_branch_ref = best_branch
        else:
            candidates = self._get_candidate_branches(active_branch_id)

            best_branch: Optional[BranchStats] = None
            best_csd    = float("inf")

            for stats in candidates:
                s = self.csd.score(emb, stats, conflict=conflict_score)
                if s < best_csd:
                    best_csd    = s
                    best_branch = stats

            # Determine action
            if best_branch is None:
                action    = "fork"
                branch_id = self._create_branch(node_type)
            else:
                action    = self.csd.decide(best_csd, self.cfg, branch_type=best_branch.branch_type)
                branch_id = best_branch.branch_id
                best_branch_ref = best_branch

                if action in ("attach", "attach_tension"):
                    force_fork, reason = self._protected_attach_requires_fork(best_branch, conflict_score)
                    if force_fork:
                        action = "fork"
                        policy_note = f"; protected_gate={reason}"

                if action == "fork":
                    branch_id = self._create_branch(node_type, near=best_branch.branch_id)
                elif action == "attach_tension":
                    self._set_branch_state(best_branch.branch_id, BranchState.TENSIONED)
        # Persist node
        parent_node_id = self.db.find_node_id_by_lcm_id(parent_lcm_id) if parent_lcm_id else None
        prev_node_id = self.db.get_last_node_id(branch_id)
        update_mode = self._classify_update_mode(
            action=action,
            best_branch=best_branch_ref,
            item_emb=emb,
            conflict_score=conflict_score,
            forced_branch=forced_branch,
        )
        correction_kind = "none"
        correction_prev_id: Optional[str] = None
        correction_root_id: Optional[str] = None
        correction_version = 1

        if prev_node_id and update_mode in ("refine", "contradict", "supersede"):
            prev_meta = self.db.get_node_correction_meta(prev_node_id)
            correction_kind = str(update_mode)
            correction_prev_id = prev_node_id
            correction_root_id = (
                str(prev_meta.get("correction_root_id"))
                if prev_meta.get("correction_root_id")
                else str(prev_node_id)
            )
            correction_version = max(1, int(prev_meta.get("correction_version") or 1) + 1)

        node = MemoryNode(
            node_id    = str(uuid.uuid4()),
            lcm_id     = lcm_id,
            node_type  = node_type,
            branch_id  = branch_id,
            parent_id  = parent_node_id,
            timestamp  = time.time(),
            role       = role,
            token_count = token_count,
            embedding  = emb.tolist(),
            conflict_score = conflict_score,
            update_mode = update_mode,
            correction_kind = correction_kind,
            correction_prev_id = correction_prev_id,
            correction_root_id = correction_root_id,
            correction_version = correction_version,
        )
        self.db.insert_node(node)

        # Build temporal chain inside each branch.
        if prev_node_id and prev_node_id != node.node_id:
            self.add_temporal_edge(prev_node_id, node.node_id)

        # Optional explicit contradiction hint from caller signal.
        if parent_node_id and conflict_score > 0.0:
            self.db.add_edge(parent_node_id, node.node_id, EdgeType.CONTRADICTS, weight=float(conflict_score))

        # Explicit versioned correction links (v1): new node points to previous version.
        if correction_prev_id:
            if correction_kind == "refine":
                self.db.add_edge(node.node_id, correction_prev_id, EdgeType.REFINES, weight=1.0)
            elif correction_kind == "contradict":
                self.db.add_edge(node.node_id, correction_prev_id, EdgeType.CONTRADICTS, weight=max(0.01, float(conflict_score)))
            elif correction_kind == "supersede":
                self.db.add_edge(node.node_id, correction_prev_id, EdgeType.SUPERSEDES, weight=max(0.01, float(conflict_score)))

        # Adiabatically update branch geometry
        updated = self.db.load_branch(branch_id)
        if updated:
            if updated.state == BranchState.DORMANT:
                updated.state = BranchState.REACTIVATING
            self._update_branch_geometry(updated, emb)

        if forced_branch:
            rationale = (
                f"forced_branch_lock={branch_id}; "
                f"conflict={conflict_score:.2f}; branch={branch_id}; mode={update_mode}"
            )
        else:
            rationale = (
                f"CSD={best_csd:.3f} -> {action}; "
                f"conflict={conflict_score:.2f}; branch={branch_id}; mode={update_mode}{policy_note}"
            )

        return AllocationDecision(
            action         = action,
            branch_id      = branch_id,
            csd_score      = best_csd,
            conflict_score = conflict_score,
            rationale      = rationale,
            update_mode    = update_mode,
        )

    # ------------------------------------------------------------------
    # 2. Retrieval ranking
    # ------------------------------------------------------------------

    def rank_retrieval(
        self,
        query_embedding: list[float],
        historical_use:  Optional[dict[str, float]] = None,
        same_project:    Optional[dict[str, float]] = None,
        retrieval_mode:  Optional[str] = None,
    ) -> list[RetrievalCandidate]:
        """
        Returns branches sorted by retrieval priority.

        Top branches should be queried first via lcm_describe / lcm_grep.
        Only delegate lcm_expand to sub-agents for branches deep in the list.
        """
        hist_map = historical_use or {}
        proj_map = same_project or {}
        excluded = {BranchState.COLLAPSING, BranchState.SPLIT_PENDING}
        scalar_rows = self.db.list_branch_scalars(exclude_states=excluded)
        if not scalar_rows:
            return []

        prefilter_limit = max(8, int(self.cfg.retrieval_prefilter_limit))
        with_mean = [r for r in scalar_rows if int(r.get("has_mean") or 0) == 1]
        if not with_mean:
            return []

        if len(with_mean) > prefilter_limit:
            def prefilter_score(row: dict[str, Any]) -> float:
                bid = str(row.get("branch_id") or "")
                use = float(row.get("usefulness") or 0.0)
                coh = float(row.get("coherence") or 0.0)
                nodes = float(row.get("node_count") or 0.0)
                h = float(hist_map.get(bid, 0.0))
                p = float(proj_map.get(bid, 0.0))
                return (
                    0.45 * h
                    + 0.35 * p
                    + 0.12 * use
                    + 0.05 * coh
                    + 0.03 * math.log1p(max(0.0, nodes))
                )

            with_mean.sort(key=prefilter_score, reverse=True)
            shortlist_rows = with_mean[:prefilter_limit]
        else:
            shortlist_rows = with_mean

        shortlist_ids = [str(r["branch_id"]) for r in shortlist_rows]
        mean_map = self.db.load_branch_mean_vectors(shortlist_ids)
        candidates: list[BranchStats] = []
        for row in shortlist_rows:
            bid = str(row.get("branch_id") or "")
            mean_vec = mean_map.get(bid)
            if not mean_vec:
                continue
            try:
                state = BranchState(str(row.get("state") or BranchState.FORMING.value))
            except Exception:
                state = BranchState.FORMING
            try:
                regime = GeometricRegime(str(row.get("regime") or GeometricRegime.PRODUCTIVE.value))
            except Exception:
                regime = GeometricRegime.PRODUCTIVE
            candidates.append(
                BranchStats(
                    branch_id=bid,
                    branch_type=str(row.get("branch_type") or "default"),
                    state=state,
                    regime=regime,
                    mean_vec=mean_vec,
                    coherence=float(row.get("coherence") or 0.0),
                    compression_loss=float(row.get("compression_loss") or 0.0),
                    contradiction_density=float(row.get("contradiction_density") or 0.0),
                    retrieval_error=float(row.get("retrieval_error") or 0.0),
                    usefulness=float(row.get("usefulness") or 0.0),
                    node_count=int(row.get("node_count") or 0),
                    split_counter=int(row.get("split_counter") or 0),
                    merge_counter=int(row.get("merge_counter") or 0),
                    last_update_ts=float(row.get("last_update_ts") or 0.0),
                )
            )

        if not candidates:
            return []
        q_emb = np.array(query_embedding, dtype=np.float32)
        ranked = self.ranker.rank(
            q_emb,
            candidates,
            historical_use,
            same_project,
            retrieval_mode=retrieval_mode,
        )
        return ranked[: max(1, int(self.cfg.retrieval_result_limit))]

    # ------------------------------------------------------------------
    # 3. Retrieval feedback
    # ------------------------------------------------------------------

    def record_retrieval(
        self,
        query_id:  str,
        branch_id: str,
        score:     float,
        used:      bool = False,
        corrected: bool = False,
        expanded:  bool = False,
    ) -> None:
        """Log the outcome of a retrieval for usefulness-EMA updating."""
        self.db.log_retrieval(query_id, branch_id, score, used, corrected, expanded)
        # Update usefulness EMA on the branch
        fb = self.db.branch_retrieval_stats(branch_id)
        stats = self.db.load_branch(branch_id)
        if stats:
            reward = 1.0 if used and not corrected else (-0.5 if corrected else 0.0)
            lam = self.cfg.usefulness_lambda
            stats.usefulness    = (1 - lam) * stats.usefulness + lam * reward
            stats.retrieval_error = (1 - lam) * stats.retrieval_error + lam * float(corrected)
            stats.last_update_ts = time.time()
            self.db.upsert_branch(stats)

    # ------------------------------------------------------------------
    # 4. Summary health audit  (call after LCM produces a summary node)
    # ------------------------------------------------------------------

    def audit_summary(
        self,
        summary_lcm_id:    str,
        summary_embedding: list[float],
        descendant_lcm_ids: list[str],
        descendant_embeddings: list[list[float]],
        contradiction_rate: float = 0.0,
        re_expand_frequency: float = 0.0,
    ) -> dict[str, Any]:
        """
        Returns a health score + recommendation for a summary node.

        Recommendation: "healthy" | "refine" | "split_or_prevent_condensation"
        """
        if not descendant_embeddings:
            return {"health": 1.0, "recommendation": "healthy", "compression_loss": 0.0}

        s_emb   = np.array(summary_embedding,   dtype=np.float32)
        d_embs  = np.array(descendant_embeddings, dtype=np.float32)
        d_w     = np.ones(len(d_embs), dtype=np.float32)
        d_mean  = GeometryMath.weighted_mean(d_embs, d_w)

        comp    = GeometryMath.compression_loss(s_emb, d_embs, d_w,
                                                self.cfg.beta1_comp, self.cfg.beta2_comp)
        h       = self.health.score(s_emb, d_embs, d_w, d_mean,
                                    contradiction_rate, re_expand_frequency)
        rec     = self.health.recommend(h)

        return {
            "summary_lcm_id":   summary_lcm_id,
            "health":           round(h, 4),
            "compression_loss": round(comp, 4),
            "recommendation":   rec,
        }

    # ------------------------------------------------------------------
    # 5. Maintenance cycle (run periodically or via background LLM-Map)
    # ------------------------------------------------------------------

    def run_maintenance_cycle(self) -> dict[str, Any]:
        """
        Runs offline maintenance:
          - geometry recompute
          - contradiction refresh
          - split/merge scans
          - split job execution
          - reactivation scan
        """
        actions = {
            "recomputed": 0,
            "split_pending": 0,
            "split_executed": 0,
            "merge_candidates": 0,
            "merge_executed": 0,
            "dormant_marked": 0,
            "reactivated": 0,
            "split_observations": 0,
        }
        split_run_id = str(uuid.uuid4())
        actions["split_trace_run_id"] = split_run_id
        split_threshold = float(self.cfg.split_score_threshold)
        min_split_nodes = max(2, int(self.cfg.split_min_nodes))
        split_readiness_nodes = max(min_split_nodes, int(self.cfg.min_branch_size))
        split_enqueue_cap = max(0, int(self.cfg.max_split_enqueues_per_cycle))
        split_candidates: list[dict[str, Any]] = []
        split_observations: list[dict[str, Any]] = []
        scalar_rows = self.db.list_branch_scalars()
        all_stats: list[BranchStats] = []
        has_mean_by_id: dict[str, bool] = {}
        for row in scalar_rows:
            bid = str(row.get("branch_id") or "")
            if not bid:
                continue
            has_mean_by_id[bid] = bool(int(row.get("has_mean") or 0))
            all_stats.append(self._branch_stats_from_scalar_row(row))
        full_loaded_ids: set[str] = set()

        def _persist_branch(s: BranchStats) -> None:
            if s.branch_id in full_loaded_ids:
                self.db.upsert_branch(s)
            else:
                self.db.update_branch_scalars(s)

        cycle_now_ts = time.time()
        for i, s in enumerate(all_stats):
            if s is None:
                continue

            embs, weights = self.db.get_branch_embeddings(s.branch_id)
            if embs.shape[0] >= self.cfg.min_branch_size:
                full_s = self.db.load_branch(s.branch_id)
                if full_s is None:
                    continue
                full_loaded_ids.add(full_s.branch_id)
                mu = GeometryMath.weighted_mean(embs, weights)
                stats_dict = GeometryMath.compute_full_stats(embs, weights, mu)
                rho = self.cfg.stat_rho
                full_s.mean_vec = mu.tolist()
                full_s.eff_rank = (1 - rho) * full_s.eff_rank + rho * stats_dict["eff_rank"]
                full_s.trace = (1 - rho) * full_s.trace + rho * stats_dict["trace"]
                full_s.anisotropy = (1 - rho) * full_s.anisotropy + rho * stats_dict["anisotropy"]
                full_s.coherence = (1 - rho) * full_s.coherence + rho * stats_dict["coherence"]
                if full_s.anchor:
                    a_old = np.array(full_s.anchor, dtype=np.float32)
                    a_new = (1 - self.cfg.anchor_alpha) * a_old + self.cfg.anchor_alpha * mu
                    full_s.anchor = a_new.tolist()
                    full_s.anchor_drift = float(np.linalg.norm(a_new - a_old))
                else:
                    full_s.anchor = mu.tolist()
                    full_s.anchor_drift = 0.0

                var = GeometryMath.covariance_diagonal(embs, weights)
                full_s.cov_diagonal = var.tolist()
                full_s.node_count = int(embs.shape[0])
                full_s.contradiction_density = self._compute_contradiction_signals(full_s.branch_id)
                full_s.regime = self.regime.classify(full_s)
                self._update_state_from_regime(full_s)
                full_s.last_update_ts = time.time()
                self.db.upsert_branch(full_s)
                actions["recomputed"] += 1
                s = full_s
                all_stats[i] = s
            elif has_mean_by_id.get(s.branch_id, False):
                s.contradiction_density = self._compute_contradiction_signals(s.branch_id)
                s.regime = self.regime.classify(s)
                self._update_state_from_regime(s)
                s.last_update_ts = time.time()
                _persist_branch(s)
                actions["recomputed"] += 1

            prev_state = s.state
            if self._apply_dormancy_policy(s, now_ts=cycle_now_ts):
                if s.state == BranchState.DORMANT:
                    actions["dormant_marked"] += 1
                elif prev_state == BranchState.DORMANT and s.state == BranchState.REACTIVATING:
                    actions["reactivated"] += 1
                _persist_branch(s)
            if s.state == BranchState.DORMANT:
                # Keep dormant branches out of split accumulation until reactivation.
                continue

            split_counter_before = int(s.split_counter)
            state_before = s.state.value
            gate_nodes = bool(s.node_count >= split_readiness_nodes)
            real_node_count = self.db.branch_node_count(s.branch_id, require_embedding=True)
            gate_real_nodes = bool(real_node_count >= split_readiness_nodes)

            split_s = self.splitter.score(s)
            gate_score = bool(split_s > split_threshold)
            counter_reset_for_nodes = False
            if not gate_nodes or not gate_real_nodes:
                # Do not accumulate split hysteresis on branches that are below
                # either split-size gate (state node_count or real node rows).
                counter_reset_for_nodes = bool(s.split_counter != 0)
                s.split_counter = 0
            elif gate_score:
                s.split_counter += 1
            else:
                s.split_counter = max(0, s.split_counter - 1)

            gate_hysteresis = bool(s.split_counter >= self.cfg.split_hysteresis)
            should_split_now = bool(
                gate_nodes
                and gate_real_nodes
                and self.splitter.should_split(s, split_s, self.cfg)
            )
            pending_split_exists = bool(self.db.has_pending_job("split", s.branch_id))
            enqueued = False
            reason_parts: list[str] = []
            if not gate_nodes:
                reason_parts.append(f"nodes_below_min:{s.node_count}<{split_readiness_nodes}")
            if not gate_real_nodes:
                reason_parts.append(f"real_nodes_below_min:{real_node_count}<{split_readiness_nodes}")
            if counter_reset_for_nodes:
                reason_parts.append("counter_reset_for_nodes_gate")
            if not gate_score:
                reason_parts.append(f"score_below_threshold:{split_s:.4f}<={split_threshold:.4f}")
            if gate_score and gate_nodes and gate_real_nodes and not gate_hysteresis:
                reason_parts.append(
                    f"hysteresis_not_met:{s.split_counter}<{int(self.cfg.split_hysteresis)}"
                )
            if should_split_now:
                if pending_split_exists:
                    s.state = BranchState.SPLIT_PENDING
                    reason_parts.append("eligible_pending_exists")
                else:
                    reason_parts.append("eligible_candidate")
            elif not reason_parts:
                reason_parts.append("not_eligible")

            split_obs = {
                "run_id": split_run_id,
                "branch_id": s.branch_id,
                "state_before": state_before,
                "state_after": s.state.value,
                "regime": s.regime.value,
                "node_count": int(s.node_count),
                "split_score": float(split_s),
                "split_threshold": split_threshold,
                "split_counter_before": split_counter_before,
                "split_counter_after": int(s.split_counter),
                "split_hysteresis": int(self.cfg.split_hysteresis),
                "split_min_nodes": split_readiness_nodes,
                "gate_nodes": gate_nodes,
                "gate_score": gate_score,
                "gate_hysteresis": gate_hysteresis,
                "should_split": should_split_now,
                "enqueued": enqueued,
                "reason_parts": reason_parts,
            }
            split_observations.append(split_obs)
            if should_split_now and not pending_split_exists:
                split_candidates.append(
                    {
                        "branch_id": s.branch_id,
                        "split_score": float(split_s),
                        "stats": s,
                        "obs": split_obs,
                    }
                )
            _persist_branch(s)

        if split_candidates:
            split_candidates.sort(key=lambda item: float(item["split_score"]), reverse=True)
            for rank, cand in enumerate(split_candidates, start=1):
                branch_stats: BranchStats = cand["stats"]
                obs: dict[str, Any] = cand["obs"]
                reason_parts: list[str] = obs["reason_parts"]
                allowed_by_cap = bool(split_enqueue_cap > 0 and rank <= split_enqueue_cap)
                if allowed_by_cap:
                    if self.db.has_pending_job("split", branch_stats.branch_id):
                        branch_stats.state = BranchState.SPLIT_PENDING
                        reason_parts.append("eligible_pending_exists_late")
                    else:
                        self.db.enqueue_job(
                            "split",
                            target_id=branch_stats.branch_id,
                            payload={"split_score": float(cand["split_score"])},
                        )
                        actions["split_pending"] += 1
                        obs["enqueued"] = True
                        reason_parts.append(f"eligible_enqueued_rank:{rank}")
                        branch_stats.state = BranchState.SPLIT_PENDING
                else:
                    reason_parts.append(
                        f"eligible_throttled_rank:{rank}>cap:{int(split_enqueue_cap)}"
                    )
                obs["state_after"] = branch_stats.state.value
                _persist_branch(branch_stats)

        for obs in split_observations:
            self.db.log_split_observation(
                run_id=obs["run_id"],
                branch_id=obs["branch_id"],
                state_before=obs["state_before"],
                state_after=obs["state_after"],
                regime=obs["regime"],
                node_count=int(obs["node_count"]),
                split_score=float(obs["split_score"]),
                split_threshold=float(obs["split_threshold"]),
                split_counter_before=int(obs["split_counter_before"]),
                split_counter_after=int(obs["split_counter_after"]),
                split_hysteresis=int(obs["split_hysteresis"]),
                split_min_nodes=int(obs["split_min_nodes"]),
                gate_nodes=bool(obs["gate_nodes"]),
                gate_score=bool(obs["gate_score"]),
                gate_hysteresis=bool(obs["gate_hysteresis"]),
                should_split=bool(obs["should_split"]),
                enqueued=bool(obs["enqueued"]),
                reason=";".join(obs["reason_parts"]),
            )
            actions["split_observations"] += 1

        stable = [s for s in all_stats if s and s.state in (BranchState.STABLE, BranchState.DORMANT)]
        mean_map = self.db.load_branch_mean_vectors([s.branch_id for s in stable]) if stable else {}
        for s in stable:
            if not s.mean_vec:
                mv = mean_map.get(s.branch_id)
                if mv:
                    s.mean_vec = mv

        stable_by_id = {s.branch_id: s for s in stable}
        alias_cache = {s.branch_id: self._branch_aliases(s.branch_id) for s in stable}
        merge_positive_hits: set[str] = set()
        merge_decay_candidates: set[str] = set()
        for i, s1 in enumerate(stable):
            for s2 in stable[i + 1:]:
                blocked_merge, _blocked_reason = self._protected_merge_blocked(s1, s2)
                if blocked_merge:
                    if int(s1.merge_counter) > 0:
                        merge_decay_candidates.add(s1.branch_id)
                    if int(s2.merge_counter) > 0:
                        merge_decay_candidates.add(s2.branch_id)
                    continue

                topic_overlap, retrieval_co_use = self._compute_merge_signals(
                    s1,
                    s2,
                    aliases_a=alias_cache.get(s1.branch_id),
                    aliases_b=alias_cache.get(s2.branch_id),
                )
                ms = self.merger.score(
                    s1,
                    s2,
                    topic_overlap=topic_overlap,
                    retrieval_co_use=retrieval_co_use,
                )
                if self.merger.should_merge(s1, s2, ms):
                    merge_positive_hits.add(s1.branch_id)
                    merge_positive_hits.add(s2.branch_id)
                    s1.merge_counter += 1
                    s2.merge_counter += 1
                    if (
                        s1.merge_counter >= self.cfg.merge_hysteresis
                        and s2.merge_counter >= self.cfg.merge_hysteresis
                    ):
                        s1.state = BranchState.MERGE_CANDIDATE
                        s2.state = BranchState.MERGE_CANDIDATE
                        if not self.db.has_pending_job("merge", s1.branch_id):
                            self.db.enqueue_job(
                                "merge",
                                target_id=s1.branch_id,
                                payload={
                                    "merge_with": s2.branch_id,
                                    "score": float(ms),
                                    "topic_overlap": float(topic_overlap),
                                    "retrieval_co_use": float(retrieval_co_use),
                                },
                            )
                            actions["merge_candidates"] += 1
                    _persist_branch(s1)
                    _persist_branch(s2)
                else:
                    if int(s1.merge_counter) > 0:
                        merge_decay_candidates.add(s1.branch_id)
                    if int(s2.merge_counter) > 0:
                        merge_decay_candidates.add(s2.branch_id)

        for bid in sorted(merge_decay_candidates - merge_positive_hits):
            s = stable_by_id.get(bid)
            if s is None:
                continue
            s.merge_counter = max(0, int(s.merge_counter) - 1)
            _persist_branch(s)

        dormant = [s for s in all_stats if s and s.state == BranchState.DORMANT]
        for s in dormant:
            if (
                float(getattr(s, "reactivation_score", 0.0)) > float(self.cfg.reactivation_min_score)
                and self._reactivation_guard_ok(s)
            ):
                s.state = BranchState.REACTIVATING
                _persist_branch(s)
                actions["reactivated"] += 1

        actions["split_executed"] = self.execute_pending_splits(max_jobs=10)
        actions["merge_executed"] = self.execute_pending_merges(
            max_jobs=max(1, int(self.cfg.merge_max_jobs_per_cycle))
        )
        self.db.prune_split_observations(self.cfg.split_observability_keep)
        return actions

    # ------------------------------------------------------------------
    # Branch creation helpers
    # ------------------------------------------------------------------

    def _create_branch(
        self,
        node_type: NodeType,
        near:      Optional[str] = None,
    ) -> str:
        bid = self._next_conv_branch_id()
        btype = self._infer_branch_type(node_type)
        s = BranchStats(branch_id=bid, branch_type=btype)
        self.db.upsert_branch(s)
        return bid

    def _ensure_branch(self, branch_id: str, node_type: NodeType) -> BranchStats:
        existing = self.db.load_branch(branch_id)
        if existing:
            return existing
        btype = self._infer_branch_type(node_type)
        s = BranchStats(branch_id=branch_id, branch_type=btype)
        self.db.upsert_branch(s)
        return s

    def _next_conv_branch_id(self) -> str:
        return self.db.next_conv_branch_id()

    def _infer_branch_type(self, node_type: NodeType) -> str:
        if node_type in (NodeType.MESSAGE,):
            return "broad_history"
        if node_type in (NodeType.TOOL_RESULT,):
            return "project_task"
        return "default"

    def _set_branch_state(self, branch_id: str, state: BranchState) -> None:
        s = self.db.load_branch(branch_id)
        if s:
            s.state = state
            s.last_update_ts = time.time()
            self.db.upsert_branch(s)

    def _branch_activity_age_days(self, branch_id: str, now_ts: Optional[float] = None) -> Optional[float]:
        ts = float(now_ts if now_ts is not None else time.time())
        last_activity = float(self.db.branch_last_activity_ts(branch_id))
        if last_activity <= 0.0:
            return None
        return max(0.0, (ts - last_activity) / 86400.0)

    def _is_protected_branch_type(self, branch_type: str) -> bool:
        bt = str(branch_type or "").strip().lower()
        if not bt:
            return False
        raw = getattr(self.cfg, "protected_branch_types", None) or []
        for x in raw:
            if str(x).strip().lower() == bt:
                return True
        return False

    def _is_supersede_branch_type(self, branch_type: str) -> bool:
        bt = str(branch_type or "").strip().lower()
        if not bt:
            return False
        raw = getattr(self.cfg, "update_mode_supersede_branch_types", None) or []
        for x in raw:
            if str(x).strip().lower() == bt:
                return True
        return False

    def _classify_update_mode(
        self,
        *,
        action: str,
        best_branch: Optional[BranchStats],
        item_emb: np.ndarray,
        conflict_score: float,
        forced_branch: bool = False,
    ) -> str:
        act = str(action or "").strip().lower()
        if act == "fork":
            return "fork"

        sim = 0.0
        if best_branch is not None and best_branch.mean_vec:
            try:
                sim = float(
                    GeometryMath.cosine_sim(
                        item_emb,
                        np.array(best_branch.mean_vec, dtype=np.float32),
                    )
                )
            except Exception:
                sim = 0.0

        conflict = float(conflict_score)
        if (
            best_branch is not None
            and self._is_supersede_branch_type(best_branch.branch_type)
            and conflict >= float(self.cfg.update_mode_supersede_conflict_min)
            and sim >= float(self.cfg.update_mode_supersede_similarity_min)
        ):
            return "supersede"

        if conflict >= float(self.cfg.update_mode_contradict_conflict_min):
            return "contradict"

        if sim >= float(self.cfg.update_mode_refine_similarity_min):
            return "refine"

        if forced_branch and conflict <= 0.0 and sim >= 0.5:
            return "refine"
        return "attach"

    def _protected_attach_requires_fork(self, s: BranchStats, conflict_score: float) -> tuple[bool, str]:
        if not self._is_protected_branch_type(s.branch_type):
            return False, ""
        if float(conflict_score) >= float(self.cfg.protected_attach_conflict_threshold):
            return True, "protected_conflict"
        if float(s.contradiction_density) >= float(self.cfg.protected_attach_contradiction_threshold):
            return True, "protected_contradiction"
        return False, ""

    def _protected_merge_blocked(self, s1: BranchStats, s2: BranchStats) -> tuple[bool, str]:
        protected = self._is_protected_branch_type(s1.branch_type) or self._is_protected_branch_type(s2.branch_type)
        if not protected:
            return False, ""
        if bool(self.cfg.protected_merge_block):
            return True, "protected_merge_block"
        th = float(self.cfg.protected_merge_contradiction_threshold)
        if float(s1.contradiction_density) >= th or float(s2.contradiction_density) >= th:
            return True, "protected_merge_contradiction"
        return False, ""

    def _reactivation_guard_ok(
        self,
        s: BranchStats,
        query_embedding: Optional[np.ndarray] = None,
    ) -> bool:
        if not bool(getattr(self.cfg, "reactivation_guard_enabled", True)):
            return True
        if float(s.contradiction_density) > float(self.cfg.reactivation_max_contradiction):
            return False
        if float(s.retrieval_error) > float(self.cfg.reactivation_max_retrieval_error):
            return False
        if query_embedding is not None and s.mean_vec:
            try:
                sim = GeometryMath.cosine_sim(query_embedding, np.array(s.mean_vec, dtype=np.float32))
            except Exception:
                sim = 1.0
            if float(sim) < float(self.cfg.reactivation_min_similarity):
                return False
        return True

    def _apply_dormancy_policy(self, s: BranchStats, now_ts: Optional[float] = None) -> bool:
        """
        Apply inactivity/usefulness dormancy policy.
        Returns True when lifecycle state changed.
        """
        if s.branch_type == "daily_log":
            return False

        if s.state in (BranchState.COLLAPSING, BranchState.SPLIT_PENDING, BranchState.MERGE_CANDIDATE):
            return False

        age_days = self._branch_activity_age_days(s.branch_id, now_ts=now_ts)
        if age_days is None:
            return False

        dormant_after_days = max(0.25, float(self.cfg.dormant_after_days))
        reactivate_days = max(0.25, dormant_after_days * 0.25)
        min_nodes = max(1, int(self.cfg.dormant_min_nodes))
        usefulness_ceiling = float(self.cfg.dormant_usefulness_max)

        if s.state == BranchState.DORMANT and age_days <= reactivate_days:
            if self._reactivation_guard_ok(s):
                s.state = BranchState.REACTIVATING
                s.last_update_ts = float(now_ts if now_ts is not None else time.time())
                return True
            return False

        if s.state == BranchState.REACTIVATING and age_days <= reactivate_days:
            # Give branch a clean path back into active lifecycle after fresh activity.
            s.state = BranchState.ACTIVE
            s.last_update_ts = float(now_ts if now_ts is not None else time.time())
            return True

        eligible_state = s.state in (BranchState.ACTIVE, BranchState.STABLE, BranchState.TENSIONED)
        if (
            eligible_state
            and s.node_count >= min_nodes
            and s.usefulness <= usefulness_ceiling
            and age_days >= dormant_after_days
        ):
            s.state = BranchState.DORMANT
            s.split_counter = 0
            s.last_update_ts = float(now_ts if now_ts is not None else time.time())
            return True

        return False

    def _get_candidate_branches(
        self, active_branch_id: Optional[str]
    ) -> list[BranchStats]:
        excluded = {BranchState.COLLAPSING, BranchState.SPLIT_PENDING}
        cap = max(1, int(self.cfg.candidate_branch_cap))
        prefilter_limit = max(cap, int(self.cfg.candidate_prefilter_limit))
        scalar_rows = self.db.list_branch_scalars(
            exclude_states=excluded,
            limit=prefilter_limit,
        )
        if not scalar_rows:
            return []

        active_mean: Optional[np.ndarray] = None
        mean_map: dict[str, list[float]] = {}
        if active_branch_id:
            active_mean_raw = self.db.load_branch_mean_vectors([active_branch_id]).get(active_branch_id)
            if active_mean_raw:
                active_mean = np.array(active_mean_raw, dtype=np.float32)
                mean_map = self.db.load_branch_mean_vectors(
                    [str(r.get("branch_id")) for r in scalar_rows]
                )

        def candidate_score(row: dict[str, Any]) -> float:
            bid = str(row.get("branch_id") or "")
            if active_branch_id and bid == active_branch_id:
                return float("inf")

            use = float(row.get("usefulness") or 0.0)
            coh = float(row.get("coherence") or 0.0)
            nodes = float(row.get("node_count") or 0.0)
            last = float(row.get("last_update_ts") or 0.0)
            recency = 0.0
            if last > 0.0:
                age_days = max(0.0, (time.time() - last) / 86400.0)
                recency = 1.0 / (1.0 + age_days)
            base = (
                0.45 * use
                + 0.25 * coh
                + 0.10 * math.log1p(max(0.0, nodes))
                + 0.20 * recency
            )

            if active_mean is not None:
                raw = mean_map.get(bid)
                if raw:
                    try:
                        sim = GeometryMath.cosine_sim(active_mean, np.array(raw, dtype=np.float32))
                        base += 0.55 * sim
                    except Exception:
                        pass
            return base

        scalar_rows.sort(key=candidate_score, reverse=True)
        candidate_ids = [str(r.get("branch_id")) for r in scalar_rows[:cap] if str(r.get("branch_id"))]
        if active_branch_id and active_branch_id not in candidate_ids:
            active_s = self.db.load_branch(active_branch_id)
            if active_s is not None and active_s.state not in excluded:
                candidate_ids = [active_branch_id] + candidate_ids
                candidate_ids = candidate_ids[:cap]

        loaded = self.db.load_branches_by_ids(candidate_ids)
        by_id = {b.branch_id: b for b in loaded}
        return [by_id[bid] for bid in candidate_ids if bid in by_id]

    def _update_branch_geometry(self, s: BranchStats, new_emb: np.ndarray) -> None:
        """One-shot adiabatic update on insertion without reloading all embeddings."""
        rho = self.cfg.stat_rho
        alpha = self.cfg.anchor_alpha

        if not s.mean_vec:
            s.mean_vec  = new_emb.tolist()
            s.anchor    = new_emb.tolist()
            s.cov_diagonal = np.zeros_like(new_emb).tolist()
            s.node_count = 1
        else:
            mu_old = np.array(s.mean_vec, dtype=np.float32)
            n      = max(s.node_count, 1)
            mu_new = (n * mu_old + new_emb) / (n + 1)

            # Adiabatic blend of mean
            s.mean_vec = ((1 - rho) * mu_old + rho * mu_new).tolist()

            # Anchor drift
            if s.anchor:
                a_old      = np.array(s.anchor, dtype=np.float32)
                a_new      = (1 - alpha) * a_old + alpha * mu_new
                s.anchor_drift = float(np.linalg.norm(a_new - a_old))
                s.anchor   = a_new.tolist()

            s.node_count += 1

        s.last_update_ts = time.time()
        self.db.upsert_branch(s)

    def _update_state_from_regime(self, s: BranchStats) -> None:
        """Promote/demote lifecycle state based on geometric regime."""
        # Don't override terminal / pending states
        if s.state in (BranchState.SPLIT_PENDING, BranchState.MERGE_CANDIDATE,
                       BranchState.COLLAPSING):
            return

        if s.regime == GeometricRegime.UNSTABLE:
            if s.coherence < 0.3 or s.anisotropy > 0.85:
                s.state = BranchState.COLLAPSING
            else:
                s.state = BranchState.TENSIONED
        elif s.regime == GeometricRegime.RIGID:
            if s.state not in (BranchState.DORMANT, BranchState.REACTIVATING):
                s.state = BranchState.STABLE   # rigid can still be stable
        else:
            # PRODUCTIVE
            if s.node_count < self.cfg.min_branch_size:
                s.state = BranchState.FORMING
            elif s.state == BranchState.FORMING:
                s.state = BranchState.ACTIVE
            elif s.state == BranchState.ACTIVE and s.coherence > 0.70:
                s.state = BranchState.STABLE

    def add_temporal_edge(self, src_node_id: str, dst_node_id: str, weight: float = 1.0) -> None:
        if not src_node_id or not dst_node_id or src_node_id == dst_node_id:
            return
        self.db.add_edge(src_node_id, dst_node_id, EdgeType.TEMPORAL_NEXT, weight=float(weight))

    def _branch_aliases(self, branch_id: str) -> set[str]:
        aliases: set[str] = {branch_id}
        rows = self.db.list_branch_nodes(branch_id, include_embeddings=False)
        for row in rows:
            nid = str(row.get("id") or "").strip()
            lcm_id = str(row.get("lcm_id") or "").strip()
            if nid:
                aliases.add(nid)
            if lcm_id:
                aliases.add(lcm_id)
                if not lcm_id.startswith("msg_"):
                    aliases.add(f"msg_{lcm_id}")
                if not lcm_id.startswith("sn_"):
                    aliases.add(f"sn_{lcm_id}")
        return aliases

    def _compute_merge_signals(
        self,
        s1: BranchStats,
        s2: BranchStats,
        aliases_a: Optional[set[str]] = None,
        aliases_b: Optional[set[str]] = None,
    ) -> tuple[float, float]:
        if aliases_a is None:
            aliases_a = self._branch_aliases(s1.branch_id)
        if aliases_b is None:
            aliases_b = self._branch_aliases(s2.branch_id)
        edge_counts = self.db.edge_counts_between_sets(aliases_a, aliases_b)

        topic_types = {
            EdgeType.SAME_TOPIC.value,
            EdgeType.SAME_TASK.value,
            EdgeType.SAME_USER_FACT.value,
            EdgeType.DERIVED_FROM.value,
            EdgeType.SUMMARIZES.value,
            EdgeType.REFINES.value,
            EdgeType.SEMANTIC_NEIGHBOR.value,
        }
        topic_raw = float(sum(edge_counts.get(et, 0) for et in topic_types))
        temporal_raw = float(edge_counts.get(EdgeType.TEMPORAL_NEXT.value, 0))
        norm = max(1.0, float(min(len(aliases_a), len(aliases_b))))
        topic_overlap = min(1.0, (topic_raw + 0.25 * temporal_raw) / norm)

        retrieval_co_use = self.db.retrieval_co_use_score(
            s1.branch_id,
            s2.branch_id,
            lookback=self.cfg.merge_signal_lookback,
        )
        return float(topic_overlap), float(retrieval_co_use)

    def _temporal_stratified_indices(self, total: int, target: int) -> list[int]:
        n = max(0, int(total))
        k = max(1, int(target))
        if n <= 0:
            return []
        if k >= n:
            return list(range(n))
        # Deterministic temporal stratification: preserve branch timeline coverage.
        picks = np.linspace(0, n - 1, num=k, dtype=np.int64)
        uniq = sorted({int(x) for x in picks if 0 <= int(x) < n})
        if len(uniq) >= k:
            return uniq[:k]
        used = set(uniq)
        i = 0
        while len(uniq) < k and i < n:
            if i not in used:
                uniq.append(i)
            i += 1
        return sorted(uniq[:k])

    def _compute_contradiction_signals(self, branch_id: str) -> float:
        rows = self.db.list_branch_nodes(branch_id, include_embeddings=True)
        all_node_ids: list[str] = []
        all_vecs: list[np.ndarray] = []
        for row in rows:
            emb = row.get("embedding") or []
            if not emb:
                continue
            try:
                v = np.array(emb, dtype=np.float32)
            except Exception:
                continue
            if v.size <= 0:
                continue
            all_node_ids.append(str(row.get("id")))
            all_vecs.append(v)

        if len(all_vecs) < 2:
            if all_node_ids:
                self.db.remove_edges_for_nodes(all_node_ids, EdgeType.CONTRADICTS)
            return 0.0

        pair_cap = max(1, int(self.cfg.contradiction_edge_max_pairs))
        dynamic_min = max(2, int(math.sqrt(max(2, pair_cap * 2))))
        configured_min = max(2, int(self.cfg.contradiction_sample_min_nodes))
        configured_max = int(self.cfg.contradiction_sample_max_nodes)
        if configured_max > 0:
            sample_target = min(
                len(all_vecs),
                max(dynamic_min, min(configured_max, len(all_vecs)), configured_min),
            )
        else:
            sample_target = min(len(all_vecs), max(dynamic_min, configured_min))

        if len(all_vecs) > sample_target:
            keep_idx = self._temporal_stratified_indices(len(all_vecs), sample_target)
            node_ids = [all_node_ids[i] for i in keep_idx]
            vecs = [all_vecs[i] for i in keep_idx]
        else:
            node_ids = all_node_ids
            vecs = all_vecs

        embs = np.stack(vecs, axis=0)
        norms = np.linalg.norm(embs, axis=1, keepdims=True)
        norms = np.maximum(norms, GeometryMath.EPS)
        normed = embs / norms
        sim = np.matmul(normed, normed.T)

        triu_i, triu_j = np.triu_indices(sim.shape[0], k=1)
        pair_total = len(triu_i)
        if pair_total <= 0:
            return 0.0

        th = float(self.cfg.contradiction_sim_threshold)
        sims = sim[triu_i, triu_j]
        contrad_mask = sims < th
        contrad_count = int(np.count_nonzero(contrad_mask))
        density = float(contrad_count / max(1, pair_total))

        self.db.remove_edges_for_nodes(all_node_ids, EdgeType.CONTRADICTS)
        if contrad_count > 0:
            idx = np.where(contrad_mask)[0]
            if idx.size > 0:
                order = idx[np.argsort(sims[idx])]
                max_pairs = max(0, int(self.cfg.contradiction_edge_max_pairs))
                if max_pairs > 0:
                    order = order[:max_pairs]
                edges: list[tuple[str, str, EdgeType, float]] = []
                for pos in order:
                    a = node_ids[int(triu_i[pos])]
                    b = node_ids[int(triu_j[pos])]
                    w = float(abs(float(sims[pos])))
                    edges.append((a, b, EdgeType.CONTRADICTS, w))
                    edges.append((b, a, EdgeType.CONTRADICTS, w))
                self.db.add_edges_bulk(edges)

        return density

    def _refresh_branch_geometry_from_db(self, branch_id: str) -> Optional[BranchStats]:
        s = self.db.load_branch(branch_id)
        if s is None:
            return None

        embs, weights = self.db.get_branch_embeddings(branch_id)
        if embs.shape[0] <= 0:
            s.node_count = 0
            s.mean_vec = []
            s.anchor = []
            s.cov_diagonal = []
            s.eff_rank = 0.0
            s.trace = 0.0
            s.anisotropy = 0.0
            s.coherence = 0.0
            s.contradiction_density = 0.0
            s.state = BranchState.FORMING
            s.regime = GeometricRegime.PRODUCTIVE
            s.last_update_ts = time.time()
            self.db.upsert_branch(s)
            return s

        mu = GeometryMath.weighted_mean(embs, weights)
        stats_dict = GeometryMath.compute_full_stats(embs, weights, mu)
        var = GeometryMath.covariance_diagonal(embs, weights)

        s.mean_vec = mu.tolist()
        s.anchor = mu.tolist()
        s.anchor_drift = 0.0
        s.cov_diagonal = var.tolist()
        s.node_count = int(embs.shape[0])
        s.eff_rank = float(stats_dict["eff_rank"])
        s.trace = float(stats_dict["trace"])
        s.anisotropy = float(stats_dict["anisotropy"])
        s.coherence = float(stats_dict["coherence"])
        s.contradiction_density = self._compute_contradiction_signals(branch_id)
        s.regime = self.regime.classify(s)
        self._update_state_from_regime(s)
        s.last_update_ts = time.time()
        self.db.upsert_branch(s)
        return s

    def _kmeans2_labels(self, embs: np.ndarray) -> np.ndarray:
        n = int(embs.shape[0])
        if n <= 1:
            return np.zeros(n, dtype=np.int32)

        i0 = 0
        d0 = np.sum((embs - embs[i0]) ** 2, axis=1)
        i1 = int(np.argmax(d0)) if n > 1 else 0
        if i1 == i0:
            i1 = 1 if n > 1 else 0
        centers = np.stack([embs[i0], embs[i1]], axis=0).astype(np.float32)
        labels = np.zeros(n, dtype=np.int32)

        max_iter = max(4, int(self.cfg.split_kmeans_max_iter))
        for _ in range(max_iter):
            dist = np.sum((embs[:, None, :] - centers[None, :, :]) ** 2, axis=2)
            new_labels = np.argmin(dist, axis=1).astype(np.int32)
            if np.array_equal(new_labels, labels):
                break
            labels = new_labels
            for k in (0, 1):
                idx = np.where(labels == k)[0]
                if idx.size <= 0:
                    idx = np.array([int(np.argmax(np.min(dist, axis=1)))])
                    labels[idx[0]] = k
                centers[k] = np.mean(embs[idx], axis=0)
        return labels

    def _apply_split_child_priors(
        self,
        source: BranchStats,
        child_branch_id: str,
        centroid: Optional[np.ndarray] = None,
    ) -> None:
        child = self.db.load_branch(child_branch_id)
        if child is None:
            return

        if bool(self.cfg.split_child_copy_usefulness):
            child.usefulness = float(source.usefulness)
        if bool(self.cfg.split_child_copy_retrieval_error):
            child.retrieval_error = float(source.retrieval_error)

        child.split_counter = 0
        child.merge_counter = 0
        if bool(self.cfg.split_child_anchor_from_centroid) and centroid is not None:
            try:
                c = np.array(centroid, dtype=np.float32)
                if c.size > 0 and np.isfinite(c).all():
                    child.anchor = c.tolist()
                    child.anchor_drift = 0.0
            except Exception:
                pass

        child.last_update_ts = time.time()
        self.db.upsert_branch(child)

    def execute_pending_splits(self, max_jobs: int = 10) -> int:
        pending = [j for j in self.db.pending_jobs() if str(j.get("job_type")) == "split"]
        if not pending:
            return 0

        done = 0
        for job in pending[:max(1, int(max_jobs))]:
            jid = str(job.get("id"))
            branch_id = str(job.get("target_id") or "")
            if not branch_id:
                self.db.complete_job(jid)
                continue

            source = self.db.load_branch(branch_id)
            if source is None:
                self.db.complete_job(jid)
                continue

            rows = self.db.list_branch_nodes(branch_id, include_embeddings=True)
            node_ids: list[str] = []
            vecs: list[np.ndarray] = []
            for row in rows:
                emb = row.get("embedding") or []
                if not emb:
                    continue
                node_ids.append(str(row.get("id")))
                vecs.append(np.array(emb, dtype=np.float32))

            if len(node_ids) < max(4, int(self.cfg.split_min_nodes)):
                source.state = BranchState.ACTIVE if source.state == BranchState.SPLIT_PENDING else source.state
                source.split_counter = max(0, int(source.split_counter) - 1)
                source.last_update_ts = time.time()
                self.db.upsert_branch(source)
                self.db.complete_job(jid)
                continue

            embs = np.stack(vecs, axis=0)
            labels = self._kmeans2_labels(embs)
            idx_a = np.where(labels == 0)[0].tolist()
            idx_b = np.where(labels == 1)[0].tolist()
            if len(idx_a) < 2 or len(idx_b) < 2:
                source.state = BranchState.TENSIONED
                source.last_update_ts = time.time()
                self.db.upsert_branch(source)
                self.db.complete_job(jid)
                continue

            child_a = self._create_branch(NodeType.MESSAGE, near=branch_id)
            child_b = self._create_branch(NodeType.MESSAGE, near=branch_id)

            moved_a = [node_ids[i] for i in idx_a]
            moved_b = [node_ids[i] for i in idx_b]
            self.db.reassign_nodes_to_branch(moved_a, child_a)
            self.db.reassign_nodes_to_branch(moved_b, child_b)
            centroid_a = np.mean(embs[idx_a], axis=0) if idx_a else None
            centroid_b = np.mean(embs[idx_b], axis=0) if idx_b else None

            self.db.add_edge(branch_id, child_a, EdgeType.REFINES, weight=1.0)
            self.db.add_edge(branch_id, child_b, EdgeType.REFINES, weight=1.0)

            self._refresh_branch_geometry_from_db(child_a)
            self._refresh_branch_geometry_from_db(child_b)
            self._apply_split_child_priors(source, child_a, centroid=centroid_a)
            self._apply_split_child_priors(source, child_b, centroid=centroid_b)

            source.state = BranchState.COLLAPSING
            source.node_count = 0
            source.mean_vec = []
            source.anchor = []
            source.cov_diagonal = []
            source.eff_rank = 0.0
            source.trace = 0.0
            source.anisotropy = 0.0
            source.coherence = 0.0
            source.contradiction_density = 0.0
            source.last_update_ts = time.time()
            self.db.upsert_branch(source)

            self.db.complete_job(jid)
            done += 1

        return done

    def execute_pending_merges(self, max_jobs: int = 5) -> int:
        mode = str(getattr(self.cfg, "merge_execution_mode", "soft") or "soft").strip().lower()
        if mode in ("off", "disabled", "none"):
            return 0

        pending = [j for j in self.db.pending_jobs() if str(j.get("job_type")) == "merge"]
        if not pending:
            return 0

        done = 0
        for job in pending[: max(1, int(max_jobs))]:
            jid = str(job.get("id") or "")
            source_id = str(job.get("target_id") or "")
            payload_raw = job.get("payload_json")
            payload: dict[str, Any] = {}
            if isinstance(payload_raw, str) and payload_raw.strip():
                try:
                    payload = json.loads(payload_raw)
                except Exception:
                    payload = {}
            merge_with = str(payload.get("merge_with") or "")

            if not jid:
                continue
            if not source_id or not merge_with or source_id == merge_with:
                self.db.complete_job(jid)
                continue

            s1 = self.db.load_branch(source_id)
            s2 = self.db.load_branch(merge_with)
            if s1 is None or s2 is None:
                self.db.complete_job(jid)
                continue
            if s1.state == BranchState.COLLAPSING or s2.state == BranchState.COLLAPSING:
                self.db.complete_job(jid)
                continue

            if mode == "soft":
                weight = float(payload.get("score") or self.cfg.merge_soft_edge_weight)
                weight = max(0.05, min(3.0, weight))
                # Soft merge: persist explicit affinity edge and reset merge hysteresis.
                self.db.add_edge(source_id, merge_with, EdgeType.SAME_TOPIC, weight=weight)
                self.db.add_edge(merge_with, source_id, EdgeType.SAME_TOPIC, weight=weight)

                now_ts = time.time()
                for s in (s1, s2):
                    if s.state == BranchState.MERGE_CANDIDATE:
                        s.state = BranchState.STABLE
                    s.merge_counter = 0
                    s.last_update_ts = now_ts
                    self.db.upsert_branch(s)

            self.db.complete_job(jid)
            done += 1

        return done

    def mark_branch_agent_interest(self, agent_id: str, branch_id: str, weight: float = 1.0) -> None:
        src = f"agent:{agent_id}"
        self.db.add_edge(src, branch_id, EdgeType.CROSS_AGENT, weight=float(weight))

    def add_cross_agent_shared_edge(self, source_branch_id: str, target_branch_id: str, weight: float = 1.0) -> None:
        self.db.add_edge(source_branch_id, target_branch_id, EdgeType.CROSS_AGENT, weight=float(weight))

    def list_cross_agent_links(self, branch_id: Optional[str] = None, limit: int = 200) -> list[dict[str, Any]]:
        return self.db.list_cross_agent_edges(branch_id=branch_id, limit=limit)

    def health_report(self) -> dict[str, Any]:
        all_stats = self.db.list_branch_scalars()
        state_counts: dict[str, int] = {}
        regime_counts: dict[str, int] = {}
        pending_split = 0
        pending_merge = 0
        mean_coh = 0.0
        mean_comp = 0.0
        total_nodes = 0

        for s in all_stats:
            state = str(s.get("state") or BranchState.FORMING.value)
            regime = str(s.get("regime") or GeometricRegime.PRODUCTIVE.value)
            state_counts[state] = state_counts.get(state, 0) + 1
            regime_counts[regime] = regime_counts.get(regime, 0) + 1
            if state == BranchState.SPLIT_PENDING.value:
                pending_split += 1
            if state == BranchState.MERGE_CANDIDATE.value:
                pending_merge += 1
            mean_coh += float(s.get("coherence") or 0.0)
            mean_comp += float(s.get("compression_loss") or 0.0)
            total_nodes += int(s.get("node_count") or 0)

        n = len(all_stats)
        jobs = self.db.pending_jobs()
        pending_jobs = len(jobs)
        pending_job_types: dict[str, int] = {}
        for j in jobs:
            jt = str(j.get("job_type") or "unknown")
            pending_job_types[jt] = pending_job_types.get(jt, 0) + 1

        return {
            "branches": n,
            "states": state_counts,
            "regimes": regime_counts,
            "pending_split_branches": pending_split,
            "pending_merge_branches": pending_merge,
            "pending_jobs": pending_jobs,
            "pending_job_types": pending_job_types,
            "mean_coherence": (mean_coh / n) if n > 0 else 0.0,
            "mean_compression_loss": (mean_comp / n) if n > 0 else 0.0,
            "total_node_count": total_nodes,
            "cross_agent_edges": len(self.list_cross_agent_links(limit=10000)),
        }

    # ------------------------------------------------------------------
    # Public utility: mark reactivation interest
    # ------------------------------------------------------------------

    def signal_branch_relevance(
        self,
        branch_id: str,
        signal: float = 0.3,
        query_embedding: Optional[list[float]] = None,
    ) -> None:
        """Called when a query semantically aligns with a dormant branch.
        Accumulates reactivation_score adiabatically."""
        s = self.db.load_branch(branch_id)
        if s:
            lam = self.cfg.usefulness_lambda
            s.reactivation_score = (1 - lam) * s.reactivation_score + lam * signal
            q_emb: Optional[np.ndarray] = None
            if query_embedding is not None:
                try:
                    q_emb = np.array(query_embedding, dtype=np.float32)
                except Exception:
                    q_emb = None
            if (
                s.state == BranchState.DORMANT
                and s.reactivation_score > float(self.cfg.reactivation_min_score)
            ):
                if self._reactivation_guard_ok(s, query_embedding=q_emb):
                    s.state = BranchState.REACTIVATING
                else:
                    # Keep signal near threshold if guard fails to avoid unstable wake-flap.
                    s.reactivation_score = min(
                        float(s.reactivation_score),
                        float(self.cfg.reactivation_min_score) * 0.98,
                    )
            self.db.upsert_branch(s)

    # ------------------------------------------------------------------
    # Daily log sidecar (additive; does not change LCM flows)
    # ------------------------------------------------------------------

    def add_daily_log_entry(
        self,
        text: str,
        date_str: Optional[str] = None,
        source: str = "manual_log",
        embedding: Optional[list[float]] = None,
        token_count: Optional[int] = None,
    ) -> dict[str, Any]:
        """
        Append one daily-log entry to branch `day_YYYY-MM-DD`.

        Content text is stored in sidecar table `daily_log_content`,
        while geometry node metadata remains in `memory_nodes`.
        """
        clean_text = str(text or "").strip()
        if not clean_text:
            raise ValueError("add_daily_log_entry requires non-empty text")

        if date_str is None:
            day = time.strftime("%Y-%m-%d", time.localtime())
        else:
            # Validate format strictly to keep branch IDs stable.
            try:
                time.strptime(str(date_str), "%Y-%m-%d")
            except ValueError as exc:
                raise ValueError("date_str must be YYYY-MM-DD") from exc
            day = str(date_str)

        branch_id = f"day_{day}"
        now_ts = time.time()
        node_id = f"node_{day.replace('-', '')}_{int(now_ts * 1000)}"
        lcm_id = f"dailylog_{day.replace('-', '')}_{int(now_ts * 1000)}"

        if embedding is not None:
            emb = np.array(embedding, dtype=np.float32)
        elif self.embedding_provider is not None:
            emb = np.array(self.embedding_provider.embed(clean_text), dtype=np.float32)
        else:
            # Keep operation available even without embedding provider.
            emb = np.zeros(self.cfg.embedding_dim, dtype=np.float32)

        s = self._ensure_branch(branch_id, NodeType.DAILY_LOG_ENTRY)
        s.branch_type = "daily_log"
        if s.state == BranchState.FORMING:
            s.state = BranchState.ACTIVE
        s.last_update_ts = now_ts
        self.db.upsert_branch(s)

        prev_node_id = self.db.get_last_node_id(branch_id)
        n = MemoryNode(
            node_id=node_id,
            lcm_id=lcm_id,
            node_type=NodeType.DAILY_LOG_ENTRY,
            branch_id=branch_id,
            parent_id=None,
            timestamp=now_ts,
            role="system",
            token_count=int(token_count if token_count is not None else max(1, len(clean_text) // 4)),
            embedding=emb.tolist(),
        )
        self.db.insert_node(n)
        self.db.upsert_daily_log_content(
            node_id=node_id,
            text=clean_text,
            source=source,
            created_ts=now_ts,
        )

        if prev_node_id and prev_node_id != node_id:
            self.add_temporal_edge(prev_node_id, node_id)

        updated = self.db.load_branch(branch_id)
        if updated is not None:
            updated.state = BranchState.ACTIVE
            self._update_branch_geometry(updated, emb)

        return {
            "branch_id": branch_id,
            "node_id": node_id,
            "lcm_id": lcm_id,
            "source": source,
            "timestamp": now_ts,
        }

    def daily_log_entries(
        self,
        branch_id: str,
        limit: int = 200,
        max_chars: int = 500,
    ) -> list[dict[str, Any]]:
        if not str(branch_id).startswith("day_"):
            return []
        return self.db.list_daily_log_content(branch_id=branch_id, limit=limit, max_chars=max_chars)

    # ------------------------------------------------------------------
    # Inspection helpers (non-mutating)
    # ------------------------------------------------------------------

    def branch_report(self, branch_id: str) -> Optional[dict[str, Any]]:
        s = self.db.load_branch(branch_id)
        if not s:
            return None
        out: dict[str, Any] = {
            "branch_id":         s.branch_id,
            "state":             s.state.value,
            "regime":            s.regime.value,
            "eff_rank":          round(s.eff_rank, 3),
            "trace":             round(s.trace, 4),
            "anisotropy":        round(s.anisotropy, 3),
            "anchor_drift":      round(s.anchor_drift, 4),
            "coherence":         round(s.coherence, 3),
            "compression_loss":  round(s.compression_loss, 3),
            "node_count":        s.node_count,
            "usefulness":        round(s.usefulness, 3),
            "contradiction_density": round(s.contradiction_density, 3),
            "update_mode_counts": self.db.branch_update_mode_counts(branch_id),
            "correction_counts": self.db.branch_correction_counts(branch_id),
            "recent_corrections": self.db.recent_corrections(branch_id, limit=8),
        }
        if str(branch_id).startswith("day_"):
            out["daily_log_entries"] = self.db.count_daily_log_content(branch_id)
            latest = self.db.latest_daily_log_content(branch_id, max_chars=200)
            if latest:
                out["latest_daily_log"] = latest.get("text", "")
                out["latest_daily_log_ts"] = latest.get("timestamp")
        return out

    def pending_maintenance(self) -> list[dict]:
        return self.db.pending_jobs()

    def split_observations(
        self,
        limit: int = 200,
        run_id: Optional[str] = None,
        branch_id: Optional[str] = None,
    ) -> list[dict[str, Any]]:
        return self.db.list_split_observations(limit=limit, run_id=run_id, branch_id=branch_id)

    def _default_backfill_error_log_path(self) -> str:
        base = os.path.dirname(self.db.db_path) or "."
        return os.path.join(base, "backfill_errors.log")

    def _append_backfill_error(
        self,
        log_path: str,
        *,
        conv_id: int,
        branch_id: str,
        message_count: int,
        exc: Exception,
    ) -> None:
        os.makedirs(os.path.dirname(log_path) or ".", exist_ok=True)
        ts = time.strftime("%Y-%m-%d %H:%M:%S", time.localtime())
        stack = traceback.format_exc().strip()
        with open(log_path, "a", encoding="utf-8") as fh:
            fh.write(
                f"[{ts}] conv_id={conv_id} branch_id={branch_id} "
                f"messages={message_count} error={type(exc).__name__}: {exc}\n"
            )
            if stack:
                fh.write(stack + "\n")
            fh.write("---\n")

    # ------------------------------------------------------------------
    # Fix 3 — Incremental backfill from LCM (resume-safe)
    # ------------------------------------------------------------------

    def backfill_from_lcm(
        self,
        lcm_db_path:  str,
        max_per_conv: int = 200,
        resume:       bool = True,
        progress_cb = None,   # Optional[Callable[[int,int], None]]
        error_log_path: Optional[str] = None,
    ) -> dict[str, int]:
        """
        Batch-import all conversations from an LCM database into the geometry DB.

        Safe to re-run: branches already in branch_states are skipped when resume=True.
        Large conversations are stratified-sampled to max_per_conv messages.

        Requires self.embedding_provider to be set (pass to GeometryController).

        Returns:
            {"processed": N, "skipped": M, "failed": F, "sampled": K, "errors_logged": E}
        """
        lcm = sqlite3.connect(lcm_db_path)
        lcm.row_factory = sqlite3.Row

        rows = lcm.execute(
            "SELECT message_id, conversation_id, role, content, token_count, created_at "
            "FROM messages ORDER BY conversation_id, created_at"
        ).fetchall()

        convs: dict[int, list] = {}
        for r in rows:
            cid = r["conversation_id"]
            if cid not in convs:
                convs[cid] = []
            convs[cid].append(r)

        already_done = (
            {b.branch_id for b in self.db.all_branches()}
            if resume else set()
        )

        total = len(convs)
        stats = {"processed": 0, "skipped": 0, "failed": 0, "sampled": 0, "errors_logged": 0}
        error_log = error_log_path or self._default_backfill_error_log_path()

        for i, (cid, msgs) in enumerate(convs.items()):
            branch_id = f"conv_{cid}"
            if resume and branch_id in already_done:
                stats["skipped"] += 1
                continue

            if len(msgs) > max_per_conv:
                stats["sampled"] += 1
                keep = max(2, max_per_conv)
                step = len(msgs) / keep
                msgs = [msgs[0], msgs[-1]] + [
                    msgs[int(j * step)] for j in range(1, int(keep) - 1)
                ]
                msgs = msgs[:max_per_conv]

            if not msgs:
                continue

            try:
                if self.embedding_provider is None:
                    raise ValueError(
                        "backfill_from_lcm requires self.embedding_provider. "
                        "Pass embedding_provider=EmbeddingProvider() to GeometryController."
                    )
                texts = [m["content"] or "" for m in msgs]
                embeddings = self.embedding_provider.embed_batch(texts)
                for msg, emb in zip(msgs, embeddings):
                    self.on_new_item(
                        lcm_id=msg["message_id"],
                        node_type=NodeType.MESSAGE,
                        embedding=emb,
                        role=msg["role"] or "user",
                        token_count=msg["token_count"] or 0,
                        force_branch_id=branch_id,
                    )
                stats["processed"] += 1
                if progress_cb:
                    progress_cb(i + 1, total)
            except Exception as exc:
                stats["failed"] += 1
                self._append_backfill_error(
                    error_log,
                    conv_id=int(cid),
                    branch_id=branch_id,
                    message_count=len(msgs),
                    exc=exc,
                )
                stats["errors_logged"] += 1

        lcm.close()
        return stats
    # ------------------------------------------------------------------
    # Incremental polling API (real-time LCM ingest)
    # ------------------------------------------------------------------

    def poll_lcm_for_new_items(
        self,
        lcm_db_path: str,
        since_rowid: int = 0,
        limit: int = 200,
        conversation_id: Optional[int] = None,
    ) -> dict[str, Any]:
        """
        Incrementally ingest new LCM messages since a rowid cursor.
        """
        if self.embedding_provider is None:
            raise ValueError(
                "poll_lcm_for_new_items requires self.embedding_provider. "
                "Pass embedding_provider=EmbeddingProvider() to GeometryController."
            )
        with self._poll_lock:
            safe_limit = max(1, int(limit))
            safe_since = max(0, int(since_rowid))
            lcm = sqlite3.connect(lcm_db_path)
            lcm.row_factory = sqlite3.Row

            try:
                if conversation_id is None:
                    rows = lcm.execute(
                        "SELECT rowid AS _rid, message_id, conversation_id, seq, role, content, token_count, created_at "
                        "FROM messages WHERE rowid > ? ORDER BY rowid ASC LIMIT ?",
                        (safe_since, safe_limit),
                    ).fetchall()
                else:
                    rows = lcm.execute(
                        "SELECT rowid AS _rid, message_id, conversation_id, seq, role, content, token_count, created_at "
                        "FROM messages WHERE rowid > ? AND conversation_id = ? ORDER BY rowid ASC LIMIT ?",
                        (safe_since, int(conversation_id), safe_limit),
                    ).fetchall()

                if not rows:
                    return {
                        "polled": 0,
                        "processed": 0,
                        "failed": 0,
                        "skipped_duplicates": 0,
                        "since_rowid": safe_since,
                        "next_rowid": safe_since,
                        "has_more": False,
                    }

                lcm_ids = [str(r["message_id"]) for r in rows if r["message_id"] is not None]
                existing_ids = self.db.existing_lcm_ids(lcm_ids, node_type=NodeType.MESSAGE)

                processed = 0
                failed = 0
                skipped_duplicates = 0
                next_rowid = safe_since

                candidate_rows: list[sqlite3.Row] = []
                for row in rows:
                    rid = int(row["_rid"])
                    next_rowid = max(next_rowid, rid)
                    msg_id = row["message_id"]
                    if msg_id is None:
                        failed += 1
                        continue
                    msg_lcm_id = str(msg_id)
                    if msg_lcm_id in existing_ids:
                        skipped_duplicates += 1
                        continue
                    candidate_rows.append(row)

                if candidate_rows:
                    texts = [(r["content"] or "")[:1000] for r in candidate_rows]
                    embeddings = self.embedding_provider.embed_batch(texts)
                else:
                    embeddings = []

                for row, emb in zip(candidate_rows, embeddings):
                    try:
                        msg_lcm_id = str(row["message_id"])
                        cid = row["conversation_id"]
                        branch_id = f"conv_{int(cid)}" if cid is not None else None
                        self.on_new_item(
                            lcm_id=msg_lcm_id,
                            node_type=NodeType.MESSAGE,
                            embedding=emb,
                            role=row["role"] or "user",
                            token_count=int(row["token_count"] or 0),
                            force_branch_id=branch_id,
                        )
                        existing_ids.add(msg_lcm_id)
                        processed += 1
                    except Exception:
                        failed += 1

                return {
                    "polled": len(rows),
                    "processed": processed,
                    "failed": failed,
                    "skipped_duplicates": skipped_duplicates,
                    "since_rowid": safe_since,
                    "next_rowid": next_rowid,
                    "has_more": len(rows) >= safe_limit,
                }
            finally:
                lcm.close()

    # ------------------------------------------------------------------
    # Fix 4 - Import DAG edges from LCM
    # ------------------------------------------------------------------

    def import_dag_edges_from_lcm(self, lcm_db_path: str) -> dict[str, int]:
        """
        Import summary DAG edges from lcm.db into the geometry DB.

        Reads:
          - summary_parents: parent_id → child_id (DERIVED_FROM edges)
          - summary_messages: summary_id → message_id (SUMMARIZES edges)

        This populates memory_edges so that merge/split scorers can compute
        topic_overlap and retrieval_co_use.

        Returns:
            {"derived_from": N, "summarizes": M, "skipped": K}
        """
        lcm = sqlite3.connect(lcm_db_path)
        lcm.row_factory = sqlite3.Row
        stats = {"derived_from": 0, "summarizes": 0, "skipped": 0}

        # summary_parents: (summary_id, parent_summary_id) — DERIVED_FROM
        for row in lcm.execute("SELECT summary_id, parent_summary_id FROM summary_parents"):
            if row["summary_id"] and row["parent_summary_id"]:
                try:
                    self.db.add_edge(
                        f"sn_{row['parent_summary_id']}", f"sn_{row['summary_id']}",
                        EdgeType.DERIVED_FROM,
                    )
                    stats["derived_from"] += 1
                except Exception:
                    stats["skipped"] += 1

        for row in lcm.execute("SELECT summary_id, message_id FROM summary_messages"):
            if row["summary_id"] and row["message_id"]:
                try:
                    self.db.add_edge(
                        f"sn_{row['summary_id']}", f"msg_{row['message_id']}",
                        EdgeType.SUMMARIZES,
                    )
                    stats["summarizes"] += 1
                except Exception:
                    stats["skipped"] += 1

        lcm.close()
        return stats




# ---------------------------------------------------------------------------
# Quick-start helper
# ---------------------------------------------------------------------------

def create_geometry_controller(
    db_path:            str = "lcm_geometry.db",
    embedding_dim:       int = 384,
    embedding_provider = None,   # Optional[EmbeddingProvider]
    **cfg_overrides:    Any,
) -> GeometryController:
    """
    Convenience factory.

    Example::

        from lcm_geometry_controller import create_geometry_controller, EmbeddingProvider, NodeType

        gc = create_geometry_controller(
            "my_agent.db",
            embedding_provider=EmbeddingProvider(),
        )

        # With text auto-embedding:
        decision = gc.on_new_item(
            lcm_id="msg_001",
            node_type=NodeType.MESSAGE,
            text="Hello, world!",          # ← auto-embeds via provider
            role="user",
            token_count=5,
        )
        print(decision.action, decision.branch_id)

        # With pre-computed embedding:
        decision = gc.on_new_item(
            lcm_id="msg_002",
            node_type=NodeType.MESSAGE,
            embedding=my_embed_fn("What's the weather?"),
            role="user",
            token_count=6,
        )
    """
    cfg = GeometryConfig(embedding_dim=embedding_dim, **cfg_overrides)
    return GeometryController(db_path, cfg, embedding_provider=embedding_provider)


# ---------------------------------------------------------------------------
# CLI smoke-test
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    import tempfile, os

    print("=== LCM Geometry Controller – smoke test ===\n")

    with tempfile.TemporaryDirectory() as td:
        db = os.path.join(td, "test.db")
        gc = create_geometry_controller(db, embedding_dim=8)

        rng = np.random.default_rng(42)

        def fake_embed(seed: int) -> list[float]:
            rng2 = np.random.default_rng(seed)
            v = rng2.standard_normal(8).astype(np.float32)
            return (v / (np.linalg.norm(v) + 1e-9)).tolist()

        # Ingest 12 similar messages → should attach to one branch
        branch_ids = set()
        for i in range(12):
            d = gc.on_new_item(
                lcm_id=f"msg_{i:03}",
                node_type=NodeType.MESSAGE,
                embedding=fake_embed(i % 3),  # 3 tight clusters
                role="user",
                token_count=20 + i,
                conflict_score=0.0,
            )
            branch_ids.add(d.branch_id)
            print(f"  msg_{i:03} → {d.action:17s} branch={d.branch_id} csd={d.csd_score:.3f}")

        print(f"\n  Distinct branches used: {len(branch_ids)}")

        # Force a divergent message → should fork
        d_fork = gc.on_new_item(
            lcm_id="msg_divergent",
            node_type=NodeType.MESSAGE,
            embedding=fake_embed(99),
            role="assistant",
            token_count=100,
            conflict_score=0.8,
        )
        print(f"\n  Divergent msg → {d_fork.action} (expected fork or attach_tension)")

        # Rank retrieval
        ranked = gc.rank_retrieval(fake_embed(1))
        print(f"\n  Retrieval ranking (top 3):")
        for r in ranked[:3]:
            print(f"    branch={r.branch_id}  total={r.total_score:.3f}")

        # Maintenance
        counts = gc.run_maintenance_cycle()
        print(f"\n  Maintenance cycle: {counts}")

        # Report on first branch
        bid = list(branch_ids)[0]
        rpt = gc.branch_report(bid)
        print(f"\n  Branch report ({bid}):")
        for k, v in rpt.items():
            print(f"    {k}: {v}")

    print("\n=== smoke test passed ===")
