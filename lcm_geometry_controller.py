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
import sqlite3
import time
import threading
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


class EdgeType(str, Enum):
    SUMMARIZES       = "summarizes"
    DERIVED_FROM     = "derived_from"
    TEMPORAL_NEXT    = "temporal_next"
    SEMANTIC_NEIGHBOR = "semantic_neighbor"
    CONTRADICTS      = "contradicts"
    REFINES          = "refines"
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

    # ── Retrieval ranker ───────────────────────────────────────────────────
    alpha_sem:   float = 0.60
    beta_trust:  float = 0.25
    delta_react: float = 0.15
    kappa_coherence:     float = 0.30
    kappa_comp_loss:     float = 0.20
    kappa_contradiction: float = 0.25
    kappa_ret_error:     float = 0.25

    # ── Split scorer ───────────────────────────────────────────────────────
    split_hysteresis: int = 3
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


@dataclass
class AllocationDecision:
    action:         str          # "attach" | "attach_tension" | "fork"
    branch_id:      str
    csd_score:      float
    conflict_score: float
    rationale:      str


@dataclass
class RetrievalCandidate:
    branch_id:   str
    sem_score:   float
    trust_score: float
    react_score: float
    total_score: float


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

CREATE INDEX IF NOT EXISTS idx_nodes_branch ON memory_nodes(branch_id);
CREATE INDEX IF NOT EXISTS idx_nodes_lcm    ON memory_nodes(lcm_id);
CREATE INDEX IF NOT EXISTS idx_edges_src    ON memory_edges(src_id);
CREATE INDEX IF NOT EXISTS idx_feedback_branch ON retrieval_feedback(branch_id);
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

    def __init__(self, cfg: GeometryConfig):
        self.cfg = cfg
        # Per-branch EMA of historical deformations (in-memory only; reset on restart)
        self._history: dict[str, dict[str, float]] = {}

    def _get_ema(self, branch_id: str, key: str) -> float:
        return self._history.get(branch_id, {}).get(key, 0.0)

    def _update_ema(self, branch_id: str, key: str, value: float) -> None:
        if branch_id not in self._history:
            self._history[branch_id] = {}
        prev = self._history[branch_id].get(key, value)
        self._history[branch_id][key] = (1 - self.cfg.stat_rho) * prev + self.cfg.stat_rho * value

    def _residual(self, delta: float, branch_id: str, key: str) -> float:
        expected = self._get_ema(branch_id, key) + self.cfg.EPS if hasattr(self.cfg, "EPS") else delta + 1e-9
        # fallback: use small constant to avoid div-by-zero on first call
        expected = max(expected, 1e-6)
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
        csd = (
            c.gamma_mu       * R_mu
            + c.gamma_r      * R_r
            + c.gamma_A      * R_A
            + c.gamma_tau    * R_tau
            + c.gamma_sem    * sem_dist
            + c.gamma_conflict * conflict
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
    ) -> str:
        """'attach' | 'attach_tension' | 'fork'"""
        c = cfg or self.cfg
        if csd_score < c.attach_threshold:
            return "attach"
        if csd_score < c.tension_threshold:
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
        cos = GeometryMath.cosine_sim(mu1, mu2)
        drift_mismatch = abs(s1.anchor_drift - s2.anchor_drift)
        score = (
            c.eta_cos          * cos
            + c.eta_topic      * topic_overlap
            + c.eta_co_use     * retrieval_co_use
            + c.eta_contradiction * ((s1.contradiction_density + s2.contradiction_density) / 2)
            + c.eta_drift      * drift_mismatch
        )
        return float(score)

    def should_merge(self, s1: BranchStats, s2: BranchStats, score: float) -> bool:
        states_ok = s1.state in (BranchState.STABLE, BranchState.DORMANT)
        states_ok = states_ok and s2.state in (BranchState.STABLE, BranchState.DORMANT)
        return states_ok and score > 0.55


class SplitScorer:
    def __init__(self, cfg: GeometryConfig):
        self.cfg = cfg

    def score(self, s: BranchStats) -> float:
        c = self.cfg
        return float(
            c.zeta_incoherence   * (1.0 - s.coherence)
            + c.zeta_anisotropy  * s.anisotropy
            + c.zeta_comp_loss   * s.compression_loss
            + c.zeta_ret_error   * s.retrieval_error
            + c.zeta_contradiction * s.contradiction_density
        )

    def should_split(self, s: BranchStats, score: float, cfg: GeometryConfig) -> bool:
        return (
            score > 0.55
            and s.split_counter >= cfg.split_hysteresis
        )


# ---------------------------------------------------------------------------
# Retrieval Ranker
# ---------------------------------------------------------------------------

class RetrievalRanker:
    def __init__(self, cfg: GeometryConfig):
        self.cfg = cfg

    def rank(
        self,
        query_emb: np.ndarray,
        candidates: list[BranchStats],
        historical_use: Optional[dict[str, float]] = None,
        same_project: Optional[dict[str, float]] = None,
    ) -> list[RetrievalCandidate]:
        results = []
        c = self.cfg
        for s in candidates:
            if not s.mean_vec:
                continue
            mu = np.array(s.mean_vec, dtype=np.float32)

            sem   = GeometryMath.cosine_sim(query_emb, mu)
            trust = (
                c.kappa_coherence     * s.coherence
                + c.kappa_comp_loss   * s.compression_loss
                + c.kappa_contradiction * s.contradiction_density
                + c.kappa_ret_error   * s.retrieval_error
            )
            hist  = (historical_use or {}).get(s.branch_id, 0.0)
            proj  = (same_project  or {}).get(s.branch_id, 0.0)
            react = (
                0.5 * sem
                + c.kappa_coherence * hist     # reuse kappa slot
                + 0.3 * proj
            )

            total = c.alpha_sem * sem + c.beta_trust * trust + c.delta_react * react

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
            self._conn.commit()
        return self._conn

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
                 retrieval_error, usefulness,
                 split_counter, merge_counter, last_update_ts)
            VALUES
                (:branch_id, :branch_type, :state, :regime,
                 :mean_vec, :anchor, :cov_diagonal,
                 :eff_rank, :trace, :anisotropy, :anchor_drift,
                 :coherence, :compression_loss,
                 :node_count, :contradiction_density,
                 :retrieval_error, :usefulness,
                 :split_counter, :merge_counter, :last_update_ts)
        """, d)
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

    # ---- Nodes ----

    def insert_node(self, n: MemoryNode) -> None:
        emb_blob = json.dumps(n.embedding).encode() if n.embedding else None
        self.conn.execute("""
            INSERT OR REPLACE INTO memory_nodes
            (id, lcm_id, node_type, parent_id, branch_id, timestamp,
             role, token_count, embedding,
             importance_score, novelty_score, conflict_score,
             coherence_score, compression_loss, reactivation_score,
             stability_state)
            VALUES (?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?)
        """, (
            n.node_id, n.lcm_id, n.node_type, n.parent_id, n.branch_id,
            n.timestamp, n.role, n.token_count, emb_blob,
            n.importance_score, n.novelty_score, n.conflict_score,
            n.coherence_score, n.compression_loss, n.reactivation_score,
            n.stability_state,
        ))
        self.conn.commit()

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

    # ---- Edges ----

    def add_edge(self, src: str, dst: str, etype: EdgeType, weight: float = 1.0) -> None:
        self.conn.execute("""
            INSERT OR REPLACE INTO memory_edges (src_id, dst_id, edge_type, weight)
            VALUES (?, ?, ?, ?)
        """, (src, dst, etype.value, weight))
        self.conn.commit()

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
        self.csd               = CSDScorer(self.cfg)
        self.merger            = MergeScorer(self.cfg)
        self.splitter          = SplitScorer(self.cfg)
        self.ranker            = RetrievalRanker(self.cfg)
        self.health            = SummaryHealthChecker(self.cfg)
        self.regime            = RegimeClassifier(self.cfg)
        self.embedding_provider = embedding_provider  # may be None
        self.db.connect()

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
            action    = self.csd.decide(best_csd, self.cfg)
            branch_id = best_branch.branch_id

            if action == "fork":
                branch_id = self._create_branch(node_type, near=best_branch.branch_id)
            elif action == "attach_tension":
                self._set_branch_state(best_branch.branch_id, BranchState.TENSIONED)

        # Persist node
        node = MemoryNode(
            node_id    = str(uuid.uuid4()),
            lcm_id     = lcm_id,
            node_type  = node_type,
            branch_id  = branch_id,
            parent_id  = None,   # geometry layer doesn't know LCM parent yet
            timestamp  = time.time(),
            role       = role,
            token_count = token_count,
            embedding  = emb.tolist(),
            conflict_score = conflict_score,
        )
        self.db.insert_node(node)

        # Add temporal edge from previous node if known
        # (caller may also call add_temporal_edge directly)

        # Adiabatically update branch geometry
        updated = self.db.load_branch(branch_id)
        if updated:
            self._update_branch_geometry(updated, emb)

        rationale = (
            f"CSD={best_csd:.3f} → {action}; "
            f"conflict={conflict_score:.2f}; branch={branch_id}"
        )

        return AllocationDecision(
            action         = action,
            branch_id      = branch_id,
            csd_score      = best_csd,
            conflict_score = conflict_score,
            rationale      = rationale,
        )

    # ------------------------------------------------------------------
    # 2. Retrieval ranking
    # ------------------------------------------------------------------

    def rank_retrieval(
        self,
        query_embedding: list[float],
        historical_use:  Optional[dict[str, float]] = None,
        same_project:    Optional[dict[str, float]] = None,
    ) -> list[RetrievalCandidate]:
        """
        Returns branches sorted by retrieval priority.

        Top branches should be queried first via lcm_describe / lcm_grep.
        Only delegate lcm_expand to sub-agents for branches deep in the list.
        """
        all_stats = self.db.all_branches()
        q_emb = np.array(query_embedding, dtype=np.float32)
        return self.ranker.rank(q_emb, all_stats, historical_use, same_project)

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

    def run_maintenance_cycle(self) -> dict[str, int]:
        """
        Runs all offline maintenance jobs:
          - geometry_recompute
          - split_scan
          - merge_scan
          - reactivation_scan
          - summary_audit (jobs queued by audit_summary)

        Returns counts of actions taken.
        """
        actions = {"recomputed": 0, "split_pending": 0,
                   "merge_candidates": 0, "reactivated": 0}

        all_stats = self.db.all_branches()

        for s in all_stats:
            if s is None:
                continue

            # --- Recompute geometry from stored embeddings ---
            embs, weights = self.db.get_branch_embeddings(s.branch_id)
            if embs.shape[0] >= self.cfg.min_branch_size:
                mu   = GeometryMath.weighted_mean(embs, weights)
                stats_dict = GeometryMath.compute_full_stats(embs, weights, mu)
                # Adiabatic update of all geometry fields
                rho = self.cfg.stat_rho
                s.mean_vec   = mu.tolist()
                s.eff_rank   = (1-rho)*s.eff_rank   + rho*stats_dict["eff_rank"]
                s.trace      = (1-rho)*s.trace      + rho*stats_dict["trace"]
                s.anisotropy = (1-rho)*s.anisotropy + rho*stats_dict["anisotropy"]
                s.coherence  = (1-rho)*s.coherence  + rho*stats_dict["coherence"]
                # anchor
                if s.anchor:
                    a_old  = np.array(s.anchor, dtype=np.float32)
                    a_new  = (1 - self.cfg.anchor_alpha)*a_old + self.cfg.anchor_alpha*mu
                    s.anchor       = a_new.tolist()
                    s.anchor_drift = float(np.linalg.norm(a_new - a_old))
                else:
                    s.anchor       = mu.tolist()
                    s.anchor_drift = 0.0

                var = GeometryMath.covariance_diagonal(embs, weights)
                s.cov_diagonal = var.tolist()
                s.node_count   = embs.shape[0]
                s.regime       = self.regime.classify(s)

                # Update lifecycle state based on regime
                self._update_state_from_regime(s)
                s.last_update_ts = time.time()
                self.db.upsert_branch(s)
                actions["recomputed"] += 1

            elif s.mean_vec:
                # Fallback: reclassify regime from stored scalar metrics.
                # This runs for branches whose geometry was set at backfill time
                # but have no (or few) memory_nodes rows — common after backfill.
                s.regime = self.regime.classify(s)
                self._update_state_from_regime(s)
                s.last_update_ts = time.time()
                self.db.upsert_branch(s)
                actions["recomputed"] += 1


            # --- Split scan ---
            split_s = self.splitter.score(s)
            if split_s > 0.55:
                s.split_counter += 1
            else:
                s.split_counter = max(0, s.split_counter - 1)

            if self.splitter.should_split(s, split_s, self.cfg):
                s.state = BranchState.SPLIT_PENDING
                self.db.enqueue_job("split", target_id=s.branch_id,
                                    payload={"split_score": split_s})
                actions["split_pending"] += 1

            self.db.upsert_branch(s)

        # --- Merge scan (pairwise, O(n²) — only on small/stable set) ---
        stable = [s for s in all_stats
                  if s and s.state in (BranchState.STABLE, BranchState.DORMANT)]
        for i, s1 in enumerate(stable):
            for s2 in stable[i+1:]:
                ms = self.merger.score(s1, s2)
                if self.merger.should_merge(s1, s2, ms):
                    s1.merge_counter += 1
                    s2.merge_counter += 1
                    if (s1.merge_counter >= self.cfg.merge_hysteresis
                            and s2.merge_counter >= self.cfg.merge_hysteresis):
                        s1.state = BranchState.MERGE_CANDIDATE
                        s2.state = BranchState.MERGE_CANDIDATE
                        self.db.enqueue_job(
                            "merge",
                            target_id=s1.branch_id,
                            payload={"merge_with": s2.branch_id, "score": ms},
                        )
                        actions["merge_candidates"] += 1
                    self.db.upsert_branch(s1)
                    self.db.upsert_branch(s2)

        # --- Reactivation scan ---
        dormant = [s for s in all_stats if s and s.state == BranchState.DORMANT]
        for s in dormant:
            if s.reactivation_score > 0.6:
                s.state = BranchState.REACTIVATING
                self.db.upsert_branch(s)
                actions["reactivated"] += 1

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

    def _next_conv_branch_id(self) -> str:
        max_conv = 0
        for stats in self.db.all_branches():
            if not stats:
                continue
            bid = stats.branch_id or ""
            if not bid.startswith("conv_"):
                continue
            suffix = bid[5:]
            if suffix.isdigit():
                max_conv = max(max_conv, int(suffix))
        return f"conv_{max_conv + 1}"

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

    def _get_candidate_branches(
        self, active_branch_id: Optional[str]
    ) -> list[BranchStats]:
        all_b = self.db.all_branches()
        # Exclude COLLAPSING, SPLIT_PENDING (no new items into broken branches)
        excluded = {BranchState.COLLAPSING, BranchState.SPLIT_PENDING}
        candidates = [b for b in all_b if b and b.state not in excluded]
        # Prioritise the currently active branch (put it first)
        if active_branch_id:
            candidates.sort(key=lambda b: 0 if b.branch_id == active_branch_id else 1)
        return candidates[:10]  # cap at 10 for performance

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

    # ------------------------------------------------------------------
    # Public utility: mark reactivation interest
    # ------------------------------------------------------------------

    def signal_branch_relevance(self, branch_id: str, signal: float = 0.3) -> None:
        """Called when a query semantically aligns with a dormant branch.
        Accumulates reactivation_score adiabatically."""
        s = self.db.load_branch(branch_id)
        if s:
            lam = self.cfg.usefulness_lambda
            s.reactivation_score = (1 - lam) * s.reactivation_score + lam * signal
            if s.state == BranchState.DORMANT and s.reactivation_score > 0.6:
                s.state = BranchState.REACTIVATING
            self.db.upsert_branch(s)

    # ------------------------------------------------------------------
    # Inspection helpers (non-mutating)
    # ------------------------------------------------------------------

    def branch_report(self, branch_id: str) -> Optional[dict[str, Any]]:
        s = self.db.load_branch(branch_id)
        if not s:
            return None
        return {
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
        }

    def pending_maintenance(self) -> list[dict]:
        return self.db.pending_jobs()

    # ------------------------------------------------------------------
    # Fix 3 — Incremental backfill from LCM (resume-safe)
    # ------------------------------------------------------------------

    def backfill_from_lcm(
        self,
        lcm_db_path:  str,
        max_per_conv: int = 200,
        resume:       bool = True,
        progress_cb = None,   # Optional[Callable[[int,int], None]]
    ) -> dict[str, int]:
        """
        Batch-import all conversations from an LCM database into the geometry DB.

        Safe to re-run: branches already in branch_states are skipped when resume=True.
        Large conversations are stratified-sampled to max_per_conv messages.

        Requires self.embedding_provider to be set (pass to GeometryController).

        Returns:
            {"processed": N, "skipped": M, "failed": 0, "sampled": K}
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
        stats = {"processed": 0, "skipped": 0, "failed": 0, "sampled": 0}

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
                        active_branch_id=branch_id,
                    )
                stats["processed"] += 1
                if progress_cb:
                    progress_cb(i + 1, total)
            except Exception:
                stats["failed"] += 1

        lcm.close()
        return stats

    # ------------------------------------------------------------------
    # Fix 4 — Import DAG edges from LCM
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
