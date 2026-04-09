#!/usr/bin/env python3
"""
LCM Geometry MCP Server  v1.2
Exposes geometry + LCM hybrid search as an MCP tool for OpenClaw.

Changelog:
  v1.0  hybrid_search, branch_report, geometry_stats
  v1.1  added conversation_content (bridges geometry  LCM text)
  v1.2  added runtime config loading + daily-log support

Usage:
    Test locally: python3 server.py
    Wire into OpenClaw:
        openclaw mcp set geometry-hybrid '{"command": "python3", "args": ["<openclaw_home>/extensions/geometry-mcp/server.py"]}'
        openclaw gateway restart
"""
import sys
import os
import json
import time
import threading
import shutil

# Resolve workspace/module path  going up from extensions/geometry-mcp/  extensions/  ~/.openclaw/
SKILL_DIR = os.path.dirname(os.path.abspath(__file__))
OPENCLAW_HOME = os.path.dirname(os.path.dirname(SKILL_DIR))  # ~/.openclaw
WORKSPACE_MODULE = os.path.join(OPENCLAW_HOME, 'workspace', 'module')
sys.path.insert(0, WORKSPACE_MODULE)

from mcp.server import Server
from mcp.server.stdio import stdio_server
from mcp.types import Tool, TextContent
import asyncio
import sqlite3
from collections import defaultdict

# DB paths
DEFAULT_GEO_DB = os.path.join(OPENCLAW_HOME, 'lcm_geometry.db')
DEFAULT_LCM_DB = os.path.join(OPENCLAW_HOME, 'lcm.db')
RUNTIME_CONFIG_PATH = os.path.join(SKILL_DIR, "runtime_config.json")


def _read_json_file(path):
    if not os.path.isfile(path):
        return {}
    try:
        with open(path, "r", encoding="utf-8") as f:
            data = json.load(f)
        return data if isinstance(data, dict) else {}
    except Exception as exc:
        print(f"[geometry-mcp] Failed to read runtime config file {path}: {exc}", file=sys.stderr)
        return {}


def _deep_merge_dict(base, updates):
    out = dict(base or {})
    for k, v in (updates or {}).items():
        if isinstance(v, dict) and isinstance(out.get(k), dict):
            out[k] = _deep_merge_dict(out[k], v)
        else:
            out[k] = v
    return out


def _load_runtime_config():
    cfg = _read_json_file(RUNTIME_CONFIG_PATH)
    raw = os.getenv("GEOMETRY_RUNTIME_CONFIG_JSON", "").strip()
    if raw:
        try:
            env_cfg = json.loads(raw)
            if isinstance(env_cfg, dict):
                cfg = _deep_merge_dict(cfg, env_cfg)
        except Exception as exc:
            print(f"[geometry-mcp] Invalid GEOMETRY_RUNTIME_CONFIG_JSON: {exc}", file=sys.stderr)
    return cfg


_RUNTIME_CFG = _load_runtime_config()
_PATHS_CFG = _RUNTIME_CFG.get("paths", {}) if isinstance(_RUNTIME_CFG.get("paths"), dict) else {}
GEO_DB = str(_PATHS_CFG.get("geo_db") or _RUNTIME_CFG.get("geo_db") or DEFAULT_GEO_DB)
LCM_DB = str(_PATHS_CFG.get("lcm_db") or _RUNTIME_CFG.get("lcm_db") or DEFAULT_LCM_DB)
EMBED_MODEL_NAME = str(_RUNTIME_CFG.get("embedding_model") or "all-MiniLM-L6-v2")
_GEOMETRY_CFG_OVERRIDES = _RUNTIME_CFG.get("geometry_config", {})
if not isinstance(_GEOMETRY_CFG_OVERRIDES, dict):
    _GEOMETRY_CFG_OVERRIDES = {}
_POLLING_CFG = _RUNTIME_CFG.get("polling", {})
if not isinstance(_POLLING_CFG, dict):
    _POLLING_CFG = {}

POLLING_ENABLED = bool(_POLLING_CFG.get("enabled", True))
POLLING_INTERVAL_SEC = float(_POLLING_CFG.get("interval_seconds", 8.0))
POLLING_LIMIT = int(_POLLING_CFG.get("limit", 200))
POLLING_CONVERSATION_ID = _POLLING_CFG.get("conversation_id")
POLLING_CURSOR_PATH = str(
    _POLLING_CFG.get("cursor_path") or os.path.join(SKILL_DIR, "poll_cursor.json")
)
POLLING_SHOW_STATUS = bool(_POLLING_CFG.get("show_status", True))
POLLING_DEBUG_LOG = bool(_POLLING_CFG.get("debug_log", False))

print(
    f"[geometry-mcp] runtime cfg loaded: geo_db={GEO_DB} lcm_db={LCM_DB} "
    f"model={EMBED_MODEL_NAME} geometry_overrides={len(_GEOMETRY_CFG_OVERRIDES)} "
    f"polling_enabled={POLLING_ENABLED} interval={POLLING_INTERVAL_SEC}s limit={POLLING_LIMIT}",
    file=sys.stderr,
)

#  Lazy-load heavy libs 
_model = None
_gc = None
_poll_lock = threading.Lock()
_last_poll_ts = 0.0

def get_model():
    global _model
    if _model is None:
        from sentence_transformers import SentenceTransformer
        _model = SentenceTransformer(EMBED_MODEL_NAME)
    return _model

def get_gc():
    global _gc
    if _gc is None:
        from lcm_geometry_controller import GeometryController, GeometryConfig, EmbeddingProvider
        provider = EmbeddingProvider(
            model_name=EMBED_MODEL_NAME,
            device=str(os.getenv("GEOMETRY_EMBED_DEVICE", "cpu") or "cpu"),
        )
        if _GEOMETRY_CFG_OVERRIDES:
            valid_keys = set(getattr(GeometryConfig, "__dataclass_fields__", {}).keys())
            safe_overrides = {}
            for k, v in _GEOMETRY_CFG_OVERRIDES.items():
                if k in valid_keys:
                    safe_overrides[k] = v
                else:
                    print(f"[geometry-mcp] Ignoring unknown geometry_config key: {k}", file=sys.stderr)
            _gc = GeometryController(
                GEO_DB,
                cfg=GeometryConfig(**safe_overrides),
                embedding_provider=provider,
            )
        else:
            _gc = GeometryController(GEO_DB, embedding_provider=provider)
    return _gc


def _poll_normalized_conversation_id():
    cid = POLLING_CONVERSATION_ID
    if cid is None or cid == "":
        return None
    try:
        return int(cid)
    except Exception:
        return None


def _load_poll_cursor() -> int:
    try:
        if not os.path.isfile(POLLING_CURSOR_PATH):
            return 0
        with open(POLLING_CURSOR_PATH, "r", encoding="utf-8") as f:
            obj = json.load(f)
        return max(0, int(obj.get("last_rowid", 0)))
    except Exception:
        return 0


def _save_poll_cursor(rowid: int) -> None:
    try:
        os.makedirs(os.path.dirname(POLLING_CURSOR_PATH) or ".", exist_ok=True)
        tmp = f"{POLLING_CURSOR_PATH}.tmp"
        payload = {
            "last_rowid": max(0, int(rowid)),
            "updated_at": time.time(),
        }
        with open(tmp, "w", encoding="utf-8") as f:
            json.dump(payload, f, ensure_ascii=False, indent=2)
            f.write("\n")
        os.replace(tmp, POLLING_CURSOR_PATH)
    except Exception as exc:
        if POLLING_DEBUG_LOG:
            print(f"[geometry-mcp] Failed to save poll cursor: {exc}", file=sys.stderr)


def _lcm_max_rowid() -> int:
    conn = sqlite3.connect(LCM_DB)
    try:
        row = conn.execute("SELECT MAX(rowid) AS m FROM messages").fetchone()
        if row is None:
            return 0
        val = row[0]
        return max(0, int(val or 0))
    finally:
        conn.close()


def _with_poll_lag(status: dict, cursor_rowid: int | None = None) -> dict:
    out = dict(status or {})
    try:
        max_rowid = _lcm_max_rowid()
        if cursor_rowid is None:
            cursor_rowid = int(out.get("next_rowid", out.get("since_rowid", _load_poll_cursor())) or 0)
        cursor_rowid = max(0, int(cursor_rowid))
        out["cursor_rowid"] = cursor_rowid
        out["lcm_max_rowid"] = max_rowid
        out["lag_rows"] = max(0, int(max_rowid - cursor_rowid))
    except Exception as exc:
        if POLLING_DEBUG_LOG:
            print(f"[geometry-mcp] Failed to compute poll lag: {exc}", file=sys.stderr)
    return out


def _backup_geo_db_copy() -> str:
    ts = time.strftime("%Y%m%d_%H%M%S")
    dst = f"{GEO_DB}.backup_dag_sync_{ts}"
    shutil.copy2(GEO_DB, dst)
    return dst


def _dag_edge_validation_summary() -> dict:
    conn = sqlite3.connect(GEO_DB)
    conn.row_factory = sqlite3.Row
    try:
        out = {
            "by_type": {},
            "orphan_by_type": {},
            "total_edges": 0,
        }
        for et in ("summarizes", "derived_from", "temporal_next", "refines"):
            total = conn.execute(
                "SELECT COUNT(*) AS c FROM memory_edges WHERE edge_type=?",
                (et,),
            ).fetchone()["c"]
            orphan = conn.execute(
                """
                SELECT COUNT(*) AS c
                FROM memory_edges e
                LEFT JOIN memory_nodes s ON s.id=e.src_id
                LEFT JOIN memory_nodes d ON d.id=e.dst_id
                WHERE e.edge_type=? AND (s.id IS NULL OR d.id IS NULL)
                """,
                (et,),
            ).fetchone()["c"]
            out["by_type"][et] = int(total or 0)
            out["orphan_by_type"][et] = int(orphan or 0)
            out["total_edges"] += int(total or 0)
        return out
    finally:
        conn.close()


def _sync_lcm_dag_edges(backup: bool = True) -> dict:
    with _poll_lock:
        backup_path = None
        if backup:
            backup_path = _backup_geo_db_copy()
        gc = get_gc()
        import_stats = gc.import_dag_edges_from_lcm(LCM_DB)
        validation = _dag_edge_validation_summary()
        return {
            "ok": True,
            "backup_path": backup_path,
            "import_stats": import_stats,
            "validation": validation,
        }


def _poll_lcm_if_due(force: bool = False, limit_override: int | None = None) -> dict:
    global _last_poll_ts
    if not POLLING_ENABLED:
        return _with_poll_lag({"enabled": False, "skipped": "disabled"})

    now = time.time()
    with _poll_lock:
        elapsed = now - float(_last_poll_ts or 0.0)
        min_interval = max(0.0, float(POLLING_INTERVAL_SEC))
        if not force and min_interval > 0.0 and elapsed < min_interval:
            return _with_poll_lag({
                "enabled": True,
                "skipped": "cooldown",
                "next_in_seconds": round(max(0.0, min_interval - elapsed), 3),
            })

        gc = get_gc()
        since_rowid = _load_poll_cursor()
        safe_limit = max(1, int(limit_override if limit_override is not None else POLLING_LIMIT))
        conv_id = _poll_normalized_conversation_id()
        try:
            out = gc.poll_lcm_for_new_items(
                lcm_db_path=LCM_DB,
                since_rowid=since_rowid,
                limit=safe_limit,
                conversation_id=conv_id,
            )
            next_rowid = int(out.get("next_rowid", since_rowid) or since_rowid)
            if next_rowid > since_rowid:
                _save_poll_cursor(next_rowid)
            _last_poll_ts = now
            out["enabled"] = True
            out["cursor_path"] = POLLING_CURSOR_PATH
            out["conversation_id"] = conv_id
            return _with_poll_lag(out, cursor_rowid=next_rowid)
        except Exception as exc:
            _last_poll_ts = now
            return _with_poll_lag({
                "enabled": True,
                "error": str(exc),
                "since_rowid": since_rowid,
                "cursor_path": POLLING_CURSOR_PATH,
            }, cursor_rowid=since_rowid)


def _poll_status_line(status: dict | None) -> str:
    if not isinstance(status, dict):
        return ""
    if not POLLING_SHOW_STATUS:
        return ""
    if status.get("skipped") == "disabled":
        return ""
    if status.get("error"):
        lag = f" lag_rows={int(status.get('lag_rows', 0) or 0)}" if "lag_rows" in status else ""
        return f" Live ingest: error={status['error']}{lag}"
    if status.get("skipped"):
        if status.get("skipped") == "cooldown":
            lag = f" lag_rows={int(status.get('lag_rows', 0) or 0)}" if "lag_rows" in status else ""
            return f" Live ingest: skipped=cooldown next_in={status.get('next_in_seconds', 0)}s{lag}"
        lag = f" lag_rows={int(status.get('lag_rows', 0) or 0)}" if "lag_rows" in status else ""
        return f" Live ingest: skipped={status.get('skipped')}{lag}"
    skipped_dup = int(status.get("skipped_duplicates", 0) or 0)
    lag = int(status.get("lag_rows", 0) or 0)
    return (
        " Live ingest: "
        f"polled={int(status.get('polled', 0) or 0)} "
        f"processed={int(status.get('processed', 0) or 0)} "
        f"failed={int(status.get('failed', 0) or 0)} "
        f"skipped_dup={skipped_dup} "
        f"next_rowid={int(status.get('next_rowid', 0) or 0)} "
        f"lag_rows={lag}"
    )


def _embed_query(query: str) -> list[float]:
    gc = get_gc()
    provider = getattr(gc, "embedding_provider", None)
    if provider is not None:
        try:
            return list(provider.embed(query))
        except Exception as exc:
            if POLLING_DEBUG_LOG:
                print(f"[geometry-mcp] Provider query embed fallback due to: {exc}", file=sys.stderr)
    model = get_model()
    return model.encode([query], normalize_embeddings=True)[0].tolist()


def _get_daily_log_content(gdb_conn, branch_id, max_entries, max_chars):
    rows = gdb_conn.execute(
        """
        SELECT mn.id AS node_id,
               mn.timestamp AS created_at,
               dl.source AS source,
               SUBSTR(dl.text, 1, ?) AS text
        FROM memory_nodes mn
        JOIN daily_log_content dl ON dl.node_id = mn.id
        WHERE mn.branch_id = ?
        ORDER BY mn.timestamp ASC, mn.rowid ASC
        LIMIT ?
        """,
        (max_chars, branch_id, max_entries),
    ).fetchall()
    return [
        {
            "type": "daily_log",
            "source": r["source"],
            "created_at": r["created_at"],
            "text": r["text"],
            "node_id": r["node_id"],
        }
        for r in rows
    ]


def _rank_collapsing_sidecar(gc, q_emb, top_n=5, retrieval_mode="balanced"):
    try:
        import numpy as np
    except Exception:
        return []

    safe_n = max(0, int(top_n or 0))
    if safe_n <= 0:
        return []

    collapsing = []
    for s in gc.db.all_branches():
        state_val = str(getattr(getattr(s, "state", None), "value", getattr(s, "state", "")) or "")
        if state_val != "COLLAPSING":
            continue
        if int(getattr(s, "node_count", 0) or 0) <= 0:
            continue
        mean_vec = getattr(s, "mean_vec", None) or []
        if not mean_vec:
            continue
        collapsing.append(s)

    if not collapsing:
        return []

    q = np.array(q_emb, dtype=np.float32)
    ranked = gc.ranker.rank(q, collapsing, retrieval_mode=retrieval_mode)
    by_id = {s.branch_id: s for s in collapsing}
    out = []
    for r in ranked[:safe_n]:
        s = by_id.get(r.branch_id)
        out.append({
            "branch_id": r.branch_id,
            "total_score": round(r.total_score, 4),
            "sem_score": round(r.sem_score, 4),
            "trust_score": round(r.trust_score, 4),
            "nodes": int(getattr(s, "node_count", 0) or 0) if s else 0,
            "coherence": round(float(getattr(s, "coherence", 0.0) or 0.0), 4) if s else 0.0,
            "eff_rank": round(float(getattr(s, "eff_rank", 0.0) or 0.0), 2) if s else 0.0,
            "state": str(getattr(getattr(s, "state", None), "value", getattr(s, "state", "unknown")) or "unknown"),
            "regime": str(getattr(getattr(s, "regime", None), "value", getattr(s, "regime", "unknown")) or "unknown"),
            "note": "excluded_from_primary_rank",
        })
    return out

#  Hybrid search 
def do_hybrid_search(query, top_n=5, retrieval_mode="balanced"):
    gc = get_gc()

    # Encode query
    q_emb = _embed_query(query)

    # Geometry ranking
    ranked = gc.rank_retrieval(q_emb, retrieval_mode=retrieval_mode)
    geo_results = []
    for r in ranked[:top_n]:
        b = gc.db.load_branch(r.branch_id)
        geo_results.append({
            'branch_id': r.branch_id,
            'total_score': round(r.total_score, 4),
            'sem_score': round(r.sem_score, 4),
            'trust_score': round(r.trust_score, 4),
            'nodes': b.node_count if b else 0,
            'coherence': round(b.coherence, 4) if b else 0,
            'eff_rank': round(b.eff_rank, 2) if b else 0,
            'state': b.state.value if b and b.state else 'unknown',
            'regime': b.regime.value if b and b.regime else 'unknown',
        })
    collapsing_sidecar = _rank_collapsing_sidecar(
        gc,
        q_emb,
        top_n=top_n,
        retrieval_mode=retrieval_mode,
    )

    # LCM keyword search
    keywords = [w.strip() for w in query.split() if len(w.strip()) >= 3]
    conn = sqlite3.connect(LCM_DB)
    conn.row_factory = sqlite3.Row
    gconn = sqlite3.connect(GEO_DB)
    gconn.row_factory = sqlite3.Row

    results_by_conv = defaultdict(list)
    for kw in keywords:
        cur = conn.execute('''
            SELECT message_id, conversation_id, role,
                   SUBSTR(content, 1, 120) as snippet, token_count, created_at
            FROM messages
            WHERE content LIKE ?
            ORDER BY created_at DESC
            LIMIT 50
        ''', [f'%{kw}%'])
        for r in cur.fetchall():
            results_by_conv[r['conversation_id']].append({
                'message_id': r['message_id'],
                'role': r['role'],
                'snippet': r['snippet'][:100],
                'keyword_matched': kw,
                'created_at': r['created_at']
            })
    conn.close()

    scored = []
    for conv_id, matches in results_by_conv.items():
        most_recent = max(m['created_at'] for m in matches)
        scored.append({
            'conv_id': conv_id,
            'match_count': len(matches),
            'unique_keywords_matched': len(set(m['keyword_matched'] for m in matches)),
            'total_matches': len(matches),
            'best_snippet': matches[0]['snippet'][:80],
        })
    scored.sort(key=lambda x: (x['unique_keywords_matched'], x['total_matches']), reverse=True)
    lcm_results = scored[:top_n]

    # Daily log sidecar search (keyword + semantic)
    daily_keyword = []
    for kw in keywords:
        rows = gconn.execute(
            """
            SELECT mn.branch_id, mn.id AS node_id, mn.timestamp AS created_at,
                   SUBSTR(dl.text, 1, 140) AS snippet
            FROM daily_log_content dl
            JOIN memory_nodes mn ON mn.id = dl.node_id
            WHERE dl.text LIKE ?
            ORDER BY mn.timestamp DESC, mn.rowid DESC
            LIMIT ?
            """,
            (f"%{kw}%", max(5, int(top_n) * 4)),
        ).fetchall()
        for r in rows:
            daily_keyword.append({
                "branch_id": r["branch_id"],
                "node_id": r["node_id"],
                "created_at": r["created_at"],
                "keyword_matched": kw,
                "snippet": r["snippet"],
            })
    # De-duplicate by node_id preserving first hit.
    seen_nodes = set()
    dedup_keyword = []
    for row in daily_keyword:
        nid = row["node_id"]
        if nid in seen_nodes:
            continue
        seen_nodes.add(nid)
        dedup_keyword.append(row)
    daily_keyword = dedup_keyword[:top_n]

    daily_semantic = []
    try:
        import numpy as np
        q = np.array(q_emb, dtype=np.float32)
        rows = gconn.execute(
            """
            SELECT mn.branch_id, mn.id AS node_id, mn.timestamp AS created_at,
                   mn.embedding AS embedding,
                   SUBSTR(dl.text, 1, 140) AS snippet
            FROM daily_log_content dl
            JOIN memory_nodes mn ON mn.id = dl.node_id
            WHERE mn.embedding IS NOT NULL
            ORDER BY mn.timestamp DESC, mn.rowid DESC
            LIMIT 2000
            """
        ).fetchall()
        for r in rows:
            blob = r["embedding"]
            if not blob:
                continue
            try:
                v = np.array(json.loads(blob.decode()), dtype=np.float32)
            except Exception:
                continue
            den = (np.linalg.norm(q) * np.linalg.norm(v))
            if den <= 1e-9:
                continue
            sim = float(np.dot(q, v) / den)
            daily_semantic.append({
                "branch_id": r["branch_id"],
                "node_id": r["node_id"],
                "created_at": r["created_at"],
                "sem_score": round(sim, 4),
                "snippet": r["snippet"],
            })
        daily_semantic.sort(key=lambda x: x["sem_score"], reverse=True)
        daily_semantic = daily_semantic[:top_n]
    except Exception:
        daily_semantic = []
    finally:
        gconn.close()

    # Recommendation
    lcm_count = len(lcm_results)
    geo_top = geo_results[0]['sem_score'] if geo_results else 0

    if lcm_count == 0 and geo_top > 0.3:
        recommendation = "geometry"
    elif geo_top < 0.25:
        recommendation = "lcm"
    elif lcm_count > 0 and geo_top > 0.3:
        recommendation = "both"
    else:
        recommendation = "geometry"

    return {
        'query': query,
        'retrieval_mode': retrieval_mode,
        'recommendation': recommendation,
        'geometry': {'results': geo_results},
        'collapsing_sidecar': {
            'note': 'COLLAPSING branches are excluded from primary geometry ranking; shown as fallback only.',
            'results': collapsing_sidecar,
        },
        'lcm': {'conversations': lcm_results, 'keywords': keywords},
        'daily_logs': {
            'keyword_results': daily_keyword,
            'semantic_results': daily_semantic,
        },
    }

#  Branch report 
def do_branch_report(branch_id):
    gc = get_gc()
    rpt = gc.branch_report(branch_id)
    if not rpt or 'error' in rpt:
        return rpt
    # Also get node info from DB
    conn = sqlite3.connect(GEO_DB)
    conn.row_factory = sqlite3.Row
    cur = conn.execute("SELECT COUNT(*) as n FROM memory_nodes WHERE branch_id = ?", (branch_id,))
    node_count = cur.fetchone()['n']
    daily_count = 0
    latest_daily = None
    if str(branch_id).startswith("day_"):
        cur = conn.execute(
            """
            SELECT COUNT(*) AS n
            FROM daily_log_content dl
            JOIN memory_nodes mn ON mn.id = dl.node_id
            WHERE mn.branch_id = ?
            """,
            (branch_id,),
        )
        daily_count = cur.fetchone()["n"]
        cur = conn.execute(
            """
            SELECT SUBSTR(dl.text, 1, 160) AS text
            FROM daily_log_content dl
            JOIN memory_nodes mn ON mn.id = dl.node_id
            WHERE mn.branch_id = ?
            ORDER BY mn.timestamp DESC, mn.rowid DESC
            LIMIT 1
            """,
            (branch_id,),
        )
        row = cur.fetchone()
        latest_daily = row["text"] if row else None
    conn.close()
    out = {**rpt, 'db_node_count': node_count}
    if str(branch_id).startswith("day_"):
        out["daily_log_entries"] = int(daily_count)
        if latest_daily:
            out["latest_daily_log"] = latest_daily
    return out

#  Geometry stats 
def do_geometry_stats():
    conn = sqlite3.connect(GEO_DB)
    conn.row_factory = sqlite3.Row

    branches = conn.execute("SELECT COUNT(*) FROM branch_states").fetchone()[0]
    nodes = conn.execute("SELECT COUNT(*) FROM memory_nodes").fetchone()[0]

    states = {r['state']: r['cnt'] for r in conn.execute(
        "SELECT state, COUNT(*) as cnt FROM branch_states GROUP BY state").fetchall()}
    regimes = {r['regime']: r['cnt'] for r in conn.execute(
        "SELECT regime, COUNT(*) as cnt FROM branch_states GROUP BY regime").fetchall()}

    r = conn.execute("SELECT AVG(eff_rank) as a, AVG(coherence) as c FROM branch_states").fetchone()
    avg_rank = round(r['a'], 2) if r['a'] else 0
    avg_coh = round(r['c'], 4) if r['c'] else 0

    conn.close()
    return {
        'total_branches': branches,
        'total_nodes': nodes,
        'states': states,
        'regimes': regimes,
        'avg_eff_rank': avg_rank,
        'avg_coherence': avg_coh,
        'embedding_model': EMBED_MODEL_NAME,
        'embedding_dim': 384
    }


#  Conversation content (geometry  LCM bridge) 
def _get_lcm_content(lcm_conn, conv_id, content_type, max_entries, max_chars):
    """Get content from LCM database for a conversation."""
    entries = []
    if content_type in ("summaries", "both"):
        cur = lcm_conn.execute(
            "SELECT kind, SUBSTR(content, 1, ?) as text "
            "FROM summaries WHERE conversation_id = ? ORDER BY created_at",
            (max_chars, conv_id)
        )
        for r in cur.fetchall():
            entries.append({
                "type": "summary",
                "kind": r["kind"],
                "conversation_id": int(conv_id),
                "text": r["text"],
            })

    if content_type in ("messages", "both"):
        cur = lcm_conn.execute(
            "SELECT role, SUBSTR(content, 1, ?) as text, created_at "
            "FROM messages WHERE conversation_id = ? ORDER BY seq LIMIT ?",
            (max_chars, conv_id, max_entries)
        )
        for r in cur.fetchall():
            entries.append({
                "type": "message",
                "role": r["role"],
                "conversation_id": int(conv_id),
                "created_at": r["created_at"],
                "text": r["text"],
            })

    return entries


def _parse_conv_branch_id(branch_id):
    try:
        text = str(branch_id or "")
        if not text.startswith("conv_"):
            return None
        return int(text.replace("conv_", "", 1))
    except Exception:
        return None


def _chunked(seq, size=400):
    safe = max(1, int(size))
    for i in range(0, len(seq), safe):
        yield seq[i:i + safe]


def _ordered_unique(seq):
    out = []
    seen = set()
    for x in seq:
        if x in seen:
            continue
        seen.add(x)
        out.append(x)
    return out


def _append_resolution_warning(meta, warning_text):
    if not warning_text:
        return
    existing = str(meta.get("warning", "") or "").strip()
    if not existing:
        meta["warning"] = str(warning_text)
    elif warning_text not in existing:
        meta["warning"] = f"{existing}; {warning_text}"


def _resolve_branch_lcm_ids(gdb_conn, branch_id):
    rows = gdb_conn.execute(
        "SELECT lcm_id, node_type FROM memory_nodes "
        "WHERE branch_id = ? ORDER BY timestamp ASC, rowid ASC",
        (branch_id,),
    ).fetchall()
    message_ids = []
    summary_ids = []
    for r in rows:
        lcm_id = r["lcm_id"]
        node_type = str(r["node_type"] or "").strip().lower()
        if lcm_id is None:
            continue
        lid = str(lcm_id).strip()
        if not lid:
            continue
        if node_type == "message":
            try:
                message_ids.append(int(lid))
            except Exception:
                continue
            continue
        if node_type in ("summary", "leaf_summary", "condensed_summary"):
            summary_ids.append(lid)
            continue
        if lid.startswith("sum_"):
            summary_ids.append(lid)
            continue
        try:
            message_ids.append(int(lid))
        except Exception:
            continue
    return _ordered_unique(message_ids), _ordered_unique(summary_ids)


def _count_resolved_conversations(lcm_conn, message_ids, summary_ids):
    counts = defaultdict(int)
    for chunk in _chunked(message_ids):
        placeholders = ",".join("?" for _ in chunk)
        sql = (
            "SELECT conversation_id, COUNT(*) AS n FROM messages "
            f"WHERE message_id IN ({placeholders}) GROUP BY conversation_id"
        )
        for r in lcm_conn.execute(sql, chunk).fetchall():
            counts[int(r["conversation_id"])] += int(r["n"] or 0)
    for chunk in _chunked(summary_ids):
        placeholders = ",".join("?" for _ in chunk)
        sql = (
            "SELECT conversation_id, COUNT(*) AS n FROM summaries "
            f"WHERE summary_id IN ({placeholders}) GROUP BY conversation_id"
        )
        for r in lcm_conn.execute(sql, chunk).fetchall():
            counts[int(r["conversation_id"])] += int(r["n"] or 0)
    return counts


def _get_branch_lineage_content(gdb_conn, lcm_conn, branch_id, content_type, max_entries, max_chars):
    message_ids, summary_ids = _resolve_branch_lcm_ids(gdb_conn, branch_id)
    conv_counts = _count_resolved_conversations(lcm_conn, message_ids, summary_ids)
    ordered_conv_ids = [
        int(k) for k, _v in sorted(conv_counts.items(), key=lambda kv: (-kv[1], kv[0]))
    ]

    meta = {
        "resolution_mode": "branch_lineage",
        "resolved_conversation_ids": ordered_conv_ids,
    }

    entries = []
    if content_type in ("summaries", "both"):
        summary_rows = []
        for chunk in _chunked(summary_ids):
            placeholders = ",".join("?" for _ in chunk)
            sql = (
                "SELECT summary_id, conversation_id, kind, created_at, "
                "SUBSTR(content, 1, ?) AS text "
                f"FROM summaries WHERE summary_id IN ({placeholders})"
            )
            params = [max_chars]
            params.extend(chunk)
            summary_rows.extend(lcm_conn.execute(sql, params).fetchall())
        summary_rows.sort(key=lambda r: (str(r["created_at"] or ""), str(r["summary_id"] or "")))
        for r in summary_rows:
            entries.append({
                "type": "summary",
                "kind": r["kind"],
                "conversation_id": int(r["conversation_id"]),
                "text": r["text"],
            })

    if content_type in ("messages", "both"):
        message_rows = []
        for chunk in _chunked(message_ids):
            placeholders = ",".join("?" for _ in chunk)
            sql = (
                "SELECT message_id, conversation_id, role, seq, created_at, "
                "SUBSTR(content, 1, ?) AS text "
                f"FROM messages WHERE message_id IN ({placeholders})"
            )
            params = [max_chars]
            params.extend(chunk)
            message_rows.extend(lcm_conn.execute(sql, params).fetchall())
        message_rows.sort(key=lambda r: (str(r["created_at"] or ""), int(r["seq"] or 0)))
        for r in message_rows:
            entries.append({
                "type": "message",
                "role": r["role"],
                "conversation_id": int(r["conversation_id"]),
                "created_at": r["created_at"],
                "text": r["text"],
            })

    if len(ordered_conv_ids) > 1:
        _append_resolution_warning(
            meta,
            f"mixed_branch_content:{','.join(str(x) for x in ordered_conv_ids[:8])}",
        )

    suffix_conv_id = _parse_conv_branch_id(branch_id)
    if suffix_conv_id is not None and ordered_conv_ids:
        if suffix_conv_id != ordered_conv_ids[0]:
            _append_resolution_warning(
                meta,
                f"branch_suffix_mismatch:conv_{suffix_conv_id}->conv_{ordered_conv_ids[0]}",
            )

    # Fallback path for branches with empty lineage (e.g. collapsed roots).
    if not entries and suffix_conv_id is not None:
        fallback_entries = _get_lcm_content(
            lcm_conn, suffix_conv_id, content_type, max_entries=max_entries, max_chars=max_chars
        )
        if fallback_entries:
            meta["resolution_mode"] = "suffix_fallback"
            meta["resolved_conversation_ids"] = [suffix_conv_id]
            _append_resolution_warning(meta, "lineage_empty_used_suffix_fallback")
            entries = fallback_entries

    return entries[:max_entries], meta

def do_conversation_content(branch_id=None, state=None, content_type="summaries", max_entries=100, max_chars=250):
    """Retrieve actual conversation text from LCM for geometry-identified branches.

    Modes:
      - Single branch:   branch_id="conv_148"
      - By state:        state="ACTIVE" | "STABLE" | "FORMING" | "ALL"
      - All branches:    no filters (implicit state="ALL")
    """
    gdb = sqlite3.connect(GEO_DB)
    gdb.row_factory = sqlite3.Row
    lcm = sqlite3.connect(LCM_DB)
    lcm.row_factory = sqlite3.Row

    results = []

    if branch_id:
        # Single branch mode
        b = gdb.execute("SELECT * FROM branch_states WHERE branch_id = ?", (branch_id,)).fetchone()
        if not b:
            gdb.close(); lcm.close()
            return {"error": f"Branch {branch_id} not found"}
        if str(branch_id).startswith("day_") or content_type == "logs":
            entries = _get_daily_log_content(gdb, branch_id, max_entries, max_chars)
            resolution_meta = {
                "resolution_mode": "daily_log",
                "resolved_conversation_ids": [],
            }
        else:
            entries, resolution_meta = _get_branch_lineage_content(
                gdb, lcm, branch_id, content_type, max_entries, max_chars
            )
        results.append({
            "branch_id": branch_id,
            "state": b["state"],
            "regime": b["regime"],
            "entries_returned": len(entries),
            "resolution_mode": resolution_meta.get("resolution_mode", "unknown"),
            "resolved_conversation_ids": resolution_meta.get("resolved_conversation_ids", []),
            "warning": resolution_meta.get("warning"),
            "content": entries
        })
    else:
        # Multi-branch mode (filtered by state or ALL)
        filter_state = (state or "ALL").upper()
        if content_type == "logs":
            if filter_state != "ALL":
                branches = gdb.execute(
                    "SELECT branch_id, state, regime FROM branch_states "
                    "WHERE state = ? AND branch_id LIKE 'day_%' ORDER BY node_count DESC",
                    (filter_state,),
                ).fetchall()
            else:
                branches = gdb.execute(
                    "SELECT branch_id, state, regime FROM branch_states "
                    "WHERE branch_id LIKE 'day_%' ORDER BY node_count DESC"
                ).fetchall()
        else:
            if filter_state != "ALL":
                branches = gdb.execute(
                    "SELECT branch_id, state, regime FROM branch_states "
                    "WHERE state = ? AND branch_id LIKE 'conv_%' ORDER BY node_count DESC",
                    (filter_state,),
                ).fetchall()
            else:
                branches = gdb.execute(
                    "SELECT branch_id, state, regime FROM branch_states "
                    "WHERE branch_id LIKE 'conv_%' ORDER BY node_count DESC"
                ).fetchall()

        if not branches:
            gdb.close(); lcm.close()
            return {"error": f"No branches found{f' with state={state}' if state else ''}"}

        per_branch = max(max_entries // len(branches), 3)
        for b in branches:
            if str(b["branch_id"]).startswith("day_") or content_type == "logs":
                entries = _get_daily_log_content(gdb, b["branch_id"], per_branch, max_chars)
                resolution_meta = {
                    "resolution_mode": "daily_log",
                    "resolved_conversation_ids": [],
                }
            else:
                entries, resolution_meta = _get_branch_lineage_content(
                    gdb, lcm, b["branch_id"], content_type, per_branch, max_chars
                )
            results.append({
                "branch_id": b["branch_id"],
                "state": b["state"],
                "regime": b["regime"],
                "entries_returned": len(entries),
                "resolution_mode": resolution_meta.get("resolution_mode", "unknown"),
                "resolved_conversation_ids": resolution_meta.get("resolved_conversation_ids", []),
                "warning": resolution_meta.get("warning"),
                "content": entries
            })

    gdb.close()
    lcm.close()

    total_entries = sum(r["entries_returned"] for r in results)

    # Trim if overflow
    if total_entries > max_entries:
        trimmed = []
        remaining = max_entries
        for r in results:
            if remaining <= 0:
                break
            keep = r["content"][:remaining]
            trimmed.append({**r, "content": keep, "entries_returned": len(keep)})
            remaining -= len(keep)
        results = trimmed
        total_entries = sum(r["entries_returned"] for r in results)

    return {
        "mode": "single" if branch_id else "multi",
        "content_type": content_type,
        "state_filter": state or "ALL",
        "branches_returned": len(results),
        "total_entries": total_entries,
        "results": results
    }


#  MCP Server 
server = Server("geometry-hybrid")

@server.list_tools()
async def list_tools():
    return [
        Tool(
            name="hybrid_search",
            description=(
                "Search both LCM (keyword) and Geometry DB (semantic similarity). "
                "Use for recall, topic exploration, finding related conversations. "
                "Returns combined results from both systems with a recommendation on which to trust."
            ),
            inputSchema={
                "type": "object",
                "properties": {
                    "query": {"type": "string", "description": "Search query  question, topic, or keyword phrase"},
                    "top_n": {"type": "integer", "description": "Results per system (default 5)"},
                    "retrieval_mode": {
                        "type": "string",
                        "enum": ["balanced", "factual", "exploratory"],
                        "description": "Geometry ranking mode. factual=more reliability filtering, exploratory=more novelty tolerance."
                    }
                },
                "required": ["query"]
            }
        ),
        Tool(
            name="branch_report",
            description="Get detailed geometry metrics for a specific branch (e.g. 'conv_186' or 'day_2026-04-07').",
            inputSchema={
                "type": "object",
                "properties": {
                    "branch_id": {"type": "string", "description": "Branch ID like 'conv_186' or 'day_YYYY-MM-DD'"}
                },
                "required": ["branch_id"]
            }
        ),
        Tool(
            name="geometry_stats",
            description="Get overall geometry database statistics  branch count, state distribution, average metrics.",
            inputSchema={"type": "object", "properties": {}}
        ),
        Tool(
            name="sync_lcm_ingest",
            description=(
                "Force an immediate incremental poll of new LCM messages into geometry DB. "
                "Uses persistent rowid cursor and returns ingest status."
            ),
            inputSchema={
                "type": "object",
                "properties": {
                    "limit": {
                        "type": "integer",
                        "description": "Optional max messages to ingest in this forced poll (default runtime polling.limit)."
                    }
                }
            }
        ),
        Tool(
            name="sync_lcm_dag_edges",
            description=(
                "Rebuild summary DAG edges (summarizes/derived_from) from lcm.db into geometry DB "
                "using real geometry node IDs, then return validation counters (including orphan counts)."
            ),
            inputSchema={
                "type": "object",
                "properties": {
                    "backup": {
                        "type": "boolean",
                        "description": "Create a DB backup before rebuild (default true)."
                    }
                }
            }
        ),
        Tool(
            name="conversation_content",
            description=(
                "Retrieve actual conversation text from the LCM database for geometry-identified branches. "
                "Bridges the gap between geometry metadata (branch IDs, scores) and real text content. "
                "Modes: (1) single branch by ID, (2) all branches filtered by state (ACTIVE/STABLE/FORMING/ALL). "
                "Default returns summaries only to keep output compact."
            ),
            inputSchema={
                "type": "object",
                "properties": {
                    "branch_id": {
                        "type": "string",
                        "description": "Single branch ID like 'conv_148'. If set, state filter is ignored."
                    },
                    "state": {
                        "type": "string",
                        "enum": ["ACTIVE", "STABLE", "FORMING", "ALL"],
                        "description": "Filter by branch lifecycle state. Default: ALL."
                    },
                    "content_type": {
                        "type": "string",
                        "enum": ["summaries", "messages", "both", "logs"],
                        "description": "What to retrieve. Default: summaries (compact)."
                    },
                    "max_entries": {
                        "type": "integer",
                        "description": "Maximum total entries across all results (default 100)."
                    },
                    "max_chars": {
                        "type": "integer",
                        "description": "Max characters per entry snippet (default 250)."
                    }
                }
            }
        )
    ]


@server.call_tool()
async def call_tool(name, arguments):
    try:
        auto_poll_status = None
        if name not in ("sync_lcm_ingest", "sync_lcm_dag_edges"):
            auto_poll_status = _poll_lcm_if_due(force=False)
        poll_line = _poll_status_line(auto_poll_status)

        if name == "hybrid_search":
            query = arguments.get("query", "")
            top_n = arguments.get("top_n", 5)
            retrieval_mode = str(arguments.get("retrieval_mode", "balanced") or "balanced").strip().lower()
            if retrieval_mode not in ("balanced", "factual", "exploratory"):
                retrieval_mode = "balanced"
            result = do_hybrid_search(query, top_n, retrieval_mode=retrieval_mode)

            lines = [
                f" Hybrid Search: \"{query}\"",
                f" Retrieval mode: {result.get('retrieval_mode', 'balanced')}",
                f" Recommendation: use {result['recommendation'].upper()}",
                "",
            ]
            if poll_line:
                lines.insert(1, poll_line)
            lines.append(" GEOMETRY DB (semantic similarity):")
            for i, r in enumerate(result['geometry']['results'], 1):
                lines.append(f"  {i}. {r['branch_id']} | sem={r['sem_score']} | trust={r['trust_score']} | nodes={r['nodes']} | eff_rank={r['eff_rank']} | {r['state']}/{r['regime']}")
            lines.append("")
            lines.append(" COLLAPSING SIDECAR (excluded from primary geometry ranking):")
            sidecar_rows = result.get("collapsing_sidecar", {}).get("results", [])
            if sidecar_rows:
                for i, r in enumerate(sidecar_rows, 1):
                    lines.append(
                        f"  {i}. {r['branch_id']} | sem={r['sem_score']} | trust={r['trust_score']} "
                        f"| nodes={r['nodes']} | eff_rank={r['eff_rank']} | {r['state']}/{r['regime']}"
                    )
            else:
                lines.append("  (no collapsing candidates)")
            lines.append("")
            lines.append(" LCM (keyword matches):")
            for i, c in enumerate(result['lcm']['conversations'], 1):
                lines.append(f"  {i}. conv_{c['conv_id']} | {c['unique_keywords_matched']} kw matched | {c['total_matches']} total hits")
                lines.append(f"     \"{c['best_snippet']}\"")
            lines.append("")
            lines.append(" DAILY LOGS (sidecar):")
            sem_rows = result.get("daily_logs", {}).get("semantic_results", [])
            kw_rows = result.get("daily_logs", {}).get("keyword_results", [])
            if sem_rows:
                lines.append("  Semantic:")
                for i, r in enumerate(sem_rows, 1):
                    lines.append(f"   {i}. {r['branch_id']} | sem={r['sem_score']} | {r['snippet']}")
            if kw_rows:
                lines.append("  Keyword:")
                for i, r in enumerate(kw_rows, 1):
                    lines.append(f"   {i}. {r['branch_id']} | kw={r['keyword_matched']} | {r['snippet']}")
            if not sem_rows and not kw_rows:
                lines.append("  (no daily-log matches)")
            return [TextContent(type="text", text="\n".join(lines))]

        elif name == "branch_report":
            branch_id = arguments.get("branch_id", "")
            rpt = do_branch_report(branch_id)
            if 'error' in rpt:
                return [TextContent(type="text", text=f"Error: {rpt['error']}")]
            lines = [f" Branch Report: {rpt['branch_id']}"]
            if poll_line:
                lines.append(poll_line)
            for k, v in rpt.items():
                if k != 'branch_id':
                    lines.append(f"  {k}: {v}")
            return [TextContent(type="text", text="\n".join(lines))]

        elif name == "geometry_stats":
            stats = do_geometry_stats()
            lines = [" Geometry DB Stats",
                     f"  Branches: {stats['total_branches']}",
                     f"  Nodes: {stats['total_nodes']}",
                     f"  States: {stats['states']}",
                     f"  Regimes: {stats['regimes']}",
                     f"  Avg eff_rank: {stats['avg_eff_rank']}",
                     f"  Avg coherence: {stats['avg_coherence']}",
                     f"  Embedding: {stats['embedding_model']} ({stats['embedding_dim']}d)"]
            if poll_line:
                lines.insert(1, poll_line)
            return [TextContent(type="text", text="\n".join(lines))]

        elif name == "sync_lcm_ingest":
            limit_raw = arguments.get("limit")
            limit = None
            if limit_raw is not None:
                try:
                    limit = max(1, int(limit_raw))
                except Exception:
                    limit = None
            status = _poll_lcm_if_due(force=True, limit_override=limit)
            lines = [" LCM Ingest Sync"]
            lines.append(_poll_status_line(status) or " Live ingest: no status")
            lines.append(f"  Cursor path: {status.get('cursor_path', POLLING_CURSOR_PATH)}")
            if "since_rowid" in status:
                lines.append(f"  Since rowid: {status.get('since_rowid')}")
            if "next_rowid" in status:
                lines.append(f"  Next rowid: {status.get('next_rowid')}")
            if "has_more" in status:
                lines.append(f"  Has more: {status.get('has_more')}")
            if "lcm_max_rowid" in status:
                lines.append(f"  LCM max rowid: {status.get('lcm_max_rowid')}")
            if "lag_rows" in status:
                lines.append(f"  Lag rows: {status.get('lag_rows')}")
            if "skipped_duplicates" in status:
                lines.append(f"  Skipped duplicates: {status.get('skipped_duplicates')}")
            return [TextContent(type="text", text="\n".join(lines))]

        elif name == "sync_lcm_dag_edges":
            backup = arguments.get("backup", True)
            do_backup = True if backup is None else bool(backup)
            result = _sync_lcm_dag_edges(backup=do_backup)
            if not result.get("ok"):
                return [TextContent(type="text", text=f"Error: {result.get('error', 'unknown error')}")]
            stats = result.get("import_stats", {})
            val = result.get("validation", {})
            by_type = val.get("by_type", {})
            orphan = val.get("orphan_by_type", {})
            lines = [" LCM DAG Edge Sync"]
            lines.append(f"  Backup: {result.get('backup_path') or '(disabled)'}")
            lines.append(
                "  Imported: "
                f"summarizes={stats.get('summarizes', 0)} "
                f"derived_from={stats.get('derived_from', 0)} "
                f"skipped={stats.get('skipped', 0)} "
                f"purged={stats.get('purged', 0)}"
            )
            lines.append(
                "  Indexed nodes: "
                f"summary={stats.get('summary_nodes_indexed', 0)} "
                f"message={stats.get('message_nodes_indexed', 0)}"
            )
            lines.append(
                "  Validation totals: "
                f"summarizes={by_type.get('summarizes', 0)} "
                f"derived_from={by_type.get('derived_from', 0)} "
                f"temporal_next={by_type.get('temporal_next', 0)} "
                f"refines={by_type.get('refines', 0)}"
            )
            lines.append(
                "  Validation orphans: "
                f"summarizes={orphan.get('summarizes', 0)} "
                f"derived_from={orphan.get('derived_from', 0)} "
                f"temporal_next={orphan.get('temporal_next', 0)} "
                f"refines={orphan.get('refines', 0)}"
            )
            return [TextContent(type="text", text="\n".join(lines))]

        elif name == "conversation_content":
            branch_id = arguments.get("branch_id")
            state = arguments.get("state")
            content_type = arguments.get("content_type", "summaries")
            max_entries = arguments.get("max_entries", 100)
            max_chars = arguments.get("max_chars", 250)

            result = do_conversation_content(branch_id, state, content_type, max_entries, max_chars)

            if 'error' in result:
                return [TextContent(type="text", text=f"Error: {result['error']}")]

            lines = [
                f" Conversation Content",
                f"  Mode: {result['mode']}",
                f"  Content type: {result['content_type']}",
                f"  State filter: {result['state_filter']}",
                f"  Branches: {result['branches_returned']}",
                f"  Total entries: {result['total_entries']}",
                ""
            ]
            if poll_line:
                lines.insert(1, poll_line)

            for r in result['results']:
                resolved_ids = r.get("resolved_conversation_ids") or []
                resolved_text = ",".join(str(x) for x in resolved_ids[:8]) if resolved_ids else "-"
                lines.append(
                    f"--- {r['branch_id']} | {r['state']}/{r['regime']} | "
                    f"{r['entries_returned']} entries | mode={r.get('resolution_mode','unknown')} "
                    f"| resolved_conv={resolved_text} ---"
                )
                if r.get("warning"):
                    lines.append(f"  [warning] {r['warning']}")
                for e in r['content']:
                    if e['type'] == 'summary':
                        conv_note = f" conv={e['conversation_id']}" if "conversation_id" in e else ""
                        lines.append(f"\n  [{e['type'].upper()}/{e['kind']}{conv_note}] {e['text']}")
                    elif e['type'] == 'daily_log':
                        lines.append(f"\n  [DAILY_LOG/{e.get('source','manual_log')}] {e['text']}")
                    else:
                        conv_note = f" conv={e['conversation_id']}" if "conversation_id" in e else ""
                        lines.append(f"\n  [{e['role']}{conv_note}] {e['text']}")
                lines.append("")

            return [TextContent(type="text", text="\n".join(lines))]

        else:
            return [TextContent(type="text", text=f"Unknown tool: {name}")]

    except Exception as e:
        import traceback
        return [TextContent(type="text", text=f"Error in {name}: {e}\n{traceback.format_exc()}")]


#  Main 
async def main():
    async with stdio_server() as (read_stream, write_stream):
        await server.run(read_stream, write_stream, server.create_initialization_options())

if __name__ == "__main__":
    asyncio.run(main())

