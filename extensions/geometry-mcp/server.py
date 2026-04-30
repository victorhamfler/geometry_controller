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
import math
import time
import threading
import shutil
import uuid
from datetime import datetime, timezone

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


def _coerce_float(value, default=0.0):
    try:
        if value is None or value == "":
            return default
        return float(value)
    except Exception:
        return default


def _coerce_int(value, default=0):
    try:
        if value is None or value == "":
            return default
        return int(value)
    except Exception:
        return default


def _coerce_timestamp(value):
    if value is None or value == "":
        return None
    if isinstance(value, (int, float)):
        ts = float(value)
        if ts > 10_000_000_000:
            ts = ts / 1000.0
        return ts
    text = str(value).strip()
    if not text:
        return None
    try:
        ts = float(text)
        if ts > 10_000_000_000:
            ts = ts / 1000.0
        return ts
    except Exception:
        pass
    try:
        iso = text.replace("Z", "+00:00")
        if len(iso) == 10 and iso[4] == "-" and iso[7] == "-":
            iso = iso + "T00:00:00+00:00"
        dt = datetime.fromisoformat(iso)
        if dt.tzinfo is None:
            dt = dt.replace(tzinfo=timezone.utc)
        return dt.timestamp()
    except Exception:
        return None


def _date_bounds_from_args(
    min_age_days=None,
    max_age_days=None,
    updated_within_days=None,
    updated_after=None,
    updated_before=None,
    date_from=None,
    date_to=None,
):
    now_ts = time.time()
    after_ts = _coerce_timestamp(updated_after)
    before_ts = _coerce_timestamp(updated_before)
    date_from_ts = _coerce_timestamp(date_from)
    date_to_ts = _coerce_timestamp(date_to)
    if date_from_ts is not None:
        after_ts = max(after_ts, date_from_ts) if after_ts is not None else date_from_ts
    if date_to_ts is not None:
        # A bare YYYY-MM-DD should include the whole day.
        if isinstance(date_to, str) and len(date_to.strip()) == 10:
            date_to_ts += 86400.0 - 0.001
        before_ts = min(before_ts, date_to_ts) if before_ts is not None else date_to_ts

    min_age = None if min_age_days in (None, "") else _coerce_float(min_age_days, None)
    max_age_raw = max_age_days if max_age_days not in (None, "") else updated_within_days
    max_age = None if max_age_raw in (None, "") else _coerce_float(max_age_raw, None)
    if max_age is not None and max_age >= 0.0:
        age_after = now_ts - (max_age * 86400.0)
        after_ts = max(after_ts, age_after) if after_ts is not None else age_after
    if min_age is not None and min_age >= 0.0:
        age_before = now_ts - (min_age * 86400.0)
        before_ts = min(before_ts, age_before) if before_ts is not None else age_before
    return after_ts, before_ts


def _timestamp_in_bounds(ts, after_ts=None, before_ts=None):
    if after_ts is None and before_ts is None:
        return True
    t = _coerce_timestamp(ts)
    if t is None:
        return False
    if after_ts is not None and t < after_ts:
        return False
    if before_ts is not None and t > before_ts:
        return False
    return True


def _recency_score(ts, half_life_days=14.0, now_ts=None):
    t = _coerce_timestamp(ts)
    if t is None:
        return 0.0
    now = time.time() if now_ts is None else float(now_ts)
    age_days = max(0.0, (now - t) / 86400.0)
    half_life = max(0.01, float(half_life_days or 14.0))
    return float(2.0 ** (-age_days / half_life))


def _recency_meta(ts, half_life_days=14.0, now_ts=None):
    t = _coerce_timestamp(ts)
    if t is None:
        return {
            "last_update_ts": None,
            "source_timestamp": None,
            "last_updated": None,
            "age_days": None,
            "recency_score": 0.0,
            "recency_label": "unknown",
        }
    now = time.time() if now_ts is None else float(now_ts)
    age = max(0.0, (now - t) / 86400.0)
    if age < 1.0:
        label = "today"
    elif age < 2.0:
        label = "1 day ago"
    else:
        label = f"{int(round(age))} days ago"
    return {
        "last_update_ts": t,
        "source_timestamp": t,
        "last_updated": datetime.fromtimestamp(t, timezone.utc).isoformat().replace("+00:00", "Z"),
        "age_days": round(age, 2),
        "recency_score": round(_recency_score(t, half_life_days, now), 4),
        "recency_label": label,
    }


def _rerank_rows_by_recency(rows, score_key, ts_key, recency_boost=0.0, half_life_days=14.0):
    boost = max(0.0, min(1.0, _coerce_float(recency_boost, 0.0)))
    if not rows:
        return rows
    scores = [_coerce_float(row.get(score_key), 0.0) for row in rows]
    lo = min(scores)
    hi = max(scores)
    span = hi - lo
    now_ts = time.time()
    for row in rows:
        raw = _coerce_float(row.get(score_key), 0.0)
        relevance = 1.0 if span <= 1e-12 else (raw - lo) / span
        recency = _recency_score(row.get(ts_key), half_life_days, now_ts)
        final = (1.0 - boost) * relevance + boost * recency if boost > 0.0 else raw
        row["recency_score"] = round(recency, 4)
        row["final_score"] = round(final, 4)
    rows.sort(
        key=lambda r: (
            _coerce_float(r.get("final_score"), 0.0),
            _coerce_float(r.get(score_key), 0.0),
            _coerce_float(_coerce_timestamp(r.get(ts_key)), 0.0),
        ),
        reverse=True,
    )
    return rows


def _chunked(values, size=900):
    for i in range(0, len(values), size):
        yield values[i:i + size]


def _numeric_lcm_id(value):
    text = str(value or "").strip()
    if not text.isdigit():
        return None
    try:
        return int(text)
    except Exception:
        return None


def _conv_id_from_branch(branch_id):
    text = str(branch_id or "").strip()
    if not text.startswith("conv_"):
        return None
    suffix = text[5:]
    if not suffix.isdigit():
        return None
    try:
        return int(suffix)
    except Exception:
        return None


def _min_ts(current, candidate):
    ts = _coerce_timestamp(candidate)
    if ts is None:
        return current
    if current is None or ts < current:
        return ts
    return current


def _set_recency_candidate(out, branch_id, candidate, source):
    bid = str(branch_id or "")
    ts = _coerce_timestamp(candidate)
    if not bid or ts is None:
        return
    current = out.get(bid)
    if current is None:
        out[bid] = {
            "source_timestamp": float(ts),
            "last_source_timestamp": float(ts),
            "timestamp_source": source,
            "last_timestamp_source": source,
        }
        return
    source_ts = _coerce_timestamp(current.get("source_timestamp"))
    last_ts = _coerce_timestamp(current.get("last_source_timestamp"))
    if source_ts is None or ts < source_ts:
        current["source_timestamp"] = float(ts)
        current["timestamp_source"] = source
    if last_ts is None or ts > last_ts:
        current["last_source_timestamp"] = float(ts)
        current["last_timestamp_source"] = source


def _activity_meta(ts, half_life_days=14.0, now_ts=None):
    meta = _recency_meta(ts, half_life_days, now_ts)
    return {
        "last_source_timestamp": meta["source_timestamp"],
        "last_source_updated": meta["last_updated"],
        "activity_age_days": meta["age_days"],
        "activity_score": meta["recency_score"],
        "activity_label": meta["recency_label"],
    }


def _normalize_state_filter(value):
    if value in (None, "", "ALL", "all", "*"):
        return None
    raw_values = value if isinstance(value, list) else str(value).replace(",", " ").split()
    states = []
    seen = set()
    for raw in raw_values:
        state = str(raw or "").strip().upper()
        if not state or state in ("ALL", "*") or state in seen:
            continue
        seen.add(state)
        states.append(state)
    return states or None


def _activity_filter_from_args(activity_state=None, activity_within_days=None, state_group=None):
    days = _coerce_float(activity_within_days, 14.0)
    if days < 0.0:
        days = 14.0
    threshold = time.time() - (days * 86400.0)
    activity = str(activity_state or "").strip().lower()
    group = str(state_group or "").strip().lower()
    if not activity and group == "working":
        activity = "recent"
    elif not activity and group in ("settled", "dormant"):
        activity = "stale"

    after_ts = None
    before_ts = None
    if activity in ("recent", "active", "current"):
        after_ts = threshold
    elif activity in ("stale", "dormant", "old", "inactive"):
        before_ts = threshold
    return activity or None, days, after_ts, before_ts


def _state_filter_from_group(state=None, state_group=None):
    explicit = _normalize_state_filter(state)
    if explicit:
        return explicit
    group = str(state_group or "").strip().lower()
    if group == "working":
        return ["FORMING", "ACTIVE", "REACTIVATING"]
    if group in ("settled", "dormant"):
        return ["STABLE"]
    return None


def _branch_ids_for_activity(metadata, after_ts=None, before_ts=None):
    if after_ts is None and before_ts is None:
        return None
    ids = []
    for bid, meta in (metadata or {}).items():
        if _timestamp_in_bounds(meta.get("last_source_timestamp"), after_ts, before_ts):
            ids.append(str(bid))
    return ids


def _activity_state_for_ts(ts, activity_within_days=14.0, now_ts=None):
    t = _coerce_timestamp(ts)
    if t is None:
        return "unknown"
    now = time.time() if now_ts is None else float(now_ts)
    days = max(0.0, _coerce_float(activity_within_days, 14.0))
    return "recent" if t >= now - (days * 86400.0) else "stale"


def _load_geometry_recency_metadata(gc=None, branch_rows=None):
    """
    Return branch_id -> source-time metadata.

    branch_states.last_update_ts is an ingestion/maintenance timestamp, so it is
    too fresh after polling. Prefer actual LCM message/conversation time for
    conv_* branches, daily_log_content.created_ts for daily logs, and only then
    fall back to geometry timestamps.
    """
    out = {}
    if branch_rows is None:
        try:
            branch_rows = gc.db.list_branch_scalars()
        except Exception:
            branch_rows = []
    branch_ids = [str(r.get("branch_id") or "") for r in branch_rows if str(r.get("branch_id") or "").strip()]
    branch_set = set(branch_ids)
    fallback_update = {
        str(r.get("branch_id") or ""): _coerce_timestamp(r.get("last_update_ts"))
        for r in branch_rows
    }

    geo_rows = []
    try:
        gconn = sqlite3.connect(GEO_DB)
        gconn.row_factory = sqlite3.Row
        try:
            for r in gconn.execute(
                """
                SELECT mn.branch_id, MIN(dl.created_ts) AS first_ts, MAX(dl.created_ts) AS last_ts
                FROM daily_log_content dl
                JOIN memory_nodes mn ON mn.id = dl.node_id
                GROUP BY mn.branch_id
                """
            ).fetchall():
                bid = str(r["branch_id"] or "")
                if bid:
                    _set_recency_candidate(out, bid, r["first_ts"], "daily_log_content")
                    _set_recency_candidate(out, bid, r["last_ts"], "daily_log_content")
        except Exception:
            pass

        try:
            geo_rows = gconn.execute(
                """
                SELECT branch_id, lcm_id, timestamp
                FROM memory_nodes
                WHERE branch_id IS NOT NULL
                """
            ).fetchall()
        except Exception:
            geo_rows = []
        finally:
            gconn.close()
    except Exception:
        geo_rows = []

    msg_to_branches = defaultdict(set)
    geom_min = {}
    geom_max = {}
    for r in geo_rows:
        bid = str(r["branch_id"] or "")
        if not bid:
            continue
        geom_min[bid] = _min_ts(geom_min.get(bid), r["timestamp"])
        ts = _coerce_timestamp(r["timestamp"])
        if ts is not None and (geom_max.get(bid) is None or ts > geom_max[bid]):
            geom_max[bid] = ts
        mid = _numeric_lcm_id(r["lcm_id"])
        if mid is not None:
            msg_to_branches[mid].add(bid)

    try:
        if os.path.isfile(LCM_DB):
            lconn = sqlite3.connect(LCM_DB)
            lconn.row_factory = sqlite3.Row
            try:
                message_ids = list(msg_to_branches.keys())
                for chunk in _chunked(message_ids):
                    ph = ",".join(["?"] * len(chunk))
                    rows = lconn.execute(
                        f"SELECT message_id, created_at FROM messages WHERE message_id IN ({ph})",
                        tuple(chunk),
                    ).fetchall()
                    for row in rows:
                        ts = _coerce_timestamp(row["created_at"])
                        if ts is None:
                            continue
                        for bid in msg_to_branches.get(int(row["message_id"]), ()):
                            _set_recency_candidate(out, bid, ts, "lcm_messages")

                missing_conv = {}
                for bid in branch_ids:
                    if bid in out:
                        continue
                    cid = _conv_id_from_branch(bid)
                    if cid is not None:
                        missing_conv[cid] = bid
                conv_ids = list(missing_conv.keys())
                for chunk in _chunked(conv_ids):
                    ph = ",".join(["?"] * len(chunk))
                    rows = lconn.execute(
                        f"""
                        SELECT conversation_id,
                               MIN(created_at) AS first_at,
                               MAX(created_at) AS last_at
                        FROM messages
                        WHERE conversation_id IN ({ph})
                        GROUP BY conversation_id
                        """,
                        tuple(chunk),
                    ).fetchall()
                    for row in rows:
                        bid = missing_conv.get(int(row["conversation_id"]))
                        if bid:
                            _set_recency_candidate(out, bid, row["first_at"], "lcm_messages")
                            _set_recency_candidate(out, bid, row["last_at"], "lcm_messages")

                missing_conv = {
                    cid: bid for cid, bid in missing_conv.items()
                    if bid not in out
                }
                conv_ids = list(missing_conv.keys())
                for chunk in _chunked(conv_ids):
                    ph = ",".join(["?"] * len(chunk))
                    rows = lconn.execute(
                        f"SELECT conversation_id, created_at FROM conversations WHERE conversation_id IN ({ph})",
                        tuple(chunk),
                    ).fetchall()
                    for row in rows:
                        bid = missing_conv.get(int(row["conversation_id"]))
                        if bid:
                            _set_recency_candidate(out, bid, row["created_at"], "lcm_conversations")
            finally:
                lconn.close()
    except Exception:
        pass

    for bid in branch_set:
        if bid not in out:
            fallback_first = geom_min.get(bid) or fallback_update.get(bid) or 0.0
            fallback_last = geom_max.get(bid) or fallback_update.get(bid) or fallback_first
            _set_recency_candidate(out, bid, fallback_first, "geometry_last_update")
            _set_recency_candidate(out, bid, fallback_last, "geometry_last_update")
    return {
        bid: meta
        for bid, meta in out.items()
        if _coerce_timestamp(meta.get("source_timestamp")) is not None
    }


def _load_geometry_recency_timestamps(gc):
    return {
        bid: float(meta["source_timestamp"])
        for bid, meta in _load_geometry_recency_metadata(gc).items()
    }

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
_EMBED_CFG = _RUNTIME_CFG.get("embedding", {}) if isinstance(_RUNTIME_CFG.get("embedding"), dict) else {}
EMBED_MODEL_NAME = str(
    _EMBED_CFG.get("model")
    or _RUNTIME_CFG.get("embedding_model")
    or "all-MiniLM-L6-v2"
)
_backend_raw = str(
    _EMBED_CFG.get("backend")
    or _RUNTIME_CFG.get("embedding_backend")
    or ""
).strip().lower().replace("-", "_")
if not _backend_raw:
    _backend_raw = "llama_cpp" if EMBED_MODEL_NAME.lower().endswith(".gguf") else "sentence_transformers"
if _backend_raw in ("st", "sentence_transformer"):
    _backend_raw = "sentence_transformers"
if _backend_raw in ("gguf", "llama"):
    _backend_raw = "llama_cpp"
EMBED_BACKEND = _backend_raw
EMBED_DEVICE = str(_EMBED_CFG.get("device") or os.getenv("GEOMETRY_EMBED_DEVICE", "cpu") or "cpu")
EMBED_DIM = int(_EMBED_CFG.get("dim") or _RUNTIME_CFG.get("embedding_dim") or 384)
EMBED_GGUF_PATH = str(
    _EMBED_CFG.get("gguf_path")
    or _RUNTIME_CFG.get("embedding_gguf_path")
    or ""
).strip() or None
EMBED_GGUF_N_CTX = int(_EMBED_CFG.get("gguf_n_ctx") or _RUNTIME_CFG.get("embedding_gguf_n_ctx") or 2048)
_threads_raw = _EMBED_CFG.get("gguf_n_threads", _RUNTIME_CFG.get("embedding_gguf_n_threads"))
EMBED_GGUF_N_THREADS = int(_threads_raw) if _threads_raw is not None else None
EMBED_HTTP_URL = str(
    _EMBED_CFG.get("http_url")
    or _RUNTIME_CFG.get("embedding_http_url")
    or ""
).strip() or None
EMBED_HTTP_TIMEOUT_SEC = float(
    _EMBED_CFG.get("http_timeout_sec")
    or _RUNTIME_CFG.get("embedding_http_timeout_sec")
    or 30.0
)
_GEOMETRY_CFG_OVERRIDES = _RUNTIME_CFG.get("geometry_config", {})
if not isinstance(_GEOMETRY_CFG_OVERRIDES, dict):
    _GEOMETRY_CFG_OVERRIDES = {}
_POLLING_CFG = _RUNTIME_CFG.get("polling", {})
if not isinstance(_POLLING_CFG, dict):
    _POLLING_CFG = {}

POLLING_ENABLED = bool(_POLLING_CFG.get("enabled", True))
POLLING_INTERVAL_SEC = float(_POLLING_CFG.get("interval_seconds", 8.0))
POLLING_LIMIT = int(_POLLING_CFG.get("limit", 200))
_auto_limit_default = 3 if EMBED_BACKEND == "llama_cpp" else POLLING_LIMIT
try:
    POLLING_AUTO_LIMIT = max(1, int(_POLLING_CFG.get("auto_limit", _auto_limit_default)))
except Exception:
    POLLING_AUTO_LIMIT = max(1, int(_auto_limit_default))
_auto_tools_raw = _POLLING_CFG.get("auto_tools")
if isinstance(_auto_tools_raw, (list, tuple, set)):
    AUTO_POLL_TOOLS = {str(x).strip() for x in _auto_tools_raw if str(x).strip()}
else:
    AUTO_POLL_TOOLS = {"hybrid_search"}
if not AUTO_POLL_TOOLS:
    AUTO_POLL_TOOLS = {"hybrid_search"}
POLLING_CONVERSATION_ID = _POLLING_CFG.get("conversation_id")
POLLING_CURSOR_PATH = str(
    _POLLING_CFG.get("cursor_path") or os.path.join(SKILL_DIR, "poll_cursor.json")
)
POLLING_SHOW_STATUS = bool(_POLLING_CFG.get("show_status", True))
POLLING_DEBUG_LOG = bool(_POLLING_CFG.get("debug_log", False))
_STARTUP_CFG = _RUNTIME_CFG.get("startup", {})
if not isinstance(_STARTUP_CFG, dict):
    _STARTUP_CFG = {}
WARMUP_GC_ENABLED = bool(_STARTUP_CFG.get("warmup_gc", EMBED_BACKEND == "llama_cpp"))
WARMUP_PROBE_EMBED = bool(_STARTUP_CFG.get("warmup_probe_embed", EMBED_BACKEND != "llama_cpp"))
WARMUP_QUERY = str(_STARTUP_CFG.get("warmup_query") or "geometry-mcp-warmup")

print(
    f"[geometry-mcp] runtime cfg loaded: geo_db={GEO_DB} lcm_db={LCM_DB} "
    f"embed_backend={EMBED_BACKEND} model={EMBED_MODEL_NAME} "
    f"geometry_overrides={len(_GEOMETRY_CFG_OVERRIDES)} "
    f"polling_enabled={POLLING_ENABLED} interval={POLLING_INTERVAL_SEC}s "
    f"limit={POLLING_LIMIT} auto_limit={POLLING_AUTO_LIMIT} auto_tools={sorted(AUTO_POLL_TOOLS)}",
    file=sys.stderr,
)

#  Lazy-load heavy libs 
_gc = None
_gc_lock = threading.Lock()
_warmup_thread = None
_warmup_state = {
    "started": False,
    "completed": False,
    "ok": False,
    "error": None,
    "started_at": 0.0,
    "finished_at": 0.0,
}
_poll_lock = threading.Lock()
_last_poll_ts = 0.0


def _is_warmup_running() -> bool:
    return bool(
        WARMUP_GC_ENABLED
        and _warmup_state.get("started")
        and not _warmup_state.get("completed")
    )


def get_gc():
    global _gc
    if _gc is not None:
        return _gc
    with _gc_lock:
        if _gc is not None:
            return _gc
        from lcm_geometry_controller import GeometryController, GeometryConfig, EmbeddingProvider

        provider = EmbeddingProvider(
            model_name=EMBED_MODEL_NAME,
            device=EMBED_DEVICE,
            backend=EMBED_BACKEND,
            gguf_model_path=EMBED_GGUF_PATH,
            gguf_n_ctx=EMBED_GGUF_N_CTX,
            gguf_n_threads=EMBED_GGUF_N_THREADS,
            http_url=EMBED_HTTP_URL,
            http_timeout_sec=EMBED_HTTP_TIMEOUT_SEC,
        )

        valid_keys = set(getattr(GeometryConfig, "__dataclass_fields__", {}).keys())
        safe_overrides = {}
        for k, v in (_GEOMETRY_CFG_OVERRIDES or {}).items():
            if k in valid_keys:
                safe_overrides[k] = v
            else:
                print(f"[geometry-mcp] Ignoring unknown geometry_config key: {k}", file=sys.stderr)

        provider_dim = int(provider.embedding_dim)

        cfg_dim_raw = (
            _RUNTIME_CFG.get("embedding_dim")
            if _RUNTIME_CFG.get("embedding_dim") is not None
            else _EMBED_CFG.get("dim")
        )
        if cfg_dim_raw is None:
            cfg_dim_raw = safe_overrides.get("embedding_dim")
        if cfg_dim_raw is None:
            cfg_dim_raw = EMBED_DIM
        try:
            cfg_dim = int(cfg_dim_raw)
        except Exception:
            cfg_dim = None

        if cfg_dim is None:
            safe_overrides["embedding_dim"] = provider_dim
        elif cfg_dim != provider_dim:
            print(
                f"[geometry-mcp] embedding_dim mismatch in runtime_config: "
                f"cfg={cfg_dim} provider={provider_dim}; forcing provider dim",
                file=sys.stderr,
            )
            safe_overrides["embedding_dim"] = provider_dim
        else:
            safe_overrides["embedding_dim"] = cfg_dim

        _gc = GeometryController(
            GEO_DB,
            cfg=GeometryConfig(**safe_overrides),
            embedding_provider=provider,
        )
    return _gc


def _warmup_gc_background():
    _warmup_state["started"] = True
    _warmup_state["started_at"] = time.time()
    try:
        gc = get_gc()
        if WARMUP_PROBE_EMBED:
            provider = getattr(gc, "embedding_provider", None)
            if provider is not None:
                provider.embed(WARMUP_QUERY)
        _warmup_state["ok"] = True
    except Exception as exc:
        _warmup_state["error"] = str(exc)
        print(f"[geometry-mcp] startup warmup failed: {exc}", file=sys.stderr)
    finally:
        _warmup_state["completed"] = True
        _warmup_state["finished_at"] = time.time()


def _start_warmup_thread():
    global _warmup_thread
    if not WARMUP_GC_ENABLED or _warmup_thread is not None:
        return
    _warmup_state["started"] = True
    _warmup_state["started_at"] = time.time()
    _warmup_thread = threading.Thread(
        target=_warmup_gc_background,
        name="geometry-mcp-warmup",
        daemon=True,
    )
    _warmup_thread.start()


_start_warmup_thread()


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


def _poll_lcm_if_due(force: bool = False, limit_override: int | None = None, auto: bool = False) -> dict:
    global _last_poll_ts
    if not POLLING_ENABLED:
        return _with_poll_lag({"enabled": False, "skipped": "disabled"})

    now = time.time()
    if auto and _is_warmup_running():
        return _with_poll_lag({
            "enabled": True,
            "skipped": "warming_up",
            "warmup_elapsed_seconds": round(
                max(0.0, now - float(_warmup_state.get("started_at") or now)), 3
            ),
        })

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
        safe_limit = max(
            1,
            int(
                limit_override
                if limit_override is not None
                else (POLLING_AUTO_LIMIT if auto else POLLING_LIMIT)
            ),
        )
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
    if provider is None:
        raise RuntimeError(
            "Geometry controller has no embedding provider configured; cannot embed query"
        )
    return list(provider.embed(query))


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


def _rank_collapsing_sidecar(
    gc,
    q_emb,
    top_n=5,
    retrieval_mode="balanced",
    recency_boost=0.0,
    recency_half_life_days=14.0,
    updated_after_ts=None,
    updated_before_ts=None,
    recency_timestamps=None,
    recency_metadata=None,
    include_states=None,
    activity_after_ts=None,
    activity_before_ts=None,
):
    try:
        import numpy as np
    except Exception:
        return []

    safe_n = max(0, int(top_n or 0))
    if safe_n <= 0:
        return []

    collapsing = []
    recency_map = recency_timestamps or {}
    include_set = set(include_states or [])
    for s in gc.db.all_branches():
        state_val = str(getattr(getattr(s, "state", None), "value", getattr(s, "state", "")) or "")
        if state_val != "COLLAPSING":
            continue
        if include_set and state_val not in include_set:
            continue
        if int(getattr(s, "node_count", 0) or 0) <= 0:
            continue
        mean_vec = getattr(s, "mean_vec", None) or []
        if not mean_vec:
            continue
        meta = (recency_metadata or {}).get(s.branch_id, {})
        effective_ts = recency_map.get(s.branch_id) or getattr(s, "last_update_ts", None)
        activity_ts = meta.get("last_source_timestamp") or effective_ts
        if not _timestamp_in_bounds(activity_ts, activity_after_ts, activity_before_ts):
            continue
        if not _timestamp_in_bounds(
            effective_ts,
            updated_after_ts,
            updated_before_ts,
        ):
            continue
        collapsing.append(s)

    if not collapsing:
        return []

    q = np.array(q_emb, dtype=np.float32)
    ranked = gc.ranker.rank(q, collapsing, retrieval_mode=retrieval_mode)
    boost = max(0.0, min(1.0, _coerce_float(recency_boost, 0.0)))
    if boost > 0.0 and ranked:
        now_ts = time.time()
        by_update = {
            s.branch_id: float(recency_map.get(s.branch_id) or getattr(s, "last_update_ts", 0.0) or 0.0)
            for s in collapsing
        }
        scores = [float(r.total_score) for r in ranked]
        lo = min(scores)
        hi = max(scores)
        span = hi - lo
        for r in ranked:
            ts = by_update.get(r.branch_id, 0.0)
            relevance = 1.0 if span <= 1e-12 else (float(r.total_score) - lo) / span
            recency = _recency_score(ts, recency_half_life_days, now_ts)
            r.recency_score = recency
            r.final_score = (1.0 - boost) * relevance + boost * recency
        ranked.sort(
            key=lambda r: (
                float(getattr(r, "final_score", r.total_score) or 0.0),
                float(r.total_score),
                by_update.get(r.branch_id, 0.0),
            ),
            reverse=True,
        )
    by_id = {s.branch_id: s for s in collapsing}
    out = []
    for r in ranked[:safe_n]:
        s = by_id.get(r.branch_id)
        meta = (recency_metadata or {}).get(r.branch_id, {})
        effective_ts = (
            meta.get("source_timestamp")
            or recency_map.get(r.branch_id)
            or (getattr(s, "last_update_ts", None) if s else None)
        )
        recency = _recency_meta(
            effective_ts,
            recency_half_life_days,
        )
        out.append({
            "branch_id": r.branch_id,
            "total_score": round(r.total_score, 4),
            "base_score": round(r.total_score, 4),
            "ranking_score": round(float(getattr(r, "final_score", r.total_score) or 0.0), 4),
            "final_score": round(float(getattr(r, "final_score", r.total_score) or 0.0), 4),
            "retrieval_kappa": round(r.total_score, 4),
            "sem_score": round(r.sem_score, 4),
            "trust_score": round(r.trust_score, 4),
            "recency_score": round(float(getattr(r, "recency_score", recency["recency_score"]) or 0.0), 4),
            "nodes": int(getattr(s, "node_count", 0) or 0) if s else 0,
            "coherence": round(float(getattr(s, "coherence", 0.0) or 0.0), 4) if s else 0.0,
            "eff_rank": round(float(getattr(s, "eff_rank", 0.0) or 0.0), 2) if s else 0.0,
            "state": str(getattr(getattr(s, "state", None), "value", getattr(s, "state", "unknown")) or "unknown"),
            "regime": str(getattr(getattr(s, "regime", None), "value", getattr(s, "regime", "unknown")) or "unknown"),
            "last_update_ts": recency["last_update_ts"],
            "source_timestamp": recency["source_timestamp"],
            "timestamp_source": meta.get("timestamp_source") or ("geometry_last_update" if not recency_map.get(r.branch_id) else "source_time"),
            "last_source_timestamp": (activity := _activity_meta(meta.get("last_source_timestamp") or effective_ts, recency_half_life_days))["last_source_timestamp"],
            "last_source_updated": activity["last_source_updated"],
            "last_timestamp_source": meta.get("last_timestamp_source") or meta.get("timestamp_source") or "geometry_last_update",
            "activity_age_days": activity["activity_age_days"],
            "activity_label": activity["activity_label"],
            "last_updated": recency["last_updated"],
            "age_days": recency["age_days"],
            "recency_label": recency["recency_label"],
            "note": "excluded_from_primary_rank",
        })
    return out

#  Hybrid search 
def do_hybrid_search(
    query,
    top_n=5,
    retrieval_mode="balanced",
    recency_boost=0.0,
    recency_half_life_days=14.0,
    min_age_days=None,
    max_age_days=None,
    updated_within_days=None,
    updated_after=None,
    updated_before=None,
    date_from=None,
    date_to=None,
    state=None,
    state_group=None,
    activity_state=None,
    activity_within_days=None,
):
    gc = get_gc()
    query_id = str(uuid.uuid4())
    safe_top_n = max(1, _coerce_int(top_n, 5))
    boost = max(0.0, min(1.0, _coerce_float(recency_boost, 0.0)))
    half_life_days = max(0.01, _coerce_float(recency_half_life_days, 14.0))
    updated_after_ts, updated_before_ts = _date_bounds_from_args(
        min_age_days=min_age_days,
        max_age_days=max_age_days,
        updated_within_days=updated_within_days,
        updated_after=updated_after,
        updated_before=updated_before,
        date_from=date_from,
        date_to=date_to,
    )

    # Encode query
    q_emb = _embed_query(query)
    geometry_recency_meta = _load_geometry_recency_metadata(gc)
    geometry_recency_ts = {
        bid: float(meta["source_timestamp"])
        for bid, meta in geometry_recency_meta.items()
    }
    include_states = _state_filter_from_group(state=state, state_group=state_group)
    normalized_state_group = str(state_group or "all").strip().lower() or "all"
    if normalized_state_group not in ("all", "working", "settled", "dormant"):
        normalized_state_group = "all"
    normalized_activity_state, activity_days, activity_after_ts, activity_before_ts = _activity_filter_from_args(
        activity_state=activity_state,
        activity_within_days=activity_within_days,
        state_group=normalized_state_group,
    )
    activity_branch_ids = _branch_ids_for_activity(
        geometry_recency_meta,
        after_ts=activity_after_ts,
        before_ts=activity_before_ts,
    )

    # Geometry ranking
    ranked = gc.rank_retrieval(
        q_emb,
        retrieval_mode=retrieval_mode,
        recency_boost=boost,
        recency_half_life_days=half_life_days,
        updated_after_ts=updated_after_ts,
        updated_before_ts=updated_before_ts,
        recency_timestamps=geometry_recency_ts,
        include_states=set(include_states) if include_states else None,
        branch_ids=activity_branch_ids,
    )
    geo_results = []
    implicit_feedback_logged = 0
    for r in ranked[:safe_top_n]:
        b = gc.db.load_branch(r.branch_id)
        recency_info = geometry_recency_meta.get(str(r.branch_id), {})
        effective_ts = (
            recency_info.get("source_timestamp")
            or geometry_recency_ts.get(str(r.branch_id))
            or (getattr(b, "last_update_ts", None) if b else None)
        )
        recency = _recency_meta(
            effective_ts,
            half_life_days,
        )
        activity = _activity_meta(
            recency_info.get("last_source_timestamp") or effective_ts,
            half_life_days,
        )
        activity_label = _activity_state_for_ts(
            activity["last_source_timestamp"],
            activity_within_days=activity_days,
        )
        try:
            gc.record_retrieval(
                query_id=query_id,
                branch_id=str(r.branch_id),
                score=float(getattr(r, "final_score", r.total_score) or r.total_score),
                used=False,
                corrected=False,
                expanded=False,
                usefulness_signal=0.5,
                feedback_source="implicit_hybrid_search",
            )
            implicit_feedback_logged += 1
        except Exception:
            pass
        geo_results.append({
            'branch_id': r.branch_id,
            'total_score': round(r.total_score, 4),
            'base_score': round(r.total_score, 4),
            'ranking_score': round(float(getattr(r, "final_score", r.total_score) or 0.0), 4),
            'final_score': round(float(getattr(r, "final_score", r.total_score) or 0.0), 4),
            'retrieval_kappa': round(r.total_score, 4),
            'sem_score': round(r.sem_score, 4),
            'trust_score': round(r.trust_score, 4),
            'recency_score': round(float(getattr(r, "recency_score", recency["recency_score"]) or 0.0), 4),
            'nodes': b.node_count if b else 0,
            'coherence': round(b.coherence, 4) if b else 0,
            'eff_rank': round(b.eff_rank, 2) if b else 0,
            'state': b.state.value if b and b.state else 'unknown',
            'regime': b.regime.value if b and b.regime else 'unknown',
            'last_update_ts': recency["last_update_ts"],
            'source_timestamp': recency["source_timestamp"],
            'timestamp_source': recency_info.get("timestamp_source") or ('source_time' if geometry_recency_ts.get(str(r.branch_id)) else 'geometry_last_update'),
            'last_source_timestamp': activity["last_source_timestamp"],
            'last_source_updated': activity["last_source_updated"],
            'last_timestamp_source': recency_info.get("last_timestamp_source") or recency_info.get("timestamp_source") or ('source_time' if geometry_recency_ts.get(str(r.branch_id)) else 'geometry_last_update'),
            'activity_age_days': activity["activity_age_days"],
            'activity_score': activity["activity_score"],
            'activity_label': activity["activity_label"],
            'activity_state': activity_label,
            'last_updated': recency["last_updated"],
            'age_days': recency["age_days"],
            'recency_label': recency["recency_label"],
            'recency_source': recency_info.get("timestamp_source") or ('source_time' if geometry_recency_ts.get(str(r.branch_id)) else 'geometry_last_update'),
        })
    collapsing_sidecar = _rank_collapsing_sidecar(
        gc,
        q_emb,
        top_n=safe_top_n,
        retrieval_mode=retrieval_mode,
        recency_boost=boost,
        recency_half_life_days=half_life_days,
        updated_after_ts=updated_after_ts,
        updated_before_ts=updated_before_ts,
        recency_timestamps=geometry_recency_ts,
        recency_metadata=geometry_recency_meta,
        include_states=include_states,
        activity_after_ts=activity_after_ts,
        activity_before_ts=activity_before_ts,
    )

    # LCM keyword search
    keywords = [w.strip() for w in query.split() if len(w.strip()) >= 3]
    conn = sqlite3.connect(LCM_DB)
    conn.row_factory = sqlite3.Row
    gconn = sqlite3.connect(GEO_DB)
    gconn.row_factory = sqlite3.Row

    results_by_conv = defaultdict(list)
    for kw in keywords:
        keyword_limit = (
            max(5000, safe_top_n * 200)
            if updated_after_ts is not None or updated_before_ts is not None
            else max(50, safe_top_n * 80)
        )
        cur = conn.execute('''
            SELECT message_id, conversation_id, role,
                   SUBSTR(content, 1, 120) as snippet, token_count, created_at
            FROM messages
            WHERE content LIKE ?
            ORDER BY created_at DESC
            LIMIT ?
        ''', [f'%{kw}%', keyword_limit])
        for r in cur.fetchall():
            if not _timestamp_in_bounds(r['created_at'], updated_after_ts, updated_before_ts):
                continue
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
        most_recent = max(
            matches,
            key=lambda m: _coerce_timestamp(m.get('created_at')) or float("-inf"),
        )['created_at']
        if not _timestamp_in_bounds(most_recent, activity_after_ts, activity_before_ts):
            continue
        recency = _recency_meta(most_recent, half_life_days)
        activity = _activity_meta(most_recent, half_life_days)
        raw_score = (
            2.0 * len(set(m['keyword_matched'] for m in matches))
            + math.log1p(len(matches))
        )
        scored.append({
            'conv_id': conv_id,
            'match_count': len(matches),
            'unique_keywords_matched': len(set(m['keyword_matched'] for m in matches)),
            'total_matches': len(matches),
            'raw_score': round(raw_score, 4),
            'best_snippet': matches[0]['snippet'][:80],
            'most_recent': most_recent,
            'last_update_ts': recency["last_update_ts"],
            'source_timestamp': recency["source_timestamp"],
            'timestamp_source': "lcm_messages",
            'last_source_timestamp': activity["last_source_timestamp"],
            'last_source_updated': activity["last_source_updated"],
            'last_timestamp_source': "lcm_messages",
            'activity_age_days': activity["activity_age_days"],
            'activity_score': activity["activity_score"],
            'activity_label': activity["activity_label"],
            'activity_state': _activity_state_for_ts(activity["last_source_timestamp"], activity_days),
            'last_updated': recency["last_updated"],
            'age_days': recency["age_days"],
            'recency_score': recency["recency_score"],
            'recency_label': recency["recency_label"],
        })
    if boost > 0.0:
        _rerank_rows_by_recency(scored, "raw_score", "most_recent", boost, half_life_days)
    else:
        scored.sort(key=lambda x: (x['unique_keywords_matched'], x['total_matches']), reverse=True)
        for row in scored:
            row["final_score"] = row["raw_score"]
    lcm_results = scored[:safe_top_n]

    # Daily log sidecar search (keyword + semantic)
    daily_keyword = []
    for kw in keywords:
        rows = gconn.execute(
            """
            SELECT mn.branch_id, mn.id AS node_id, dl.created_ts AS created_at,
                   SUBSTR(dl.text, 1, 140) AS snippet
            FROM daily_log_content dl
            JOIN memory_nodes mn ON mn.id = dl.node_id
            WHERE dl.text LIKE ?
            ORDER BY mn.timestamp DESC, mn.rowid DESC
            LIMIT ?
            """,
            (f"%{kw}%", max(5, safe_top_n * 4)),
        ).fetchall()
        for r in rows:
            if not _timestamp_in_bounds(r["created_at"], updated_after_ts, updated_before_ts):
                continue
            if not _timestamp_in_bounds(r["created_at"], activity_after_ts, activity_before_ts):
                continue
            recency = _recency_meta(r["created_at"], half_life_days)
            activity = _activity_meta(r["created_at"], half_life_days)
            daily_keyword.append({
                "branch_id": r["branch_id"],
                "node_id": r["node_id"],
                "created_at": r["created_at"],
                "keyword_matched": kw,
                "snippet": r["snippet"],
                "raw_score": 1.0,
                "last_update_ts": recency["last_update_ts"],
                "source_timestamp": recency["source_timestamp"],
                "timestamp_source": "daily_log_content",
                "last_source_timestamp": activity["last_source_timestamp"],
                "last_source_updated": activity["last_source_updated"],
                "last_timestamp_source": "daily_log_content",
                "activity_age_days": activity["activity_age_days"],
                "activity_score": activity["activity_score"],
                "activity_label": activity["activity_label"],
                "activity_state": _activity_state_for_ts(activity["last_source_timestamp"], activity_days),
                "last_updated": recency["last_updated"],
                "age_days": recency["age_days"],
                "recency_score": recency["recency_score"],
                "recency_label": recency["recency_label"],
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
    _rerank_rows_by_recency(dedup_keyword, "raw_score", "created_at", boost, half_life_days)
    daily_keyword = dedup_keyword[:safe_top_n]

    daily_semantic = []
    try:
        import numpy as np
        q = np.array(q_emb, dtype=np.float32)
        rows = gconn.execute(
            """
            SELECT mn.branch_id, mn.id AS node_id, dl.created_ts AS created_at,
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
            if not _timestamp_in_bounds(r["created_at"], updated_after_ts, updated_before_ts):
                continue
            if not _timestamp_in_bounds(r["created_at"], activity_after_ts, activity_before_ts):
                continue
            recency = _recency_meta(r["created_at"], half_life_days)
            activity = _activity_meta(r["created_at"], half_life_days)
            daily_semantic.append({
                "branch_id": r["branch_id"],
                "node_id": r["node_id"],
                "created_at": r["created_at"],
                "sem_score": round(sim, 4),
                "raw_score": sim,
                "snippet": r["snippet"],
                "last_update_ts": recency["last_update_ts"],
                "source_timestamp": recency["source_timestamp"],
                "timestamp_source": "daily_log_content",
                "last_source_timestamp": activity["last_source_timestamp"],
                "last_source_updated": activity["last_source_updated"],
                "last_timestamp_source": "daily_log_content",
                "activity_age_days": activity["activity_age_days"],
                "activity_score": activity["activity_score"],
                "activity_label": activity["activity_label"],
                "activity_state": _activity_state_for_ts(activity["last_source_timestamp"], activity_days),
                "last_updated": recency["last_updated"],
                "age_days": recency["age_days"],
                "recency_score": recency["recency_score"],
                "recency_label": recency["recency_label"],
            })
        _rerank_rows_by_recency(daily_semantic, "raw_score", "created_at", boost, half_life_days)
        daily_semantic = daily_semantic[:safe_top_n]
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
        'query_id': query_id,
        'query': query,
        'retrieval_mode': retrieval_mode,
        'filters': {
            'state': include_states or None,
            'state_group': normalized_state_group,
            'activity_state': normalized_activity_state,
            'activity_within_days': activity_days,
        },
        'activity': {
            'state': normalized_activity_state,
            'within_days': activity_days,
            'activity_after_ts': activity_after_ts,
            'activity_before_ts': activity_before_ts,
            'activity_after': _recency_meta(activity_after_ts, half_life_days)["last_updated"] if activity_after_ts is not None else None,
            'activity_before': _recency_meta(activity_before_ts, half_life_days)["last_updated"] if activity_before_ts is not None else None,
        },
        'recency': {
            'boost': boost,
            'half_life_days': half_life_days,
            'updated_after_ts': updated_after_ts,
            'updated_before_ts': updated_before_ts,
            'updated_after': _recency_meta(updated_after_ts, half_life_days)["last_updated"] if updated_after_ts is not None else None,
            'updated_before': _recency_meta(updated_before_ts, half_life_days)["last_updated"] if updated_before_ts is not None else None,
        },
        'recommendation': recommendation,
        'feedback': {
            'implicit_logged': implicit_feedback_logged,
            'implicit_usefulness_signal': 0.5,
        },
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
    provider = None
    if not _is_warmup_running() or _gc is not None:
        try:
            gc = get_gc()
            provider = getattr(gc, "embedding_provider", None)
        except Exception as exc:
            if POLLING_DEBUG_LOG:
                print(f"[geometry-mcp] geometry_stats provider unavailable: {exc}", file=sys.stderr)

    branches = conn.execute("SELECT COUNT(*) FROM branch_states").fetchone()[0]
    nodes = conn.execute("SELECT COUNT(*) FROM memory_nodes").fetchone()[0]

    states = {r['state']: r['cnt'] for r in conn.execute(
        "SELECT state, COUNT(*) as cnt FROM branch_states GROUP BY state").fetchall()}
    regimes = {r['regime']: r['cnt'] for r in conn.execute(
        "SELECT regime, COUNT(*) as cnt FROM branch_states GROUP BY regime").fetchall()}

    r = conn.execute(
        "SELECT AVG(eff_rank) as a, AVG(coherence) as c, AVG(COALESCE(role_diversity, 0.0)) as rd "
        "FROM branch_states"
    ).fetchone()
    avg_rank = round(r['a'], 2) if r['a'] else 0
    avg_coh = round(r['c'], 4) if r['c'] else 0
    avg_role_div = round(r['rd'], 4) if r['rd'] is not None else 0

    conn.close()
    embed_dim = int(
        getattr(provider, "embedding_dim", EMBED_DIM)
        if provider is not None
        else (_EMBED_CFG.get("dim") or _RUNTIME_CFG.get("embedding_dim") or EMBED_DIM)
    )
    embed_model = str(getattr(provider, "model_name", EMBED_MODEL_NAME) if provider is not None else EMBED_MODEL_NAME)
    embed_backend = str(getattr(provider, "backend", EMBED_BACKEND) if provider is not None else EMBED_BACKEND)
    return {
        'total_branches': branches,
        'total_nodes': nodes,
        'states': states,
        'regimes': regimes,
        'avg_eff_rank': avg_rank,
        'avg_coherence': avg_coh,
        'avg_role_diversity': avg_role_div,
        'embedding_backend': embed_backend,
        'embedding_model': embed_model,
        'embedding_dim': embed_dim,
        'warmup': {
            'enabled': bool(WARMUP_GC_ENABLED),
            'running': bool(_is_warmup_running()),
            'started': bool(_warmup_state.get('started')),
            'completed': bool(_warmup_state.get('completed')),
            'ok': bool(_warmup_state.get('ok')),
            'error': _warmup_state.get('error'),
        },
    }


def do_maintenance_cycle(max_branches=None, reset_chunk_cursor=False):
    gc = get_gc()
    kwargs = {"reset_chunk_cursor": bool(reset_chunk_cursor)}
    if max_branches is not None:
        kwargs["max_branches"] = max(1, int(max_branches))
    return gc.run_maintenance_cycle(**kwargs)


def do_geometry_snapshot(branch_ids=None, state=None, limit=None, include_means=False):
    gc = get_gc()

    clean_ids = None
    if isinstance(branch_ids, list):
        out = []
        seen = set()
        for raw in branch_ids:
            bid = str(raw or "").strip()
            if not bid or bid in seen:
                continue
            seen.add(bid)
            out.append(bid)
        clean_ids = out or None

    safe_state = str(state or "").strip().upper() or None

    safe_limit = None
    if limit is not None:
        try:
            safe_limit = max(1, int(limit))
        except Exception:
            safe_limit = None

    return gc.export_geometry_snapshot(
        branch_ids=clean_ids,
        state=safe_state,
        limit=safe_limit,
        include_means=bool(include_means),
    )


def do_latest_correction(node_id, branch_id=None, include_chain=False, chain_limit=64):
    gc = get_gc()
    return gc.get_latest_correction(
        node_id=str(node_id or "").strip(),
        branch_id=str(branch_id or "").strip() or None,
        include_chain=bool(include_chain),
        chain_limit=max(1, int(chain_limit or 64)),
    )


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

def _conversation_branch_candidates(gdb, content_type, state=None, state_group=None, activity_state=None, activity_within_days=None):
    safe_content = str(content_type or "summaries").strip().lower()
    branch_like = "day_%" if safe_content == "logs" else "conv_%"
    rows = gdb.execute(
        "SELECT branch_id, state, regime, node_count, last_update_ts "
        "FROM branch_states WHERE branch_id LIKE ? ORDER BY node_count DESC",
        (branch_like,),
    ).fetchall()
    branch_rows = [dict(r) for r in rows]
    include_states = _state_filter_from_group(state=state, state_group=state_group)
    if include_states:
        include_set = set(include_states)
        branch_rows = [r for r in branch_rows if str(r.get("state") or "").upper() in include_set]

    normalized_state_group = str(state_group or "all").strip().lower() or "all"
    if normalized_state_group not in ("all", "working", "settled", "dormant"):
        normalized_state_group = "all"
    normalized_activity_state, activity_days, activity_after_ts, activity_before_ts = _activity_filter_from_args(
        activity_state=activity_state,
        activity_within_days=activity_within_days,
        state_group=normalized_state_group,
    )
    recency_meta = _load_geometry_recency_metadata(branch_rows=branch_rows)
    if activity_after_ts is not None or activity_before_ts is not None:
        branch_rows = [
            r for r in branch_rows
            if _timestamp_in_bounds(
                (recency_meta.get(str(r.get("branch_id") or ""), {}) or {}).get("last_source_timestamp")
                or r.get("last_update_ts"),
                activity_after_ts,
                activity_before_ts,
            )
        ]

    return branch_rows, recency_meta, {
        "state": include_states or None,
        "state_group": normalized_state_group,
        "activity_state": normalized_activity_state,
        "activity_within_days": activity_days,
        "activity_after_ts": activity_after_ts,
        "activity_before_ts": activity_before_ts,
        "activity_after": _recency_meta(activity_after_ts)["last_updated"] if activity_after_ts is not None else None,
        "activity_before": _recency_meta(activity_before_ts)["last_updated"] if activity_before_ts is not None else None,
    }


def _conversation_fallback_args(state, state_group, activity_state, fallback_when_empty):
    if not fallback_when_empty:
        return None
    explicit_state = _normalize_state_filter(state)
    group = str(state_group or "").strip().lower()
    activity = str(activity_state or "").strip().lower()
    if explicit_state == ["ACTIVE"] and not group and not activity:
        return {
            "state": None,
            "state_group": "working",
            "activity_state": None,
            "reason": "active_empty_used_working_group",
        }
    if explicit_state == ["ACTIVE"] and activity in ("recent", "active", "current"):
        return {
            "state": None,
            "state_group": "working",
            "activity_state": activity_state,
            "reason": "active_recent_empty_used_working_group",
        }
    return None


def _build_conversation_content_results(
    gdb,
    lcm,
    branches,
    content_type,
    max_entries,
    max_chars,
    recency_meta=None,
    filters=None,
):
    results = []
    if not branches:
        return results
    per_branch = max(max_entries // len(branches), 3)
    for b in branches:
        bid = str(b["branch_id"])
        if bid.startswith("day_") or content_type == "logs":
            entries = _get_daily_log_content(gdb, bid, per_branch, max_chars)
            resolution_meta = {
                "resolution_mode": "daily_log",
                "resolved_conversation_ids": [],
            }
        else:
            entries, resolution_meta = _get_branch_lineage_content(
                gdb, lcm, bid, content_type, per_branch, max_chars
            )
            if not entries and content_type == "summaries":
                entries, resolution_meta = _get_branch_lineage_content(
                    gdb, lcm, bid, "messages", per_branch, max_chars
                )
                if entries:
                    _append_resolution_warning(resolution_meta, "summary_empty_used_messages_fallback")
        activity = _activity_meta(
            (recency_meta or {}).get(bid, {}).get("last_source_timestamp")
            or b["last_update_ts"]
        )
        results.append({
            "branch_id": bid,
            "state": b["state"],
            "regime": b["regime"],
            "last_source_timestamp": activity["last_source_timestamp"],
            "last_source_updated": activity["last_source_updated"],
            "activity_age_days": activity["activity_age_days"],
            "activity_label": activity["activity_label"],
            "activity_state": _activity_state_for_ts(
                activity["last_source_timestamp"],
                (filters or {}).get("activity_within_days", 14.0),
            ),
            "entries_returned": len(entries),
            "resolution_mode": resolution_meta.get("resolution_mode", "unknown"),
            "resolved_conversation_ids": resolution_meta.get("resolved_conversation_ids", []),
            "warning": resolution_meta.get("warning"),
            "content": entries
        })
    return results


def do_conversation_content(
    branch_id=None,
    state=None,
    content_type="summaries",
    max_entries=100,
    max_chars=250,
    state_group=None,
    activity_state=None,
    activity_within_days=None,
    fallback_when_empty=True,
):
    """Retrieve actual conversation text from LCM for geometry-identified branches.

    Modes:
      - Single branch:   branch_id="conv_148"
      - By state:        state="ACTIVE" | "STABLE" | "FORMING" | "ALL"
      - By group:        state_group="working" | "settled" | "all"
      - By activity:     activity_state="recent" | "stale"
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
            if not entries and content_type == "summaries":
                entries, resolution_meta = _get_branch_lineage_content(
                    gdb, lcm, branch_id, "messages", max_entries, max_chars
                )
                if entries:
                    _append_resolution_warning(resolution_meta, "summary_empty_used_messages_fallback")
        branch_meta = _load_geometry_recency_metadata(branch_rows=[dict(b)]).get(str(branch_id), {})
        activity = _activity_meta(branch_meta.get("last_source_timestamp") or b["last_update_ts"])
        results.append({
            "branch_id": branch_id,
            "state": b["state"],
            "regime": b["regime"],
            "last_source_timestamp": activity["last_source_timestamp"],
            "last_source_updated": activity["last_source_updated"],
            "activity_age_days": activity["activity_age_days"],
            "activity_label": activity["activity_label"],
            "activity_state": _activity_state_for_ts(activity["last_source_timestamp"]),
            "entries_returned": len(entries),
            "resolution_mode": resolution_meta.get("resolution_mode", "unknown"),
            "resolved_conversation_ids": resolution_meta.get("resolved_conversation_ids", []),
            "warning": resolution_meta.get("warning"),
            "content": entries
        })
    else:
        # Multi-branch mode (filtered by state/group/activity or ALL)
        branches, recency_meta, filters = _conversation_branch_candidates(
            gdb,
            content_type,
            state=state,
            state_group=state_group,
            activity_state=activity_state,
            activity_within_days=activity_within_days,
        )
        fallback = None
        if not branches:
            fallback_args = _conversation_fallback_args(
                state,
                state_group,
                activity_state,
                bool(fallback_when_empty),
            )
            if fallback_args:
                branches, recency_meta, filters = _conversation_branch_candidates(
                    gdb,
                    content_type,
                    state=fallback_args["state"],
                    state_group=fallback_args["state_group"],
                    activity_state=fallback_args["activity_state"],
                    activity_within_days=activity_within_days,
                )
                fallback = {
                    "used": bool(branches),
                    "reason": fallback_args["reason"],
                    "from": {
                        "state": state or "ALL",
                        "state_group": state_group or "all",
                        "activity_state": activity_state or "all",
                    },
                    "to": {
                        "state": fallback_args["state"] or "ALL",
                        "state_group": fallback_args["state_group"],
                        "activity_state": fallback_args["activity_state"] or "all",
                    },
                }

        if not branches:
            gdb.close(); lcm.close()
            return {
                "error": "No branches found for requested filters",
                "filters": filters,
                "fallback": fallback,
            }

        results = _build_conversation_content_results(
            gdb, lcm, branches, content_type, max_entries, max_chars, recency_meta, filters
        )

        if sum(r["entries_returned"] for r in results) == 0 and not fallback:
            fallback_args = _conversation_fallback_args(
                state,
                state_group,
                activity_state,
                bool(fallback_when_empty),
            )
            if fallback_args:
                branches, recency_meta, filters = _conversation_branch_candidates(
                    gdb,
                    content_type,
                    state=fallback_args["state"],
                    state_group=fallback_args["state_group"],
                    activity_state=fallback_args["activity_state"],
                    activity_within_days=activity_within_days,
                )
                fallback = {
                    "used": bool(branches),
                    "reason": fallback_args["reason"] + "_after_empty_content",
                    "from": {
                        "state": state or "ALL",
                        "state_group": state_group or "all",
                        "activity_state": activity_state or "all",
                    },
                    "to": {
                        "state": fallback_args["state"] or "ALL",
                        "state_group": fallback_args["state_group"],
                        "activity_state": fallback_args["activity_state"] or "all",
                    },
                }
                results = _build_conversation_content_results(
                    gdb, lcm, branches, content_type, max_entries, max_chars, recency_meta, filters
                )

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
        "filters": {} if branch_id else filters,
        "fallback": None if branch_id else fallback,
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
                    },
                    "recency_boost": {
                        "type": "number",
                        "description": "Optional [0,1] weight that blends freshness into ranking. Default 0 preserves relevance-only ranking."
                    },
                    "recency_half_life_days": {
                        "type": "number",
                        "description": "Freshness half-life in days for recency_boost. Default 14."
                    },
                    "max_age_days": {
                        "type": "number",
                        "description": "Only return items updated/created within this many days, e.g. 7 for the last week."
                    },
                    "updated_within_days": {
                        "type": "number",
                        "description": "Alias for max_age_days; only return items updated/created within this many days."
                    },
                    "min_age_days": {
                        "type": "number",
                        "description": "Only return items at least this many days old."
                    },
                    "updated_after": {
                        "type": "string",
                        "description": "Only return items updated/created after this ISO date/time or epoch timestamp."
                    },
                    "updated_before": {
                        "type": "string",
                        "description": "Only return items updated/created before this ISO date/time or epoch timestamp."
                    },
                    "date_from": {
                        "type": "string",
                        "description": "Alias for updated_after, convenient for YYYY-MM-DD date ranges."
                    },
                    "date_to": {
                        "type": "string",
                        "description": "Alias for updated_before, convenient for YYYY-MM-DD date ranges."
                    },
                    "state": {
                        "type": ["string", "array"],
                        "items": {"type": "string"},
                        "description": "Optional branch lifecycle state filter, e.g. ACTIVE, STABLE, FORMING, or [\"FORMING\",\"ACTIVE\"]."
                    },
                    "state_group": {
                        "type": "string",
                        "enum": ["all", "working", "settled", "dormant"],
                        "description": "Convenience filter. working=recent FORMING/ACTIVE/REACTIVATING; settled/dormant=older STABLE."
                    },
                    "activity_state": {
                        "type": "string",
                        "enum": ["recent", "stale", "dormant", "all"],
                        "description": "Filter by latest source activity, separate from geometric lifecycle state."
                    },
                    "activity_within_days": {
                        "type": "number",
                        "description": "Window for activity_state/state_group freshness checks. Default 14."
                    },
                },
                "required": ["query"]
            }
        ),
        Tool(
            name="retrieval_feedback",
            description=(
                "Record explicit retrieval feedback for a branch returned by hybrid_search. "
                "Use query_id from the hybrid_search response."
            ),
            inputSchema={
                "type": "object",
                "properties": {
                    "query_id": {"type": "string", "description": "query_id returned by hybrid_search"},
                    "branch_id": {"type": "string", "description": "Branch id, e.g. conv_186"},
                    "score": {"type": "number", "description": "Optional score for this feedback event"},
                    "used": {"type": "boolean", "description": "Set true when retrieval helped"},
                    "corrected": {"type": "boolean", "description": "Set true when retrieval was wrong/misleading"},
                    "expanded": {"type": "boolean", "description": "Set true when branch was expanded/followed-up"},
                    "usefulness_signal": {"type": "number", "description": "Optional explicit usefulness in [0,1]"},
                },
                "required": ["query_id", "branch_id"]
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
            name="maintenance_cycle",
            description=(
                "Run one geometry maintenance cycle. Supports low-RAM chunking via max_branches "
                "and optional chunk cursor reset."
            ),
            inputSchema={
                "type": "object",
                "properties": {
                    "max_branches": {
                        "type": "integer",
                        "description": "Optional branch cap for this run. Overrides configured chunk size for one cycle."
                    },
                    "reset_chunk_cursor": {
                        "type": "boolean",
                        "description": "Reset chunk cursor before running this cycle."
                    }
                }
            }
        ),
        Tool(
            name="geometry_snapshot",
            description=(
                "Export a compact branch snapshot for ops/debugging. "
                "Supports state filter, explicit branch IDs, row limit, and optional mean vectors."
            ),
            inputSchema={
                "type": "object",
                "properties": {
                    "branch_ids": {
                        "type": "array",
                        "items": {"type": "string"},
                        "description": "Optional explicit branch IDs to export (e.g. [\"conv_148\",\"day_2026-04-07\"])."
                    },
                    "state": {
                        "type": "string",
                        "description": "Optional lifecycle state filter (e.g. ACTIVE, STABLE, COLLAPSING, ALL)."
                    },
                    "limit": {
                        "type": "integer",
                        "description": "Optional max branches to return."
                    },
                    "include_means": {
                        "type": "boolean",
                        "description": "Include branch mean vectors in output (larger payload)."
                    }
                }
            }
        ),
        Tool(
            name="latest_correction",
            description=(
                "Resolve a correction chain and return the latest correction node/version for a seed node_id."
            ),
            inputSchema={
                "type": "object",
                "properties": {
                    "node_id": {
                        "type": "string",
                        "description": "Seed geometry node id from any point in a correction chain."
                    },
                    "branch_id": {
                        "type": "string",
                        "description": "Optional branch scope when same root may appear across branches."
                    },
                    "include_chain": {
                        "type": "boolean",
                        "description": "Include ordered correction chain payload."
                    },
                    "chain_limit": {
                        "type": "integer",
                        "description": "Max chain entries to return when include_chain=true (default 64)."
                    }
                },
                "required": ["node_id"]
            }
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
            name="backfill_lcm_conversations",
            description=(
                "Targeted LCM→geometry backfill for specific conversation IDs. "
                "Use this to repair zombie conv_* branches without full backfill."
            ),
            inputSchema={
                "type": "object",
                "properties": {
                    "conversation_ids": {
                        "type": "array",
                        "items": {"type": "integer"},
                        "description": "LCM conversation IDs to backfill (e.g. [57,365,376])."
                    },
                    "max_per_conv": {
                        "type": "integer",
                        "description": "Max messages per conversation (stratified sample cap). Default 200."
                    },
                    "resume": {
                        "type": "boolean",
                        "description": "Skip branches already present in branch_states. Default true."
                    },
                    "dry_run": {
                        "type": "boolean",
                        "description": "Only report what would run; do not write nodes."
                    }
                },
                "required": ["conversation_ids"]
            }
        ),
        Tool(
            name="conversation_content",
            description=(
                "Retrieve actual conversation text from the LCM database for geometry-identified branches. "
                "Bridges the gap between geometry metadata (branch IDs, scores) and real text content. "
                "Modes: (1) single branch by ID, (2) all branches filtered by state, state_group, or latest activity. "
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
                        "type": ["string", "array"],
                        "items": {"type": "string"},
                        "description": "Filter by branch lifecycle state. Default: ALL."
                    },
                    "state_group": {
                        "type": "string",
                        "enum": ["all", "working", "settled", "dormant"],
                        "description": "Convenience filter. working=recent FORMING/ACTIVE/REACTIVATING; settled/dormant=older STABLE."
                    },
                    "activity_state": {
                        "type": "string",
                        "enum": ["recent", "stale", "dormant", "all"],
                        "description": "Filter by latest source activity, separate from geometric lifecycle state."
                    },
                    "activity_within_days": {
                        "type": "number",
                        "description": "Window for activity_state/state_group freshness checks. Default 14."
                    },
                    "fallback_when_empty": {
                        "type": "boolean",
                        "description": "When true, empty ACTIVE multi-branch requests fall back explicitly to state_group=working. Default true."
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
    tool_started = time.time()
    auto_poll_status = None
    try:
        if name in AUTO_POLL_TOOLS:
            auto_poll_status = _poll_lcm_if_due(force=False, auto=True)
        poll_line = _poll_status_line(auto_poll_status)

        if name == "hybrid_search":
            query = arguments.get("query", "")
            top_n = arguments.get("top_n", 5)
            retrieval_mode = str(arguments.get("retrieval_mode", "balanced") or "balanced").strip().lower()
            if retrieval_mode not in ("balanced", "factual", "exploratory"):
                retrieval_mode = "balanced"
            recency_boost = _coerce_float(arguments.get("recency_boost"), 0.0)
            recency_half_life_days = _coerce_float(arguments.get("recency_half_life_days"), 14.0)
            result = do_hybrid_search(
                query,
                top_n,
                retrieval_mode=retrieval_mode,
                recency_boost=recency_boost,
                recency_half_life_days=recency_half_life_days,
                min_age_days=arguments.get("min_age_days"),
                max_age_days=arguments.get("max_age_days"),
                updated_within_days=arguments.get("updated_within_days"),
                updated_after=arguments.get("updated_after"),
                updated_before=arguments.get("updated_before"),
                date_from=arguments.get("date_from"),
                date_to=arguments.get("date_to"),
                state=arguments.get("state"),
                state_group=arguments.get("state_group"),
                activity_state=arguments.get("activity_state"),
                activity_within_days=arguments.get("activity_within_days"),
            )

            lines = [
                f" Hybrid Search: \"{query}\"",
                f" Query ID: {result.get('query_id', '-')}",
                f" Retrieval mode: {result.get('retrieval_mode', 'balanced')}",
                f" Recommendation: use {result['recommendation'].upper()}",
                "",
            ]
            if poll_line:
                lines.insert(1, poll_line)
            recency_meta = result.get("recency", {}) or {}
            if recency_meta.get("boost", 0.0) or recency_meta.get("updated_after") or recency_meta.get("updated_before"):
                lines.insert(
                    -1,
                    " Recency: "
                    f"boost={recency_meta.get('boost', 0.0)} "
                    f"half_life_days={recency_meta.get('half_life_days', 14.0)} "
                    f"after={recency_meta.get('updated_after') or '-'} "
                    f"before={recency_meta.get('updated_before') or '-'}"
                )
            filters_meta = result.get("filters", {}) or {}
            activity_meta = result.get("activity", {}) or {}
            if filters_meta.get("state") or filters_meta.get("state_group") != "all" or filters_meta.get("activity_state"):
                lines.insert(
                    -1,
                    " Filters: "
                    f"state={filters_meta.get('state') or 'ALL'} "
                    f"state_group={filters_meta.get('state_group') or 'all'} "
                    f"activity={filters_meta.get('activity_state') or 'all'} "
                    f"within_days={filters_meta.get('activity_within_days', 14.0)} "
                    f"after={activity_meta.get('activity_after') or '-'} "
                    f"before={activity_meta.get('activity_before') or '-'}"
                )
            lines.append(" GEOMETRY DB (semantic similarity):")
            for i, r in enumerate(result['geometry']['results'], 1):
                score_part = (
                    f"final={r.get('final_score')} | " if recency_meta.get("boost", 0.0) else ""
                )
                lines.append(
                    f"  {i}. {r['branch_id']} | {score_part}sem={r['sem_score']} "
                    f"| trust={r['trust_score']} | recency={r.get('recency_score', 0.0)} "
                    f"| updated={r.get('recency_label', 'unknown')} | nodes={r['nodes']} "
                    f"| eff_rank={r['eff_rank']} | {r['state']}/{r['regime']}"
                )
            lines.append("")
            lines.append(" COLLAPSING SIDECAR (excluded from primary geometry ranking):")
            sidecar_rows = result.get("collapsing_sidecar", {}).get("results", [])
            if sidecar_rows:
                for i, r in enumerate(sidecar_rows, 1):
                    score_part = (
                        f"final={r.get('final_score')} | " if recency_meta.get("boost", 0.0) else ""
                    )
                    lines.append(
                        f"  {i}. {r['branch_id']} | {score_part}sem={r['sem_score']} | trust={r['trust_score']} "
                        f"| recency={r.get('recency_score', 0.0)} | updated={r.get('recency_label', 'unknown')} "
                        f"| nodes={r['nodes']} | eff_rank={r['eff_rank']} | {r['state']}/{r['regime']}"
                    )
            else:
                lines.append("  (no collapsing candidates)")
            lines.append("")
            lines.append(" LCM (keyword matches):")
            for i, c in enumerate(result['lcm']['conversations'], 1):
                score_part = (
                    f"final={c.get('final_score')} | " if recency_meta.get("boost", 0.0) else ""
                )
                lines.append(
                    f"  {i}. conv_{c['conv_id']} | {score_part}{c['unique_keywords_matched']} kw matched "
                    f"| {c['total_matches']} total hits | updated={c.get('recency_label', 'unknown')}"
                )
                lines.append(f"     \"{c['best_snippet']}\"")
            lines.append("")
            lines.append(" DAILY LOGS (sidecar):")
            sem_rows = result.get("daily_logs", {}).get("semantic_results", [])
            kw_rows = result.get("daily_logs", {}).get("keyword_results", [])
            if sem_rows:
                lines.append("  Semantic:")
                for i, r in enumerate(sem_rows, 1):
                    score_part = (
                        f"final={r.get('final_score')} | " if recency_meta.get("boost", 0.0) else ""
                    )
                    lines.append(
                        f"   {i}. {r['branch_id']} | {score_part}sem={r['sem_score']} "
                        f"| updated={r.get('recency_label', 'unknown')} | {r['snippet']}"
                    )
            if kw_rows:
                lines.append("  Keyword:")
                for i, r in enumerate(kw_rows, 1):
                    score_part = (
                        f"final={r.get('final_score')} | " if recency_meta.get("boost", 0.0) else ""
                    )
                    lines.append(
                        f"   {i}. {r['branch_id']} | {score_part}kw={r['keyword_matched']} "
                        f"| updated={r.get('recency_label', 'unknown')} | {r['snippet']}"
                    )
            if not sem_rows and not kw_rows:
                lines.append("  (no daily-log matches)")
            feedback_meta = result.get("feedback", {}) or {}
            lines.append("")
            lines.append(
                f" Feedback: implicit signals logged={feedback_meta.get('implicit_logged', 0)} "
                f"(signal={feedback_meta.get('implicit_usefulness_signal', 0.5)})"
            )
            return [TextContent(type="text", text="\n".join(lines))]

        elif name == "retrieval_feedback":
            query_id = str(arguments.get("query_id", "") or "").strip()
            branch_id = str(arguments.get("branch_id", "") or "").strip()
            if not query_id or not branch_id:
                return [TextContent(type="text", text="Error: query_id and branch_id are required")]

            score_raw = arguments.get("score", 0.0)
            try:
                score = float(score_raw)
            except Exception:
                score = 0.0
            used = bool(arguments.get("used", False))
            corrected = bool(arguments.get("corrected", False))
            expanded = bool(arguments.get("expanded", False))
            usefulness_raw = arguments.get("usefulness_signal")
            usefulness_signal = None
            if usefulness_raw is not None:
                try:
                    usefulness_signal = max(0.0, min(1.0, float(usefulness_raw)))
                except Exception:
                    usefulness_signal = None

            gc = get_gc()
            gc.record_retrieval(
                query_id=query_id,
                branch_id=branch_id,
                score=score,
                used=used,
                corrected=corrected,
                expanded=expanded,
                usefulness_signal=usefulness_signal,
                feedback_source="explicit_tool_feedback",
            )
            lines = [
                " Retrieval feedback recorded",
                f"  query_id: {query_id}",
                f"  branch_id: {branch_id}",
                f"  score: {round(score, 4)}",
                f"  used: {used}",
                f"  corrected: {corrected}",
                f"  expanded: {expanded}",
                f"  usefulness_signal: {usefulness_signal if usefulness_signal is not None else '(derived)'}",
            ]
            if poll_line:
                lines.insert(1, poll_line)
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
                     f"  Avg role_diversity: {stats.get('avg_role_diversity', 0)}",
                     f"  Embedding: backend={stats.get('embedding_backend', 'unknown')} "
                     f"model={stats['embedding_model']} ({stats['embedding_dim']}d)",
                     f"  Warmup: enabled={stats.get('warmup', {}).get('enabled')} "
                     f"started={stats.get('warmup', {}).get('started')} "
                     f"completed={stats.get('warmup', {}).get('completed')} "
                     f"ok={stats.get('warmup', {}).get('ok')}"]
            if poll_line:
                lines.insert(1, poll_line)
            return [TextContent(type="text", text="\n".join(lines))]

        elif name == "maintenance_cycle":
            max_branches_raw = arguments.get("max_branches")
            max_branches = None
            if max_branches_raw is not None:
                try:
                    max_branches = max(1, int(max_branches_raw))
                except Exception:
                    max_branches = None
            reset_chunk_cursor = bool(arguments.get("reset_chunk_cursor", False))
            result = do_maintenance_cycle(
                max_branches=max_branches,
                reset_chunk_cursor=reset_chunk_cursor,
            )
            lines = [" Maintenance cycle completed"]
            if poll_line:
                lines.append(poll_line)
            lines.append(f"  recomputed: {result.get('recomputed', 0)}")
            lines.append(f"  split_pending: {result.get('split_pending', 0)}")
            lines.append(f"  split_executed: {result.get('split_executed', 0)}")
            lines.append(f"  merge_candidates: {result.get('merge_candidates', 0)}")
            lines.append(f"  merge_executed: {result.get('merge_executed', 0)}")
            lines.append(f"  dormant_marked: {result.get('dormant_marked', 0)}")
            lines.append(f"  reactivated: {result.get('reactivated', 0)}")
            lines.append(f"  refines_orphans_removed: {result.get('refines_orphans_removed', 0)}")
            lines.append(f"  retrieval_feedback_pruned: {result.get('retrieval_feedback_pruned', 0)}")
            lines.append(f"  retrieval_feedback_pruned_age: {result.get('retrieval_feedback_pruned_age', 0)}")
            lines.append(f"  retrieval_feedback_pruned_cap: {result.get('retrieval_feedback_pruned_cap', 0)}")
            chunk = result.get("maintenance_chunking", {}) or {}
            if chunk:
                lines.append(
                    "  chunking: "
                    f"enabled={chunk.get('enabled')} size={chunk.get('chunk_size')} "
                    f"selected={chunk.get('selected_branches')} wrapped={chunk.get('wrapped')} "
                    f"cursor_before={chunk.get('cursor_before')} cursor_after={chunk.get('cursor_after')}"
                )
            return [TextContent(type="text", text="\n".join(lines))]

        elif name == "geometry_snapshot":
            branch_ids = arguments.get("branch_ids")
            state = arguments.get("state")
            limit_raw = arguments.get("limit")
            include_means = bool(arguments.get("include_means", False))

            limit = None
            if limit_raw is not None:
                try:
                    limit = max(1, int(limit_raw))
                except Exception:
                    limit = None

            result = do_geometry_snapshot(
                branch_ids=branch_ids if isinstance(branch_ids, list) else None,
                state=state,
                limit=limit,
                include_means=include_means,
            )
            branches = result.get("branches", []) or []
            lines = [
                " Geometry snapshot export",
                f"  branch_count: {result.get('branch_count', len(branches))}",
                f"  state_filter: {result.get('state_filter', 'ALL')}",
                f"  include_means: {bool(result.get('include_means', False))}",
                f"  generated_ts: {result.get('generated_ts', 0)}",
                "",
            ]
            if poll_line:
                lines.insert(1, poll_line)
            for i, b in enumerate(branches[:50], 1):
                lines.append(
                    f"  {i}. {b.get('branch_id')} | {b.get('state')}/{b.get('regime')} "
                    f"| nodes={b.get('node_count', 0)} | eff_rank={round(float(b.get('eff_rank', 0.0) or 0.0), 3)} "
                    f"| drift={round(float(b.get('anchor_drift', 0.0) or 0.0), 4)} "
                    f"| role_div={round(float(b.get('role_diversity', 0.0) or 0.0), 4)}"
                )
            if len(branches) > 50:
                lines.append(f"  ... ({len(branches) - 50} more)")
            return [TextContent(type="text", text="\n".join(lines))]

        elif name == "latest_correction":
            node_id = str(arguments.get("node_id", "") or "").strip()
            branch_id = arguments.get("branch_id")
            include_chain = bool(arguments.get("include_chain", False))
            chain_limit_raw = arguments.get("chain_limit", 64)
            try:
                chain_limit = max(1, int(chain_limit_raw))
            except Exception:
                chain_limit = 64

            result = do_latest_correction(
                node_id=node_id,
                branch_id=branch_id,
                include_chain=include_chain,
                chain_limit=chain_limit,
            )
            if "error" in result:
                return [TextContent(type="text", text=f"Error: {result['error']}")]

            lines = [
                " Latest correction resolved",
                f"  root_node_id: {result.get('root_node_id')}",
                f"  branch_id: {result.get('branch_id')}",
                f"  latest_node_id: {result.get('latest_node_id')}",
                f"  latest_lcm_id: {result.get('latest_lcm_id')}",
                f"  latest_update_mode: {result.get('latest_update_mode')}",
                f"  latest_correction_kind: {result.get('latest_correction_kind')}",
                f"  latest_correction_version: {result.get('latest_correction_version')}",
                f"  latest_timestamp: {result.get('latest_timestamp')}",
                f"  chain_length: {result.get('chain_length', 0)}",
            ]
            if poll_line:
                lines.insert(1, poll_line)
            if include_chain:
                lines.append(f"  chain_returned: {result.get('chain_returned', 0)} (limit={result.get('chain_limit', chain_limit)})")
                chain_rows = result.get("chain", []) or []
                for i, row in enumerate(chain_rows[:40], 1):
                    lines.append(
                        f"   {i}. id={row.get('id')} ver={row.get('correction_version')} "
                        f"kind={row.get('correction_kind')} mode={row.get('update_mode')} ts={row.get('timestamp')}"
                    )
                if len(chain_rows) > 40:
                    lines.append(f"   ... ({len(chain_rows) - 40} more)")
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

        elif name == "backfill_lcm_conversations":
            conv_ids_raw = arguments.get("conversation_ids", [])
            if not isinstance(conv_ids_raw, list) or not conv_ids_raw:
                return [TextContent(type="text", text="Error: conversation_ids must be a non-empty array of integers")]
            conv_ids = []
            for x in conv_ids_raw:
                try:
                    conv_ids.append(int(x))
                except Exception:
                    continue
            conv_ids = [x for x in conv_ids if x > 0]
            if not conv_ids:
                return [TextContent(type="text", text="Error: no valid positive conversation_ids were provided")]

            max_per_conv_raw = arguments.get("max_per_conv", 200)
            try:
                max_per_conv = max(2, int(max_per_conv_raw))
            except Exception:
                max_per_conv = 200
            resume = bool(arguments.get("resume", True))
            dry_run = bool(arguments.get("dry_run", False))

            gc = get_gc()
            result = gc.backfill_selected_conversations_from_lcm(
                lcm_db_path=LCM_DB,
                conversation_ids=conv_ids,
                max_per_conv=max_per_conv,
                resume=resume,
                dry_run=dry_run,
            )
            title = " Targeted backfill completed"
            if result.get("aborted", False):
                title = " Targeted backfill aborted (preflight)"
            lines = [title]
            if poll_line:
                lines.append(poll_line)
            lines.append(f"  requested: {result.get('requested', 0)}")
            lines.append(f"  found_with_messages: {result.get('found_with_messages', 0)}")
            lines.append(f"  processed: {result.get('processed', 0)}")
            lines.append(f"  skipped: {result.get('skipped', 0)} (resume={result.get('skipped_resume', 0)}, empty={result.get('skipped_empty', 0)})")
            lines.append(f"  failed: {result.get('failed', 0)}")
            lines.append(f"  sampled: {result.get('sampled', 0)}")
            lines.append(f"  dry_run: {result.get('dry_run', False)}")
            if "provider_ready" in result:
                lines.append(f"  provider_ready: {result.get('provider_ready')}")
            if result.get("preflight_error"):
                lines.append(f"  preflight_error: {result.get('preflight_error')}")
            details = result.get("details", []) or []
            if details:
                lines.append("  details:")
                for d in details[:20]:
                    lines.append(
                        f"   - conv_{d.get('conversation_id')} -> {d.get('status')} "
                        f"(messages={d.get('messages')}, sampled={d.get('sampled', False)})"
                    )
                if len(details) > 20:
                    lines.append(f"   - ... ({len(details) - 20} more)")
            return [TextContent(type="text", text="\n".join(lines))]

        elif name == "conversation_content":
            branch_id = arguments.get("branch_id")
            state = arguments.get("state")
            state_group = arguments.get("state_group")
            activity_state = arguments.get("activity_state")
            activity_within_days = arguments.get("activity_within_days")
            fallback_when_empty = arguments.get("fallback_when_empty", True)
            content_type = arguments.get("content_type", "summaries")
            max_entries = arguments.get("max_entries", 100)
            max_chars = arguments.get("max_chars", 250)

            result = do_conversation_content(
                branch_id,
                state,
                content_type,
                max_entries,
                max_chars,
                state_group=state_group,
                activity_state=activity_state,
                activity_within_days=activity_within_days,
                fallback_when_empty=fallback_when_empty,
            )

            if 'error' in result:
                extra = ""
                if result.get("filters"):
                    extra += f"\nFilters: {json.dumps(result.get('filters'), ensure_ascii=False)}"
                if result.get("fallback"):
                    extra += f"\nFallback: {json.dumps(result.get('fallback'), ensure_ascii=False)}"
                return [TextContent(type="text", text=f"Error: {result['error']}{extra}")]

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
            filters = result.get("filters") or {}
            if filters:
                lines.insert(
                    -1,
                    "  Filters: "
                    f"state={filters.get('state') or 'ALL'} "
                    f"state_group={filters.get('state_group') or 'all'} "
                    f"activity={filters.get('activity_state') or 'all'} "
                    f"within_days={filters.get('activity_within_days', 14.0)}"
                )
            fallback = result.get("fallback") or {}
            if fallback.get("used"):
                lines.insert(
                    -1,
                    f"  Fallback: {fallback.get('reason')} "
                    f"from={fallback.get('from')} to={fallback.get('to')}"
                )

            for r in result['results']:
                resolved_ids = r.get("resolved_conversation_ids") or []
                resolved_text = ",".join(str(x) for x in resolved_ids[:8]) if resolved_ids else "-"
                lines.append(
                    f"--- {r['branch_id']} | {r['state']}/{r['regime']} | "
                    f"{r['entries_returned']} entries | mode={r.get('resolution_mode','unknown')} "
                    f"| activity={r.get('activity_state','unknown')} "
                    f"| updated={r.get('activity_label','unknown')} | resolved_conv={resolved_text} ---"
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
    finally:
        elapsed_ms = int((time.time() - tool_started) * 1000)
        poll_tag = "poll=none"
        if isinstance(auto_poll_status, dict):
            if auto_poll_status.get("error"):
                poll_tag = f"poll=error:{auto_poll_status.get('error')}"
            elif auto_poll_status.get("skipped"):
                poll_tag = f"poll=skip:{auto_poll_status.get('skipped')}"
            else:
                poll_tag = (
                    f"poll=ok polled={int(auto_poll_status.get('polled', 0) or 0)} "
                    f"processed={int(auto_poll_status.get('processed', 0) or 0)} "
                    f"failed={int(auto_poll_status.get('failed', 0) or 0)}"
                )
        print(
            f"[geometry-mcp] tool={name} dur_ms={elapsed_ms} {poll_tag} "
            f"warmup_running={_is_warmup_running()}",
            file=sys.stderr,
        )


#  Main 
async def main():
    try:
        async with stdio_server() as (read_stream, write_stream):
            await server.run(read_stream, write_stream, server.create_initialization_options())
    except Exception as exc:
        print(f"[geometry-mcp] fatal main error: {exc}", file=sys.stderr)
        raise

if __name__ == "__main__":
    asyncio.run(main())
