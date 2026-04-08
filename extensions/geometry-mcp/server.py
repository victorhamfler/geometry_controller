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

print(
    f"[geometry-mcp] runtime cfg loaded: geo_db={GEO_DB} lcm_db={LCM_DB} "
    f"model={EMBED_MODEL_NAME} geometry_overrides={len(_GEOMETRY_CFG_OVERRIDES)}",
    file=sys.stderr,
)

#  Lazy-load heavy libs 
_model = None
_gc = None

def get_model():
    global _model
    if _model is None:
        from sentence_transformers import SentenceTransformer
        _model = SentenceTransformer(EMBED_MODEL_NAME)
    return _model

def get_gc():
    global _gc
    if _gc is None:
        from lcm_geometry_controller import GeometryController, GeometryConfig
        if _GEOMETRY_CFG_OVERRIDES:
            valid_keys = set(getattr(GeometryConfig, "__dataclass_fields__", {}).keys())
            safe_overrides = {}
            for k, v in _GEOMETRY_CFG_OVERRIDES.items():
                if k in valid_keys:
                    safe_overrides[k] = v
                else:
                    print(f"[geometry-mcp] Ignoring unknown geometry_config key: {k}", file=sys.stderr)
            _gc = GeometryController(GEO_DB, cfg=GeometryConfig(**safe_overrides))
        else:
            _gc = GeometryController(GEO_DB)
    return _gc


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

#  Hybrid search 
def do_hybrid_search(query, top_n=5):
    model = get_model()
    gc = get_gc()

    # Encode query
    q_emb = model.encode([query], normalize_embeddings=True)[0].tolist()

    # Geometry ranking
    ranked = gc.rank_retrieval(q_emb)
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
        'recommendation': recommendation,
        'geometry': {'results': geo_results},
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
            entries.append({"type": "summary", "kind": r["kind"], "text": r["text"]})

    if content_type in ("messages", "both"):
        cur = lcm_conn.execute(
            "SELECT role, SUBSTR(content, 1, ?) as text, created_at "
            "FROM messages WHERE conversation_id = ? ORDER BY seq LIMIT ?",
            (max_chars, conv_id, max_entries)
        )
        for r in cur.fetchall():
            entries.append({"type": "message", "role": r["role"], "created_at": r["created_at"], "text": r["text"]})

    return entries

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
        else:
            cid = int(branch_id.replace("conv_", ""))
            entries = _get_lcm_content(lcm, cid, content_type, max_entries, max_chars)
        results.append({
            "branch_id": branch_id,
            "state": b["state"],
            "regime": b["regime"],
            "entries_returned": len(entries),
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
            else:
                cid = int(b["branch_id"].replace("conv_", ""))
                entries = _get_lcm_content(lcm, cid, content_type, per_branch, max_chars)
            results.append({
                "branch_id": b["branch_id"],
                "state": b["state"],
                "regime": b["regime"],
                "entries_returned": len(entries),
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
                    "top_n": {"type": "integer", "description": "Results per system (default 5)"}
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
        if name == "hybrid_search":
            query = arguments.get("query", "")
            top_n = arguments.get("top_n", 5)
            result = do_hybrid_search(query, top_n)

            lines = [f" Hybrid Search: \"{query}\"", f" Recommendation: use {result['recommendation'].upper()}", ""]
            lines.append(" GEOMETRY DB (semantic similarity):")
            for i, r in enumerate(result['geometry']['results'], 1):
                lines.append(f"  {i}. {r['branch_id']} | sem={r['sem_score']} | trust={r['trust_score']} | nodes={r['nodes']} | eff_rank={r['eff_rank']} | {r['state']}/{r['regime']}")
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

            for r in result['results']:
                lines.append(f"--- {r['branch_id']} | {r['state']}/{r['regime']} | {r['entries_returned']} entries ---")
                for e in r['content']:
                    if e['type'] == 'summary':
                        lines.append(f"\n  [{e['type'].upper()}/{e['kind']}] {e['text']}")
                    elif e['type'] == 'daily_log':
                        lines.append(f"\n  [DAILY_LOG/{e.get('source','manual_log')}] {e['text']}")
                    else:
                        lines.append(f"\n  [{e['role']}] {e['text']}")
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

