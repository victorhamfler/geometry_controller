#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import os
import sqlite3
import sys
import time
from pathlib import Path
from typing import Any


REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT))

from lcm_geometry_controller import EmbeddingProvider, GeometryController  # noqa: E402


def _default_openclaw_home() -> Path:
    return Path(os.environ.get("OPENCLAW_HOME", str(Path.home() / ".openclaw"))).expanduser()


def _query_topology(c: sqlite3.Cursor, lcm_db: Path) -> dict[str, int]:
    c.execute("ATTACH DATABASE ? AS lcm", (str(lcm_db),))
    out = {
        "lcm_conversations": int(
            c.execute("SELECT COUNT(DISTINCT conversation_id) c FROM lcm.messages").fetchone()["c"]
        ),
        "missing_conv_branches": int(
            c.execute(
                """
                SELECT COUNT(*) AS c
                FROM (SELECT DISTINCT conversation_id AS cid FROM lcm.messages) x
                LEFT JOIN branch_states bs ON bs.branch_id = ('conv_' || x.cid)
                WHERE bs.branch_id IS NULL
                """
            ).fetchone()["c"]
        ),
        "extra_branches_not_in_lcm": int(
            c.execute(
                """
                SELECT COUNT(*) AS c
                FROM branch_states bs
                LEFT JOIN (SELECT DISTINCT conversation_id AS cid FROM lcm.messages) x
                ON bs.branch_id = ('conv_' || x.cid)
                WHERE x.cid IS NULL
                """
            ).fetchone()["c"]
        ),
        "node_branch_mismatch": int(
            c.execute(
                """
                SELECT COUNT(*) AS c
                FROM memory_nodes mn
                JOIN lcm.messages m ON m.message_id = mn.lcm_id
                WHERE mn.branch_id != ('conv_' || m.conversation_id)
                """
            ).fetchone()["c"]
        ),
    }
    return out


def _query_split_trace_summary(c: sqlite3.Cursor, run_id: str) -> dict[str, int]:
    row = c.execute(
        """
        SELECT
          COUNT(*) AS obs_total,
          SUM(CASE WHEN gate_nodes=1 THEN 1 ELSE 0 END) AS gate_nodes_true,
          SUM(CASE WHEN gate_score=1 THEN 1 ELSE 0 END) AS gate_score_true,
          SUM(CASE WHEN gate_hysteresis=1 THEN 1 ELSE 0 END) AS gate_hysteresis_true,
          SUM(CASE WHEN should_split=1 THEN 1 ELSE 0 END) AS should_split_true,
          SUM(CASE WHEN enqueued=1 THEN 1 ELSE 0 END) AS enqueued_true,
          SUM(CASE WHEN reason LIKE '%eligible_candidate%' THEN 1 ELSE 0 END) AS candidate_count,
          SUM(CASE WHEN reason LIKE '%eligible_throttled%' THEN 1 ELSE 0 END) AS throttled_count,
          SUM(CASE WHEN reason LIKE '%eligible_pending_exists%' THEN 1 ELSE 0 END) AS pending_exists_count,
          SUM(CASE WHEN reason LIKE '%eligible_enqueued_rank:%' THEN 1 ELSE 0 END) AS enqueued_ranked_count
        FROM maintenance_split_observations
        WHERE run_id=?
        """,
        (run_id,),
    ).fetchone()
    return {k: int(row[k] or 0) for k in row.keys()}


def _query_final_totals(c: sqlite3.Cursor) -> dict[str, int]:
    return {
        "branches": int(c.execute("SELECT COUNT(*) c FROM branch_states").fetchone()["c"]),
        "nodes": int(c.execute("SELECT COUNT(*) c FROM memory_nodes").fetchone()["c"]),
        "edges": int(c.execute("SELECT COUNT(*) c FROM memory_edges").fetchone()["c"]),
        "pending_jobs": int(
            c.execute("SELECT COUNT(*) c FROM maintenance_jobs WHERE status='pending'").fetchone()["c"]
        ),
        "pending_split_jobs": int(
            c.execute(
                "SELECT COUNT(*) c FROM maintenance_jobs WHERE status='pending' AND job_type='split'"
            ).fetchone()["c"]
        ),
    }


def main() -> int:
    parser = argparse.ArgumentParser(description="Run full ML split regression on lcm_geometry_controller.")
    parser.add_argument("--openclaw-home", default=str(_default_openclaw_home()))
    parser.add_argument("--lcm-db", default=None)
    parser.add_argument("--test-db", default=None)
    parser.add_argument("--error-log", default=None)
    parser.add_argument("--model", default="all-MiniLM-L6-v2")
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--max-per-conv", type=int, default=200)
    parser.add_argument("--cycles", type=int, default=3)
    parser.add_argument("--keep-existing", action="store_true")
    args = parser.parse_args()

    openclaw_home = Path(args.openclaw_home).expanduser()
    lcm_db = Path(args.lcm_db) if args.lcm_db else openclaw_home / "lcm.db"
    test_db = (
        Path(args.test_db)
        if args.test_db
        else openclaw_home / "lcm_geometry.test_backfill_locked_ml.db"
    )
    error_log = (
        Path(args.error_log)
        if args.error_log
        else openclaw_home / "lcm_geometry.test_backfill_locked_ml.errors.log"
    )

    if not lcm_db.exists():
        print(f"ERROR lcm.db not found: {lcm_db}", file=sys.stderr)
        return 2

    if not args.keep_existing:
        for p in (test_db, error_log):
            try:
                p.unlink()
            except FileNotFoundError:
                pass

    print("TEST_DB", str(test_db))
    print("LCM_DB", str(lcm_db))

    provider = EmbeddingProvider(model_name=args.model, device=args.device)
    gc = GeometryController(str(test_db), embedding_provider=provider)

    t0 = time.time()
    backfill_stats = gc.backfill_from_lcm(
        lcm_db_path=str(lcm_db),
        max_per_conv=int(args.max_per_conv),
        resume=False,
        progress_cb=None,
        error_log_path=str(error_log),
    )
    print("BACKFILL_STATS", backfill_stats)
    print("BACKFILL_SECONDS", round(time.time() - t0, 2))

    edge_stats = gc.import_dag_edges_from_lcm(str(lcm_db))
    print("EDGE_IMPORT_STATS", edge_stats)

    conn = sqlite3.connect(str(test_db))
    conn.row_factory = sqlite3.Row
    c = conn.cursor()

    topology = _query_topology(c, lcm_db)
    print("TOPOLOGY_CHECK", topology)
    print(
        "CONFIG",
        {
            "split_score_threshold": gc.cfg.split_score_threshold,
            "split_hysteresis": gc.cfg.split_hysteresis,
            "split_min_nodes": gc.cfg.split_min_nodes,
            "min_branch_size": gc.cfg.min_branch_size,
            "max_split_enqueues_per_cycle": gc.cfg.max_split_enqueues_per_cycle,
        },
    )

    cycle_results: list[dict[str, Any]] = []
    trace_summaries: list[dict[str, Any]] = []
    for i in range(1, max(1, int(args.cycles)) + 1):
        out = gc.run_maintenance_cycle()
        run_id = out.get("split_trace_run_id")
        print(f"CYCLE_{i}_RESULT", out)
        cycle_results.append(out)
        if run_id:
            trace = _query_split_trace_summary(c, str(run_id))
            print(f"CYCLE_{i}_TRACE_SUMMARY", trace)
            trace_summaries.append({"cycle": i, "run_id": str(run_id), "summary": trace})

    final_totals = _query_final_totals(c)
    print("FINAL_TOTALS", final_totals)
    conn.close()

    failures: list[str] = []
    if int(backfill_stats.get("failed", 0)) != 0:
        failures.append(f"backfill_failed={backfill_stats.get('failed')}")
    for key in ("missing_conv_branches", "extra_branches_not_in_lcm", "node_branch_mismatch"):
        if int(topology.get(key, 0)) != 0:
            failures.append(f"topology_{key}={topology.get(key)}")
    if trace_summaries:
        last = trace_summaries[-1]["summary"]
        cap = int(gc.cfg.max_split_enqueues_per_cycle)
        if int(last["enqueued_true"]) > cap:
            failures.append(f"enqueued_exceeds_cap={last['enqueued_true']}>{cap}")
        if int(last["should_split_true"]) > 0 and cap > 0 and int(last["enqueued_true"]) == 0:
            failures.append("expected_some_enqueues_but_got_zero")
    if int(final_totals["pending_split_jobs"]) != 0:
        failures.append(f"pending_split_jobs={final_totals['pending_split_jobs']}")

    summary = {
        "ok": len(failures) == 0,
        "failures": failures,
        "backfill_stats": backfill_stats,
        "topology": topology,
        "last_trace": trace_summaries[-1] if trace_summaries else None,
        "final_totals": final_totals,
    }
    print("REGRESSION_SUMMARY", json.dumps(summary, ensure_ascii=True))

    if failures:
        for f in failures:
            print(f"FAIL {f}", file=sys.stderr)
        return 1

    print("REGRESSION_OK")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

