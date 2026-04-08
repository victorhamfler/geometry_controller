#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import os
import shutil
import sqlite3
import time
from pathlib import Path
from typing import Any


ROOT = Path(__file__).resolve().parents[1]
import sys

sys.path.insert(0, str(ROOT))

from lcm_geometry_controller import GeometryController  # noqa: E402


def _stats(conn: sqlite3.Connection) -> dict[str, Any]:
    c = conn.cursor()
    out: dict[str, Any] = {}
    out["branches"] = int(c.execute("SELECT COUNT(*) FROM branch_states").fetchone()[0] or 0)
    out["nodes_total"] = int(c.execute("SELECT COUNT(*) FROM memory_nodes").fetchone()[0] or 0)
    out["edges_total"] = int(c.execute("SELECT COUNT(*) FROM memory_edges").fetchone()[0] or 0)
    out["message_rows"] = int(
        c.execute("SELECT COUNT(*) FROM memory_nodes WHERE node_type='message'").fetchone()[0] or 0
    )
    out["message_distinct_lcm"] = int(
        c.execute(
            "SELECT COUNT(DISTINCT lcm_id) FROM memory_nodes WHERE node_type='message'"
        ).fetchone()[0]
        or 0
    )
    out["message_duplicate_rows"] = int(out["message_rows"] - out["message_distinct_lcm"])
    out["duplicate_lcm_ids"] = int(
        c.execute(
            """
            SELECT COUNT(*) FROM (
              SELECT lcm_id, COUNT(*) n
              FROM memory_nodes
              WHERE node_type='message'
              GROUP BY lcm_id
              HAVING n > 1
            )
            """
        ).fetchone()[0]
        or 0
    )
    out["edge_types"] = {
        str(r[0]): int(r[1] or 0)
        for r in c.execute("SELECT edge_type, COUNT(*) FROM memory_edges GROUP BY edge_type").fetchall()
    }
    out["states"] = {
        str(r[0]): int(r[1] or 0)
        for r in c.execute("SELECT state, COUNT(*) FROM branch_states GROUP BY state").fetchall()
    }
    return out


def _build_duplicate_map(conn: sqlite3.Connection) -> list[tuple[str, str]]:
    # keep_id = first row by rowid within (branch_id, lcm_id, node_type='message')
    rows = conn.execute(
        """
        WITH ranked AS (
          SELECT id, branch_id, lcm_id, rowid,
                 ROW_NUMBER() OVER (
                   PARTITION BY branch_id, lcm_id
                   ORDER BY rowid ASC
                 ) AS rn,
                 FIRST_VALUE(id) OVER (
                   PARTITION BY branch_id, lcm_id
                   ORDER BY rowid ASC
                 ) AS keep_id
          FROM memory_nodes
          WHERE node_type='message'
        )
        SELECT id AS remove_id, keep_id
        FROM ranked
        WHERE rn > 1
        """
    ).fetchall()
    return [(str(r[0]), str(r[1])) for r in rows]


def _sqlite_backup(src_db: Path, dst_db: Path) -> None:
    dst_db.parent.mkdir(parents=True, exist_ok=True)
    src = sqlite3.connect(str(src_db))
    try:
        dst = sqlite3.connect(str(dst_db))
        try:
            src.backup(dst)
        finally:
            dst.close()
    finally:
        src.close()


def _chunked(seq: list[Any], n: int) -> list[list[Any]]:
    return [seq[i : i + n] for i in range(0, len(seq), n)]


def _rebuild_message_correction_meta(conn: sqlite3.Connection) -> tuple[int, int]:
    updated = 0
    edge_rows: list[tuple[str, str, str, float]] = []
    branches = [
        str(r[0])
        for r in conn.execute(
            "SELECT DISTINCT branch_id FROM memory_nodes WHERE node_type='message' ORDER BY branch_id"
        ).fetchall()
    ]
    for bid in branches:
        rows = conn.execute(
            """
            SELECT id, update_mode, conflict_score
            FROM memory_nodes
            WHERE node_type='message' AND branch_id=?
            ORDER BY timestamp ASC, rowid ASC
            """,
            (bid,),
        ).fetchall()
        prev_id: str | None = None
        prev_root: str | None = None
        prev_ver = 1
        updates: list[tuple[str, str | None, str | None, int, str]] = []
        for r in rows:
            node_id = str(r[0])
            mode = str(r[1] or "attach").strip().lower()
            conflict = float(r[2] or 0.0)
            if prev_id and mode in ("refine", "contradict", "supersede"):
                kind = mode
                corr_prev = prev_id
                corr_root = prev_root if prev_root else prev_id
                corr_ver = max(1, int(prev_ver) + 1)
                if kind == "refine":
                    edge_type = "refines"
                    weight = 1.0
                elif kind == "contradict":
                    edge_type = "contradicts"
                    weight = max(0.01, conflict)
                else:
                    edge_type = "supersedes"
                    weight = max(0.01, conflict)
                edge_rows.append((node_id, corr_prev, edge_type, float(weight)))
            else:
                kind = "none"
                corr_prev = None
                corr_root = None
                corr_ver = 1
            updates.append((kind, corr_prev, corr_root, int(corr_ver), node_id))
            prev_id = node_id
            prev_root = corr_root
            prev_ver = int(corr_ver)

        conn.executemany(
            """
            UPDATE memory_nodes
            SET correction_kind=?,
                correction_prev_id=?,
                correction_root_id=?,
                correction_version=?
            WHERE id=?
            """,
            updates,
        )
        updated += len(updates)
    return updated, len(edge_rows)


def _rebuild_temporal_edges(conn: sqlite3.Connection) -> int:
    rows = conn.execute(
        """
        SELECT branch_id, id
        FROM memory_nodes
        ORDER BY branch_id ASC, timestamp ASC, rowid ASC
        """
    ).fetchall()
    edges: list[tuple[str, str, str, float]] = []
    prev_by_branch: dict[str, str] = {}
    for r in rows:
        bid = str(r[0])
        nid = str(r[1])
        prev = prev_by_branch.get(bid)
        if prev and prev != nid:
            edges.append((prev, nid, "temporal_next", 1.0))
        prev_by_branch[bid] = nid
    if edges:
        conn.executemany(
            "INSERT OR REPLACE INTO memory_edges (src_id, dst_id, edge_type, weight) VALUES (?, ?, ?, ?)",
            edges,
        )
    return len(edges)


def _apply_cleanup(db_path: Path) -> dict[str, Any]:
    conn = sqlite3.connect(str(db_path))
    conn.row_factory = sqlite3.Row
    out: dict[str, Any] = {}
    try:
        dup_map = _build_duplicate_map(conn)
        out["duplicate_rows_to_remove"] = len(dup_map)
        if not dup_map:
            out["changed"] = False
            return out

        conn.execute("BEGIN")
        conn.execute("CREATE TEMP TABLE tmp_dedupe_map(remove_id TEXT PRIMARY KEY, keep_id TEXT NOT NULL)")
        conn.executemany(
            "INSERT INTO tmp_dedupe_map(remove_id, keep_id) VALUES (?, ?)",
            dup_map,
        )

        # Rewire references.
        conn.execute(
            """
            UPDATE memory_nodes
            SET parent_id = (SELECT keep_id FROM tmp_dedupe_map WHERE remove_id = parent_id)
            WHERE parent_id IN (SELECT remove_id FROM tmp_dedupe_map)
            """
        )
        conn.execute(
            """
            UPDATE memory_nodes
            SET correction_prev_id = (SELECT keep_id FROM tmp_dedupe_map WHERE remove_id = correction_prev_id)
            WHERE correction_prev_id IN (SELECT remove_id FROM tmp_dedupe_map)
            """
        )
        conn.execute(
            """
            UPDATE memory_nodes
            SET correction_root_id = (SELECT keep_id FROM tmp_dedupe_map WHERE remove_id = correction_root_id)
            WHERE correction_root_id IN (SELECT remove_id FROM tmp_dedupe_map)
            """
        )

        # Remove edges touching removed nodes, then remove removed nodes.
        conn.execute(
            """
            DELETE FROM memory_edges
            WHERE src_id IN (SELECT remove_id FROM tmp_dedupe_map)
               OR dst_id IN (SELECT remove_id FROM tmp_dedupe_map)
            """
        )
        conn.execute("DELETE FROM memory_nodes WHERE id IN (SELECT remove_id FROM tmp_dedupe_map)")

        # Rebuild correction metadata + correction edges for message nodes.
        conn.execute("DELETE FROM memory_edges WHERE edge_type IN ('refines','contradicts','supersedes')")
        msg_updated, correction_edges = _rebuild_message_correction_meta(conn)

        # Rebuild temporal_next globally from remaining nodes.
        conn.execute("DELETE FROM memory_edges WHERE edge_type='temporal_next'")
        temporal_edges = _rebuild_temporal_edges(conn)

        # Insert rebuilt correction edges.
        corr_rows = conn.execute(
            """
            SELECT id, correction_kind, correction_prev_id, conflict_score
            FROM memory_nodes
            WHERE node_type='message' AND correction_prev_id IS NOT NULL
            """
        ).fetchall()
        rebuilt_corr_edges: list[tuple[str, str, str, float]] = []
        for r in corr_rows:
            node_id = str(r["id"])
            prev_id = str(r["correction_prev_id"])
            kind = str(r["correction_kind"] or "none")
            conflict = float(r["conflict_score"] or 0.0)
            if kind == "refine":
                rebuilt_corr_edges.append((node_id, prev_id, "refines", 1.0))
            elif kind == "contradict":
                rebuilt_corr_edges.append((node_id, prev_id, "contradicts", max(0.01, conflict)))
            elif kind == "supersede":
                rebuilt_corr_edges.append((node_id, prev_id, "supersedes", max(0.01, conflict)))
        if rebuilt_corr_edges:
            conn.executemany(
                "INSERT OR REPLACE INTO memory_edges (src_id, dst_id, edge_type, weight) VALUES (?, ?, ?, ?)",
                rebuilt_corr_edges,
            )

        conn.execute("DROP TABLE tmp_dedupe_map")
        conn.commit()
        out["changed"] = True
        out["message_rows_updated"] = int(msg_updated)
        out["temporal_edges_rebuilt"] = int(temporal_edges)
        out["correction_edges_rebuilt"] = int(len(rebuilt_corr_edges))
        out["correction_edges_candidates"] = int(correction_edges)
        return out
    except Exception:
        conn.rollback()
        raise
    finally:
        conn.close()


def main() -> int:
    parser = argparse.ArgumentParser(description="Clean duplicate message ingests from lcm_geometry.db")
    parser.add_argument(
        "--db",
        default="/home/victo/.openclaw/lcm_geometry.db",
        help="Path to geometry db",
    )
    parser.add_argument("--apply", action="store_true", help="Apply cleanup (default is dry-run)")
    parser.add_argument(
        "--run-maintenance",
        action="store_true",
        help="Run one maintenance cycle after cleanup",
    )
    parser.add_argument(
        "--backup",
        default="",
        help="Optional explicit backup file path. If omitted and --apply set, auto timestamped backup is created.",
    )
    parser.add_argument(
        "--report-json",
        default="",
        help="Optional file path to write JSON report",
    )
    args = parser.parse_args()

    db_path = Path(args.db).expanduser()
    if not db_path.exists():
        raise SystemExit(f"DB not found: {db_path}")

    conn = sqlite3.connect(str(db_path))
    try:
        before = _stats(conn)
        dup_map = _build_duplicate_map(conn)
    finally:
        conn.close()

    report: dict[str, Any] = {
        "db": str(db_path),
        "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
        "mode": "apply" if args.apply else "dry-run",
        "before": before,
        "duplicate_rows_to_remove": len(dup_map),
        "changed": False,
    }

    if args.apply:
        if args.backup:
            backup_path = Path(args.backup).expanduser()
        else:
            ts = time.strftime("%Y%m%d_%H%M%S")
            backup_path = db_path.with_suffix(db_path.suffix + f".backup_{ts}")
        _sqlite_backup(db_path, backup_path)
        report["backup"] = str(backup_path)

        cleanup_out = _apply_cleanup(db_path)
        report.update(cleanup_out)

        # Refresh branch stats/geometry lifecycle after dedupe.
        maintenance_out: dict[str, Any] | None = None
        if args.run_maintenance:
            gc = GeometryController(str(db_path))
            maintenance_out = gc.run_maintenance_cycle()
            report["maintenance"] = maintenance_out

        conn2 = sqlite3.connect(str(db_path))
        try:
            report["after"] = _stats(conn2)
        finally:
            conn2.close()

    print(json.dumps(report, ensure_ascii=False, indent=2))

    if args.report_json:
        rp = Path(args.report_json).expanduser()
        rp.parent.mkdir(parents=True, exist_ok=True)
        rp.write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
        print(f"REPORT_WRITTEN {rp}")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
