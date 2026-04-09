#!/usr/bin/env python3
"""
Step 2 physical rename migration:
- branch_states.contradiction_density -> branch_states.topic_drift_density (additive/backfill)
- memory_edges.edge_type 'contradicts' -> 'topic_drift' (heuristic-safe migration)

This migration preserves likely correction-lineage contradiction edges:
- src node points to dst via correction_prev_id
- dst node points to src via parent_id
"""

from __future__ import annotations

import argparse
import os
import shutil
import sqlite3
import time
from typing import Any


def _backup(db_path: str) -> str:
    ts = time.strftime("%Y%m%d_%H%M%S")
    dst = f"{db_path}.bak_step2_topic_drift_{ts}"
    shutil.copy2(db_path, dst)
    return dst


def _table_columns(conn: sqlite3.Connection, table: str) -> set[str]:
    rows = conn.execute(f"PRAGMA table_info({table})").fetchall()
    return {str(r[1]) for r in rows}


def migrate(db_path: str, make_backup: bool = True) -> dict[str, Any]:
    if not os.path.isfile(db_path):
        raise FileNotFoundError(f"DB not found: {db_path}")

    backup_path = _backup(db_path) if make_backup else ""
    conn = sqlite3.connect(db_path)
    conn.row_factory = sqlite3.Row
    out: dict[str, Any] = {"db": db_path, "backup": backup_path}
    try:
        conn.execute("BEGIN")
        cols = _table_columns(conn, "branch_states")
        if "topic_drift_density" not in cols:
            conn.execute(
                "ALTER TABLE branch_states ADD COLUMN topic_drift_density REAL DEFAULT 0.0"
            )

        cols = _table_columns(conn, "branch_states")
        if "contradiction_density" in cols:
            backfill_cur = conn.execute(
                """
                UPDATE branch_states
                SET topic_drift_density = COALESCE(contradiction_density, 0.0)
                WHERE topic_drift_density IS NULL
                   OR ABS(COALESCE(topic_drift_density, 0.0)) < 1e-12
                """
            )
            out["topic_drift_backfilled_rows"] = int(backfill_cur.rowcount or 0)
        else:
            out["topic_drift_backfilled_rows"] = 0

        total_contradicts = int(
            conn.execute(
                "SELECT COUNT(*) AS c FROM memory_edges WHERE edge_type='contradicts'"
            ).fetchone()["c"]
            or 0
        )

        # Preserve likely correction lineage contradictions.
        preserved = int(
            conn.execute(
                """
                SELECT COUNT(*) AS c
                FROM memory_edges e
                LEFT JOIN memory_nodes src ON src.id=e.src_id
                LEFT JOIN memory_nodes dst ON dst.id=e.dst_id
                WHERE e.edge_type='contradicts'
                  AND (
                    (src.correction_prev_id IS NOT NULL AND src.correction_prev_id=e.dst_id)
                    OR (dst.parent_id IS NOT NULL AND dst.parent_id=e.src_id)
                  )
                """
            ).fetchone()["c"]
            or 0
        )

        migrate_cur = conn.execute(
            """
            UPDATE memory_edges
            SET edge_type='topic_drift'
            WHERE edge_type='contradicts'
              AND NOT EXISTS (
                SELECT 1 FROM memory_nodes src
                WHERE src.id=memory_edges.src_id
                  AND src.correction_prev_id IS NOT NULL
                  AND src.correction_prev_id=memory_edges.dst_id
              )
              AND NOT EXISTS (
                SELECT 1 FROM memory_nodes dst
                WHERE dst.id=memory_edges.dst_id
                  AND dst.parent_id IS NOT NULL
                  AND dst.parent_id=memory_edges.src_id
              )
            """
        )
        migrated = int(migrate_cur.rowcount or 0)

        # Keep legacy column synchronized for compatibility window.
        if "contradiction_density" in cols:
            sync_cur = conn.execute(
                "UPDATE branch_states SET contradiction_density = COALESCE(topic_drift_density, 0.0)"
            )
            out["legacy_sync_rows"] = int(sync_cur.rowcount or 0)
        else:
            out["legacy_sync_rows"] = 0

        conn.commit()

        out["edges_contradicts_before"] = total_contradicts
        out["edges_contradicts_preserved"] = preserved
        out["edges_migrated_to_topic_drift"] = migrated
        out["edges_topic_drift_after"] = int(
            conn.execute(
                "SELECT COUNT(*) AS c FROM memory_edges WHERE edge_type='topic_drift'"
            ).fetchone()["c"]
            or 0
        )
        out["edges_contradicts_after"] = int(
            conn.execute(
                "SELECT COUNT(*) AS c FROM memory_edges WHERE edge_type='contradicts'"
            ).fetchone()["c"]
            or 0
        )
        out["branches_nonzero_topic_drift"] = int(
            conn.execute(
                "SELECT COUNT(*) AS c FROM branch_states WHERE COALESCE(topic_drift_density,0.0) > 0"
            ).fetchone()["c"]
            or 0
        )
        return out
    except Exception:
        conn.rollback()
        raise
    finally:
        conn.close()


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--db", required=True, help="Path to lcm_geometry.db")
    parser.add_argument(
        "--no-backup",
        action="store_true",
        help="Disable backup creation before migration",
    )
    args = parser.parse_args()
    result = migrate(args.db, make_backup=not bool(args.no_backup))
    for k, v in result.items():
        print(f"{k}: {v}")


if __name__ == "__main__":
    main()

