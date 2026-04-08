#!/usr/bin/env python3
from __future__ import annotations

import json
import os
import sys
import tempfile

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)

from lcm_geometry_controller import GeometryConfig, GeometryController, NodeType  # noqa: E402


def _assert(condition: bool, message: str) -> None:
    if not condition:
        raise AssertionError(message)


def main() -> int:
    cfg = GeometryConfig(
        embedding_dim=8,
        update_mode_refine_similarity_min=0.90,
        update_mode_contradict_conflict_min=0.40,
        update_mode_supersede_similarity_min=0.70,
        update_mode_supersede_conflict_min=0.80,
        update_mode_supersede_branch_types=["broad_history"],
    )

    emb_a = [0.10, 0.21, 0.31, 0.42, 0.02, 0.09, 0.19, 0.29]
    emb_b = [0.11, 0.20, 0.30, 0.41, 0.02, 0.10, 0.18, 0.30]

    with tempfile.TemporaryDirectory(prefix="lcm_geo_update_mode_") as tmp:
        db_path = os.path.join(tmp, "update_mode_test.db")
        gc = GeometryController(db_path=db_path, cfg=cfg)

        first = gc.on_new_item(
            lcm_id="msg_1",
            node_type=NodeType.MESSAGE,
            embedding=emb_a,
            role="user",
            token_count=8,
        )
        _assert(first.update_mode == "fork", f"expected fork, got {first.update_mode}")

        second = gc.on_new_item(
            lcm_id="msg_2",
            node_type=NodeType.MESSAGE,
            embedding=emb_b,
            role="assistant",
            token_count=10,
            force_branch_id=first.branch_id,
            conflict_score=0.0,
        )
        _assert(second.update_mode == "refine", f"expected refine, got {second.update_mode}")

        third = gc.on_new_item(
            lcm_id="msg_3",
            node_type=NodeType.MESSAGE,
            embedding=emb_b,
            role="assistant",
            token_count=10,
            force_branch_id=first.branch_id,
            conflict_score=0.50,
        )
        _assert(third.update_mode == "contradict", f"expected contradict, got {third.update_mode}")

        fourth = gc.on_new_item(
            lcm_id="msg_4",
            node_type=NodeType.MESSAGE,
            embedding=emb_b,
            role="assistant",
            token_count=10,
            force_branch_id=first.branch_id,
            conflict_score=0.90,
        )
        _assert(fourth.update_mode == "supersede", f"expected supersede, got {fourth.update_mode}")

        report = gc.branch_report(first.branch_id)
        _assert("error" not in report, f"branch report error: {report}")
        counts = report.get("update_mode_counts") or {}
        correction = report.get("correction_counts") or {}
        correction_by_kind = correction.get("by_kind") or {}
        expected = {
            "fork": 1,
            "refine": 1,
            "contradict": 1,
            "supersede": 1,
        }
        for mode, min_count in expected.items():
            got = int(counts.get(mode, 0) or 0)
            _assert(got >= min_count, f"expected at least {min_count} for {mode}, got {got}")
        _assert(int(correction_by_kind.get("refine", 0) or 0) >= 1, "expected refine correction count >= 1")
        _assert(int(correction_by_kind.get("contradict", 0) or 0) >= 1, "expected contradict correction count >= 1")
        _assert(int(correction_by_kind.get("supersede", 0) or 0) >= 1, "expected supersede correction count >= 1")

        # Explicit supersession chain should be represented in memory_edges.
        edge_rows = gc.db.conn.execute(
            """
            SELECT edge_type, COUNT(*) AS c
            FROM memory_edges
            WHERE edge_type IN ('refines','contradicts','supersedes')
            GROUP BY edge_type
            """
        ).fetchall()
        edge_counts = {str(r["edge_type"]): int(r["c"] or 0) for r in edge_rows}
        _assert(int(edge_counts.get("refines", 0)) >= 1, "expected at least one refines edge")
        _assert(int(edge_counts.get("contradicts", 0)) >= 1, "expected at least one contradicts edge")
        _assert(int(edge_counts.get("supersedes", 0)) >= 1, "expected at least one supersedes edge")

        out = {
            "branch_id": first.branch_id,
            "counts": counts,
            "correction_counts": correction,
            "edge_counts": edge_counts,
            "decisions": [
                {"lcm_id": "msg_1", "update_mode": first.update_mode, "action": first.action},
                {"lcm_id": "msg_2", "update_mode": second.update_mode, "action": second.action},
                {"lcm_id": "msg_3", "update_mode": third.update_mode, "action": third.action},
                {"lcm_id": "msg_4", "update_mode": fourth.update_mode, "action": fourth.action},
            ],
        }
        print("UPDATE_MODE_SUMMARY", json.dumps(out, ensure_ascii=True))
        print("UPDATE_MODE_REGRESSION_OK")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
