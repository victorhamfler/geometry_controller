#!/usr/bin/env python3
import os
import sys
import tempfile

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)

from lcm_geometry_controller import GeometryController, GeometryConfig, NodeType  # noqa: E402


def main() -> int:
    cfg = GeometryConfig(embedding_dim=8)
    with tempfile.TemporaryDirectory(prefix="lcm_geo_smoke_") as d:
        db_path = os.path.join(d, "test_geometry.db")
        gc = GeometryController(db_path=db_path, cfg=cfg)

        d1 = gc.on_new_item(
            lcm_id="msg_1",
            node_type=NodeType.MESSAGE,
            embedding=[0.1, 0.2, 0.3, 0.4, 0.0, 0.1, 0.2, 0.3],
            role="user",
            token_count=10,
        )
        d2 = gc.on_new_item(
            lcm_id="msg_2",
            node_type=NodeType.MESSAGE,
            embedding=[0.11, 0.21, 0.31, 0.41, 0.01, 0.09, 0.19, 0.29],
            role="assistant",
            token_count=12,
            active_branch_id=d1.branch_id,
        )

        assert d1.branch_id.startswith("conv_"), f"unexpected branch id: {d1.branch_id}"
        assert d2.branch_id.startswith("conv_"), f"unexpected branch id: {d2.branch_id}"

        report = gc.branch_report(d1.branch_id)
        assert "error" not in report, report

        ranked = gc.rank_retrieval([0.1, 0.2, 0.3, 0.4, 0.0, 0.1, 0.2, 0.3])
        assert ranked, "expected retrieval candidates"

        print("SMOKE_OK", d1.branch_id, len(ranked))

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
