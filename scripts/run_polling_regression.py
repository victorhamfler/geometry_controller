#!/usr/bin/env python3
from __future__ import annotations

import os
import sqlite3
import tempfile
import threading

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
import sys
sys.path.insert(0, ROOT)

from lcm_geometry_controller import GeometryConfig, GeometryController  # noqa: E402


class DummyEmbeddingProvider:
    def embed(self, text: str) -> list[float]:
        n = float(len(text or ""))
        return [n, 1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0]

    def embed_batch(self, texts: list[str]) -> list[list[float]]:
        return [self.embed(t) for t in texts]


def _assert(cond: bool, msg: str) -> None:
    if not cond:
        raise AssertionError(msg)


def _init_lcm_db(path: str) -> None:
    conn = sqlite3.connect(path)
    conn.execute(
        """
        CREATE TABLE messages (
          message_id TEXT PRIMARY KEY,
          conversation_id INTEGER,
          seq INTEGER,
          role TEXT,
          content TEXT,
          token_count INTEGER,
          created_at REAL
        )
        """
    )
    rows = [
        ("m1", 1, 1, "user", "hello world", 2, 1.0),
        ("m2", 1, 2, "assistant", "answer one", 2, 2.0),
        ("m3", 2, 1, "user", "another conversation", 3, 3.0),
    ]
    conn.executemany(
        "INSERT INTO messages (message_id, conversation_id, seq, role, content, token_count, created_at) VALUES (?, ?, ?, ?, ?, ?, ?)",
        rows,
    )
    conn.commit()
    conn.close()


def _message_stats(geo_db: str) -> dict[str, int]:
    conn = sqlite3.connect(geo_db)
    row = conn.execute(
        """
        SELECT
          COUNT(*) AS rows_total,
          COUNT(DISTINCT lcm_id) AS distinct_lcm
        FROM memory_nodes
        WHERE node_type='message'
        """
    ).fetchone()
    conn.close()
    return {
        "rows_total": int(row[0] or 0),
        "distinct_lcm": int(row[1] or 0),
    }


def main() -> int:
    with tempfile.TemporaryDirectory(prefix="lcm_geo_poll_") as tmp:
        lcm_db = os.path.join(tmp, "lcm.db")
        geo_db = os.path.join(tmp, "lcm_geometry.db")
        _init_lcm_db(lcm_db)

        gc = GeometryController(
            geo_db,
            cfg=GeometryConfig(embedding_dim=8),
            embedding_provider=DummyEmbeddingProvider(),
        )

        out1 = gc.poll_lcm_for_new_items(lcm_db_path=lcm_db, since_rowid=0, limit=2)
        _assert(out1["polled"] == 2, f"expected polled=2, got {out1}")
        _assert(out1["processed"] == 2, f"expected processed=2, got {out1}")
        _assert(out1["failed"] == 0, f"expected failed=0, got {out1}")
        _assert(bool(out1["has_more"]) is True, f"expected has_more True, got {out1}")

        out2 = gc.poll_lcm_for_new_items(lcm_db_path=lcm_db, since_rowid=int(out1["next_rowid"]), limit=2)
        _assert(out2["polled"] == 1, f"expected polled=1, got {out2}")
        _assert(out2["processed"] == 1, f"expected processed=1, got {out2}")
        _assert(out2["failed"] == 0, f"expected failed=0, got {out2}")
        _assert(bool(out2["has_more"]) is False, f"expected has_more False, got {out2}")

        # Idempotency: replaying same row range should not duplicate rows.
        out3 = gc.poll_lcm_for_new_items(lcm_db_path=lcm_db, since_rowid=0, limit=10)
        _assert(out3["polled"] == 3, f"expected replay polled=3, got {out3}")
        _assert(out3["processed"] == 0, f"expected replay processed=0, got {out3}")
        _assert(int(out3.get("skipped_duplicates", 0)) == 3, f"expected replay skipped_duplicates=3, got {out3}")

        r1 = gc.branch_report("conv_1")
        r2 = gc.branch_report("conv_2")
        _assert(r1 is not None and int(r1.get("node_count", 0)) >= 2, f"missing conv_1 report: {r1}")
        _assert(r2 is not None and int(r2.get("node_count", 0)) >= 1, f"missing conv_2 report: {r2}")
        s = _message_stats(geo_db)
        _assert(s["rows_total"] == 3, f"expected rows_total=3 after idempotency replay, got {s}")
        _assert(s["distinct_lcm"] == 3, f"expected distinct_lcm=3 after idempotency replay, got {s}")

        # Concurrency: two poll calls with same cursor should stay duplicate-free.
        geo_db2 = os.path.join(tmp, "lcm_geometry_concurrent.db")
        gc2 = GeometryController(
            geo_db2,
            cfg=GeometryConfig(embedding_dim=8),
            embedding_provider=DummyEmbeddingProvider(),
        )
        results: list[dict] = []
        lock = threading.Lock()

        def _worker() -> None:
            out = gc2.poll_lcm_for_new_items(lcm_db_path=lcm_db, since_rowid=0, limit=10)
            with lock:
                results.append(out)

        t1 = threading.Thread(target=_worker)
        t2 = threading.Thread(target=_worker)
        t1.start()
        t2.start()
        t1.join()
        t2.join()
        _assert(len(results) == 2, f"expected two polling results, got {results}")
        total_processed = sum(int(x.get("processed", 0) or 0) for x in results)
        total_failed = sum(int(x.get("failed", 0) or 0) for x in results)
        _assert(total_processed == 3, f"expected total_processed=3 in concurrency test, got {results}")
        _assert(total_failed == 0, f"expected total_failed=0 in concurrency test, got {results}")
        s2 = _message_stats(geo_db2)
        _assert(s2["rows_total"] == 3, f"expected concurrent rows_total=3, got {s2}")
        _assert(s2["distinct_lcm"] == 3, f"expected concurrent distinct_lcm=3, got {s2}")

        print("POLLING_REGRESSION_OK", out1, out2, out3, {"concurrency_results": results})

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
