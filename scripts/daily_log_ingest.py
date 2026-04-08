#!/usr/bin/env python3
"""
Append one daily-log entry into lcm_geometry.db using GeometryController API.

Usage examples:
  python3 scripts/daily_log_ingest.py "Updated geometry split thresholds"
  python3 scripts/daily_log_ingest.py --date 2026-04-07 --source agent_test "Validated sidecar retrieval"
  python3 scripts/daily_log_ingest.py --db-path ~/.openclaw/lcm_geometry.db --no-embed "Quick note"
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT))

from lcm_geometry_controller import EmbeddingProvider, GeometryController  # noqa: E402


def main() -> int:
    parser = argparse.ArgumentParser(description="Ingest one daily-log entry into geometry DB.")
    parser.add_argument("text", help="Daily log text to store")
    parser.add_argument("--date", default=None, help="Date in YYYY-MM-DD; default is local today")
    parser.add_argument("--source", default="manual_log", help="Source label (manual_log, agent_test, etc.)")
    parser.add_argument(
        "--db-path",
        default=os.path.expanduser("~/.openclaw/lcm_geometry.db"),
        help="Path to lcm_geometry.db",
    )
    parser.add_argument("--model", default="all-MiniLM-L6-v2", help="Embedding model name")
    parser.add_argument("--device", default="cpu", help="Embedding device (cpu/cuda)")
    parser.add_argument(
        "--no-embed",
        action="store_true",
        help="Insert with zero-vector fallback (skips loading embedding model)",
    )
    args = parser.parse_args()

    db_path = os.path.expanduser(args.db_path)
    provider = None if args.no_embed else EmbeddingProvider(model_name=args.model, device=args.device)
    gc = GeometryController(db_path=db_path, embedding_provider=provider)

    out = gc.add_daily_log_entry(
        text=args.text,
        date_str=args.date,
        source=args.source,
    )
    print(json.dumps(out, ensure_ascii=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

