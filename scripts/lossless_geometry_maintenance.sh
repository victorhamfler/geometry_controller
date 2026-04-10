#!/usr/bin/env bash
set -euo pipefail

LCM_DB_PATH="${LCM_DB_PATH:-/home/victo/.openclaw/lcm.db}"
GEO_DB_PATH="${GEO_DB_PATH:-/home/victo/.openclaw/lcm_geometry.db}"
MODULE_DIR="${MODULE_DIR:-/home/victo/.openclaw/workspace/module}"
RUNNER_TS="${RUNNER_TS:-/home/victo/.openclaw/workspace/scripts/lossless_doctor_clean_runner.ts}"
PYTHON_BIN="${GEOMETRY_PYTHON:-/home/victo/venvs/ml/bin/python3}"

DO_APPLY=0
DO_VACUUM=0
DO_JSON=1
FILTER_ID=""

usage() {
  cat <<'EOF'
lossless_geometry_maintenance.sh

Runs Lossless-Claw doctor clean (scan/apply) and then geometry DAG re-sync + orphan validation.

Usage:
  /home/victo/.openclaw/workspace/scripts/lossless_geometry_maintenance.sh [options]

Options:
  --apply              Apply cleanup deletion (default: dry-run scan only)
  --vacuum             Request VACUUM during apply (only valid with --apply)
  --filter <id>        Restrict to one cleaner filter
  --db <path>          LCM db path (default: /home/victo/.openclaw/lcm.db)
  --geo-db <path>      Geometry db path (default: /home/victo/.openclaw/lcm_geometry.db)
  --json               JSON output (default)
  --text               Human-readable output (default is JSON blocks)
  -h, --help           Show help

Cleaner filter IDs:
  archived_subagents, cron_sessions, null_subagent_context
EOF
}

while [[ $# -gt 0 ]]; do
  case "$1" in
    --apply)
      DO_APPLY=1
      shift
      ;;
    --vacuum)
      DO_VACUUM=1
      shift
      ;;
    --filter)
      FILTER_ID="${2:-}"
      shift 2
      ;;
    --db)
      LCM_DB_PATH="${2:-}"
      shift 2
      ;;
    --geo-db)
      GEO_DB_PATH="${2:-}"
      shift 2
      ;;
    --text)
      DO_JSON=0
      shift
      ;;
    --json)
      DO_JSON=1
      shift
      ;;
    -h|--help)
      usage
      exit 0
      ;;
    *)
      echo "[lossless-geometry-maintenance] unknown arg: $1" >&2
      usage >&2
      exit 2
      ;;
  esac
done

if [[ "$DO_APPLY" -eq 0 && "$DO_VACUUM" -eq 1 ]]; then
  echo "[lossless-geometry-maintenance] --vacuum requires --apply" >&2
  exit 2
fi

if [[ ! -f "$RUNNER_TS" ]]; then
  echo "[lossless-geometry-maintenance] missing runner: $RUNNER_TS" >&2
  exit 2
fi
if [[ ! -f "$LCM_DB_PATH" ]]; then
  echo "[lossless-geometry-maintenance] missing lcm db: $LCM_DB_PATH" >&2
  exit 2
fi
if [[ ! -f "$GEO_DB_PATH" ]]; then
  echo "[lossless-geometry-maintenance] missing geometry db: $GEO_DB_PATH" >&2
  exit 2
fi

if [[ ! -x "$PYTHON_BIN" ]]; then
  PYTHON_BIN="$(command -v python3 || true)"
fi
if [[ -z "${PYTHON_BIN:-}" ]]; then
  echo "[lossless-geometry-maintenance] python not found (set GEOMETRY_PYTHON)" >&2
  exit 2
fi

cleanup_tmp="$(mktemp)"
trap 'rm -f "$cleanup_tmp"' EXIT

cleanup_cmd=(npx --yes tsx "$RUNNER_TS" --db "$LCM_DB_PATH" --json)
if [[ "$DO_APPLY" -eq 1 ]]; then
  cleanup_cmd+=(--apply)
fi
if [[ "$DO_VACUUM" -eq 1 ]]; then
  cleanup_cmd+=(--vacuum)
fi
if [[ -n "$FILTER_ID" ]]; then
  cleanup_cmd+=(--filter "$FILTER_ID")
fi

echo "[lossless-geometry-maintenance] step 1/2: lossless doctor clean"
NODE_NO_WARNINGS=1 "${cleanup_cmd[@]}" > "$cleanup_tmp"

if [[ "$DO_APPLY" -eq 1 ]]; then
  "$PYTHON_BIN" - "$cleanup_tmp" <<'PY'
import json, sys
path=sys.argv[1]
data=json.load(open(path,'r',encoding='utf-8'))
apply=data.get("apply") or {}
if apply.get("kind") == "unavailable":
    print("[lossless-geometry-maintenance] ERROR: apply unavailable:", apply.get("reason","unknown"), file=sys.stderr)
    sys.exit(3)
PY
fi

if [[ "$DO_JSON" -eq 1 ]]; then
  echo "[lossless-geometry-maintenance] cleanup_result_json:"
  cat "$cleanup_tmp"
else
  echo "[lossless-geometry-maintenance] cleanup_result:"
  "$PYTHON_BIN" - "$cleanup_tmp" <<'PY'
import json,sys
data=json.load(open(sys.argv[1],'r',encoding='utf-8'))
scan=data.get("scan",{})
print(f"  mode={data.get('mode')}")
print(f"  distinct_conversations={scan.get('totalDistinctConversations',0)}")
print(f"  distinct_messages={scan.get('totalDistinctMessages',0)}")
for f in scan.get("filters",[]):
    print(f"  - {f.get('id')}: conversations={f.get('conversationCount',0)} messages={f.get('messageCount',0)}")
apply=data.get("apply")
if isinstance(apply,dict):
    print(f"  apply.kind={apply.get('kind')}")
    if apply.get("kind")=="applied":
        print(f"  deleted_conversations={apply.get('deletedConversations',0)}")
        print(f"  deleted_messages={apply.get('deletedMessages',0)}")
        print(f"  backup_path={apply.get('backupPath')}")
PY
fi

echo "[lossless-geometry-maintenance] step 2/2: geometry DAG sync + orphan validation"
"$PYTHON_BIN" - "$MODULE_DIR" "$GEO_DB_PATH" "$LCM_DB_PATH" "$DO_JSON" <<'PY'
import json, sqlite3, sys
module_dir, geo_db, lcm_db, json_mode = sys.argv[1], sys.argv[2], sys.argv[3], int(sys.argv[4])
sys.path.insert(0, module_dir)
from lcm_geometry_controller import GeometryController

gc = GeometryController(geo_db)
dag_stats = gc.import_dag_edges_from_lcm(lcm_db)

conn = sqlite3.connect(geo_db)
conn.row_factory = sqlite3.Row
rows = conn.execute(
    """
    SELECT
      edge_type,
      COUNT(*) AS total_edges,
      SUM(CASE WHEN s.id IS NULL OR d.id IS NULL THEN 1 ELSE 0 END) AS orphan_edges
    FROM memory_edges e
    LEFT JOIN memory_nodes s ON s.id = e.src_id
    LEFT JOIN memory_nodes d ON d.id = e.dst_id
    WHERE edge_type IN ('derived_from','summarizes')
    GROUP BY edge_type
    ORDER BY edge_type
    """
).fetchall()
conn.close()

by_type = {}
for r in rows:
    by_type[str(r["edge_type"])] = {
        "total_edges": int(r["total_edges"] or 0),
        "orphan_edges": int(r["orphan_edges"] or 0),
    }

payload = {"dag_sync": dag_stats, "orphan_validation": by_type}
if json_mode:
    print("[lossless-geometry-maintenance] geometry_sync_json:")
    print(json.dumps(payload, indent=2, ensure_ascii=False))
else:
    print("[lossless-geometry-maintenance] geometry_sync_result:")
    print(f"  derived_from={dag_stats.get('derived_from',0)} summarizes={dag_stats.get('summarizes',0)} skipped={dag_stats.get('skipped',0)} purged={dag_stats.get('purged',0)}")
    for k, v in by_type.items():
        print(f"  - {k}: total_edges={v['total_edges']} orphan_edges={v['orphan_edges']}")
PY

echo "[lossless-geometry-maintenance] done"
