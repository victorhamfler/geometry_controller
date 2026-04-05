#!/usr/bin/env python3
"""
LCM Geometry Backfill - Full DB with sentence-transformers embeddings
Run: python3 lcm_geometry_backfill.py (optionally set OPENCLAW_HOME and GEOMETRY_MODULE_HOME)
"""
import sqlite3, time, tempfile, os, sys, numpy as np, json, shutil, random
from collections import defaultdict
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent
OPENCLAW_HOME = Path(os.environ.get('OPENCLAW_HOME', str(Path.home() / '.openclaw'))).expanduser()
MODULE_HOME = Path(os.environ.get('GEOMETRY_MODULE_HOME', str(OPENCLAW_HOME / 'workspace' / 'module'))).expanduser()

# Allow imports when running from either the repository root or deployed OpenClaw module folder.
sys.path.insert(0, str(REPO_ROOT))
sys.path.insert(0, str(MODULE_HOME))

from lcm_geometry_controller import (
    GeometryController, GeometryConfig, NodeType, create_geometry_controller,
    BranchState, GeometricRegime, BranchStats, MemoryNode, GeometryMath
)
from sentence_transformers import SentenceTransformer

OUTPUT_DB = str(OPENCLAW_HOME / 'lcm_geometry.db')
PROGRESS_FILE = str(MODULE_HOME / 'backfill_progress.json')
LOG_FILE = str(MODULE_HOME / 'backfill.log')
LCM_DB = str(OPENCLAW_HOME / 'lcm.db')
MAX_MSGS_PER_CONV = 200

MODULE_HOME.mkdir(parents=True, exist_ok=True)
if not Path(LCM_DB).exists():
    raise FileNotFoundError(f"LCM database not found: {LCM_DB}")

def log(msg):
    ts = time.strftime('%H:%M:%S')
    line = f"[{ts}] {msg}"
    print(line)
    with open(LOG_FILE, 'a') as f:
        f.write(line + '\n')

DIM = 384

log("Loading model...")
t0 = time.time()
model = SentenceTransformer('all-MiniLM-L6-v2')
log(f"Model loaded in {time.time()-t0:.1f}s | dim={DIM}")

log("Loading LCM data...")
lcm_conn = sqlite3.connect(LCM_DB)
cur = lcm_conn.execute('''
    SELECT m.message_id, m.conversation_id, m.seq, m.role, m.content, m.token_count
    FROM messages m ORDER BY m.conversation_id, m.seq
''')
messages = [{'message_id':r[0],'conv_id':r[1],'seq':r[2],'role':r[3],
             'content':(r[4] or '')[:1000],'tokens':r[5] or 0} for r in cur.fetchall()]
cur = lcm_conn.execute('''
    SELECT summary_id, conversation_id, kind, depth, content, token_count, descendant_count
    FROM summaries ORDER BY conversation_id, depth
''')
summaries = [{'summary_id':r[0],'conv_id':r[1],'kind':r[2],'depth':r[3],
              'content':(r[4] or '')[:1000],'tokens':r[5] or 0,'desc_count':r[6] or 0}
             for r in cur.fetchall()]
lcm_conn.close()
log(f"Loaded {len(messages)} msgs + {len(summaries)} sums")

msg_by_conv = defaultdict(list)
for m in messages: msg_by_conv[m['conv_id']].append(m)
sum_by_conv = defaultdict(list)
for s in summaries: sum_by_conv[s['conv_id']].append(s)
convs = sorted(msg_by_conv.keys())
log(f"Conversations: {len(convs)}")

large = [cid for cid in convs if len(msg_by_conv[cid]) > MAX_MSGS_PER_CONV]
log(f"Large convs (> {MAX_MSGS_PER_CONV} msgs, will sample): {len(large)}")
for cid in large[:5]:
    log(f"  conv_{cid}: {len(msg_by_conv[cid])} msgs -> {MAX_MSGS_PER_CONV} sampled")

log("=== Encoding per conversation ===")
t0 = time.time()
all_bs = []
all_mn = []

for ci, conv_id in enumerate(convs):
    msgs = msg_by_conv[conv_id]
    sums = sum_by_conv[conv_id]

    if ci % 20 == 0 or len(msgs) > MAX_MSGS_PER_CONV:
        elapsed = time.time() - t0
        rate = ci / elapsed if elapsed > 0 else 0
        eta_s = (len(convs) - ci) / rate if rate > 0 else 0
        tag = f" [conv_{conv_id}={len(msgs)}msgs SAMPLED]" if len(msgs) > MAX_MSGS_PER_CONV else ""
        log(f"  conv {ci}/{len(convs)} [{rate:.2f}/s] ETA={eta_s/60:.0f}min{tag}")
        with open(PROGRESS_FILE, 'w') as f:
            json.dump({'done': ci, 'total': len(convs), 'elapsed_s': elapsed}, f)

    # Stratified sample if large
    work_msgs = msgs
    if len(msgs) > MAX_MSGS_PER_CONV:
        n = MAX_MSGS_PER_CONV
        step = len(msgs) / n
        indices = sorted(set(int(step * i) for i in range(n)))
        work_msgs = [msgs[i] for i in indices]

    texts = [m['content'] for m in work_msgs]
    embs = model.encode(texts, convert_to_numpy=True, show_progress_bar=False,
                        batch_size=64, normalize_embeddings=True)

    weights = np.ones(len(embs), dtype=np.float32)
    mu = GeometryMath.weighted_mean(embs.astype(np.float32), weights)
    vd = GeometryMath.covariance_diagonal(embs.astype(np.float32), weights)
    sd = GeometryMath.compute_full_stats(embs.astype(np.float32), weights, mu)

    bs = BranchStats(
        branch_id=f"conv_{conv_id}", branch_type='conversation',
        state=BranchState.ACTIVE, regime=GeometricRegime.PRODUCTIVE,
        mean_vec=mu.tolist(), anchor=mu.tolist(), cov_diagonal=vd.tolist(),
        eff_rank=float(sd['eff_rank']), trace=float(sd['trace']),
        anisotropy=float(sd['anisotropy']), anchor_drift=0.0,
        coherence=float(sd['coherence']), compression_loss=0.0,
        node_count=len(msgs), last_update_ts=time.time())
    all_bs.append(bs)

    if sums:
        sum_texts = [s['content'] for s in sums]
        sum_embs_raw = model.encode(sum_texts, convert_to_numpy=True,
                                    show_progress_bar=False, batch_size=64,
                                    normalize_embeddings=True)
        for si, s in enumerate(sums):
            se = sum_embs_raw[si].astype(np.float32)
            se_blend = (0.7 * se + 0.3 * mu.astype(np.float32)).astype(np.float32)
            nt = NodeType.LEAF_SUMMARY if s['depth'] == 0 else NodeType.CONDENSED_SUMMARY
            all_mn.append(MemoryNode(
                node_id=f"sn_{conv_id}_{si}", lcm_id=s['summary_id'],
                node_type=nt, branch_id=bs.branch_id, parent_id=None,
                timestamp=time.time(), role='summary', token_count=s['tokens'],
                embedding=se_blend.tolist()))

enc_time = time.time() - t0
log(f"Geometry computed in {enc_time:.0f}s: {len(all_bs)} branches, {len(all_mn)} nodes")

log("Writing geometry DB...")
t0 = time.time()
with tempfile.TemporaryDirectory() as td:
    tmp_db = os.path.join(td, 'lcm_geometry.db')
    gc = create_geometry_controller(tmp_db, embedding_dim=DIM)
    for bs in all_bs:
        gc.db.upsert_branch(bs)
    for mn in all_mn:
        gc.db.insert_node(mn)
    log(f"  Inserted {len(all_bs)} branches + {len(all_mn)} nodes")
    counts = gc.run_maintenance_cycle()
    log(f"  Maintenance: {counts}")
    shutil.copy(tmp_db, OUTPUT_DB)

sz = os.path.getsize(OUTPUT_DB) / 1024
all_b = gc.db.all_branches()
from collections import Counter
states = Counter(b.state.value for b in all_b if b)
regimes = Counter(b.regime.value for b in all_b if b)

log(f"\n{'='*50}")
log(f"DONE - {OUTPUT_DB} ({sz:.0f} KB)")
log(f"Branches: {len(all_b)} | States: {dict(states)} | Regimes: {dict(regimes)}")
log(f"Total time: {enc_time + time.time()-t0:.0f}s")
for bs in all_bs[:5]:
    rpt = gc.branch_report(bs.branch_id)
    log(f"  {bs.branch_id}: {rpt['state']}/{rpt['regime']} nodes={bs.node_count} "
        f"eff_rank={rpt['eff_rank']:.3f} coh={rpt['coherence']:.3f}")
q = np.zeros(DIM, dtype=np.float32)
ranked = gc.rank_retrieval(q.tolist())
log(f"Retrieval: {len(ranked)} candidates | top: {ranked[0].branch_id if ranked else 'none'}")


