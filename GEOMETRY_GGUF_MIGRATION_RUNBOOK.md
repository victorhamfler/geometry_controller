# Geometry EmbeddingGemma GGUF Migration Runbook

This runbook performs a safe cutover from legacy `sentence_transformers` embeddings (for example `all-MiniLM-L6-v2`) to `llama_cpp` GGUF embeddings (for example `embeddinggemma-300m`).

## 1. Preflight

- Confirm live MCP server is the updated version (`server.py` supports `embedding.backend`).
- Confirm Python env has `llama-cpp-python` installed.
- Confirm target GGUF file exists on disk.

Recommended target:

- Model: `unsloth/embeddinggemma-300m-GGUF`
- Quantization: start with `Q8_0` for quality-first baseline.
- Embedding dimension: `768`.

## 2. Backup

Backup before any runtime change:

```bash
cp /home/victo/.openclaw/lcm_geometry.db /home/victo/.openclaw/lcm_geometry.db.bak_$(date +%Y%m%d_%H%M%S)
cp /home/victo/.openclaw/extensions/geometry-mcp/runtime_config.json /home/victo/.openclaw/extensions/geometry-mcp/runtime_config.json.bak_$(date +%Y%m%d_%H%M%S)
```

## 3. Runtime Config Cutover

Edit `/home/victo/.openclaw/extensions/geometry-mcp/runtime_config.json` and set:

```json
{
  "embedding": {
    "backend": "llama_cpp",
    "model": "embedding-gemma-300M-Q8_0.gguf",
    "gguf_path": "/home/victo/models/embedding-gemma-300M-Q8_0.gguf",
    "gguf_n_ctx": 2048,
    "gguf_n_threads": 8,
    "dim": 768
  }
}
```

Notes:

- Keep legacy `embedding_model` key only for backward compatibility; `embedding` block is authoritative.
- `gguf_n_threads` should match your CPU profile (8 is a safe starting point).

## 4. Mandatory Geometry DB Rebuild

Do not reuse old vectors across different embedding backends/dimensions.

```bash
mv /home/victo/.openclaw/lcm_geometry.db /home/victo/.openclaw/lcm_geometry.db.pre_gemma_$(date +%Y%m%d_%H%M%S)
```

Then rebuild from LCM history:

```bash
cd /home/victo/.openclaw/workspace/module
source .venv_mlcpu/bin/activate
python lcm_geometry_backfill.py --lcm-db /home/victo/.openclaw/lcm.db --geo-db /home/victo/.openclaw/lcm_geometry.db
```

## 5. Restart + Smoke Validation

```bash
openclaw gateway restart
openclaw mcp list
```

Recommended checks:

- `geometry_stats`: verify `embedding_backend=llama_cpp`, model and dim `768`.
- `geometry_query`: verify retrieval works and returns results.
- no startup errors about embedding runtime signature mismatch.

## 6. Rollback

If quality or latency is unacceptable:

1. Restore old `runtime_config.json` backup.
2. Restore old geometry DB backup.
3. Restart gateway.

## 7. Future Fine-Tuning Path

Fine-tuning from geometry feedback is feasible, but recommended flow is:

1. Build supervised embedding dataset from feedback events.
2. Fine-tune from trainable HF checkpoint (not directly on GGUF).
3. Export tuned checkpoint to GGUF for production inference.
4. Repeat this runbook with a fresh geometry DB rebuild.
