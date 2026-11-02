"""Verification only: export Python MLX results for the independent Elixir port."""
import json
import sys
import time
from pathlib import Path
import mlx.core as mx
import numpy as np

root = Path(__file__).resolve().parents[1]
path = root / "models/clef-flash-8bit"
sys.path.insert(0, str(path))
from clef_mlx import load, encode_record
from mlx_lm.models.base import create_attention_mask
from mlx_lm.models.gated_delta import gated_delta_ops

out = root / "artifacts"
out.mkdir(exist_ok=True)
model, tok, head, lex = load(str(path))
results = []
for index, record in enumerate(json.loads((root / "fixtures/records.json").read_text())):
    start = time.perf_counter()
    ids, qs = encode_record(tok, record)
    backbone = model.language_model.model
    x = backbone.embed_tokens(mx.array(ids)[None])
    traces = {"embedding": x}
    mask = create_attention_mask(x, None)
    for i, layer in enumerate(backbone.layers):
        x = layer(x, mask=None if layer.is_linear else mask)
        if i in (0, 3, 31):
            traces[f"layer_{i}"] = x
    hidden = backbone.norm(x)[0]
    traces["hidden"] = hidden
    logits = head(hidden, ids, qs, lex)
    probs = [np.array(mx.softmax(l)).tolist() for l in logits]
    mx.save_safetensors(str(out / f"reference_{index}.safetensors"), traces)
    results.append({"ids": ids, "fields": qs, "probabilities": dict(zip(record["questions"], [dict(zip(q["oids"], p)) for q, p in zip(qs, probs)]))})
    print(f"reference {index}: {len(ids)} tokens, {time.perf_counter()-start:.2f}s, {results[-1]['probabilities']}", flush=True)
(out / "reference.json").write_text(json.dumps(results, ensure_ascii=False, indent=2))

# Small independent recurrence fixture: batches, repeated key heads and multiple timesteps.
rng = np.random.default_rng(42)
q = mx.array(rng.normal(0, 0.1, (2, 5, 2, 128)).astype(np.float32))
k = mx.array(rng.normal(0, 0.1, q.shape).astype(np.float32))
v = mx.array(rng.normal(size=(2, 5, 4, 8)).astype(np.float32))
g = mx.array(rng.uniform(0.1, 0.95, (2, 5, 4)).astype(np.float32))
beta = mx.array(rng.uniform(0.1, 0.95, g.shape).astype(np.float32))
y, _ = gated_delta_ops(q, k, v, g, beta, mx.zeros((2, 4, 8, 128)), None)
mx.save_safetensors(str(root / "fixtures/delta.safetensors"), dict(q=q, k=k, v=v, g=g, beta=beta, y=y))
(root / "fixtures/reference.json").write_text(json.dumps(results, ensure_ascii=False, indent=2))
