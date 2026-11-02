"""Independent KnowLine prompt/probability fixtures. Python is verification-only.

Uses the publisher's MIT-licensed prompt renderer at a pinned revision and
mlx-lm for inference; writes source hashes into the generated fixture.
"""
import argparse
import hashlib
import importlib.metadata
import json
import sys
import urllib.request
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
MODEL = ROOT / 'models/knowline-4b-gen5.5-mlx-q5'
SOURCE = ROOT / 'artifacts/knowline-reference-source'

RECORDS = [
    {'state': 'The production database is down and nobody can sign in.', 'questions': {
        'urgent': {'type': 'noul', 'instructions': 'Does this require immediate attention?'},
        'team': {'type': 'choice', 'instructions': 'Which team should respond?', 'criteria': {'z_engineering': 'Software and infrastructure', 'a_sales': 'New customers', 'm_billing': 'Invoices'}},
        'severity': {'type': 'score', 'instructions': 'How severe is the incident?', 'criteria': ['minor', 'degraded', 'outage']}}},
    {'state': {'customer': 'café 中文', 'paid': True, 'extra': None, 'amount': 1.0e20, 'note': 'first\nsecond\n', 'items': ['refund', {'status': 'broken'}]}, 'questions': {
        'refund': {'type': 'choice', 'criteria': {'yes': None, 'no': 'Do not refund'}},
        'q': {'type': 'noul', 'criteria': {'true': 'eligible', 'false': 'ineligible'}}}},
    {'state': {'messages': [{'role': 'system', 'content': '  You classify support requests.  '}, {'role': 'user', 'content': 'Hello'}, {'role': 'assistant', 'content': '<think>old reasoning</think>\n\nHow can I help?'}, {'role': 'user', 'content': 'My order arrived broken.'}]}, 'questions': {'refund': {'type': 'noul', 'instructions': 'Should we offer a refund?'}}},
    {'state': 'Routine status update. ' * 1100, 'questions': {'urgent': {'type': 'noul', 'instructions': 'Is this urgent?'}}},
]

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--prompts-only', action='store_true')
    args = parser.parse_args()
    SOURCE.mkdir(parents=True, exist_ok=True)
    revision = '779944d2cc8ffa19cef386aee83c83bcb967ade2'
    hashes = {}
    for filename in ['knowline_server.py', 'knowline_layout.py']:
        data = urllib.request.urlopen(f'https://huggingface.co/PelaAI/KnowLine-4B-Gen5.5/resolve/{revision}/{filename}').read()
        (SOURCE / filename).write_bytes(data)
        hashes[filename] = hashlib.sha256(data).hexdigest()
    sys.path.insert(0, str(SOURCE))
    import knowline_server as ks
    from knowline_layout import Layout, render_request
    from transformers import AutoTokenizer
    tokenizer = AutoTokenizer.from_pretrained(MODEL)
    _, probe, _ = ks.render(tokenizer, 'x', {'q': {'type': 'noul'}}, ['A', 'B'])
    labels, label_ids = ks.find_labels(tokenizer, probe['q'][0])
    fixture = {'source_revision': revision, 'source_sha256': hashes, 'model': str(MODEL.name), 'runtime': {p: importlib.metadata.version(p) for p in ['mlx', 'mlx-lm', 'transformers']}, 'records': []}
    if not args.prompts_only:
        import mlx.core as mx
        if mx.__version__ != '0.32.0':
            raise RuntimeError('Probability fixtures require MLX 0.32.0 to match EMLX; use the pinned uv command in README.md')
        from mlx_lm import load
        mx.set_cache_limit(256 * 1024**2)
        model, _ = load(str(MODEL))
    traces = {}
    for ri, record in enumerate(RECORDS):
        r = render_request(record, tokenizer, Layout(read_twice=True), labels=labels)
        out = {'record': record, 'twice': r['twice'], 'once_tokens': r['once_tokens'], 'rows': []}
        for qid, question in record['questions'].items():
            prompt, keys = r['prompts'][qid][0]
            ids = tokenizer.encode(prompt, add_special_tokens=False)
            row = {'id': qid, 'prompt': prompt, 'ids': ids, 'keys': keys, 'label_ids': label_ids[:len(keys)]}
            if not args.prompts_only:
                # Independent mlx-lm backbone, selecting only last-position readout.
                hidden = model.language_model.model(mx.array(ids)[None])[:, -1:, :]
                logits = model.language_model.model.embed_tokens.as_linear(hidden).astype(mx.float32)[0, 0, row['label_ids']]
                probs = mx.softmax(logits)
                mx.eval(probs)
                row['probabilities'] = dict(zip(keys, probs.tolist()))
                row['logits'] = logits.tolist()
                row['answer'] = ks.answer(question, keys, probs.tolist())
                traces[f'{ri}_{qid}'] = hidden.astype(mx.float32)
                mx.eval(traces[f'{ri}_{qid}'])
                print(qid, len(ids), row['probabilities'], flush=True)
                del hidden, logits, probs
                mx.clear_cache()
            out['rows'].append(row)
        fixture['records'].append(out)
    if not args.prompts_only:
        mx.save_safetensors(str(ROOT / 'artifacts/knowline_hidden.safetensors'), traces)
    target = ROOT / 'fixtures' / ('knowline_prompts.json' if args.prompts_only else 'knowline_reference.json')
    target.write_text(json.dumps(fixture, ensure_ascii=False, indent=2) + '\n')
    print(target, flush=True)

if __name__ == '__main__':
    main()
