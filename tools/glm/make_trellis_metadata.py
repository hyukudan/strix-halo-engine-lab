"""Convert GLM EXL3 fixture safetensors (tools/glm fixtures) into strix-trellis' real-fixture layout: one raw file
per tensor + a metadata JSON ({model, revision, url, tensors: {name: {dtype, shape, file, bytes, sha256}}}) that
strix-trellis tools/make_dense_gpu_fixtures.py consumes (gate D: device decode / transforms vs CPU, via the library's
own STGPU001 harness tests/gpu_runtime.hip).
usage: make_trellis_metadata.py FIXTURE_DIR OUT_DIR
"""
import hashlib, json, struct, sys
from pathlib import Path

src, out = Path(sys.argv[1]), Path(sys.argv[2])
out.mkdir(parents=True, exist_ok=True)
tensors = {}
for f in sorted(src.glob('*.safetensors')):
    with open(f, 'rb') as fh:
        n = struct.unpack('<Q', fh.read(8))[0]
        h = json.loads(fh.read(n))
        for k, v in h.items():
            if k == '__metadata__' or not k.endswith(('.trellis', '.suh', '.svh', '.mul1')):
                continue
            a, b = v['data_offsets']
            fh.seek(8 + n + a)
            data = fh.read(b - a)
            name = k.replace('model.language_model.', '')
            (out / name).write_bytes(data)
            tensors[name] = dict(dtype=v['dtype'], shape=v['shape'], file=name, bytes=b - a,
                                 sha256=hashlib.sha256(data).hexdigest())
meta = dict(model='turboderp/GLM-5.3-Flash-exl3', revision='2a30229e67012798ba9f0cd832bb78abf4c363d5',
            url='https://huggingface.co/turboderp/GLM-5.3-Flash-exl3', tensors=tensors)
(out / 'metadata.json').write_text(json.dumps(meta, indent=1))
groups = sorted({k.rsplit('.', 1)[0] for k in tensors})
print(len(tensors), 'tensors', len(groups), 'groups')
print('\n'.join(groups))
