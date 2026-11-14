"""Generate private geometry fixtures from an explicitly supplied read-only numerical source.

Only the geometry definitions are extracted; no module imports, model initialization,
tensor libraries, device discovery, or weight loading are executed. Requires NumPy.
Keep generated source-identified fixtures outside version control.
"""
import argparse
import ast
import hashlib
import json
import math
from pathlib import Path
from types import SimpleNamespace


def generate(source, output):
    import numpy as np
    text = source.read_text(encoding="utf-8")
    names = {"_axis_from_sqrt_area", "_frame_grid", "_video_t_spans", "_video_t_grid",
             "_ref_t_span", "_audio_grid", "_video_grid", "PackedLayout"}
    definitions = [n for n in ast.parse(text).body if isinstance(n, (ast.FunctionDef, ast.ClassDef)) and n.name in names]
    if {n.name for n in definitions} != names:
        raise ValueError("reference geometry interface changed")
    shim = SimpleNamespace(float64=np.float64, bool=np.bool_, long=np.int64, int64=np.int64,
                           arange=np.arange, tensor=np.array, zeros=lambda *s, dtype=None: np.zeros(s, dtype=dtype),
                           ones=lambda *s, dtype=None: np.ones(s, dtype=dtype),
                           empty=lambda *s, dtype=None: np.empty(s, dtype=dtype),
                           meshgrid=np.meshgrid, stack=lambda x, dim=0: np.stack(x, axis=dim),
                           cat=lambda x, dim=0: np.concatenate(x, axis=dim))
    scope = {"torch": shim, "math": math, "FRAME_RESCALE": 5.0 / 3, "FRAME_PER_TOKEN": [1, 4, 4, 4, 4]}
    exec(compile(ast.Module(body=definitions, type_ignores=[]), "geometry_reference", "exec"), scope)
    tests = []
    for text_len, width, height, refs in [(0,32,32,[]),(3,64,96,[]),(2,64,32,[(4,8),(2,2)]),
                                          (7,96,64,[(2,6)]*16),(1,128,32,[(4,2),(2,8)])]:
        layout = scope["PackedLayout"](text_len,1,height//16,width//16,0,
                    refs=[{"kind":"image","latent_h":h,"latent_w":w} for h,w in refs])
        tests.append({"text":text_len,"width":width,"height":height,"refs":refs,
                      "positions":layout.position_ids.reshape(-1).tolist()})
    output.parent.mkdir(parents=True,exist_ok=True)
    output.write_text(json.dumps(tests,indent=2)+"\n",encoding="utf-8")
    output.with_suffix(".provenance.json").write_text(json.dumps({"source_sha256":hashlib.sha256(text.encode()).hexdigest(),
            "geometry_definitions":sorted(names),"cases":len(tests),"device_work":False},indent=2)+"\n",encoding="utf-8")
    print(f"generated {len(tests)} private geometry cases")


if __name__ == "__main__":
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source",type=Path,required=True)
    parser.add_argument("--output",type=Path,required=True)
    args=parser.parse_args()
    generate(args.source,args.output)
