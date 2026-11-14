"""Host-only comparison of normalized FP32 latents and planar RGB in [0,1]."""
import argparse
import array
import json
import math
from pathlib import Path
import sys


def read(path):
    raw=Path(path).read_bytes()
    if len(raw)%4:
        raise ValueError("unaligned FP32 payload")
    values=array.array("f")
    values.frombytes(raw)
    if sys.byteorder!="little": values.byteswap()
    return values


def metrics(got,ref):
    if not got or len(got)!=len(ref) or any(not math.isfinite(x) for x in got) or any(not math.isfinite(x) for x in ref):
        raise ValueError("empty, non-finite or mismatched comparison")
    error=math.fsum((g-r)**2 for g,r in zip(got,ref))
    norm=math.fsum(r*r for r in ref)
    return {"elements":len(got),"rel_l2":math.sqrt(error/norm) if norm else (0.0 if not error else None),
            "psnr":10*math.log10(len(got)/error) if error else "infinity",
            "bitwise":got.tobytes()==ref.tobytes()}


def png_metrics(got, ref):
    # Optional operator dependency; FP32 comparison stays standard-library only.
    from PIL import Image
    values = []
    size = None
    for path in (got, ref):
        with Image.open(path) as image:
            if size is not None and size != image.size:
                raise ValueError("PNG dimensions differ")
            size = image.size
            values.append(array.array("f", (x / 255.0 for x in image.convert("RGB").tobytes())))
    return {**metrics(*values), "width": size[0], "height": size[1]}


if __name__=="__main__":
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument("--latent",nargs=2,required=True,metavar=("GOT","REFERENCE"))
    images=p.add_mutually_exclusive_group(required=True)
    images.add_argument("--rgb",nargs=2,metavar=("GOT","REFERENCE"))
    images.add_argument("--png",nargs=2,metavar=("GOT","REFERENCE"),help="RGB8 PNG comparison (requires Pillow)")
    p.add_argument("--max-latent-rel-l2",type=float,default=0.02)
    p.add_argument("--min-image-psnr",type=float,default=35.0)
    a=p.parse_args()
    if a.max_latent_rel_l2<0 or not math.isfinite(a.max_latent_rel_l2) or not math.isfinite(a.min_image_psnr):
        p.error("invalid acceptance threshold")
    latent=metrics(*(read(f) for f in a.latent))
    rgb=png_metrics(*a.png) if a.png else metrics(*(read(f) for f in a.rgb))
    passed=latent["rel_l2"] is not None and latent["rel_l2"]<=a.max_latent_rel_l2 and (rgb["psnr"]=="infinity" or rgb["psnr"]>=a.min_image_psnr)
    print(json.dumps({"latent":latent,"image":rgb,"verdict":"PASS" if passed else "MISMATCH"}))
    sys.exit(0 if passed else 1)
