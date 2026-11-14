"""Header-only audit of image checkpoint assets; never imports a device library."""
import argparse
import collections
import json
import math
from pathlib import Path
import struct


def headers(folder):
    tensors={}
    files=sorted(folder.glob("*.safetensors")) if folder.is_dir() else ([folder] if folder.is_file() else [])
    for file in files:
        with file.open("rb") as f:
            prefix=f.read(8)
            if len(prefix)!=8: raise ValueError("short checkpoint header")
            size=struct.unpack("<Q",prefix)[0]
            if not 0<size<1<<28: raise ValueError("invalid header length")
            header=json.loads(f.read(size))
        for key,value in header.items():
            if key=="__metadata__": continue
            if key in tensors: raise ValueError("duplicate tensor across checkpoint shards")
            begin,end=value["data_offsets"]
            if not 0<=begin<=end<=file.stat().st_size-8-size: raise ValueError("incomplete tensor payload")
            tensors[key]=value
    return tensors,len(files)


def audit(root):
    result={"payload_read":False,"device_work":False}
    required={"transformer":["video_patch_proj.weight","condition_proj.weight","time_embedder.proj_in.weight",
                             "time_embedder.proj_out.weight","rope.inv_freq","blocks.49.attn.qkv_proj.weight"],
              "vae":["encoder.conv_in.weight","quant_conv.weight","post_quant_conv.weight",
                     "decoder.proj_out.weight","decoder.transformer_blocks.23.norm1.weight","latents_mean","latents_std"],
              "pdd_fused_heads.safetensors":["video_out.weight","video_out.bias"]}
    ready=True
    for component,names in required.items():
        tensors,files=headers(root/component)
        missing=[key for key in names if key not in tensors]
        result[component]={"files":files,"tensors":len(tensors),"missing_contract_keys":missing,
                           "dtype_counts":dict(collections.Counter(t["dtype"] for t in tensors.values())),
                           "selected_shapes":{key:tensors[key]["shape"] for key in names if key in tensors}}
        ready &= not missing
    result["status"]="HEADERS_PRESENT_NATIVE_GATES_PENDING" if ready else "INCOMPLETE_ASSETS"
    return result,ready


if __name__=="__main__":
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument("--model",type=Path,required=True)
    p.add_argument("--require",action="store_true")
    a=p.parse_args()
    result,ready=audit(a.model)
    print(json.dumps(result,sort_keys=True))
    raise SystemExit(1 if a.require and not ready else 0)
