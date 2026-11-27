"""Inventory and score a role-isolated Inkling 2 Mini CPU pack study."""
import argparse
import json
import math
import re
import struct
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path[:0] = [str(ROOT), str(ROOT / "tools" / "strix-gate")]
import gate_v2 as gate
from tools.qualification.option_qual.reduce import reduce_pair

PREFIX = "model.llm."
ATTENTION = {
    "wq_du.weight": "query", "wk_dv.weight": "key", "wv_dv.weight": "value",
    "wo_ud.weight": "attention_output", "wr_du.weight": "relative_features",
    "attn_sink_proj.weight": "attention_sink", "gate_attn.weight": "attention_gate",
    "k_sconv.weight_proj": "key_taps", "v_sconv.weight_proj": "value_taps",
}


def checkpoint_header(path):
    with path.open("rb") as stream:
        framing = stream.read(8)
        if len(framing) != 8:
            raise ValueError("truncated checkpoint")
        size = struct.unpack("<Q", framing)[0]
        if not 0 < size <= 64 << 20:
            raise ValueError("invalid checkpoint header size")
        return json.loads(stream.read(size))


def tensor_role(name):
    if name == PREFIX + "embed.weight":
        return "embedding"
    if name == PREFIX + "unembed.weight":
        return "head"
    suffix = re.sub(r"^model\.llm\.layers\.[0-9]+\.", "", name)
    if suffix == "mlp.gate.weight":
        return "router"
    if suffix.startswith("controllers.") and suffix.endswith(".down_weight"):
        return "controller_down"
    return ATTENTION.get(suffix.removeprefix("attn.")) if suffix.startswith("attn.") else None


def inventory(checkpoint):
    header = checkpoint_header(checkpoint)
    matrices, retained, groups = [], [], []
    for name, spec in header.items():
        if name == "__metadata__":
            continue
        begin, end = spec["data_offsets"]
        role = tensor_role(name)
        if ".mlp.experts." in name and (
            name.endswith(("w13_weight", "w2_weight", "_weight_scale", "_weight_amax",
                           "_weight_outer_scale_denominator"))
        ):
            continue
        if role is None:
            retained.append({"name": name, "bytes": end - begin})
            continue
        rows, columns = spec["shape"]
        if spec["dtype"] != "BF16" or columns % 128:
            raise ValueError(f"unsupported matrix {name}")
        matrices.append(dict(name=name, role=role, rows=rows, columns=columns, source_bytes=end - begin))
    for layer in range(24):
        layer_name = f"{PREFIX}layers.{layer}.mlp.experts."
        group = {"first_layer": layer, "layers": 1, "experts": 128, "projections": []}
        for role, tensor, rows, columns in (
            ("expert_gate", "w13_weight", 1152, 2304),
            ("expert_up", "w13_weight", 1152, 2304),
            ("expert_down", "w2_weight", 2304, 1152),
        ):
            source = header[layer_name + tensor]
            scale_tensor = header[layer_name + tensor + "_scale"]
            code_begin, code_end = source["data_offsets"]
            scale_begin, scale_end = scale_tensor["data_offsets"]
            halves = 2 if tensor == "w13_weight" else 1
            source_bytes = (code_end - code_begin + scale_end - scale_begin) // (128 * halves)
            group["projections"].append(dict(
                name=f"{PREFIX}layers.{{layer}}.mlp.experts.{{expert}}.{role.split('_')[1]}.weight",
                role=role, rows=rows, columns=columns, source_bytes=source_bytes,
            ))
        for projection in ("w13_weight", "w2_weight"):
            for suffix in ("_amax", "_outer_scale_denominator"):
                name = layer_name + projection + suffix
                if name in header:
                    begin, end = header[name]["data_offsets"]
                    retained.append({"name": name, "bytes": end - begin})
        groups.append(group)
    return {"schema": "strix-pack-model/v1", "matrices": matrices, "expert_groups": groups, "retained": retained}


def description(inventory_data, choices, routes=None):
    # Copy so one role description and both combined descriptions can use one inventory.
    data = json.loads(json.dumps(inventory_data))
    projections = [p for group in data["expert_groups"] for p in group["projections"]]
    for item in data["matrices"] + projections:
        fmt, coder = choices.get(item["role"], ("keep", None))
        item["format"] = fmt
        if coder is not None:
            item["coder"] = coder
            if routes is not None and coder == "importance" and item["role"].startswith("expert_"):
                item["unobserved_coder"] = "rtn"
    if routes is not None:
        data["routes"] = routes
    return data


def bytes_per_step(inventory_data, role, fmt):
    blocks = {"sp2": 36, "sp3": 52, "sp4": 68}
    visits = [1] * 6 + [3] * 12 + [1] * 6

    def compact(item):
        return item["columns"] // 128 * blocks[fmt]

    exact_bytes, packed_bytes = 0, 0
    for item in inventory_data["matrices"]:
        if item["role"] != role:
            continue
        rows = item["rows"]
        layer = re.match(r"model\.llm\.layers\.(\d+)\.", item["name"])
        count = visits[int(layer[1])] if layer else 1
        if role == "controller_down":
            count -= 1  # The initial visit has no previous-pass message.
        if role == "embedding":
            rows = 1
        elif role == "head":
            rows = 200058  # Only the scored output vocabulary is projected.
        exact_bytes += rows * item["columns"] * 2 * count
        packed_bytes += rows * compact(item) * count
    for group in inventory_data["expert_groups"]:
        for item in group["projections"]:
            if item["role"] != role:
                continue
            count = visits[group["first_layer"]] * 8
            exact_bytes += item["source_bytes"] * count
            packed_bytes += item["rows"] * compact(item) * count
    if exact_bytes == 0:
        raise ValueError(f"unknown role {role}")
    return {"exact_bytes": exact_bytes, "packed_bytes": packed_bytes, "saved": exact_bytes - packed_bytes}


def scores(manifest, reference, candidate, null=None):
    pooled, windows = [], {}
    for name, ids in gate.read_manifest(manifest).items():
        ref = reference / (name + ".f32")
        cand = candidate / (name + ".f32")
        values, info = reduce_pair(ref, cand, "head", ids=ids, rows=len(ids))
        if info["vocab"] != 200058:
            raise ValueError(f"unexpected logits vocabulary for {name}")
        d = values["nll_c"] - values["nll_r"]
        if not all(math.isfinite(float(x)) for field in ("kl_rc", "nll_c", "nll_r") for x in values[field]):
            raise ValueError(f"nonfinite scores for {name}")
        window = {
            "rows": info["rows"], "kl": float(values["kl_rc"].mean()),
            "top1": float(values["agree"].mean()), "dnll": float(d.mean()),
            "dnll_se": gate.mean_se(d)[1] if d.size > 1 else None,
            "exact_nll": float(values["nll_r"].mean()),
        }
        ref_routes, cand_routes = [directory / (name + ".rte") for directory in (reference, candidate)]
        if ref_routes.exists() and cand_routes.exists():
            flips = gate.route_flips(ref_routes, cand_routes)
            window["route_flips"] = flips
        if null is not None:
            null_values, _ = reduce_pair(ref, null / (name + ".f32"), "head", ids=ids, rows=len(ids))
            for tier, limits in gate.TIERS.items():
                judged = gate.window_stats(name, values["kl_rc"], values["agree"], values["nll_r"],
                                           values["nll_c"], [null_values["kl_rc"].mean()],
                                           limits=limits, null_nlls=null_values["nll_c"])
                window[tier] = {"state": judged["state"], "limits": {
                    k: judged[k] for k in ("kl_limit", "top1_limit", "dnll_limit")
                }}
        windows[name] = window
        pooled.append(window)
    count = sum(w["rows"] for w in pooled)
    result = {"rows": count, "windows": windows}
    result.update({k: sum(w[k] * w["rows"] for w in pooled) / count for k in ("kl", "top1", "dnll")})
    result["absolute_bars"] = {
        tier: result["kl"] <= limits["max_kl"] and result["top1"] >= limits["min_top1"] and
        result["dnll"] <= limits["max_dnll"] for tier, limits in gate.TIERS.items()
    }
    result["qualification"] = "provisional" if count < gate.LIMITS["min_rows"] or null is None else "see gate v2"
    return result


def choose(rows, tier):
    limits = gate.TIERS[tier]
    available = {}
    for row in rows:
        if row["format"] == "keep":
            raise ValueError("source precision is an implicit candidate")
        available.setdefault(row["role"], []).append(row)
    roles = sorted(available)
    for role in roles:
        available[role].append(dict(format="keep", coder=None, saved=0, kl=0, top1=1, dnll=0))
        available[role].sort(key=lambda row: -row["saved"])
    upper = [0] * (len(roles) + 1)
    for index in range(len(roles) - 1, -1, -1):
        upper[index] = upper[index + 1] + available[roles[index]][0]["saved"]
    best_saved, best_totals, best_choices = -1, None, None

    def search(index, saved, kl, disagreement, dnll, picked):
        nonlocal best_saved, best_totals, best_choices
        if (saved + upper[index] <= best_saved or kl > limits["max_kl"] or
                disagreement > 1 - limits["min_top1"] or dnll > limits["max_dnll"]):
            return
        if index == len(roles):
            best_saved, best_totals, best_choices = saved, [kl, 1 - disagreement, dnll], picked
            return
        for row in available[roles[index]]:
            search(index + 1, saved + row["saved"], kl + row["kl"], disagreement + 1 - row["top1"],
                   dnll + max(0, row["dnll"]), picked + [(row["format"], row["coder"])])

    search(0, 0, 0, 0, 0, [])
    return dict(zip(roles, best_choices)), dict(zip(("kl", "top1", "dnll", "saved"), best_totals + [best_saved]))


def write_json(path, value):
    with path.open("x", encoding="utf-8") as output:
        json.dump(value, output, indent=2, allow_nan=False)
        output.write("\n")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest="command", required=True)
    inv = commands.add_parser("inventory")
    inv.add_argument("checkpoint", type=Path)
    inv.add_argument("output", type=Path)
    arm = commands.add_parser("arm")
    arm.add_argument("inventory", type=Path)
    arm.add_argument("role")
    arm.add_argument("format", choices=["sp2", "sp3", "sp4"])
    arm.add_argument("coder", choices=["rtn", "importance"])
    arm.add_argument("output", type=Path)
    arm.add_argument("--routes", type=Path)
    score = commands.add_parser("score")
    for name in ("manifest", "reference", "candidate", "output"):
        score.add_argument(name, type=Path)
    score.add_argument("--null", type=Path)
    score.add_argument("--inventory", type=Path)
    score.add_argument("--role")
    score.add_argument("--format", choices=["sp2", "sp3", "sp4"])
    score.add_argument("--coder", choices=["rtn", "importance"])
    plan = commands.add_parser("choose")
    plan.add_argument("inventory", type=Path)
    plan.add_argument("measurements", type=Path)
    plan.add_argument("tier", choices=gate.TIERS)
    plan.add_argument("output", type=Path)
    plan.add_argument("--routes", type=Path)
    args = parser.parse_args()
    if args.command == "inventory":
        result = inventory(args.checkpoint)
    elif args.command == "score":
        result = scores(args.manifest, args.reference, args.candidate, args.null)
        if any((args.inventory, args.role, args.format, args.coder)):
            if not all((args.inventory, args.role, args.format, args.coder)):
                parser.error("role scoring needs inventory, role, format and coder")
            data = json.loads(args.inventory.read_text(encoding="utf-8"))
            result.update(bytes_per_step(data, args.role, args.format), role=args.role, format=args.format,
                          coder=args.coder)
    else:
        data = json.loads(args.inventory.read_text(encoding="utf-8"))
        routes = json.loads(args.routes.read_text(encoding="utf-8"))["routes"] if args.routes else None
        if args.command == "arm":
            result = description(data, {args.role: (args.format, args.coder)}, routes)
        else:
            measurements = json.loads(args.measurements.read_text(encoding="utf-8"))
            choices, predicted = choose(measurements, args.tier)
            result = description(data, choices, routes)
            result["prediction"] = predicted
            result["quality"] = "candidate until measured end to end"
    write_json(args.output, result)


if __name__ == "__main__":
    main()
