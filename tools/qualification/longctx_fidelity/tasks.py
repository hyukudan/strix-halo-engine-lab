"""Pure symbolic tasks; text ordering never substitutes for declared semantics."""
import json
import math
try:
    from .pcg32 import PCG32
except ImportError:
    from pcg32 import PCG32

SYMBOLS = tuple("ABCDEFGH")
FAMILIES = ("aggregate", "multihop", "multikey", "niah_single", "vartrack")


def latest(records):
    result = {}
    for key, version, value in records:
        if key in result and version == result[key][0]:
            raise ValueError("ambiguous equal version")
        if key not in result or version > result[key][0]:
            result[key] = (version, value)
    return {k: v[1] for k, v in result.items()}


def track(initial, events):
    registers = list(initial)
    if len(registers) != 16 or len({e[0] for e in events}) != len(events):
        raise ValueError("register count or duplicate sequence")
    for _, op, a, b in sorted(events):
        if not 0 <= a < 16:
            raise ValueError("invalid register")
        if op == "copy":
            registers[a] = registers[b]
        elif op == "swap":
            registers[a], registers[b] = registers[b], registers[a]
        elif op == "inc":
            registers[a] = (registers[a] + b) % 97
        else:
            raise ValueError("invalid operation")
    return registers


def walk(edges, namespace, start, hops):
    graph = {}
    for ns, a, b in edges:
        if (ns, a) in graph:
            raise ValueError("ambiguous mapping")
        graph[ns, a] = b
    visited = set()
    for _ in range(hops):
        if start in visited:
            raise ValueError("chain cycle")
        visited.add(start)
        start = graph[namespace, start]
    if start in visited:
        raise ValueError("chain cycle")
    return start


def aggregate(events, group, operation):
    # Every row is an event (identical rows count separately).
    values = [v for g, flag, v in events if g == group and flag == "keep"]
    if operation == "count":
        return len(values)
    if operation == "sum":
        return sum(values)
    if operation == "min":
        return min(values) if values else 0
    raise ValueError("invalid aggregation")


def reasoning_protocol(family, variant="normal"):
    return variant != "symbol" and family.startswith(("vartrack", "multihop", "aggregate"))


def default_generate(family, variant="normal"):
    # Measured on hardware: the model reasons step by step even in non-thinking mode;
    # 64-event vartrack and 8-key multikey need more than 1024/512 tokens, also short.
    if reasoning_protocol(family, variant):
        return 8192 if family == "vartrack256" else 4096 if family.startswith("vartrack") else 1024
    if variant != "symbol" and family.startswith("multikey"):
        return 8192 if family == "multikey32" else 2048
    return 32


def generate(family, depth, position, seed, variant="normal"):
    if family.split("_")[0] not in ("niah",) and family not in (
            *FAMILIES, "multikey32", "vartrack256", "multihop2", "multihop8", "aggregate128"):
        raise ValueError("unknown family")
    if depth <= 0 or not math.isfinite(position) or not 0 <= position <= 1:
        raise ValueError("invalid depth/position")
    if variant not in ("normal", "symbol"):
        raise ValueError("invalid answer variant")
    rng = PCG32(seed)
    facts, data = [], {}
    rule = ""
    if family == "niah_single":
        key, value = rng.opaque(), rng.opaque()[:16]
        facts = [f"vault key={key} value={value}."]
        rule = "A vault record binds its exact key to its value."
        question, answer = f"Value for exact vault key {key}?", value
        data = {"key": key, "value": value}
    elif family.startswith("multikey"):
        count = 32 if family == "multikey32" else 8
        prefix = rng.opaque()
        keys = [prefix + f"-{i:02d}" for i in range(count)]
        records = []
        used = set()
        for key in keys:
            for version in (9, 2):  # obsolete row deliberately later in text
                value = rng.opaque()[:8]
                while value in used:
                    value = rng.opaque()[:8]
                used.add(value)
                records.append((key, version, value))
            records.append((key + "0", 12, rng.opaque()[:8]))
        blocks = 3 + rng.below(6)
        facts = ["\n".join(f"catalog key={k} version={v} value={x}."
                             for k, v, x in records[i::blocks]) for i in range(blocks)]
        oracle = latest(records)
        pairs = [[k, oracle[k]] for k in keys]
        answer = pairs
        rule = "For an exact catalog key, use the greatest numeric version, regardless of text order."
        question = "Return ordered JSON [[key,value],...] pairs for: " + ", ".join(keys)
        if variant == "symbol":
            chosen = rng.choice(keys)
            question, answer = "Value for exact catalog key " + chosen + "?", oracle[chosen]
        data = {"records": records, "keys": keys, "blocks": blocks}
    elif family.startswith("vartrack"):
        initial = [rng.below(97) for _ in range(16)]
        events = [(i, ("copy", "swap", "inc")[i % 3], rng.below(16),
                   rng.below(16) if i % 3 != 2 else rng.below(193) - 96)
                  for i in range(1, (256 if family == "vartrack256" else 64) + 1)]
        final = track(initial, events)
        facts = ["initial " + " ".join(f"r{i}={v}" for i, v in enumerate(initial)) + "."]
        rows = [f"seq={s} {op} r{a} " + (str(b) if op == "inc" else f"r{b}") + "."
                for s, op, a, b in reversed(events)]
        facts += ["\n".join(rows[i::8]) for i in range(8)]
        target = rng.below(16)
        rule = ("Start from initial registers. Execute increasing seq, not text order. "
                "copy rA rB assigns B to A; swap exchanges; inc rA N adds N modulo 97.")
        question, answer = f"Final integer in r{target}?", str(final[target])
        data = {"initial": initial, "events": events, "target": target}
    elif family.startswith("multihop"):
        hops = 2 if family == "multihop2" else 8 if family == "multihop8" else 4
        nodes = [rng.opaque()[:16] for _ in range(hops + 1)]
        if len(set(nodes)) != len(nodes):
            raise ValueError("node collision")
        edges = [("primary", a, b) for a, b in zip(nodes, nodes[1:])]
        edges += [("other", a, rng.opaque()[:16]) for a in nodes[:-1]]
        answer = walk(edges, "primary", nodes[0], hops)
        facts = [f"map namespace={ns} from={a} to={b}." for ns, a, b in edges]
        # Each evidence block holds one primary and one same-label distractor.
        facts = [facts[i] + "\n" + facts[i + hops] for i in range(hops)]
        if seed % 3 == 1:
            facts.reverse()
        elif seed % 3 == 2:
            facts = facts[::2] + facts[1::2]
        rule = "Mappings are unique within each namespace. Follow only primary mappings."
        question = f"Terminal label after exactly {hops} links from {nodes[0]} in primary?"
        data = {"edges": edges, "start": nodes[0], "hops": hops}
    elif family.startswith("aggregate"):
        count = 128 if family == "aggregate128" else 32
        events = [("g" + str(i % 8), "keep" if rng.below(3) else "drop", rng.below(199) - 99)
                  for i in range(count)]
        group, operation = "g" + str(rng.below(8)), rng.choice(("sum", "count", "min"))
        answer = str(aggregate(events, group, operation))
        facts = ["\n".join(f"event group={g} flag={f} amount={v}." for g, f, v in events[i::8])
                 for i in range(8)]
        rule = "Each event counts once. Filter exact group and flag=keep. Empty sum/count/min is 0."
        # Unambiguous wording: "Integer count of amount" was read as a sum on hardware.
        question = {"sum": f"What is the sum of the amount values of the events with group={group} and flag=keep?",
                    "count": f"How many events have group={group} and flag=keep? Count the events, not their amounts.",
                    "min": f"What is the smallest amount value among the events with group={group} and flag=keep?"}[operation]
        data = {"events": events, "group": group, "operation": operation}
    else:
        raise ValueError("unknown family")
    base_answer = answer
    if variant == "symbol":
        # Opaque values get an explicit codebook; integers use a complete mod-8 rule.
        if family.startswith(("vartrack", "aggregate")):
            answer = SYMBOLS[int(answer) % len(SYMBOLS)]
            question += " Map the integer modulo 8 via 0=A,1=B,2=C,3=D,4=E,5=F,6=G,7=H."
        else:
            alternatives = [str(base_answer)]
            while len(alternatives) < len(SYMBOLS):
                value = rng.opaque()[:len(str(base_answer))]
                if value not in alternatives:
                    alternatives.append(value)
            shift = rng.below(8)
            alternatives = alternatives[shift:] + alternatives[:shift]
            answer = SYMBOLS[alternatives.index(str(base_answer))]
            question += " Use codebook " + "; ".join(f"{v}={s}" for v, s in zip(alternatives, SYMBOLS)) + "."
        question += " Return only one symbol."
    elif reasoning_protocol(family, variant):
        question += " You may reason briefly. End your reply with one final line: Answer: <value>"
    elif not isinstance(answer, list):
        question += " Return only the value."
    # Spread blocks across distinct bands, with the requested anchor included.
    n = len(facts)
    if n == 1:
        positions = [position]
    else:
        # First block is anchored exactly. Other blocks occupy successive decile
        # centers; no modulo wrap to position zero (inside template/instructions).
        decile = min(9, int(position * 10))
        positions = [position] + [(((decile + i) % 10) + .5) / 10 for i in range(1, n)]
    return {"family": family, "depth": depth, "position": position, "seed": seed,
            "variant": variant, "rule": rule, "question": question,
            "facts": facts, "positions": positions, "answer": answer,
            "base_answer": base_answer, "oracle_data": data}


def filler_units(seed, stratum="prose"):
    """Infinite deterministic units. Unique serial and 128-bit tags prevent block tiling.

    Vocabulary contains observations only, never instructions, catalog/vault/map/event
    syntax, query keys or generated answer values. Padding is reported separately.
    """
    if stratum not in ("prose", "records", "similar-records"):
        raise ValueError("invalid filler stratum")
    rng = PCG32(seed, stream=71)
    adjectives = ("quiet", "amber", "distant", "silver", "coastal", "green", "mild", "pale")
    nouns = ("garden", "harbor", "meadow", "hill", "river", "cloud", "forest", "shore")
    verbs = ("bordered", "shaded", "surrounded", "crossed", "overlooked", "framed")
    i = 0
    while True:
        tag = rng.opaque()
        if stratum == "prose":
            # Short 32-bit tag: unique units without a hex-dominated haystack that
            # resembles the 128-bit needle keys (the records strata cover that case).
            yield (f"The {rng.choice(adjectives)} {rng.choice(nouns)} {rng.choice(verbs)} "
                   f"a {rng.choice(adjectives)} {rng.choice(nouns)} near survey {i}:{tag[:8]}.\n")
        elif stratum == "records":
            yield f"observation serial={i} tag={tag} hue={rng.choice(adjectives)} level={rng.below(10000)}.\n"
        else:
            yield f"archive item={tag} revision={rng.below(100)} reading={rng.below(100000)} serial={i}.\n"
        i += 1
