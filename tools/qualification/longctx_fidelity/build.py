"""Exact token-budget construction; generated filler alone may be shortened.

Usage: python -m tools.qualification.longctx_fidelity.build --help
"""
import argparse
import json
import math
from pathlib import Path
try:
    from .common import VERSION, NATIVE_CAP, canonical, sha, atomic, ids_bytes, file_sha, case_id
    from .tasks import generate, filler_units
    from .tok import FakeTokenizer, StrixTokenize, PREFIX, SUFFIX, TEMPLATE_SHA256, validate_symbols
except ImportError:
    from common import VERSION, NATIVE_CAP, canonical, sha, atomic, ids_bytes, file_sha, case_id
    from tasks import generate, filler_units
    from tok import FakeTokenizer, StrixTokenize, PREFIX, SUFFIX, TEMPLATE_SHA256, validate_symbols


def source_identity():
    return {p.name: file_sha(p) for p in sorted(Path(__file__).parent.glob("*.py"))}


def build_case(family, depth, position, seed, tokenizer, generate_tokens=32,
               stratum="prose", variant="normal"):
    if not isinstance(depth, int) or not isinstance(generate_tokens, int) or depth < 1 or generate_tokens < 1:
        raise ValueError("D and G must be positive integers")
    if depth + generate_tokens > NATIVE_CAP:
        raise ValueError("native budget requires D + G <= 262144")
    if depth == 262143 and (generate_tokens != 1 or variant != "symbol"):
        raise ValueError("last-token edge probe requires G=1 and validated symbol answers")
    task = generate(family, depth, position, seed, variant)
    header = PREFIX + task["rule"] + "\nEvidence and unrelated observations follow.\n"
    # Server strips user content before appending im_end: no trailing user newline.
    tail = "\nQuestion: " + task["question"] + SUFFIX
    ordered = sorted(zip(task["positions"], task["facts"], range(len(task["facts"]))))
    # Each immutable fact gets delimiter newlines. Target offsets include all template tokens.
    facts = ["\n" + fact + "\n" for _, fact, _ in ordered]
    desired = [round(a * depth) for a, _, _ in ordered]
    fixed = header + "".join(facts) + tail
    fixed_count = tokenizer.count(fixed)
    if fixed_count > depth:
        raise ValueError("depth too short for immutable task/template")
    ratio = len(fixed) / max(1, fixed_count)
    # Deterministic high-entropy reservoir: never tile an existing block.
    reservoir = ""
    units = filler_units(seed, stratum)
    def filler(chars):
        nonlocal reservoir
        while len(reservoir) < chars:
            reservoir += next(units)
        return reservoir[:chars]
    # Separate reservoirs offsets avoid repeating the same filler in different gaps.
    lengths = []
    previous = len(header) / ratio
    for target, fact in zip(desired, facts):
        lengths.append(max(0, round((target - previous) * ratio)))
        previous = max(target, previous) + len(fact) / ratio
    lengths.append(max(0, round((depth - previous - len(tail) / ratio) * ratio)))
    pad = ""
    text = ""
    # Close to the budget, the final filler gap is cut at a line boundary and the
    # remainder is a pad line of " x" units. A mid-word cut makes the count jump by
    # two per character (BPE merges across the cut), so no exact length exists there.
    snap = False
    for attempt in range(96):
        total = sum(lengths)
        pool = filler(total)
        cursor = 0
        pieces, char_spans = [header], []
        char_count = len(header)
        for i, fact in enumerate(facts):
            gap = pool[cursor:cursor + lengths[i]]
            cursor += lengths[i]
            pieces.append(gap)
            char_count += len(gap)
            # Span excludes delimiter newlines, covers the immutable fact exactly.
            char_spans.append((char_count + 1, char_count + len(fact) - 1))
            pieces.append(fact)
            char_count += len(fact)
        last = pool[cursor:cursor + lengths[-1]]
        if snap:
            last = last[:last.rfind("\n") + 1]
        pieces.extend([last, pad, tail])
        text = "".join(pieces)
        ids = tokenizer.encode(text)
        count = len(ids)
        error = depth - count
        if error == 0:
            measured = tokenizer.boundaries(ids, [a for a, _ in char_spans])
            deltas = [want - actual for want, actual in zip(desired, measured)]
            if any(abs(d) > 8 for d in deltas):
                # Move only filler between adjacent immutable blocks. Include the
                # final gap so placement correction does not change total budget.
                changes = [deltas[0]] + [b-a for a, b in zip(deltas, deltas[1:])] + [-deltas[-1]]
                scale = len(text) / max(1, count)
                adjusted = [n + round(d * scale) for n, d in zip(lengths, changes)]
                if min(adjusted) < 0:
                    raise ValueError("requested evidence placement cannot fit immutable task")
                lengths = adjusted
                snap, pad = False, ""
                continue
            break
        # Full-prompt re-encoding checks the one-token-per-unit assumption every time.
        if attempt >= 12 and abs(error) <= 256:
            # The pad line always keeps one unit: with no pad, the gap's final "\n"
            # and the tail's leading "\n" merge into one token, so the first unit
            # would cost two and an error of one would be unreachable.
            if not snap:
                snap, pad = True, " x"
            elif error > 0:
                pad += " x" * error
            elif len(pad) - 2 >= -2 * error:
                pad = pad[:len(pad) + 2 * error]
            else:
                # Drop the last whole filler line and refill with pad units.
                pad = " x"
                lengths[-1] = max(0, len(last) - 1)
        else:
            correction = round(error * (len(text) / max(1, count)))
            # Preserve approximate fractional placement by distributing correction
            # across filler gaps in proportion to their current character budgets.
            weight = max(1, sum(lengths))
            corrections = [round(correction * n / weight) for n in lengths]
            corrections[-1] += correction - sum(corrections)
            lengths = [max(0, n + d) for n, d in zip(lengths, corrections)]
    else:
        raise ValueError("token budget drift: no exact construction; no needle truncated")
    if tokenizer.decode(ids) != text:
        raise ValueError("tokenizer roundtrip drift")
    forbidden = []
    data = task["oracle_data"]
    if family == "niah_single":
        forbidden = [data["key"], data["value"]]
    elif family.startswith("multikey"):
        forbidden = [x for k, _, v in data["records"] for x in (k, v)]
    elif family.startswith("multihop"):
        forbidden = [x for _, a, b in data["edges"] for x in (a, b)]
    if any(value in pool for value in forbidden):
        raise ValueError("answer/key collision with filler")
    starts = tokenizer.boundaries(ids, [a for a, _ in char_spans])
    ends = tokenizer.boundaries(ids, [b for _, b in char_spans])
    spans = []
    for i, ((a, b), start, end) in enumerate(zip(char_spans, starts, ends)):
        # End boundary may be inside a token, requiring one extra covering token.
        if len(tokenizer.decode(ids[:end])) < b:
            end += 1
        recovered = tokenizer.decode(ids[start:end])
        fact = ordered[i][1]
        if fact not in recovered or text[a:b] != fact:
            raise ValueError("needle span recovery drift")
        spans.append({"fact_index": ordered[i][2], "start": start, "end": end,
                      "char_start": a, "char_end": b, "sha256": sha(fact.encode()),
                      "target_start": desired[i], "actual_fraction": start / depth})
    symbols = validate_symbols(tokenizer, text) if variant == "symbol" else None
    answer = task["answer"]
    if isinstance(answer, str) and len(answer) > 24:
        raise ValueError("normal scalar answer exceeds 24 characters")
    identifier = case_id(family, depth, position, seed, stratum, variant)
    manifest = {"schema": 1, "generator_version": VERSION, "case_id": identifier,
                "mode": "native", "family": family, "seed": seed, "position": position,
                "depth": depth, "generate": generate_tokens, "stratum": stratum,
                "variant": variant, "status": "PLANNED", "ground_truth": answer,
                "ground_truth_sha256": sha(canonical(answer)), "text_sha256": sha(text.encode()),
                "ids_sha256": sha(ids_bytes(ids)), "prompt_tokens": len(ids),
                "last_input_position": depth - 1, "first_output_position": depth,
                "chat_template_sha256": TEMPLATE_SHA256, "tokenizer": tokenizer.identity,
                "source_sha256": source_identity(), "spans": spans,
                "rng": {"algorithm": "pcg-xsh-rr-64-32", "multiplier": 6364136223846793005,
                        "stream": 54, "increment": 109, "filler_stream": 71},
                "symbol_ids": symbols, "acceptable_token_ids": [symbols[answer]] if symbols else None,
                "filler": {"generated_chars": sum(lengths), "padding_chars": len(pad),
                           "unique_grammar_units": reservoir.count("\n"),
                           "touched_ple_pages": None},
                "oracle_data": task["oracle_data"],
                "requested": {"context": "auto", "native_limit": NATIVE_CAP,
                              "generate": generate_tokens, "rope_mode": "native"},
                "resolved": None, "reference_sha256": None, "state_lineage": None}
    return text, ids, manifest


def write_case(directory, text, ids, manifest, manifest_only=False):
    directory = Path(directory)
    stem = manifest["case_id"]
    if not manifest_only:
        atomic(directory / (stem + ".txt"), text.encode())
        atomic(directory / (stem + ".u32"), ids_bytes(ids))
    atomic(directory / (stem + ".json"), canonical(manifest))
    return directory / (stem + ".json")


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--family", default="niah_single")
    p.add_argument("--depth", type=int, required=True)
    p.add_argument("--position", type=float, default=.5)
    p.add_argument("--seed", type=int, default=104729)
    p.add_argument("--generate", type=int, default=32)
    p.add_argument("--stratum", choices=("prose", "records", "similar-records"), default="prose")
    p.add_argument("--variant", choices=("normal", "symbol"), default="normal")
    p.add_argument("--fake", action="store_true")
    p.add_argument("--tokenizer")
    p.add_argument("--model")
    p.add_argument("--model-receipt")
    p.add_argument("--out", required=True)
    p.add_argument("--manifest-only", action="store_true")
    a = p.parse_args()
    tok = FakeTokenizer() if a.fake else StrixTokenize(a.tokenizer, a.model, a.model_receipt)
    result = build_case(a.family, a.depth, a.position, a.seed, tok, a.generate, a.stratum, a.variant)
    print(write_case(a.out, *result, manifest_only=a.manifest_only))


if __name__ == "__main__":
    main()
