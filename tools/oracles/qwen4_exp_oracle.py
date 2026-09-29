#!/usr/bin/env python3
"""Small dependency-free numerical oracles for the first target model.

These functions are deliberately simple and use Python's double-precision float
arithmetic. They are references for correctness tests, not implementations for
runtime inference.
"""

from __future__ import annotations

import math
from copy import deepcopy
from typing import Iterable

_MASK64 = (1 << 64) - 1
_SPLITMIX_GAMMA = 0x9E3779B97F4A7C15
_SPLITMIX_M1 = 0xBF58476D1CE4E5B9
_SPLITMIX_M2 = 0x94D049BB133111EB
_PRIME_1 = 10007


def _require_same_length(left: Iterable[object], right: Iterable[object], what: str) -> None:
    if len(left) != len(right):
        raise ValueError(f"{what}: length mismatch")


def rms_norm(
    values: list[float],
    weight: list[float],
    *,
    eps: float = 1e-6,
    group_size: int | None = None,
) -> list[float]:
    """Centred-weight RMSNorm: norm(x) * (1 + weight).

    The optional group size normalizes consecutive independent groups, matching
    the grouped residual-stream form used by the target architecture.
    """

    _require_same_length(values, weight, "rms_norm")
    if not values:
        return []
    size = len(values) if group_size is None else group_size
    if size <= 0 or len(values) % size != 0:
        raise ValueError("group_size must divide the vector length")

    output: list[float] = []
    for start in range(0, len(values), size):
        group = [float(value) for value in values[start : start + size]]
        mean_square = sum(value * value for value in group) / float(size)
        inv_rms = 1.0 / math.sqrt(mean_square + eps)
        for offset, value in enumerate(group):
            index = start + offset
            output.append(value * inv_rms * (1.0 + float(weight[index])))
    return output


def l2_norm(values: list[float], *, eps: float = 1e-6) -> list[float]:
    if not values:
        raise ValueError("l2_norm requires a non-empty vector")
    inv_norm = 1.0 / math.sqrt(sum(float(x) * float(x) for x in values) + eps)
    return [float(x) * inv_norm for x in values]


def rotate_half(values: list[float]) -> list[float]:
    if len(values) % 2 != 0:
        raise ValueError("rotate_half requires an even vector length")
    half = len(values) // 2
    return [-float(x) for x in values[half:]] + [float(x) for x in values[:half]]


def apply_partial_rope(
    values: list[float],
    cos: list[float],
    sin: list[float],
) -> list[float]:
    """Apply RoPE to the prefix covered by cos/sin and preserve the tail."""

    _require_same_length(cos, sin, "apply_partial_rope")
    rotary_dim = len(cos)
    if rotary_dim == 0 or rotary_dim % 2 != 0 or rotary_dim > len(values):
        raise ValueError("rotary dimension must be non-zero, even and within the vector")
    prefix = [float(x) for x in values[:rotary_dim]]
    rotated = rotate_half(prefix)
    rope = [
        prefix[index] * float(cos[index]) + rotated[index] * float(sin[index])
        for index in range(rotary_dim)
    ]
    return rope + [float(x) for x in values[rotary_dim:]]



def _linear(weight: list[list[float]], values: list[float]) -> list[float]:
    output: list[float] = []
    for row in weight:
        if len(row) != len(values):
            raise ValueError("linear weight width does not match input")
        output.append(sum(float(w) * float(x) for w, x in zip(row, values)))
    return output


def _sigmoid(value: float) -> float:
    if value >= 0.0:
        z = math.exp(-value)
        return 1.0 / (1.0 + z)
    z = math.exp(value)
    return z / (1.0 + z)


def _silu(value: float) -> float:
    return value * _sigmoid(value)


def gated_residual(
    hyper_input: list[float],
    *,
    hc_count: int,
    hidden_size: int,
    norm_weight: list[float],
    down_weight: list[list[float]],
    up_weight: list[list[float]],
    inject_weight: list[list[float]] | None = None,
    eps: float = 1e-6,
) -> tuple[list[float], list[float] | None]:
    """Reference gated residual mix for one logical row.

    This follows the declared numerical order: grouped centred RMSNorm, low-rank
    down projection divided by hc_count, SiLU, up projection, sigmoid, per-stream
    elementwise weighting and mean. Optional injection weights are
    2 * sigmoid(linear(normed) / hc_count).
    """

    if hc_count <= 0 or hidden_size <= 0:
        raise ValueError("hc_count and hidden_size must be positive")
    expected = hc_count * hidden_size
    if len(hyper_input) != expected or len(norm_weight) != expected:
        raise ValueError("hyper-connection width does not match configuration")

    normed = rms_norm(
        hyper_input, norm_weight, eps=eps, group_size=hidden_size
    )
    low_rank = [
        _silu(value / float(hc_count))
        for value in _linear(down_weight, normed)
    ]
    expanded = [_sigmoid(value) for value in _linear(up_weight, low_rank)]
    if len(expanded) != expected:
        raise ValueError("up projection does not produce hc_count * hidden_size")

    mixed: list[float] = []
    for feature in range(hidden_size):
        mixed.append(
            sum(
                expanded[stream * hidden_size + feature]
                * normed[stream * hidden_size + feature]
                for stream in range(hc_count)
            )
            / float(hc_count)
        )

    injection = None
    if inject_weight is not None:
        raw = _linear(inject_weight, normed)
        if len(raw) != hc_count:
            raise ValueError("inject projection does not produce hc_count values")
        injection = [
            2.0 * _sigmoid(value / float(hc_count)) for value in raw
        ]
    return mixed, injection


def qsa_select_tokens_no_ties(
    query_heads: list[list[float]],
    raw_keys: list[list[float]],
    visible_indices: list[int],
    *,
    compress_ratio: int,
    token_budget: int,
    key_norm_weight: list[float],
    eps: float = 1e-6,
) -> list[int]:
    """Reference QSA block pooling/scoring for a no-tie fixture.

    query_heads are the already-normalized/rotated query heads. raw_keys are the
    cached unpooled key rows before block pooling. The function intentionally
    refuses a score tie that could affect top-k membership because upstream tie
    semantics are not yet a canonical contract.
    """

    if compress_ratio <= 0 or token_budget <= 0:
        raise ValueError("compress_ratio and token_budget must be positive")
    if token_budget % compress_ratio != 0:
        raise ValueError("token_budget must be divisible by compress_ratio")
    if not query_heads or not query_heads[0]:
        raise ValueError("query heads must not be empty")
    head_dim = len(query_heads[0])
    if len(key_norm_weight) != head_dim:
        raise ValueError("key norm weight width does not match head dimension")
    if any(len(head) != head_dim for head in query_heads):
        raise ValueError("query head dimensions do not match")
    if any(len(key) != head_dim for key in raw_keys):
        raise ValueError("raw key dimensions do not match")
    if any(index < 0 or index >= len(raw_keys) for index in visible_indices):
        raise ValueError("visible key index is out of range")

    complete = len(visible_indices) // compress_ratio
    selected: list[int] = []
    if complete:
        blocks = [
            visible_indices[
                block * compress_ratio : (block + 1) * compress_ratio
            ]
            for block in range(complete)
        ]
        scores: list[float] = []
        for block in blocks:
            pooled = [
                sum(float(raw_keys[index][feature]) for index in block)
                / float(compress_ratio)
                for feature in range(head_dim)
            ]
            pooled = rms_norm(pooled, key_norm_weight, eps=eps)
            score = 0.0
            for query in query_heads:
                dot = sum(float(q) * float(k) for q, k in zip(query, pooled))
                score += max(0.0, dot)
            scores.append(score / math.sqrt(float(head_dim)))

        block_topk = min(token_budget // compress_ratio, complete)
        ordered = sorted(range(complete), key=lambda index: -scores[index])
        if block_topk < complete and block_topk > 0:
            cutoff = scores[ordered[block_topk - 1]]
            next_score = scores[ordered[block_topk]]
            if cutoff == next_score:
                raise ValueError("QSA top-k cutoff has unresolved score tie")
        for block_index in ordered[:block_topk]:
            selected.extend(blocks[block_index])

    selected.extend(visible_indices[complete * compress_ratio :])
    return selected


def moe_route_no_ties(
    logits: list[float],
    *,
    top_k: int,
    normalize_selected: bool = True,
) -> tuple[list[int], list[float]]:
    """Reference MoE routing for one row with an explicit no-tie contract."""

    if top_k <= 0 or top_k > len(logits):
        raise ValueError("top_k is out of range")
    if not logits:
        raise ValueError("router logits must not be empty")
    if any(not math.isfinite(float(value)) for value in logits):
        raise ValueError("router logits must be finite")

    maximum = max(float(value) for value in logits)
    exponentials = [math.exp(float(value) - maximum) for value in logits]
    denominator = sum(exponentials)
    probabilities = [value / denominator for value in exponentials]

    ordered = sorted(
        range(len(probabilities)),
        key=lambda index: -probabilities[index],
    )
    if top_k < len(ordered):
        cutoff = probabilities[ordered[top_k - 1]]
        next_value = probabilities[ordered[top_k]]
        if cutoff == next_value:
            raise ValueError("MoE top-k cutoff has unresolved score tie")

    indices = ordered[:top_k]
    selected = [probabilities[index] for index in indices]
    if normalize_selected:
        selected_sum = sum(selected)
        selected = [value / selected_sum for value in selected]
    return indices, selected

def stable_top_k(values: list[float], k: int) -> list[int]:
    """Candidate deterministic top-k policy: larger value, then lower index."""

    if k < 0 or k > len(values):
        raise ValueError("k is out of range")
    if any(math.isnan(float(value)) for value in values):
        raise ValueError("top-k input contains NaN")
    return sorted(range(len(values)), key=lambda index: (-float(values[index]), index))[:k]


def recurrent_gated_delta_rule(
    query: list[list[list[float]]],
    key: list[list[list[float]]],
    value: list[list[list[float]]],
    decay_log: list[list[float]],
    beta: list[list[float]],
    *,
    initial_state: list[list[list[float]]] | None = None,
    normalize_qk: bool = True,
    l2_eps: float = 1e-6,
) -> tuple[list[list[list[float]]], list[list[list[float]]]]:
    """Sequential Gated DeltaNet reference.

    Shapes:
      query/key: [sequence][heads][key_dim]
      value:     [sequence][heads][value_dim]
      decay_log: [sequence][heads]
      beta:      [sequence][heads]
      state:     [heads][key_dim][value_dim]

    The caller is expected to have repeated key/query heads to the value-head
    count when the model configuration requires it.
    """

    sequence = len(query)
    if sequence == 0:
        raise ValueError("sequence must not be empty")
    if len(key) != sequence or len(value) != sequence or len(decay_log) != sequence or len(beta) != sequence:
        raise ValueError("sequence dimensions do not match")

    heads = len(query[0])
    if heads == 0:
        raise ValueError("head count must not be zero")
    key_dim = len(query[0][0])
    value_dim = len(value[0][0])
    if key_dim == 0 or value_dim == 0:
        raise ValueError("head dimensions must not be zero")

    for position in range(sequence):
        if (
            len(query[position]) != heads
            or len(key[position]) != heads
            or len(value[position]) != heads
            or len(decay_log[position]) != heads
            or len(beta[position]) != heads
        ):
            raise ValueError("head dimensions do not match")
        for head in range(heads):
            if len(query[position][head]) != key_dim or len(key[position][head]) != key_dim:
                raise ValueError("key dimensions do not match")
            if len(value[position][head]) != value_dim:
                raise ValueError("value dimensions do not match")

    if initial_state is None:
        state = [[[0.0 for _ in range(value_dim)] for _ in range(key_dim)] for _ in range(heads)]
    else:
        state = deepcopy(initial_state)
        if len(state) != heads:
            raise ValueError("initial state head count does not match")
        for head_state in state:
            if len(head_state) != key_dim or any(len(row) != value_dim for row in head_state):
                raise ValueError("initial state dimensions do not match")

    outputs: list[list[list[float]]] = []
    query_scale = 1.0 / math.sqrt(float(key_dim))

    for position in range(sequence):
        position_output: list[list[float]] = []
        for head in range(heads):
            q = [float(x) for x in query[position][head]]
            k = [float(x) for x in key[position][head]]
            if normalize_qk:
                q = l2_norm(q, eps=l2_eps)
                k = l2_norm(k, eps=l2_eps)
            q = [x * query_scale for x in q]

            decay = math.exp(float(decay_log[position][head]))
            for key_index in range(key_dim):
                for value_index in range(value_dim):
                    state[head][key_index][value_index] *= decay

            remembered = [
                sum(
                    state[head][key_index][value_index] * k[key_index]
                    for key_index in range(key_dim)
                )
                for value_index in range(value_dim)
            ]
            rate = float(beta[position][head])
            delta = [
                (float(value[position][head][value_index]) - remembered[value_index]) * rate
                for value_index in range(value_dim)
            ]

            for key_index in range(key_dim):
                for value_index in range(value_dim):
                    state[head][key_index][value_index] += k[key_index] * delta[value_index]

            position_output.append(
                [
                    sum(
                        state[head][key_index][value_index] * q[key_index]
                        for key_index in range(key_dim)
                    )
                    for value_index in range(value_dim)
                ]
            )
        outputs.append(position_output)

    return outputs, state


def _splitmix64(value: int) -> int:
    value = (value + _SPLITMIX_GAMMA) & _MASK64
    value = ((value ^ (value >> 30)) * _SPLITMIX_M1) & _MASK64
    value = ((value ^ (value >> 27)) * _SPLITMIX_M2) & _MASK64
    return (value ^ (value >> 31)) & _MASK64


def _is_prime(value: int) -> bool:
    if value < 2:
        return False
    if value % 2 == 0:
        return value == 2
    divisor = 3
    while divisor * divisor <= value:
        if value % divisor == 0:
            return False
        divisor += 2
    return True


def _nth_prime_after(start: int, count: int) -> int:
    prime = start
    for _ in range(count):
        prime += 1
        while not _is_prime(prime):
            prime += 1
    return prime


def ngram_layer_multipliers(
    unigram_vocab_size: int,
    ngram_size: int,
    ple_layer_index: int,
    seed: int,
) -> list[int]:
    if unigram_vocab_size <= 0 or ngram_size < 2 or ple_layer_index < 0:
        raise ValueError("invalid n-gram configuration")
    maximum = (1 << 63) - 1
    multiplier_max = maximum // unigram_vocab_size
    half_bound = max(1, multiplier_max // 2)
    base_seed = seed + _PRIME_1 * ple_layer_index
    return [
        2
        * (
            _splitmix64(
                (base_seed + _SPLITMIX_GAMMA * (index + 1)) & _MASK64
            )
            % half_bound
        )
        + 1
        for index in range(ngram_size)
    ]


def _shift_right_ignore_eos(tokens: list[int], shift: int, eos_token_id: int) -> list[int]:
    if shift == 0:
        return list(tokens)
    output: list[int] = []
    previous_eos = -1
    for position in range(len(tokens)):
        if position > 0 and tokens[position - 1] == eos_token_id:
            previous_eos = position - 1
        position_in_segment = position - (previous_eos + 1)
        source = position - shift
        valid = position_in_segment >= shift and source >= 0
        output.append(tokens[source] if valid else eos_token_id)
    return output


def ngram_ids(
    input_ids: list[int],
    *,
    eos_token_id: int,
    unigram_vocab_size: int,
    ngram_size: int,
    heads_per_ngram: int,
    ngram_vocab_size_base: int,
    ple_layer_index: int,
    seed: int,
    previous_context: list[int] | None = None,
) -> tuple[list[list[int]], list[int]]:
    """Return hashed PLE row ids and the context required for a resumed call."""

    if ngram_size < 2 or heads_per_ngram <= 0:
        raise ValueError("invalid n-gram geometry")
    context_len = ngram_size - 1
    if previous_context is None:
        previous_context = [eos_token_id] * context_len
    if len(previous_context) != context_len:
        raise ValueError("previous context has the wrong length")

    history = list(previous_context) + [int(token) for token in input_ids]
    ngram_heads = context_len * heads_per_ngram

    sizes: list[int] = []
    offsets: list[int] = []
    total = 0
    for head_index in range(ngram_heads):
        global_head = ple_layer_index * ngram_heads + head_index
        size = _nth_prime_after(ngram_vocab_size_base - 1, global_head + 1)
        sizes.append(size)
        offsets.append(total)
        total += size

    multipliers = ngram_layer_multipliers(
        unigram_vocab_size, ngram_size, ple_layer_index, seed
    )
    shifted = [
        _shift_right_ignore_eos(history, amount, eos_token_id)
        for amount in range(ngram_size)
    ]

    all_rows: list[list[int]] = []
    for position in range(len(history)):
        row: list[int] = []
        for ngram in range(2, ngram_size + 1):
            start = (ngram - 2) * heads_per_ngram
            mixed = shifted[0][position] * multipliers[0]
            for component in range(1, ngram):
                mixed ^= shifted[component][position] * multipliers[component]
            for head in range(start, start + heads_per_ngram):
                row.append(mixed % sizes[head] + offsets[head])
        all_rows.append(row)

    current_rows = all_rows[-len(input_ids) :] if input_ids else []
    next_context = history[-context_len:] if context_len else []
    return current_rows, next_context
