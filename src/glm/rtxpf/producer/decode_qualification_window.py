"""Decode exactness gate (glm-dec-rtx-native-v1, draft), plan-only unless --allow-gpu.

One resident service load (capacity 32768, forced file tier) driven over
stdin/stdout. For every tested step t the incremental session state after t
generated tokens must equal a fresh prefill of prompt+generated[:t]: per-rank
payload SHA-256, component extents/order and the KDA replay checkpoint digest;
then both arms consume the pending token and must agree on the complete logits
digest and greedy token. Session prefill must also equal the unchanged prefill
producer's per-rank state. Timing arms restore one prefilled session snapshot
(cold/warm order interleaved) and must agree on tokens, every complete logits
digest and the final state/replay digests before their timings are accepted.

Required coverage (else INCOMPLETE, never EXACT): prompts of exactly 1024 and
8192 tokens; consumed positions crossing a B32 block boundary (residues 31, 0, 1);
sparse DSA selection (>512 complete pools) at 8K; nonzero selected-expert
file-tier reads during generation; every planned chat case present and run.
Exactness of completed cases is reported separately from workload completion.
Expected wall budget: 20-28 minutes; hard deadline at most 30 minutes plus owned
cleanup.
"""
import argparse
import copy
import hashlib
import importlib.util
import json
import os
from pathlib import Path
import queue
import re
import struct
import subprocess
import sys
import threading
import time

IDENTITY = 'glm-dec-rtx-native-v1'
CAPACITY = 32768
# Whole-card runs below the full card plan at the capacity they need: activations
# and attention/KV state scale with it. 8K prompts + up to 64 steps, 32-aligned.
WHOLE_CARD_CAPACITY = 8320


def capacity(a):
    """The service capacity this run plans and starts with."""
    return getattr(a, 'capacity', None) or CAPACITY
QUALIFY = (1024, 8192)
CHAT_KINDS = ('explanation', 'coding', 'arithmetic', 'extraction', 'summarization', 'instruction', 'multiturn')
MAX_SECONDS = 30 * 60
CLEANUP_MARGIN = 150  # service exit after releasing device + pinned memory
PRIOR_SECONDS = {('qualify', 1024): 240, ('qualify', 8192): 480, ('bench', 1024): 120, ('bench', 8192): 200,
                 ('compare', 1024): 120, ('compare', 8192): 240,
                 # TTFT priors: about twice the measured resident-prefill times (27-34 / 35-41 / 71-81 s)
                 ('ttft', 1024): 60, ('ttft', 8192): 90, ('ttft', 32000): 180}
# 'Full card' mode (whole RTX PRO, cached path only): prefill TTFT at these prompt
# lengths (q1024/q8192 and ttft-32000.u32; cold cache each) + decode cold/warm on
# these chat kinds at 1K/8K, after the exactness gate; sized for <= 30 min.
TTFT = (1024, 8192, 32000)
# Service startup (unit-file hash: 156.6 GB every start without a USN journal on its
# volume; source recall; loads): a prior until the receipts' measured startup_s replaces it.
STARTUP_PRIOR = 300
# Full card: prefill at resident speed (experts through the cache, ~half on the device).
FULL_CARD_PRIORS = {('qualify', 1024): 150, ('qualify', 8192): 300, ('compare', 1024): 120,
                    ('bench', 1024): 100, ('bench', 8192): 150,
                    # a full-card run measured TTFT 8.03 s (1K) / 17.05 s (8K) at tile 2048; each arm also
                    # exports the prefill state for its digest (untimed). 32K is not measured yet.
                    ('ttft', 1024): 30, ('ttft', 8192): 60, ('ttft', 32000): 180}
# Full card: TTFT tile arms per prompt length (None = the service's allocated tile:
# the default rule, the largest tile <= 8192 whose growth over 2048 costs <= 5% of
# the expert arena; a number is capped at it, since a per-request tile cannot exceed
# the startup allocation, and the receipt says so). The
# prefill state digests of the arms of one length are compared bitwise: equal means
# the tile is a pure speed choice; different means it changes reduction order, so
# the larger tile is a FAST-profile lever, not the default.
FULL_CARD_TTFT_TILES = {1024: (None,), 8192: (8192, 2048), 32000: (None, 2048)}
# Tile sweep (full card, --tile-sweep 2048,4096,8192): one service start per tile
# (startup allocation --row-tile T); the exactness gate (q1024 + q8192 + the untimed
# q1024 compare) once, on the first tile; per tile TTFT q8192 + ttft-32000 and the
# decode bench (cold/warm) on one chat kind at 1K/8K, all with state/logits digests;
# cross-tile digest comparison in the summary. Priors from a full-card run (qualify 82 /
# 147 s, TTFT 8K 23 s, bench 26 / 28 s, startup 180 s incl. the 104 s unit-file hash)
# with margin; restarts recall the unit-file record (no hash).
SWEEP_PRIORS = {('qualify', 1024): 100, ('qualify', 8192): 180, ('compare', 1024): 120,
                ('ttft', 8192): 35, ('ttft', 32000): 120, ('bench', 1024): 40, ('bench', 8192): 45}
SWEEP_FIRST_START = 200  # the first start hashes the unit file once (then the record is recalled)
SWEEP_RESTART = 100      # a later start: loads only
SWEEP_STOP = 60          # service exit between tiles (device + pinned memory release)
SWEEP_DROP_FIRST = 4096  # dropped first when the sweep does not fit the deadline


def sweep_segments(a):
    """[(tile, items)] of a tile sweep: the gate on the first tile, the per-tile arms on every tile."""
    d = a.prompt_dir
    gate = [('qualify', n, d / f'q{n}.u32') for n in QUALIFY]
    gate += [('compare', n, d / (f'q{n}.u32' if kind == 'q' else f'chat-{kind}-{n}.u32')) for kind, n in COMPARE]
    per = [('ttft', 8192, d / 'q8192.u32', None), ('ttft', 32000, d / 'ttft-32000.u32', None)]
    per += [('bench', n, d / f'chat-{FULL_CARD_CHATS[0]}-{n}.u32', ['cold', 'warm']) for n in QUALIFY]
    return [(tile, (gate if k == 0 else []) + per) for k, tile in enumerate(a.tile_sweep)]


def sweep_seconds(a, tiles):
    """Whole-sweep estimate: every start and stop, every item, the final cleanup."""
    b = dict(vars(a))
    b['tile_sweep'] = list(tiles)
    items = [i for _, seg in sweep_segments(argparse.Namespace(**b)) for i in seg]
    starts = SWEEP_FIRST_START + (len(tiles) - 1) * (SWEEP_RESTART + SWEEP_STOP)
    return starts + sum(SWEEP_PRIORS[(i[0], i[1])] for i in items) + CLEANUP_MARGIN


def fit_sweep(a, tiles):
    """(tiles, note): drops the 4096 arm first when the sweep exceeds the deadline; refuses if it still does."""
    tiles, note = list(tiles), ''
    if sweep_seconds(a, tiles) > a.timeout and SWEEP_DROP_FIRST in tiles and len(tiles) > 1:
        note = f'4096 arm dropped: the sweep needed {sweep_seconds(a, tiles)} s > {a.timeout} s'
        tiles.remove(SWEEP_DROP_FIRST)
    if sweep_seconds(a, tiles) > a.timeout:
        raise ValueError(f'tile sweep {tiles} needs {sweep_seconds(a, tiles)} s > --timeout {a.timeout} s')
    return tiles, note


def priors(a):
    """Per-item seconds for this run's mode."""
    p = dict(PRIOR_SECONDS)
    if getattr(a, 'full_card', False):
        p.update(FULL_CARD_PRIORS)
    if getattr(a, 'tile_sweep', None):
        p.update(SWEEP_PRIORS)
    return p


def whole_card(a):
    """A whole-device run below the full card (e.g. 24 GiB + 32 GiB)."""
    return getattr(a, 'vram_total_gib', None) is not None and not getattr(a, 'full_card', False)
FULL_CARD_CHATS = ('explanation',)  # one chat kind at 1K and 8K: the schedule fits the 30 min deadline
FULL_CARD_VRAM_GIB = 93.0  # the usable PRO device (99.86e9 B = 93.0 GiB), the runtime's limit
FULL_CARD_HOST_MARGIN_GIB = 16


def host_total_gib():
    """Physical host memory in GiB (the --full-card RAM cap is this minus a margin)."""
    if os.name == 'nt':
        import ctypes

        class Status(ctypes.Structure):
            _fields_ = [('dwLength', ctypes.c_ulong), ('dwMemoryLoad', ctypes.c_ulong), ('ullTotalPhys', ctypes.c_ulonglong),
                        ('ullAvailPhys', ctypes.c_ulonglong), ('ullTotalPageFile', ctypes.c_ulonglong), ('ullAvailPageFile', ctypes.c_ulonglong),
                        ('ullTotalVirtual', ctypes.c_ulonglong), ('ullAvailVirtual', ctypes.c_ulonglong), ('ullAvailExtendedVirtual', ctypes.c_ulonglong)]
        s = Status()
        s.dwLength = ctypes.sizeof(Status)
        if not ctypes.windll.kernel32.GlobalMemoryStatusEx(ctypes.byref(s)):
            raise OSError('GlobalMemoryStatusEx failed')
        return s.ullTotalPhys / (1 << 30)
    return os.sysconf('SC_PAGE_SIZE') * os.sysconf('SC_PHYS_PAGES') / (1 << 30)
# Reference-backend comparison (tierstream only): identical continuations under
# the uncached reference tier and the tierstream cache, UNTIMED and on the
# smallest prompt only (no timed uncached/reference arms in any PRO
# window; every reported tok/s and TTFT is the cached tierstream path). Run
# order: qualification (q1024, q8192) -> comparison (q1024) -> timing arms
# (all chat cases, tierstream only).
COMPARE = (('q', 1024),)
# 19 device + 22 pinned of 43 routed layers: two file-tier layers covering both ring parities.
FILE_TIER_FLAGS = ['--max-device-layers', '19', '--max-pinned-layers', '22', '--require-file-tier']
# Tierstream mode: the cache owns the whole expert budget (--tier-vram-gib/--tier-ram-gib);
# prefill keeps no resident/pinned expert copies and streams all 43 layers (both ring parities).
TIERSTREAM_PREFILL_FLAGS = ['--max-device-layers', '0', '--max-pinned-layers', '0', '--require-file-tier']
SPARSE_POOLS = 512  # selection is sparse once more than 512 complete pools are scored


def _window():
    spec = importlib.util.spec_from_file_location('glm_qualification_window', Path(__file__).with_name('qualification_window.py'))
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


window = _window()
Deadline, PinnedExecutables, digest, stop, start = (window.Deadline, window.PinnedExecutables, window.digest,
                                                    window.stop, window.start)


def read_u32(path, length):
    """Exactly `length` little-endian u32 IDs inside the vocabulary."""
    raw = Path(path).read_bytes()
    if len(raw) != 4 * length:
        raise ValueError(f'prompt {Path(path).name}: must contain exactly {length} tokens, found {len(raw) / 4:g}')
    toks = list(struct.unpack(f'<{length}I', raw))
    if any(t >= 154880 for t in toks):
        raise ValueError(f'prompt {Path(path).name}: token outside vocabulary')
    return toks, hashlib.sha256(raw).hexdigest()


def prompt_plan(a):
    """Every planned (op, length, path[, order]); chat cases are planned whether present or not."""
    if getattr(a, 'tile_sweep', None):
        return [i for _, seg in sweep_segments(a) for i in seg]
    if a.expert_tier != 'tierstream':
        # The reference backend runs only inside the untimed q1024 compare.
        raise ValueError('only the cached tierstream backend runs standalone (reference only inside its compare)')
    items = [('qualify', n, a.prompt_dir / f'q{n}.u32') for n in QUALIFY]
    items += [('compare', n, a.prompt_dir / (f'q{n}.u32' if kind == 'q' else f'chat-{kind}-{n}.u32')) for kind, n in COMPARE]
    # Every run: startup -> exactness gate (q1024, q8192) -> untimed q1024 compare -> TTFT
    # once each (32K only with --full-card) -> decode cold/warm on one chat kind at 1K/8K.
    # Sized with the startup prior so the printed estimate fits the deadline.
    lengths = TTFT if getattr(a, 'full_card', False) else QUALIFY
    for n in lengths:
        path = a.prompt_dir / (f'q{n}.u32' if n in QUALIFY else f'ttft-{n}.u32')
        tiles = FULL_CARD_TTFT_TILES[n] if getattr(a, 'full_card', False) else (None,)
        items += [('ttft', n, path, tile) for tile in tiles]
    for n in QUALIFY:
        for i, kind in enumerate(FULL_CARD_CHATS):
            items.append(('bench', n, a.prompt_dir / f'chat-{kind}-{n}.u32', ['cold', 'warm'] if i % 2 == 0 else ['warm', 'cold']))
    return items


def check_inputs(a):
    """Qualification prompts are required; a missing chat prompt is recorded (INCOMPLETE later)."""
    pins = {}
    for item in prompt_plan(a):
        op, n, path = item[:3]
        if path.name in pins:
            continue  # comparison / TTFT items reuse the qualification/chat prompts
        if not path.is_file():
            if op == 'qualify':
                raise ValueError(f'missing qualification prompt {path.name}')
            pins[path.name] = dict(missing=True)
            continue
        toks, sha = read_u32(path, n)
        if n + (a.steps if op == 'qualify' else 1 if op == 'ttft' else a.bench_steps) + 1 > capacity(a):
            raise ValueError(f'{path.name}: steps exceed capacity')
        pins[path.name] = dict(tokens=len(toks), sha256=sha)
    if a.expert_tier == 'tierstream':
        if not a.unit_file or not Path(a.unit_file).is_file() or not Path(str(a.unit_file) + '.units.json').is_file():
            raise ValueError('--expert-tier tierstream needs --unit-file PRIMARY with PRIMARY.units.json beside it')
        digest(a.unit_manifest_sha256 or '')  # the pinned manifest digest printed by glm_unit_file copy
        pins['unit_file'] = dict(path=str(a.unit_file), bytes=Path(a.unit_file).stat().st_size)
    if a.steps < 3:
        raise ValueError('--steps must be >= 3 so consumed positions cross a B32 boundary (residues 31, 0, 1)')
    return pins


def service_command(a, binary):
    source = (['--model', str(a.model)] if a.model else
              ['--cache', str(a.cache), '--cache-manifest-sha256', digest(a.cache_manifest_sha256)])
    tier = ['--expert-tier', a.expert_tier]
    if a.expert_tier == 'tierstream':
        tier += ['--unit-file', str(a.unit_file), '--unit-manifest-sha256', str(a.unit_manifest_sha256), '--reference-compare']
        for flag, value in (('--vram-total-gib', getattr(a, 'vram_total_gib', None)), ('--ram-total-gib', getattr(a, 'ram_total_gib', None)),
                            ('--tier-vram-gib', a.tier_vram_gib), ('--tier-ram-gib', a.tier_ram_gib),
                            ('--tier-spare-percent', getattr(a, 'tier_spare_percent', None)),
                            ('--tier-policy', getattr(a, 'tier_policy', None)),
                            ('--tier-predict', getattr(a, 'tier_predict', None)),
                            ('--tier-flush-ready', getattr(a, 'tier_flush_ready', None)),
                            ('--skip-empty-expert-groups', getattr(a, 'skip_empty_expert_groups', None)),
                            ('--persistent-dense-fp16', getattr(a, 'persistent_dense_fp16', None)),
                            ('--expert-residency', getattr(a, 'expert_residency', None)),
                            ('--expert-routes', getattr(a, 'expert_routes', None)),
                            ('--routes-out', getattr(a, 'routes_out', None)),
                            ('--row-tile', getattr(a, 'row_tile', None)),
                            ('--profile', getattr(a, 'profile', None))):
            if value is not None:
                tier += [flag, str(value)]
        if getattr(a, 'allow_pending', False):
            tier += ['--allow-pending']
        if getattr(a, 'decode_profile', False):
            tier += ['--decode-profile']
    prefill = TIERSTREAM_PREFILL_FLAGS if a.expert_tier == 'tierstream' else FILE_TIER_FLAGS
    return [str(binary), *source, '--capacity', str(capacity(a)), *prefill, '--decode-commands', *tier, '--allow-gpu']


def print_effective_config(a):
    """The service's --profile resolution (profile.hpp: the profile and every lever row, value / status / effective /
    kind / source, pending rows): the service prints it and exits before any model read or CUDA call."""
    cmd = service_command(a, a.build / 'glm_native_service.exe')
    cmd[-1:] = ['--print-effective-config']
    done = subprocess.run(cmd, capture_output=True, text=True, timeout=60)
    sys.stdout.write(done.stdout)
    sys.stderr.write(done.stderr)
    return done.returncode


def plan_budget(a):
    """Dry tierstream budget plan: the service reads shard headers only and prints the
    planned VRAM/host split (no --allow-gpu, no hashing, no unit-file read, no CUDA calls)."""
    if a.expert_tier != 'tierstream' or not a.model:
        raise ValueError('--plan-budget plans the tierstream budget from --model shards')
    cmd = service_command(a, a.build / 'glm_native_service.exe')
    cmd[-1:] = ['--plan-budget']
    for flag, value in (('--device-total-gib', getattr(a, 'device_total_gib', None)),
                        ('--device-free-gib', getattr(a, 'device_free_gib', None))):
        if value is not None:
            cmd += [flag, str(value)]
    done = subprocess.run(cmd, capture_output=True, text=True, timeout=300)
    sys.stdout.write(done.stdout)
    sys.stderr.write(done.stderr)
    return done.returncode


def expected_seconds(a):
    if getattr(a, 'tile_sweep', None):
        return sweep_seconds(a, a.tile_sweep)
    return expected_seconds_single(a)


def expected_seconds_single(a):
    """The planned workload's estimate: per-item priors plus the cleanup margin (startup excluded)."""
    p = priors(a)
    return STARTUP_PRIOR + sum(p[(i[0], i[1])] for i in prompt_plan(a)) + CLEANUP_MARGIN


def plan(a):
    return dict(mode='DECODE-SELF-CONSISTENCY', identity=IDENTITY, capacity=capacity(a), steps=a.steps,
                bench_steps=a.bench_steps, expert_tier=a.expert_tier,
                command=service_command(a, Path('BUILD') / 'glm_native_service.exe'),
                items=[dict(op=i[0], tokens=i[1], prompt=i[2].name) for i in prompt_plan(a)],
                expected_wall_seconds=expected_seconds(a), aggregate_timeout_seconds=a.timeout,
                estimate_basis=f'startup prior {STARTUP_PRIOR} s (unit-file hash, source recall, loads; the receipt '
                               f'records the measured startup_s) + per-item priors + cleanup {CLEANUP_MARGIN} s; the '
                               'deadline skips items that no longer fit (INCOMPLETE)',
                gates=['per-rank state SHA incremental == fresh prefill at every tested t',
                       'component extents/order and KDA replay checkpoint digests equal',
                       'complete logits digest and greedy token equal after consuming the pending token',
                       'session prefill == unchanged prefill producer state (per rank)',
                       'receiver snapshot layout/hash of the 1K incremental export',
                       'timing arms: labels, token counts, every logits digest and final state/replay digests equal',
                       'tierstream: reference-backend comparison of identical continuations (tokens, logits, state, replay)'],
                run_order=' -> '.join(['startup'] + [f"{i[0]} {i[2].name}" for i in prompt_plan(a)]),
                cache='tierstream: cold arm = reset_open, warm arm = after an untimed warm-up pass; misses/token, '
                      'hits per tier and bytes/token per arm' if a.expert_tier == 'tierstream' else 'reference (uncached test backend)',
                coverage=['prompts exactly 1024 and 8192 tokens', 'B32 boundary residues 31/0/1 consumed',
                          'sparse DSA selection at 8K', 'nonzero file-tier expert reads during generation',
                          f"untimed reference compare on {', '.join(sorted({i[2].name for i in prompt_plan(a) if i[0] == 'compare'}))}",
                          f"TTFT at {', '.join(str(i[1]) for i in prompt_plan(a) if i[0] == 'ttft')} tokens (cold cache)",
                          f"{sum(1 for i in prompt_plan(a) if i[0] == 'bench')} planned chat case(s) run cold/warm: "
                          f"{', '.join(i[2].name for i in prompt_plan(a) if i[0] == 'bench')}"],
                verdicts=['EXACT', 'MISMATCH', 'NO-REFERENCE', 'INCOMPLETE'],
                qualification='DRAFT; node R1b greedy comparison is a separate measured result')


def ranks(d):
    return [d.get('rank0_sha256'), d.get('rank1_sha256')]


def file_bytes(steps):
    return sum(int(s.get('experts', {}).get('file_bytes', 0)) for s in steps)


# ---- Evidence: ONE strict validator per record kind ---------------------------------
# Each validator takes the PLANNED request (prompt n, steps, arm labels, expected tile,
# startup allocation) and returns canonical evidence: every hash a validated lowercase
# SHA-256, byte counts > 0, the replay checkpoint, frontier == the planned frontier,
# non-empty ordered component extents, token IDs = ints in [0, vocab) of the planned
# length, every generated token bound to its step (index 0 included), and for qualify
# the initial state. Any defect raises Evidence -> NO-REFERENCE with the reason. Every
# judge and comparison works ONLY on these objects, never on raw response fields.
VOCAB = 154880


class Evidence(ValueError):
    """Incomplete or malformed evidence: the record is NO-REFERENCE."""


def ev_hex(x, what):
    if not isinstance(x, str) or not re.fullmatch('[0-9a-f]{64}', x):
        raise Evidence(f'{what}: not a lowercase SHA-256')
    return x


def ev_int(x, what, lo=0, hi=None):
    if type(x) is not int or x < lo or (hi is not None and x >= hi):
        raise Evidence(f'{what}: {x!r} is not an integer in [{lo}, {hi if hi is not None else "inf"})')
    return x


def ev_pos(x, what):
    if type(x) not in (int, float) or not 0 < x < float('inf'):
        raise Evidence(f'{what}: {x!r} is not a positive number')
    return x


def ev_list(x, what, length=None):
    if not isinstance(x, list) or (length is not None and len(x) != length):
        raise Evidence(f'{what}: expected a list' + (f' of {length}' if length is not None else ''))
    return x


def ev_dict(x, what):
    if not isinstance(x, dict):
        raise Evidence(f'{what}: expected an object')
    return x


def ev_tokens(x, what, length):
    return [ev_int(v, f'{what}[{i}]', 0, VOCAB) for i, v in enumerate(ev_list(x, what, length))]


def ev_planned(rec, n, what):
    """The record answers the PLANNED prompt (its reported length must equal n)."""
    ev_int(n, f'{what}: planned prompt length', 1)
    if type(rec.get('prompt_tokens')) is not int or rec['prompt_tokens'] != n:
        raise Evidence(f"{what}: reported prompt length {rec.get('prompt_tokens')!r} != planned {n}")


def ev_state(s, frontier, what):
    """Canonical state signature (session_digest): both ranks' SHA-256 and bytes, the KDA
    replay checkpoint, frontier == the planned frontier, non-empty ordered components."""
    s = ev_dict(s, what)
    if type(s.get('frontier')) is not int or s['frontier'] != frontier:
        raise Evidence(f"{what}.frontier {s.get('frontier')!r} != planned {frontier}")
    comps = ev_list(s.get('components'), f'{what}.components')
    if not comps:
        raise Evidence(f'{what}.components: empty')
    canon = []
    for i, c in enumerate(comps):
        c = ev_list(c, f'{what}.components[{i}]', 3)
        canon.append([ev_int(c[0], f'{what}.components[{i}].layer'), ev_int(c[1], f'{what}.components[{i}].component'),
                      ev_int(c[2], f'{what}.components[{i}].bytes', 1)])
    bytes0 = ev_int(s.get('rank0_bytes'), f'{what}.rank0_bytes', 1)
    if sum(c[2] for c in canon) != bytes0:  # session_digest lists rank 0's components with their sizes
        raise Evidence(f'{what}: rank0_bytes {bytes0} != the sum of its component extents')
    return dict(rank0=ev_hex(s.get('rank0_sha256'), f'{what}.rank0_sha256'), rank1=ev_hex(s.get('rank1_sha256'), f'{what}.rank1_sha256'),
                bytes0=bytes0, bytes1=ev_int(s.get('rank1_bytes'), f'{what}.rank1_bytes', 1),
                replay=ev_hex(s.get('replay_sha256'), f'{what}.replay_sha256'), frontier=frontier, components=canon)


def evidence_qualify(rec, n, steps):
    """Qualification: initial state (t=0, frontier n-1), steps+1 generated tokens each bound
    to its decode step (index 0 included, position n-1+i), incremental and fresh states at
    n-1+t for t=1..steps, fresh next token/logits, the prefill producer's rank digests."""
    rec = ev_dict(rec, 'qualify')
    ev_planned(rec, n, 'qualify')
    ev_int(steps, 'planned steps', 1)
    generated = ev_tokens(rec.get('generated'), 'generated', steps + 1)
    step = []
    for i, s in enumerate(ev_list(rec.get('incremental_steps'), 'incremental_steps', steps + 1)):
        s = ev_dict(s, f'incremental_steps[{i}]')
        tok = ev_int(s.get('token'), f'incremental_steps[{i}].token', 0, VOCAB)
        if tok != generated[i]:
            raise Evidence(f'incremental_steps[{i}].token {tok} != generated[{i}] {generated[i]}')
        if type(s.get('position')) is not int or s['position'] != n - 1 + i:
            raise Evidence(f"incremental_steps[{i}].position {s.get('position')!r} != {n - 1 + i}")
        step.append(dict(token=tok, position=n - 1 + i, logits=ev_hex(s.get('logits_sha256'), f'incremental_steps[{i}].logits_sha256')))

    def by_t(items, what, ts):
        out = {}
        for i, e in enumerate(ev_list(items, what, len(ts))):
            e = ev_dict(e, f'{what}[{i}]')
            k = e.get('t')
            if type(k) is not int or k not in ts or k in out:
                raise Evidence(f'{what}[{i}].t {k!r}: not a distinct planned step')
            out[k] = e
        return out
    inc = by_t(rec.get('incremental_exports'), 'incremental_exports', range(steps + 1))
    fresh = by_t(rec.get('fresh'), 'fresh', range(1, steps + 1))
    legacy = ev_dict(rec.get('legacy_prefill'), 'legacy_prefill')
    fresh_c = []
    for k in range(1, steps + 1):
        nx = ev_dict(fresh[k].get('next'), f'fresh[t={k}].next')
        if type(nx.get('position')) is not int or nx['position'] != n - 1 + k:
            raise Evidence(f"fresh[t={k}].next.position {nx.get('position')!r} != {n - 1 + k}")
        fresh_c.append(dict(state=ev_state(fresh[k], n - 1 + k, f'fresh[t={k}]'), next_position=n - 1 + k,
                            next_token=ev_int(nx.get('token'), f'fresh[t={k}].next.token', 0, VOCAB),
                            next_logits=ev_hex(nx.get('logits_sha256'), f'fresh[t={k}].next.logits_sha256')))
    return dict(kind='qualify', n=n, steps=steps,
                legacy=dict(rank0=ev_hex(legacy.get('rank0_sha256'), 'legacy_prefill.rank0_sha256'),
                            rank1=ev_hex(legacy.get('rank1_sha256'), 'legacy_prefill.rank1_sha256')),
                initial=ev_state(inc[0], n - 1, 'incremental_exports[t=0]'), generated=generated, step=step,
                incremental=[ev_state(inc[k], n - 1 + k, f'incremental_exports[t={k}]') for k in range(1, steps + 1)],
                fresh=fresh_c)


def evidence_arms(rec, n, steps, labels, kind, backends=None, timed=False):
    """Two-arm records (compare, bench): per arm the planned labels (and backends), steps
    token IDs and logits digests each bound to its step record, and the final state at
    frontier n - 1 + steps; timed arms also need positive timings."""
    rec = ev_dict(rec, kind)
    ev_planned(rec, n, kind)
    ev_int(steps, 'planned steps', 1)
    arms = []
    for i, (x, label) in enumerate(zip(ev_list(rec.get('arms'), f'{kind}.arms', len(labels)), labels)):
        x = ev_dict(x, f'{kind}.arms[{i}]')
        if x.get('arm') != label or (backends and x.get('backend') != backends[i]):
            raise Evidence(f"{kind}.arms[{i}]: arm/backend {x.get('arm')!r}/{x.get('backend')!r} != planned {label!r}")
        toks = ev_tokens(x.get('tokens'), f'{label}.tokens', steps)
        logits = [ev_hex(h, f'{label}.logits_sha256[{j}]') for j, h in enumerate(ev_list(x.get('logits_sha256'), f'{label}.logits_sha256', steps))]
        for j, s in enumerate(ev_list(x.get('steps'), f'{label}.steps', steps)):
            s = ev_dict(s, f'{label}.steps[{j}]')
            if type(s.get('token')) is not int or s['token'] != toks[j] or s.get('logits_sha256') != logits[j]:
                raise Evidence(f'{label}.steps[{j}]: token/logits not bound to tokens[{j}]/logits_sha256[{j}]')
            if type(s.get('position')) is not int or s['position'] != n - 1 + j:
                raise Evidence(f"{label}.steps[{j}].position {s.get('position')!r} != {n - 1 + j}")
        arm = dict(arm=label, tokens=toks, logits=logits, positions=[n - 1 + j for j in range(steps)],
                   final=ev_state(x.get('final_state'), n - 1 + steps, f'{label}.final_state'))
        if timed:
            arm.update(generation_loop_s=ev_pos(x.get('generation_loop_s'), f'{label}.generation_loop_s'),
                       tokens_per_s=ev_pos(x.get('tokens_per_s'), f'{label}.tokens_per_s'))
        arms.append(arm)
    return dict(kind=kind, n=n, steps=steps, arms=arms)


def evidence_compare(rec, n, steps):
    return evidence_arms(rec, n, steps, ('reference', 'tierstream'), 'compare', backends=('reference', 'tierstream'))


def evidence_bench(rec, n, steps, order):
    return evidence_arms(rec, n, steps, tuple(order), 'bench', timed=True)


def evidence_ttft(rec, n, expected_tile, allocated):
    """TTFT on the cached path: prefill state at frontier n-1, the first token, positive
    timings, the startup allocation (a positive multiple of 32) and the reported tile
    equal to the effective requested tile."""
    rec = ev_dict(rec, 'ttft')
    ev_planned(rec, n, 'ttft')
    if rec.get('backend') != 'tierstream':
        raise Evidence('ttft: not measured on the cached tierstream backend')
    alloc = ev_int(allocated, 'startup allocation (row tile)', 32)
    expected = ev_int(expected_tile, 'expected row tile', 32)
    if alloc % 32 or expected % 32 or expected > alloc:
        raise Evidence(f'ttft: expected tile {expected} / allocation {alloc} not multiples of 32 within the allocation')
    tile = ev_int(rec.get('row_tile'), 'ttft.row_tile', 32)
    if tile != expected:
        raise Evidence(f'ttft.row_tile {tile} != effective requested tile {expected}')
    if type(rec.get('allocated_row_tile')) is not int or rec['allocated_row_tile'] != alloc:
        raise Evidence(f"ttft.allocated_row_tile {rec['allocated_row_tile']!r} != startup allocation {alloc}")
    return dict(kind='ttft', n=n, row_tile=tile, allocated=alloc, token=ev_int(rec.get('token'), 'ttft.token', 0, VOCAB),
                ttft_s=ev_pos(rec.get('ttft_s'), 'ttft.ttft_s'), prefill_s=ev_pos(rec.get('prefill_s'), 'ttft.prefill_s'),
                state=ev_state(rec.get('prefill_state'), n - 1, 'ttft.prefill_state'))


def evidence_or_reason(build):
    """(evidence, None) or (None, reason); never raises."""
    try:
        return build(), None
    except Evidence as exc:
        return None, str(exc)
    except Exception as exc:  # noqa: BLE001 - malformed shapes are evidence defects too
        return None, f'malformed evidence: {type(exc).__name__}: {exc}'


def same_state(a, b):
    return [what for what, ok in (('per-rank state SHA/bytes', (a['rank0'], a['rank1'], a['bytes0'], a['bytes1']) ==
                                   (b['rank0'], b['rank1'], b['bytes0'], b['bytes1'])),
                                  ('component extents/order', a['components'] == b['components']),
                                  ('KDA replay checkpoint', a['replay'] == b['replay'])) if not ok]


def judge_qualify(rec, steps, n):
    """Incremental vs fresh prefill on canonical evidence only. Missing evidence fails."""
    e, why = evidence_or_reason(lambda: evidence_qualify(rec, n, steps))
    if e is None:
        return 'NO-REFERENCE', [why]
    problems = []
    if (e['legacy']['rank0'], e['legacy']['rank1']) != (e['initial']['rank0'], e['initial']['rank1']):
        problems.append('session prefill state != prefill producer state')
    for k in range(1, steps + 1):
        a, f = e['incremental'][k - 1], e['fresh'][k - 1]
        problems += [f't={k}: {w} differs' for w in same_state(a, f['state'])]
        if f['next_logits'] != e['step'][k]['logits']:
            problems.append(f't={k}: logits digest differs')
        if f['next_token'] != e['generated'][k]:
            problems.append(f't={k}: greedy token differs')
    return ('MISMATCH', problems) if problems else ('EXACT', [])


def judge_arms(e, label):
    a, b = e['arms']
    problems = [f'{label}: {w}' for w, ok in (('tokens', a['tokens'] == b['tokens']), ('logits digests', a['logits'] == b['logits']))
                if not ok]
    return problems + [f'{label}: final state ({w})' for w in same_state(a['final'], b['final'])]


def judge_compare(rec, steps, n):
    """Reference tier vs tierstream on the identical continuation (canonical evidence)."""
    e, why = evidence_or_reason(lambda: evidence_compare(rec, n, steps))
    if e is None:
        return 'NO-REFERENCE', [why]
    problems = judge_arms(e, 'backends differ')
    return ('MISMATCH', problems) if problems else ('EXACT', [])


def judge_bench(rec, order, steps, n):
    """(exactness, problems, missing coverage). Cold vs warm on canonical evidence; timings
    count only for EXACT arms."""
    e, why = evidence_or_reason(lambda: evidence_bench(rec, n, steps, order))
    if e is None:
        return 'NO-REFERENCE', [why], []
    problems = judge_arms(e, 'arms differ')
    try:  # coverage from the cache counters (not exactness evidence)
        missing = [f'{x["arm"]}: no file-tier expert reads' for x in rec['arms'] if file_bytes(x['steps']) <= 0]
    except Exception as exc:  # noqa: BLE001
        missing = [f'file-tier coverage evidence malformed: {exc}']
    return ('MISMATCH' if problems else 'EXACT'), problems, missing


def judge_ttft(rec, n, expected_tile, allocated):
    """A TTFT record counts only with complete canonical evidence (evidence_ttft)."""
    e, why = evidence_or_reason(lambda: evidence_ttft(rec, n, expected_tile, allocated))
    return ('EXACT', []) if e is not None else ('NO-REFERENCE', [why])


def evidence_key(r):
    """Comparison key of a judged result's stored canonical evidence (raises if absent)."""
    e = ev_dict(r.get('evidence'), 'stored evidence')
    if e.get('kind') == 'ttft':
        return json.dumps([e['state'], e['token']], sort_keys=True)
    if e.get('kind') == 'bench':
        return json.dumps([[x['arm'], x['tokens'], x['logits'], x['final']] for x in e['arms']], sort_keys=True)
    raise Evidence(f"stored evidence kind {e.get('kind')!r}")


def tile_compare(results):
    """Per TTFT length with two or more tile arms (one service): canonical prefill state and
    first token bitwise equal across tiles? Informational; never raises."""
    out = []
    try:
        lengths = sorted({r['tokens'] for r in results if isinstance(r, dict) and r.get('op') == 'ttft' and not r.get('sweep_tile')})
    except Exception as exc:  # noqa: BLE001
        return [dict(verdict='NO-REFERENCE', reason=f'results: {exc}', internal_error=True)]
    for n in lengths:
        try:
            arms = [r for r in results if isinstance(r, dict) and r.get('op') == 'ttft' and r.get('tokens') == n
                    and not r.get('sweep_tile') and r.get('verdict') == 'EXACT']
            tiles = [ev_dict(r.get('evidence'), 'stored evidence')['row_tile'] for r in arms]
            if len(set(tiles)) < 2:
                continue
            keys = {evidence_key(r) for r in arms}
            timing = [dict(row_tile=r['evidence']['row_tile'], prefill_s=r['evidence']['prefill_s'], ttft_s=r['evidence']['ttft_s'])
                      for r in arms]
        except Exception as exc:  # noqa: BLE001
            out.append(dict(tokens=n, verdict='NO-REFERENCE', reason=str(exc), internal_error=True))
            continue
        equal = len(keys) == 1
        out.append(dict(tokens=n, row_tiles=tiles, timing=timing, verdict='BITWISE-EQUAL' if equal else 'DIFFERENT',
                        meaning=('the tile is a pure speed choice' if equal else
                                 'the tile changes reduction order: the larger tile is a FAST-profile lever, not the default')))
    return out


def sweep_compare(results):
    """Across the sweep's tiles (one service start each): canonical TTFT state + first token
    and canonical bench arms bitwise; timings per tile from the evidence. Never raises."""
    out = []
    try:
        items = sorted({(r['op'], r['tokens']) for r in results
                        if isinstance(r, dict) and r.get('op') in ('ttft', 'bench') and r.get('sweep_tile')})
    except Exception as exc:  # noqa: BLE001
        return [dict(verdict='NO-REFERENCE', reason=f'results: {exc}', internal_error=True)]
    for op, n in items:
        try:
            done = [r for r in results if isinstance(r, dict) and r.get('op') == op and r.get('tokens') == n
                    and r.get('sweep_tile') and r.get('verdict') == 'EXACT']
            keys = {r['sweep_tile']: evidence_key(r) for r in done}
            if op == 'ttft':
                timing = {r['sweep_tile']: dict(ttft_s=r['evidence']['ttft_s'], prefill_s=r['evidence']['prefill_s']) for r in done}
            else:
                timing = {r['sweep_tile']: {x['arm']: x['tokens_per_s'] for x in r['evidence']['arms']} for r in done}
        except Exception as exc:  # noqa: BLE001
            out.append(dict(op=op, tokens=n, verdict='NO-REFERENCE', reason=str(exc), internal_error=True))
            continue
        if len(keys) < 2:
            out.append(dict(op=op, tokens=n, tiles=sorted(keys), verdict='NOT-COMPARED', timing=timing))
            continue
        equal = len(set(keys.values())) == 1
        out.append(dict(op=op, tokens=n, tiles=sorted(keys), verdict='BITWISE-EQUAL' if equal else 'DIFFERENT', timing=timing,
                        meaning=('the tile is a pure speed choice' if equal else
                                 'tiles differ bitwise (reduction order): a larger tile is a FAST-profile lever, not the default')))
    return out


def qualify_coverage(rec, n, steps):
    """Missing required coverage of one qualification case (empty list = covered)."""
    missing = []
    try:
        positions = [int(s['position']) for s in rec['incremental_steps']]
        if rec['prompt_tokens'] != n or positions != list(range(n - 1, n + steps)):
            missing.append(f'{n}: consumed positions do not start at the full prompt frontier')
        if not {31, 0, 1} <= {p % 32 for p in positions}:
            missing.append(f'{n}: no B32 boundary crossing (residues 31/0/1)')
        if n == 8192 and not all((p + 1) // 4 > SPARSE_POOLS for p in positions):
            missing.append(f'{n}: sparse DSA selection not exercised')
        if file_bytes(rec['incremental_steps']) <= 0:
            missing.append(f'{n}: no file-tier expert reads during generation')
    except (KeyError, TypeError, ValueError, AttributeError) as exc:
        missing.append(f'{n}: coverage evidence malformed: {exc}')
    return missing


def cache_summary(arm, steps):
    """Every aggregated counter of a timing arm, unchanged (tierstream: lookups,
    misses, device_hits, resurrected, covered, fetched, prefetched, bank_serves,
    ram_copies, file_reads, file_bytes, file_read_ns, h2d_bytes, evicted,
    disk_drops, writebacks, stalled, ...; reference: per-tier selections and
    bytes), normalized names across backends, and a per-token rate for each."""
    totals = dict(arm.get('expert_totals', {}))
    out = dict(totals)
    out.setdefault('device_hits', totals.get('device_selections', 0))
    out.setdefault('ram_copies', totals.get('pinned_selections', 0))
    out.setdefault('file_reads', totals.get('file_selections', 0))
    for key in ('lookups', 'misses', 'resurrected', 'covered'):
        out.setdefault(key, 0)
    out['per_token'] = {k: v / steps for k, v in totals.items()} if steps else {}
    rate = lambda k: out.get(k, 0) / steps if steps else 0
    out.update(misses_per_token=rate('misses'), file_bytes_per_token=rate('file_bytes'), h2d_bytes_per_token=rate('h2d_bytes'))
    return out


def ttft_tile(requested, allocated):
    """(row_tile to send or None, note) for a TTFT tile arm: per-request tiles cannot
    exceed the tile the service allocated at startup."""
    if requested is None:
        return None, f'allocated tile {allocated} (the default rule; per-request tiles cannot exceed it)'
    if allocated and requested > allocated:
        return allocated, f'requested {requested} exceeds the startup allocation {allocated}: ran at {allocated}'
    return requested, f'requested {requested}'


def health_coverage(health, a):
    decode = health.get('residency', {}).get('decode', {})
    if decode.get('expert_tier') != a.expert_tier:
        raise ValueError('service expert tier differs from the requested backend')
    gaps = []
    if a.expert_tier == 'tierstream':
        plan = decode.get('tier_plan', {})
        if not plan.get('three_tier'):
            gaps.append('tierstream plan has no file tier (lower --tier-ram-gib to exercise the disk tier)')
        if not plan.get('reference_compare'):
            gaps.append('no reference backend for the comparison (--reference-compare)')
        if plan.get('unit_manifest_sha256') != a.unit_manifest_sha256:
            raise ValueError('service unit manifest pin differs from the requested pin')
        residency = health.get('residency', {})
        if residency.get('device_layers', 0) or residency.get('pinned_layers', 0):
            gaps.append('prefill keeps resident/pinned expert copies beside the tierstream budget')
        whole = any(caps(a))  # whole-device mode: the arena is what the caps leave (checked by memory_gate)
        for key, gib in (('device_budget_bytes', a.tier_vram_gib), ('host_budget_bytes', a.tier_ram_gib)):
            if not whole and gib is not None and key in plan and plan[key] != int(gib * (1 << 30)):
                gaps.append(f'tierstream {key} {plan.get(key)} is not the requested {gib} GiB')
    return gaps


def caps(a):
    """Whole-device caps in bytes (0 = none)."""
    gib = lambda v: int(v * (1 << 30)) if v is not None else 0
    return gib(getattr(a, 'vram_total_gib', None)), gib(getattr(a, 'ram_total_gib', None))


def memory_gate(rec, a, peaks):
    """Measured peaks of every receipt against --vram-total-gib / --ram-total-gib."""
    vcap, rcap = caps(a)
    m = rec.get('memory')
    if not m:
        return ['receipt has no memory report'] if (vcap or rcap) else []
    # host_peak_commit_bytes is OUR host commit (raw private minus the WDDM backing
    # of our device allocations); the raw peak and the backing are kept beside it.
    for key in ('measured_device_peak_bytes', 'host_peak_commit_bytes',
                'process_peak_commit_bytes_raw', 'wddm_device_backing_bytes',
                'host_peak_sampled_bytes', 'host_peak_os_bound_bytes'):
        peaks[key] = max(peaks.get(key, 0), int(m.get(key, 0)))
    gaps = []
    if vcap and int(m.get('measured_device_peak_bytes', 1 << 62)) > vcap:
        gaps.append(f'measured device peak {m.get("measured_device_peak_bytes")} B exceeds --vram-total-gib {a.vram_total_gib}')
    if rcap and int(m.get('host_peak_commit_bytes', 1 << 62)) > rcap:
        gaps.append(f'host peak commit {m.get("host_peak_commit_bytes")} B exceeds --ram-total-gib {a.ram_total_gib}')
    if m.get('within_caps') is False and not gaps:
        gaps.append('service reports memory outside the whole-device caps')
    return gaps


def summarize(results, extra_gaps=()):
    """Exactness of completed cases and completion of the planned workload, separately."""
    done = [r for r in results if r['verdict'] not in ('INCOMPLETE',)]
    exactness = ('MISMATCH' if any(r['verdict'] == 'MISMATCH' for r in done) else
                 'NO-REFERENCE' if any(r['verdict'] == 'NO-REFERENCE' for r in done) else
                 'EXACT' if done else 'NO-CASES-COMPLETED')
    gaps = [f'{r["prompt"]}: {r.get("reason", "not run")}' for r in results if r['verdict'] == 'INCOMPLETE']
    gaps += [m for r in done for m in r.get('missing', [])] + list(extra_gaps)
    for n in QUALIFY:
        if not any(r['op'] == 'qualify' and r['tokens'] == n and r['verdict'] == 'EXACT' for r in results):
            gaps.append(f'qualification at {n} not EXACT')
    completion = 'COMPLETE' if not gaps else 'INCOMPLETE'
    verdict = (exactness if exactness in ('MISMATCH', 'NO-REFERENCE') else
               'EXACT' if exactness == 'EXACT' and completion == 'COMPLETE' else 'INCOMPLETE')
    return dict(verdict=verdict, exactness_of_completed=exactness, completion=completion, gaps=gaps,
                completed_cases=len(done), planned_cases=len(results))


class Service:
    """Owned service process; stdout lines through a reader thread (deadline-bounded)."""
    def __init__(self, command, log):
        self.proc = start(command, stdin=subprocess.PIPE, stdout=subprocess.PIPE, stderr=log, text=True)
        self.lines = queue.Queue()
        threading.Thread(target=self._pump, daemon=True).start()

    def _pump(self):
        for line in self.proc.stdout:
            self.lines.put(line)
        self.lines.put(None)

    def read(self, deadline):
        while True:
            try:
                line = self.lines.get(timeout=min(5, deadline.left()))
            except queue.Empty:
                continue
            if line is None:
                raise RuntimeError(f'service exited: {self.proc.poll()}')
            if line.startswith('{'):
                return json.loads(line)

    def send(self, command):
        self.proc.stdin.write(json.dumps(command) + '\n')
        self.proc.stdin.flush()


def estimate(observed, op, n, a=None):
    seen = observed.get((op, n))
    return 1.25 * max(seen) if seen else (priors(a) if a is not None else PRIOR_SECONDS)[(op, n)]


def snapshot_check(a, out, rec, steps):
    """Independent CPU receiver: layout validation + payload hash of the export."""
    final = {e['t']: e for e in rec['incremental_exports']}[steps]
    P = final['frontier']
    for rank in (0, 1):
        path = out / 'export-1024' / f'state-r{rank}.bin'
        result = subprocess.run([str(window.executable(a, 'receiver')), '--snapshot', str(path), str(rank), str(P), str(P + 1)],
                                capture_output=True, text=True, timeout=300)
        words = result.stdout.split()
        path.unlink(missing_ok=True)
        if result.returncode or len(words) != 3 or words[0] != 'SNAPSHOT' or words[1] != final[f'rank{rank}_sha256'] \
                or int(words[2]) != final[f'rank{rank}_bytes']:
            raise ValueError(f'receiver snapshot rank {rank} rejected/mismatched: {result.stdout} {result.stderr}')


def execute(a, out, deadline, candidate, pins):
    # Safe finalization defaults first: everything after them runs inside the outer try.
    results, observed, stopped, plan_gaps, peaks, startup_s, starts = [], {}, False, [], {}, None, []
    sweep, segments, done, failure = bool(getattr(a, 'tile_sweep', None)), [], set(), None
    try:
        (out / 'provenance.json').write_text(json.dumps(dict(candidate=candidate, prompts=pins, plan=plan(a)), indent=2, default=str))
        # One segment = one service start. A normal run is one segment; a tile sweep starts
        # the service once per tile (--row-tile T) and writes each tile under tile-T/.
        segments = sweep_segments(a) if sweep else [(None, prompt_plan(a))]
        for k, (tile, items) in enumerate(segments):
            seg = out / f'tile-{tile}' if sweep else out
            seg.mkdir(parents=True, exist_ok=True)
            a_seg = copy.copy(a)
            if sweep:
                a_seg.row_tile = tile
                if k and deadline.left() - CLEANUP_MARGIN < SWEEP_RESTART + SWEEP_STOP + sum(estimate(observed, i[0], i[1], a) for i in items):
                    for j, item in enumerate(items):
                        results.append(dict(op=item[0], tokens=item[1], prompt=item[2].name, sweep_tile=tile, verdict='INCOMPLETE',
                                            reason='remaining time below the estimate for this tile (restart + items)'))
                        done.add((k, j))
                    continue
            with (seg / 'service.log').open('w', encoding='utf-8') as log:
                service = None
                try:
                    service = Service(service_command(a_seg, window.executable(a, 'service')), log)
                    ready = service.read(deadline)
                    if not ready.get('ready') or ready.get('producer_executable_sha256') != candidate['service_sha256'] \
                            or ready['health'].get('decode_identity') != IDENTITY:
                        raise ValueError('service identity/executable/decode capability mismatch')
                    (seg / 'health.json').write_text(json.dumps(ready, indent=2))
                    seg_startup = (ready['health'].get('startup_ms') or 0) / 1000  # measured: replaces the priors next time
                    startup_s = seg_startup if startup_s is None else startup_s
                    allocated = ready['health'].get('row_tile')
                    starts.append(dict(tile=tile, allocated_row_tile=allocated, startup_s=seg_startup, profile=ready.get('profile')))
                    if sweep and allocated != tile:
                        plan_gaps.append(f'tile {tile}: the service allocated row tile {allocated!r}')
                    plan_gaps += [g for g in health_coverage(ready['health'], a_seg) if g not in plan_gaps]
                    for j, item in enumerate(items):
                        op, n, path = item[:3]
                        entry = dict(op=op, tokens=n, prompt=path.name)
                        if sweep:
                            entry['sweep_tile'] = tile
                        if stopped:
                            results.append(dict(entry, verdict='INCOMPLETE', reason='qualification stopped by a failed gate'))
                            done.add((k, j))
                            continue
                        if pins.get(path.name, {}).get('missing') or not path.is_file():
                            results.append(dict(entry, verdict='INCOMPLETE', reason='planned prompt missing'))
                            done.add((k, j))
                            continue
                        if deadline.left() - CLEANUP_MARGIN < estimate(observed, op, n, a):
                            results.append(dict(entry, verdict='INCOMPLETE', reason='remaining time below estimate'))
                            done.add((k, j))
                            continue
                        command = dict(op=op, prompt=str(path), steps=a.bench_steps if op == 'bench' else 1 if op == 'ttft' else a.steps)
                        expected_tile = None
                        if op == 'bench':
                            command['order'] = item[3]
                        if op == 'ttft':
                            tile_req, note = ttft_tile(item[3] if len(item) > 3 else None, allocated if type(allocated) is int else 0)
                            if tile_req:
                                command['row_tile'] = tile_req
                            expected_tile = tile_req or allocated  # the effective requested tile
                            entry.update(row_tile_requested=item[3] if len(item) > 3 and item[3] else 'allocated', row_tile_note=note)
                        if op == 'qualify' and n == 1024:
                            command['export_dir'] = str(seg / 'export-1024')
                        begin = time.monotonic()
                        service.send(command)
                        rec = service.read(deadline)
                        observed.setdefault((op, n), []).append(time.monotonic() - begin)
                        try:  # materialization: a defect here makes THIS case NO-REFERENCE, the run continues
                            plan_gaps += [f'{path.name}: {g}' for g in memory_gate(rec, a, peaks)]
                            (seg / f'{op}-{path.stem}.json').write_text(json.dumps(rec, indent=2))
                            results.append(materialize(a, op, n, item, rec, entry, seg, observed[(op, n)][-1], expected_tile, allocated))
                        except Exception as exc:  # noqa: BLE001
                            results.append(dict(entry, verdict='NO-REFERENCE', problems=[f'materialization: {type(exc).__name__}: {exc}'],
                                                missing=[]))
                        done.add((k, j))
                        if op in ('qualify', 'compare') and results[-1]['verdict'] != 'EXACT':
                            stopped = True  # a failed exactness gate stops qualification
                        (out / 'results.json').write_text(json.dumps(results, indent=2))
                    service.send(dict(op='quit'))
                finally:
                    try:
                        (out / 'results.json').write_text(json.dumps(results, indent=2))
                    finally:  # the service is stopped unconditionally
                        stop(service.proc if service else None, wait_seconds=window.SERVICE_EXIT_SECONDS)
    except Exception as exc:  # noqa: BLE001 - one outer finalization: record, then still summarize
        failure = dict(type=type(exc).__name__, error=str(exc))
    return finalize(a, out, results, plan_gaps, peaks, startup_s, starts, segments, done, failure, sweep)


def materialize(a, op, n, item, rec, entry, seg, wall_s, expected_tile, allocated):
    """The result of one case: its judge's verdict and, for EXACT records, the canonical
    evidence the comparisons use (they never read raw response fields)."""
    if op == 'qualify':
        v, problems = judge_qualify(rec, a.steps, n)
        missing = qualify_coverage(rec, n, a.steps) if v == 'EXACT' else []
        if v == 'EXACT' and n == 1024:
            snapshot_check(a, seg, rec, a.steps)
        timing = dict(wall_s=wall_s) if a.expert_tier == 'tierstream' else dict(timed=False)
        return dict(entry, verdict=v, problems=problems, missing=missing, generated=evidence_qualify(rec, n, a.steps)['generated']
                    if v == 'EXACT' else None, **timing)
    if op == 'ttft':
        v, problems = judge_ttft(rec, n, expected_tile, allocated)
        ev = evidence_ttft(rec, n, expected_tile, allocated) if v == 'EXACT' else None
        return dict(entry, verdict=v, problems=problems, missing=[], evidence=ev, row_tile=ev and ev['row_tile'],
                    ttft_s=ev and ev['ttft_s'], prefill_s=ev and ev['prefill_s'], token=ev and ev['token'], wall_s=wall_s)
    if op == 'compare':
        v, problems = judge_compare(rec, a.steps, n)
        return dict(entry, verdict=v, problems=problems, missing=[], tokens_compared=a.steps, timed=False)  # exactness only
    v, problems, missing = judge_bench(rec, item[3], a.bench_steps, n)
    ev = evidence_bench(rec, n, a.bench_steps, item[3]) if v == 'EXACT' else None
    arms = [dict(arm=x['arm'], tokens_per_s=x['tokens_per_s'], generation_loop_s=x['generation_loop_s'], tokens=x['tokens'],
                 file_bytes=file_bytes(r['steps']), cache=cache_summary(r, a.bench_steps))
            for x, r in zip(ev['arms'], rec['arms'])] if ev else []  # timings only for agreeing arms
    return dict(entry, verdict=v, problems=problems, missing=missing, evidence=ev, arms=arms, wall_s=wall_s)


def finalize(a, out, results, plan_gaps, peaks, startup_s, starts, segments, done, failure, sweep):
    """Always prints a summary. Unfinished cases are INCOMPLETE; a recorded failure is never
    EXACT (exit non-zero)."""
    try:
        if failure is not None:
            for k, (tile, items) in enumerate(segments):
                for j, item in enumerate(items):
                    if (k, j) not in done:
                        r = dict(op=item[0], tokens=item[1], prompt=getattr(item[2], 'name', str(item[2])), verdict='INCOMPLETE',
                                 reason=f"not run: {failure['type']}: {failure['error']}")
                        if sweep:
                            r['sweep_tile'] = tile
                        results.append(r)
            (out / 'results.json').write_text(json.dumps(results, indent=2))
    except Exception as exc:  # noqa: BLE001
        failure = dict(failure or {}, finalize_error=str(exc))
    builder_errors = []  # every summary-stage error is a structured run failure (non-zero exit)
    try:
        summary = summarize(results, plan_gaps)
    except Exception as exc:  # noqa: BLE001 - a summary is always printed
        summary = dict(verdict='NO-REFERENCE', exactness_of_completed='NO-REFERENCE', completion='UNKNOWN', gaps=list(plan_gaps),
                       summary_error=f'summarize: {exc}')
        builder_errors.append(f'summarize: {type(exc).__name__}: {exc}')
    builders = [('tile_compare', lambda: tile_compare(results))]
    if sweep:
        builders.append(('tile_sweep', lambda: dict(tiles=list(a.tile_sweep), note=getattr(a, 'sweep_note', ''), starts=starts,
                                                    comparison=sweep_compare(results))))
    for key, build in builders:
        try:
            summary[key] = build()
            # A comparison only considers EXACT results: an item it could not compare is an
            # internal inconsistency, recorded as a run failure as well.
            entries = summary[key].get('comparison', []) if isinstance(summary[key], dict) else summary[key]
            builder_errors += [f"{key}: {x.get('op', 'ttft')} {x.get('tokens')}: {x.get('reason')}" for x in entries
                               if isinstance(x, dict) and x.get('internal_error')]
        except Exception as exc:  # noqa: BLE001 - recorded as a run failure below (non-zero exit)
            summary[key] = dict(verdict='NO-REFERENCE', reason=str(exc))
            builder_errors.append(f'{key}: {type(exc).__name__}: {exc}')
    try:
        vcap, rcap = caps(a)
        memory = dict(peaks, vram_total_cap_bytes=vcap, ram_total_cap_bytes=rcap)
        expected = expected_seconds(a)
    except Exception as exc:  # noqa: BLE001
        memory, expected = dict(peaks, error=str(exc)), None
        builder_errors.append(f'metadata: {type(exc).__name__}: {exc}')
    if builder_errors:
        failure = dict(failure or dict(type='SummaryError', error='summary builders failed'), summary_builders=builder_errors)
    if starts:
        summary['profile'] = starts[0].get('profile')  # the service's --profile resolution (its ready event)
    if failure is not None:
        summary['failure'] = failure
        if summary.get('verdict') == 'EXACT':
            summary['verdict'] = 'INCOMPLETE'  # a failed run is never EXACT (exit 4)
    print(json.dumps(dict(summary, identity=IDENTITY, output=str(out), results=results, memory=memory, startup_s=startup_s,
                          starts=starts, expected_wall_seconds=expected, qualification='DRAFT; decode self-consistency only'),
                     default=str))
    return 0 if summary.get('verdict') == 'EXACT' else (4 if summary.get('verdict') == 'INCOMPLETE' else 3)


def worker_options(a):
    """Every run option the worker re-parses: caps, tiers, placement, the startup tile,
    the fitted tile sweep (and the note of an arm the parent dropped)."""
    argv = []
    for name in ('capacity', 'vram_total_gib', 'ram_total_gib', 'tier_vram_gib', 'tier_ram_gib', 'tier_spare_percent',
                 'tier_policy', 'tier_predict', 'tier_flush_ready', 'skip_empty_expert_groups', 'persistent_dense_fp16', 'expert_residency', 'expert_routes', 'routes_out', 'row_tile', 'profile'):
        if getattr(a, name, None) is not None:
            argv += ['--' + name.replace('_', '-'), str(getattr(a, name))]
    if getattr(a, 'full_card', False):
        argv += ['--full-card']
    if getattr(a, 'allow_pending', False):
        argv += ['--allow-pending']
    if getattr(a, 'decode_profile', False):
        argv += ['--decode-profile']
    if getattr(a, 'tile_sweep', None):
        argv += ['--tile-sweep', ','.join(str(x) for x in a.tile_sweep)]
        if getattr(a, 'sweep_note', ''):
            argv += ['--sweep-note', a.sweep_note]
    return argv


def run_worker(a, out, end):
    deadline = Deadline(a.timeout, end=end)
    pins = check_inputs(a)
    with PinnedExecutables(a.build, out) as pinned:
        a.pinned_build = pinned.root
        if not a.allow_gpu:
            print(json.dumps(dict(verdict='INPUTS-VALID', candidate=pinned.digests, prompts=pins)))
            return 0
        return execute(a, out, deadline, pinned.digests, pins)


def run(a):
    if not 1 <= a.timeout <= MAX_SECONDS:
        raise ValueError('aggregate deadline must be 1..1800 seconds')
    if (a.model is None) == (a.cache is None):
        raise ValueError('exactly one of --model or --cache')
    if getattr(a, 'plan_budget', False):
        return plan_budget(a)
    if getattr(a, 'print_effective_config', False):
        return print_effective_config(a)
    if not a.allow_gpu and not a.check_inputs:
        print(json.dumps(plan(a), indent=2, default=str))
        return 0
    if a.allow_gpu and not re.fullmatch(r'GPU-[0-9a-fA-F]{8}(?:-[0-9a-fA-F]{4}){3}-[0-9a-fA-F]{12}',
                                       os.environ.get('CUDA_VISIBLE_DEVICES', '')):
        raise ValueError('set CUDA_VISIBLE_DEVICES to one full authorized PRO UUID')
    if os.name != 'nt':
        raise RuntimeError('hard-deadline qualification requires Windows job objects')
    if expected_seconds(a) > a.timeout:
        raise ValueError(f'the planned schedule needs ~{expected_seconds(a)} s (startup prior included) > --timeout {a.timeout} s')
    deadline = Deadline(a.timeout)
    out = a.build / ('decode-qualification-' + time.strftime('%Y%m%d-%H%M%S') + '-' + os.urandom(6).hex())
    out.mkdir()
    argv = ['--build', str(a.build), '--prompt-dir', str(a.prompt_dir), '--steps', str(a.steps),
            '--bench-steps', str(a.bench_steps), '--timeout', str(a.timeout), '--expert-tier', a.expert_tier]
    argv += ['--model', str(a.model)] if a.model else ['--cache', str(a.cache), '--cache-manifest-sha256', a.cache_manifest_sha256]
    if a.unit_file:
        argv += ['--unit-file', str(a.unit_file)]
    if a.unit_manifest_sha256:
        argv += ['--unit-manifest-sha256', a.unit_manifest_sha256]
    argv += worker_options(a)
    argv += ['--allow-gpu' if a.allow_gpu else '--check-inputs']
    code = ('import runpy,sys; from pathlib import Path; m=runpy.run_path(sys.argv[1]); '
            'sys.exit(m["run_worker"](m["arguments"](sys.argv[4:]),Path(sys.argv[2]),float(sys.argv[3])))')
    try:
        with window.windows_job()() as job:
            process, _ = job.start([sys.executable, '-B', '-c', code, str(Path(__file__).resolve()),
                                    str(out.resolve()), str(deadline.end), *argv])
            result = job.wait(process, deadline.left())
            deadline.left()
            return result
    except TimeoutError:
        print('INCOMPLETE: absolute deadline; complete owned job terminated; no qualification pass')
        return 4


def arguments(argv=None):
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--build', type=Path, required=True)
    p.add_argument('--model', type=Path, help='direct shards including root lm_head')
    p.add_argument('--cache', type=Path, help='decode-capable cache (cache.py --with-head)')
    p.add_argument('--cache-manifest-sha256')
    p.add_argument('--prompt-dir', type=Path, required=True,
                   help='private: q1024.u32, q8192.u32 and chat-<kind>-<1024|8192>.u32 (exact lengths)')
    p.add_argument('--steps', type=int, default=4)
    p.add_argument('--bench-steps', type=int, default=4)
    p.add_argument('--expert-tier', default='tierstream', choices=['tierstream'],
                   help='tierstream: the product cache (needs --unit-file); the reference backend runs only inside its untimed q1024 compare')
    p.add_argument('--unit-file', type=Path, help='GLM unit file (glm_unit_file copy output) for the disk tier')
    p.add_argument('--unit-manifest-sha256', help='pinned digest of PRIMARY.units.json (printed by glm_unit_file copy)')
    p.add_argument('--vram-total-gib', type=float,
                   help='WHOLE-device cap (as if the card had N GiB): context + trunk + KV + workspaces + rings + expert arena; default none')
    p.add_argument('--ram-total-gib', type=float, help='WHOLE host cap incl. pinned (process commit); default none')
    p.add_argument('--tier-vram-gib', type=float, help='arena-only cap of the tierstream VRAM arena (default: all that remains)')
    p.add_argument('--tier-ram-gib', type=float, help='arena-only cap of the pinned RAM tier (default 32 for tierstream without --ram-total-gib)')
    # Placement / retention / prefetch (service defaults when omitted: 5% spares,
    # SCORED_AUTO, predict off, residency file <unit-file>.residency, no census).
    p.add_argument('--tier-spare-percent', type=float, help='arena spares as a percent of the arena (default 5)')
    p.add_argument('--tier-policy', choices=['lru', 'scored', 'auto'], help='boundary victims (default auto = SCORED_AUTO)')
    p.add_argument('--tier-predict', choices=['0', '1'], help='overlapped next-layer prefetch (default 0)')
    p.add_argument('--tier-flush-ready', choices=['0', '1'], help='flush the cache demand stream after each ready signal (WDDM submission; default 1 = profile row flush_ready)')
    p.add_argument('--expert-residency', help='residency file (default <unit-file>.residency; "none" = off)')
    p.add_argument('--expert-routes', help='routes census file(s), comma-separated (glm_route_census.py output)')
    p.add_argument('--routes-out', help='record the routes of every decoded token (DS routes format) to this file; receipts carry them')
    p.add_argument('--row-tile', type=int,
                   help='startup prefill tile allocation (multiple of 32, <= min(capacity, 32768)); default: the 5%% rule '
                        '(largest tile <= 8192 whose growth over 2048 costs <= 5%% of the expert arena). E.g. 8192 measures '
                        'the whole 8K prompt in one piece and the 32K arm at 8192 vs 2048, at the arena cost the plan reports')
    p.add_argument('--tile-sweep', help='full card: comma-separated startup row tiles (e.g. 2048,4096,8192), one service '
                   'start each; the exactness gate on the first, TTFT 8K/32K and the decode bench on every tile; the 4096 arm '
                   'is dropped first if the whole sweep does not fit --timeout')
    p.add_argument('--profile', choices=['default', 'fast', 'quality'],
                   help='the service profile (one switch for every lever; default = today; per-lever options override)')
    p.add_argument('--skip-empty-expert-groups', choices=['0', '1'], help='routed MoE: launch only expert groups holding routed rows (exact; pending row)')
    p.add_argument('--persistent-dense-fp16', choices=['0', '1'], help='shared experts / dense FFNs decoded once at startup (exact; pending row; ~2.8 GiB)')
    p.add_argument('--decode-profile', action='store_true',
                   help='decode timeline: CUDA events per phase per layer in every decode step record and per-arm totals '
                        '(outputs unchanged; timings include the events)')
    p.add_argument('--allow-pending', action='store_true', help='run pending profile rows (benchmarks only: receipts unqualified)')
    p.add_argument('--print-effective-config', action='store_true',
                   help='print the service profile resolution (every lever row) and exit; no model read, no CUDA call')
    p.add_argument('--sweep-note', default='', help=argparse.SUPPRESS)  # parent -> worker: the arm the parent dropped
    p.add_argument('--timeout', type=int, default=MAX_SECONDS)
    mode = p.add_mutually_exclusive_group()
    mode.add_argument('--allow-gpu', action='store_true')
    mode.add_argument('--check-inputs', action='store_true', help='CPU-only: validate prompts and pin executables')
    p.add_argument('--capacity', type=int,
                   help=f'service capacity (default {CAPACITY}; {WHOLE_CARD_CAPACITY} with --vram-total-gib below the full card)')
    p.add_argument('--full-card', action='store_true',
                   help=f'whole RTX PRO: --vram-total-gib {FULL_CARD_VRAM_GIB:g} and --ram-total-gib = host RAM - '
                        f'{FULL_CARD_HOST_MARGIN_GIB} GiB unless given; TTFT 1K/8K/32K + decode cold/warm, cached path only')
    mode.add_argument('--plan-budget', action='store_true',
                      help='dry: print the planned VRAM/host split (shard headers only; no hashing, no CUDA calls)')
    p.add_argument('--device-total-gib', type=float, help='--plan-budget: device total (default: idle PRO at the cap)')
    p.add_argument('--device-free-gib', type=float, help='--plan-budget: device free (default: the total)')
    a = p.parse_args(argv)
    if a.full_card:
        if a.expert_tier != 'tierstream':
            p.error('--full-card measures the cached tierstream path only')
        if a.vram_total_gib is None:
            a.vram_total_gib = FULL_CARD_VRAM_GIB
        if a.ram_total_gib is None:
            a.ram_total_gib = float(int(host_total_gib()) - FULL_CARD_HOST_MARGIN_GIB)
    if a.capacity is None:
        a.capacity = WHOLE_CARD_CAPACITY if (a.vram_total_gib is not None and not a.full_card) else CAPACITY
    if a.capacity % 32 or not 8320 <= a.capacity <= 131072 or (a.full_card and a.capacity < 32768):
        p.error('--capacity: a multiple of 32 in 8320..131072 (>= 32768 with --full-card for the 32K TTFT)')
    if a.row_tile is not None and (a.row_tile % 32 or not 32 <= a.row_tile <= min(a.capacity, 32768)):
        p.error('--row-tile: a multiple of 32 in 32..min(capacity, 32768)')
    if a.tile_sweep is not None:
        try:
            tiles = [int(x) for x in a.tile_sweep.split(',') if x.strip()]
        except ValueError:
            p.error('--tile-sweep: comma-separated integers')
        if not a.full_card or a.row_tile is not None:
            p.error('--tile-sweep needs --full-card and replaces --row-tile')
        if not tiles or len(set(tiles)) != len(tiles) or any(x % 32 or not 32 <= x <= min(a.capacity, 32768) for x in tiles):
            p.error('--tile-sweep: distinct multiples of 32 in 32..min(capacity, 32768)')
        try:
            given = a.sweep_note
            a.tile_sweep, a.sweep_note = fit_sweep(a, tiles)
            a.sweep_note = given or a.sweep_note
        except ValueError as e:
            p.error(str(e))
    if a.expert_tier == 'tierstream' and a.tier_ram_gib is None and a.ram_total_gib is None:
        a.tier_ram_gib = 32.0  # arena-only default; with --ram-total-gib the pinned tier is what the cap leaves
    if (a.vram_total_gib is not None or a.ram_total_gib is not None) and a.expert_tier != 'tierstream':
        p.error('--vram-total-gib/--ram-total-gib apply to --expert-tier tierstream')
    if not 3 <= a.steps <= 16 or not 1 <= a.bench_steps <= 64:
        p.error('--steps 3..16 (B32 boundary coverage), --bench-steps 1..64')
    return a


if __name__ == '__main__':
    raise SystemExit(run(arguments()))
