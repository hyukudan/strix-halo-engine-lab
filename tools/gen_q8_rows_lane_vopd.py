#!/usr/bin/env python3
# Generates src/hip_q8_rows_lane_vopd.inc: two steps of the lane-output q8_rows prompt kernel
# (project_wide_rows_lane_kernel in hip_q8_rows_gemv.hpp) as hand-paired VOPD.
#
# Lane L of a wave owns outputs o0 = L and o1 = 32 + L of a 64-output group and 4 rows r. For the leaf
# subset of a step it runs 16 chains per (output, row), chain m being leaf lambda = j + 16 m; a step
# is one 256-feature chunk k of that subset, in two halves of chains m = 8 h .. 8 h + 7. Every chain
# runs acc = fma(x(r, f), c(o, f), acc) for f = lambda + 256 k in ascending k (v_dual_fmac_f32, the
# single-rounding FMA of the sites; chunk 0 uses v_dual_fmaak_f32 with K = +0, i.e. fma(x, c, +0), the
# first FMA from the zero accumulator). x(r, f) is the same in every lane, so it is an SGPR; a VOPD pair
# is the two outputs of one (row, chain), both reading that SGPR:
#   X: v_dual_fmac_f32 acc(o0, r, m), s_x(r, m), c(o0, m)
#   Y: v_dual_fmac_f32 acc(o1, r, m), s_x(r, m), c(o1, m)
# The pair's four VGPRs sit in four banks (register % 4): c(o0, mm) in bank mm % 4, c(o1, mm) in bank
# (mm + 1) % 4 (half h's coefficient quads load to v64 / v73 or v84 / v93), acc(o0, r, m) in bank
# (mm + 2) % 4 and acc(o1, r, m) in (mm + 3) % 4 (mm = m % 8), which also gives the destinations opposite
# parities. The 128 accumulators fill v128..v255.
#
# The block loads its own operands (per half: two s_load_b512 of its 32 x values into s40.. / s72..,
# four ds_load_b128 of its 16 coefficients into the set of that half's parity), each half's loads issued
# before the previous half's FMAs. Only the accumulators cross the asm boundary (scalar values handed
# between asm statements are kept in VGPRs by the compiler). The first step of a subset (chunk 0) is
# the first step of a pair (the chunk count is even).
# Usage: python tools/gen_q8_rows_lane_vopd.py > src/hip_q8_rows_lane_vopd.inc
ACC = 128
C0 = (64, 84)       # c(o0, 0..7) of half 0 / 1
C1 = (73, 93)       # c(o1, 0..7)
XS = (40, 72)       # x(r, mm) = s(XS + 8 r + mm) of half 0 / 1

acc_reg = {}
slot = {b: 0 for b in range(4)}
for o in range(2):
    for r in range(4):
        for m in range(16):
            b = (m % 8 + 2 + o) % 4
            acc_reg[(o, r, m)] = ACC + 4 * slot[b] + b
            slot[b] += 1
assert all(v == 32 for v in slot.values()) and sorted(acc_reg.values()) == list(range(ACC, ACC + 128))

def index(o, r, m): return (o * 4 + r) * 16 + m

def loads(n):
    # Half n = 2 t + h (step t of the pair, half h) into register set n % 2: x at xp + 256 t + 128 h,
    # coefficients at LDS byte 4096 t + 2048 h + 1024 quad + 512 (o / 32).
    t, h, b = n // 2, n % 2, n % 2
    xo, co = 256 * t + 128 * h, 4096 * t + 2048 * h
    return [f"s_load_b512 s[{XS[b]}:{XS[b] + 15}], %[xp], {hex(xo)}",
            f"s_load_b512 s[{XS[b] + 16}:{XS[b] + 31}], %[xp], {hex(xo + 64)}",
            f"ds_load_b128 v[{C0[b]}:{C0[b] + 3}], %[lds] offset:{co}",
            f"ds_load_b128 v[{C0[b] + 4}:{C0[b] + 7}], %[lds] offset:{co + 1024}",
            f"ds_load_b128 v[{C1[b]}:{C1[b] + 3}], %[lds] offset:{co + 512}",
            f"ds_load_b128 v[{C1[b] + 4}:{C1[b] + 7}], %[lds] offset:{co + 1536}"]

def fmas(n, first):
    h, b = n % 2, n % 2
    lines = []
    for r in range(4):
        for mm in range(8):
            m = 8 * h + mm
            a0, a1 = acc_reg[(0, r, m)], acc_reg[(1, r, m)]
            c0, c1 = C0[b] + mm, C1[b] + mm
            assert a0 % 2 != a1 % 2 and len({a0 % 4, a1 % 4, c0 % 4, c1 % 4}) == 4
            x = XS[b] + 8 * r + mm
            if first:
                lines.append(f"v_dual_fmaak_f32 v{a0}, s{x}, v{c0}, 0x0 :: v_dual_fmaak_f32 v{a1}, s{x}, v{c1}, 0x0")
            else:
                lines.append(f"v_dual_fmac_f32 v{a0}, s{x}, v{c0} :: v_dual_fmac_f32 v{a1}, s{x}, v{c1}")
    return lines

def emit(first, prefetch, ablate=''):
    # Two steps (four halves): half n + 1's loads are in flight during half n's FMAs.
    def keep(line):
        return not (("x" in ablate and line.startswith("s_load")) or ("c" in ablate and line.startswith("ds_load")))
    lines = loads(0)
    for n in range(4):
        lines.append("s_waitcnt lgkmcnt(0)")
        if n + 1 < 4: lines += loads(n + 1)
        elif prefetch:
            # Scalar-cache prefetch of the next pair's x (eight 64-byte lines) into the null SGPR: nothing is
            # written after the block ends (the compiler may reuse any SGPR then); the next block's first
            # wait covers the requests.
            lines += [f"s_load_b32 null, %[xn], {hex(64 * i)}" for i in range(8)]
        lines += fmas(n, first and n < 2)
    lines = [l for l in lines if keep(l)]
    mode = "=" if first else "+"
    acc_operands = ", ".join(f'"{mode}{{v{acc_reg[(o, r, m)]}}}"(acc[{index(o, r, m)}])'
                             for o in range(2) for r in range(4) for m in range(16))
    clobbers = [f'"s{XS[h] + i}"' for h in range(2) for i in range(32)]
    clobbers += [f'"v{C0[h] + i}"' for h in range(2) for i in range(8)] + [f'"v{C1[h] + i}"' for h in range(2) for i in range(8)]
    name = ("lane_steps_first" if first else "lane_steps") + ("_prefetch" if prefetch else "") + (f"_no{ablate}" if ablate else "")
    print(f"__device__ __forceinline__ void {name}(float (&acc)[128], const void* xp, const void* xn, unsigned lds) {{")
    print('    asm volatile("' + "\\n\\t".join(lines) + '"')
    print("        : " + acc_operands)
    print('        : [xp] "s"(xp), [xn] "s"(xn), [lds] "v"(lds)')
    print("        : " + ", ".join(clobbers) + ', "memory");')
    print("}")

def emit_tree(upper_first):
    # Subset tree levels lambda bits 7..4 (chains m, m + 8; m, m + 4; m, m + 2; m, m + 1, lower m first in
    # the tree) in place on the pinned accumulators: each pair's value ends in its chain 0 register. A VOPD
    # pair is the same addition for outputs o0 / o1 (banks (mm + 2) % 4 / (mm + 3) % 4: legal, opposite
    # parities). v_dual_add_f32 is v_add_f32; src0 is the lower operand, or the upper one for sites whose
    # tree addition reads the upper value first (Site::tree_upper_src0).
    lines = []
    for half in (8, 4, 2, 1):
        for r in range(4):
            for m in range(half):
                ops = []
                for o in range(2):
                    lo, hi = acc_reg[(o, r, m)], acc_reg[(o, r, m + half)]
                    a, b = (hi, lo) if upper_first else (lo, hi)
                    ops.append(f"v_dual_add_f32 v{lo}, v{a}, v{b}")
                x, y = ops
                assert acc_reg[(0, r, m)] % 2 != acc_reg[(1, r, m)] % 2
                lines.append(f"{x} :: {y}")
    acc_operands = ", ".join(f'"+{{v{acc_reg[(o, r, m)]}}}"(acc[{index(o, r, m)}])' for o in range(2) for r in range(4) for m in range(16))
    name = "lane_tree_upper_first" if upper_first else "lane_tree_lower_first"
    print(f"__device__ __forceinline__ void {name}(float (&acc)[128]) {{")
    print('    asm volatile("' + "\\n\\t".join(lines) + '"')
    print("        : " + acc_operands + ");")
    print("}")

print("// Generated by tools/gen_q8_rows_lane_vopd.py; do not edit.")
for prefetch in (False, True):
    emit(True, prefetch)
    emit(False, prefetch)
# Timing ablations (STRIX_Q8_ROWS_LANE_EXP 4 / 8: no scalar / no LDS operand loads; wrong results).
emit(False, False, "x")
emit(False, False, "c")
emit_tree(False)
emit_tree(True)
