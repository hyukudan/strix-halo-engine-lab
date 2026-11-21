// DS41 exact prefill variants share the grouped integer decoder traits.

// ---- Split kernels v2 (STRIX_DS41_PREFILL_MOE_V2, IQ2_XXS gate/up and Q2_K down) ----
//
// The same declared products as split_gate_up_kernel / split_down_kernel (forms 6/7), bit for bit:
// - Class map: class-wave q owns the lane classes q + 8k (k = 0..3), i.e. group q + 8k of every chunk, so the class bits
//   held in a wave are the tree's FIRST two levels (bits 4 and 3): each wave folds its four class sums in registers
//   ((q + q+16) + (q+8 + q+24)) and only one value per (row, output) meets the other waves in levels 3-5 (class bits 2,
//   1, 0 = wave bits), through LDS: a quarter of the exchange of split_tree, one barrier pair per level for both tiles.
//   It also balances a partial last chunk (down width 2304 = 72 groups: every wave runs 9 groups, not 12 / 8).
// - Two LDS-only barriers per chunk: the stage's global reads go out before the next weight tile's (the in-order load
//   counter never makes the stage wait for the prefetch).
// - V2_DECODE (IQ2_XXS): the sign of each grid byte is applied by one v_perm_b32 that picks the byte from the grid
//   entry or from its negation (both in the LDS grid table): the integers of apply_signs(grid, byte_masks(signs)).
// - V2_FMA (Q2_K): the minimum term's integer B = m0 * S0 + m1 * S1 (|B| <= 60960) as fma(m1, S1, fma(m0, S0, +0)) on
//   exact float operands: every intermediate is an exact integer, and +0 for B = 0 (the int-to-float of B).
// - V2_PART (Q2_K): a partial last chunk loads only the 256-weight blocks it has (Q2_K down: 1 of 4 in chunk 2).
constexpr unsigned v2_on = 1U, v2_decode = 2U, v2_fma = 4U, v2_part = 8U;
struct alignas(16) Grid2 { unsigned pos_lo, pos_hi, neg_lo, neg_hi; };
__device__ __forceinline__ void load_grid2(Grid2* grid) {
    for (unsigned t = threadIdx.x; t < 256U; t += blockDim.x) {
        const unsigned long long g = prefill_iq2_grid[t];
        const unsigned lo = static_cast<unsigned>(g), hi = static_cast<unsigned>(g >> 32U);
        // apply_signs(m, 0xffffffff) = (m ^ 0xffffffff) + 0x01010101: the all-negative word (grid bytes are 8, 25 or 43:
        // no byte carries, so each byte is the byte of apply_signs under any mask)
        grid[t] = Grid2{lo, hi, static_cast<unsigned>(apply_signs(lo, 0xffffffffU)), static_cast<unsigned>(apply_signs(hi, 0xffffffffU))};
    }
}
// v_perm_b32 selector: byte j = j (positive byte j) or 4 + j (negated byte j) for bit j of bits4.
__device__ __forceinline__ unsigned sign_select(unsigned bits4) { return ((bits4 * 0x00810204U) & 0x04040404U) | 0x03020100U; }
__device__ __forceinline__ Operands decode_v2(const Raw<WeightFormat::iq2_xxs>& raw, const Grid2* grid) {
    Operands out{};
    const float d = half_bits(raw.d);
    const unsigned aux0 = static_cast<unsigned>(raw.aux), aux1 = static_cast<unsigned>(raw.aux >> 32U);
    out.scale = d * (0.5F + static_cast<float>(aux1 >> 28U)) * 0.25F;
    int w[8];
#pragma unroll
    for (unsigned l = 0; l < 4U; ++l) {
        const Grid2 g = grid[(aux0 >> (8U * l)) & 0xffU];
        const unsigned signs = sign_byte(aux1 >> (7U * l));
        w[2U * l] = static_cast<int>(__builtin_amdgcn_perm(g.neg_lo, g.pos_lo, sign_select(signs & 15U)));
        w[2U * l + 1U] = static_cast<int>(__builtin_amdgcn_perm(g.neg_hi, g.pos_hi, sign_select(signs >> 4U)));
    }
    out.b0 = v4i{w[0], w[1], w[2], w[3]};
    out.b1 = v4i{w[4], w[5], w[6], w[7]};
    return out;
}

// V2_SIGNS (IQ2_XXS): the byte masks of both words of a grid lookup from a 128-entry LDS table indexed by the 7-bit sign
// index (entry i = byte_masks(sign_byte(i) & 15), byte_masks(sign_byte(i) >> 4)), instead of the parity, spread and
// v_perm per word: apply_signs(grid, mask) = (grid ^ mask) + (mask & 0x01010101) on the same masks, the same integers.
constexpr unsigned v2_signs = 32U, v2_signs16 = 64U; // v2_signs16: the 16-byte entries of form 2 (masks and their low bits)
__device__ __forceinline__ void load_signs2(uint2* signs) {
    for (unsigned t = threadIdx.x; t < 128U; t += blockDim.x) {
        const unsigned b = sign_byte(t);
        signs[t] = uint2{byte_masks(b & 15U), byte_masks(b >> 4U)};
    }
}
__device__ __forceinline__ Operands decode_signs2(const Raw<WeightFormat::iq2_xxs>& raw, const unsigned long long* grid, const uint2* signs) {
    Operands out{};
    const float d = half_bits(raw.d);
    const unsigned aux0 = static_cast<unsigned>(raw.aux), aux1 = static_cast<unsigned>(raw.aux >> 32U);
    out.scale = d * (0.5F + static_cast<float>(aux1 >> 28U)) * 0.25F;
    int w[8];
#pragma unroll
    for (unsigned l = 0; l < 4U; ++l) {
        const unsigned long long g = grid[(aux0 >> (8U * l)) & 0xffU];
        const uint2 m = signs[(aux1 >> (7U * l)) & 127U];
        w[2U * l] = apply_signs(static_cast<unsigned>(g), m.x);
        w[2U * l + 1U] = apply_signs(static_cast<unsigned>(g >> 32U), m.y);
    }
    out.b0 = v4i{w[0], w[1], w[2], w[3]};
    out.b1 = v4i{w[4], w[5], w[6], w[7]};
    return out;
}

// V2_PIPE (IQ2_XXS gate/up): the decode in two steps, the LDS reads of group k + 1 (its raw bytes and four grid entries)
// issued before the products of group k, the integer steps after: decode_lean's operations on the same values.
constexpr unsigned v2_pipe = 128U;
struct Iq2Pre { unsigned short d; unsigned aux1; unsigned long long g[4]; };
__device__ __forceinline__ Iq2Pre iq2_prefetch(const Raw<WeightFormat::iq2_xxs>& raw, const unsigned long long* grid) {
    Iq2Pre p;
    p.d = raw.d;
    const unsigned aux0 = static_cast<unsigned>(raw.aux);
    p.aux1 = static_cast<unsigned>(raw.aux >> 32U);
#pragma unroll
    for (unsigned l = 0; l < 4U; ++l) p.g[l] = grid[(aux0 >> (8U * l)) & 0xffU];
    return p;
}
__device__ __forceinline__ Operands iq2_finish(const Iq2Pre& p) {
    Operands out{};
    const float d = half_bits(p.d);
    out.scale = d * (0.5F + static_cast<float>(p.aux1 >> 28U)) * 0.25F;
    int w[8];
#pragma unroll
    for (unsigned l = 0; l < 4U; ++l) {
        const unsigned signs = sign_byte(p.aux1 >> (7U * l));
        w[2U * l] = apply_signs(static_cast<unsigned>(p.g[l]), byte_masks_lean(signs & 15U));
        w[2U * l + 1U] = apply_signs(static_cast<unsigned>(p.g[l] >> 32U), byte_masks_lean(signs >> 4U));
    }
    out.b0 = v4i{w[0], w[1], w[2], w[3]};
    out.b1 = v4i{w[4], w[5], w[6], w[7]};
    return out;
}

// Stage of v2: codes [2 * 32 groups][16 rows] v4i, deltas [32][16] (tile_position order), Q2_K: S0 and S1 [32][16]
// (tile order) as int16 pairs (form 6) or, with V2_FMA, two float arrays.
template <bool Q2, unsigned L> inline constexpr unsigned v2_sum_bytes = Q2 ? ((L & v2_fma) ? 2U * split_sum_bytes : split_sum_bytes) : 0U;
template <bool Q2, unsigned L> inline constexpr unsigned v2_stage_bytes = split_code_bytes + split_delta_bytes + v2_sum_bytes<Q2, L>;
template <WeightFormat Format, unsigned L> inline constexpr unsigned v2_lds =
    v2_stage_bytes<Format == WeightFormat::q2_k, L> + (Format == WeightFormat::q8_0 ? 0U : tile_rows * chunk_row_bytes<Format>);
template <bool Q2, unsigned L>
__device__ __forceinline__ void v2_store_items(unsigned char* lds, const StagedItems& x, unsigned chunk, unsigned width) {
    const unsigned groups = width / 32U;
    auto* a = reinterpret_cast<v4i*>(lds);
    auto* d = reinterpret_cast<float*>(lds + split_code_bytes);
#pragma unroll
    for (unsigned i = 0; i < 2U; ++i) {
        const unsigned item = threadIdx.x + split_threads * i;
        const unsigned r = item % split_rows, kh = (item / split_rows) % 2U, gg = item / (2U * split_rows);
        if (split_chunk_groups * chunk + gg >= groups) continue;
        const v4i v = x.v[i];
        a[(2U * gg + kh) * split_rows + r] = v;
        const unsigned at = tile_position(r);
        if (kh == 0U) d[gg * split_rows + at] = x.d[i];
        if constexpr (Q2) {
            const int ones = 0x01010101;
            int sum = 0;
            sum = STRIX_HIP_SUDOT4(false, ones, true, v.x, sum, false);
            sum = STRIX_HIP_SUDOT4(false, ones, true, v.y, sum, false);
            sum = STRIX_HIP_SUDOT4(false, ones, true, v.z, sum, false);
            sum = STRIX_HIP_SUDOT4(false, ones, true, v.w, sum, false);
            if constexpr ((L & v2_fma) != 0U) {
                auto* s = reinterpret_cast<float*>(lds + split_code_bytes + split_delta_bytes);
                s[kh * split_chunk_groups * split_rows + gg * split_rows + at] = static_cast<float>(sum); // exact (|sum| <= 2032)
            } else {
                auto* s = reinterpret_cast<short*>(lds + split_code_bytes + split_delta_bytes);
                s[2U * (gg * split_rows + at) + kh] = static_cast<short>(sum);
            }
        }
    }
}

// Weight tile loads of the v2 / R2 kernels: the two runs of 32 rows (gate and up, or the two output pairs of down) share
// one set of 32-bit per-lane offsets (row * row_bytes + 4 * word) from two uniform bases, so the loop keeps a few offsets
// instead of one 64-bit address per load (which spilled). A partial last chunk (V2_PART) loads only its blocks.
template <WeightFormat Format> inline constexpr unsigned run_words = 32U * chunk_row_bytes<Format> / 4U;
template <WeightFormat Format> inline constexpr unsigned run_loads = (run_words<Format> + split_threads - 1U) / split_threads;
template <WeightFormat Format> struct TileRegs2 { unsigned w[2][run_loads<Format>]; };
template <WeightFormat Format, unsigned L>
__device__ __forceinline__ TileRegs2<Format> load_tile2(const TileRows& t, unsigned chunk, unsigned groups) {
    constexpr unsigned row_words = chunk_row_bytes<Format> / 4U;
    constexpr unsigned block = Format == WeightFormat::q2_k ? q2_k_block_bytes : iq2_xxs_block_bytes;
    TileRegs2<Format> r{};
    const unsigned first = split_chunk_groups * chunk;
    if (first >= groups) return r;
    const unsigned left = groups - first;
    const auto* b0 = t.base[0] + static_cast<std::uint64_t>(chunk) * chunk_row_bytes<Format>;
    const auto* b1 = t.base[1] + static_cast<std::uint64_t>(chunk) * chunk_row_bytes<Format>;
    const auto rb = static_cast<unsigned>(t.row_bytes);
    if ((L & v2_part) == 0U || left >= split_chunk_groups) {
#pragma unroll
        for (unsigned i = 0; i < run_loads<Format>; ++i) {
            const unsigned word = threadIdx.x + split_threads * i;
            if (word < run_words<Format>) {
                const unsigned o = word / row_words * rb + 4U * (word % row_words);
                r.w[0][i] = *reinterpret_cast<const unsigned*>(b0 + o);
                r.w[1][i] = *reinterpret_cast<const unsigned*>(b1 + o);
            }
        }
    } else {
        const unsigned used = ((left + 7U) / 8U * block + 3U) / 4U; // words of the chunk's blocks in each row (uniform)
#pragma unroll
        for (unsigned i = 0; i < run_loads<Format>; ++i) {
            const unsigned word = threadIdx.x + split_threads * i;
            if (word < run_words<Format> && word % row_words < used) {
                const unsigned o = word / row_words * rb + 4U * (word % row_words);
                r.w[0][i] = *reinterpret_cast<const unsigned*>(b0 + o);
                r.w[1][i] = *reinterpret_cast<const unsigned*>(b1 + o);
            }
        }
    }
    return r;
}
template <WeightFormat Format>
__device__ __forceinline__ void store_tile2(unsigned char* tile, const TileRegs2<Format>& r) {
#pragma unroll
    for (unsigned i = 0; i < run_loads<Format>; ++i) {
        const unsigned word = threadIdx.x + split_threads * i;
        if (word < run_words<Format>) {
            reinterpret_cast<unsigned*>(tile)[word] = r.w[0][i];
            reinterpret_cast<unsigned*>(tile)[run_words<Format> + word] = r.w[1][i];
        }
    }
}

// v_permlanex16 without a copy of its source into the destination first: every lane is active, so the instruction's
// old-value operand is never read (an empty asm defines it).
#ifndef STRIX_DS41_V2_XCHG
#define STRIX_DS41_V2_XCHG 1
#endif
#ifndef STRIX_DS41_V2_MAGIC
#define STRIX_DS41_V2_MAGIC 1
#endif
__device__ __forceinline__ int v2_other_half(int v) {
    if constexpr (STRIX_DS41_V2_XCHG != 0) {
        int old;
        asm volatile("" : "=v"(old));
        return static_cast<int>(STRIX_HIP_EXCHANGE16(static_cast<unsigned>(old), static_cast<unsigned>(v),
                                                             false, false));
    } else {
        return other_half(v);
    }
}
__device__ __forceinline__ v4i v2_other_half(v4i v) { return v4i{v2_other_half(v.x), v2_other_half(v.y), v2_other_half(v.z), v2_other_half(v.w)}; }
__device__ __forceinline__ float v2_other_half(float v) { return __int_as_float(v2_other_half(__float_as_int(v))); }
// group_dot with the magic accumulator kept in registers (opaque to rematerialization: no 8 moves per chain).
__device__ __forceinline__ v8i v2_group_dot(v4i a0, v4i b0, v4i a1, v4i b1, const v8i& magic) {
    const v8i c = STRIX_HIP_WMMA_IU8(true, a0, true, b0, magic, false);
    return STRIX_HIP_WMMA_IU8(true, a1, true, b1, c, false);
}
// V2_PIN: the two WMMAs of v2_group_dot as inline asm on pinned registers so that A, B and C of every WMMA start in
// three different VGPR banks (bank = register % 4; gfx1151 issues a WMMA whose A, B and C share a bank in 34 cycles, not
// 32): magic accumulator v[0:7] (bank 0), A v[163:170] (bank 3), B v[154:161] (bank 2), D of output tile t v[137 + 8t]
// (bank 1, disjoint from A and B). The same two instructions on the same values (signed A and B: neg_lo:[1,1,0]).
constexpr unsigned v2_pin = 16U;
template <unsigned T>
__device__ __forceinline__ v8i v2_group_dot_pin(v4i a0, v4i b0, v4i a1, v4i b1, const v8i& magic) {
    v8i d;
    if constexpr (T == 0U) {
        STRIX_HIP_RDNA3_ASM(asm("v_wmma_i32_16x16x16_iu8 %0, %1, %2, %3 neg_lo:[1,1,0]" : "={v[137:144]}"(d) : "{v[163:166]}"(a0), "{v[154:157]}"(b0), "{v[0:7]}"(magic)));
        STRIX_HIP_RDNA3_ASM(asm("v_wmma_i32_16x16x16_iu8 %0, %1, %2, %0 neg_lo:[1,1,0]" : "+{v[137:144]}"(d) : "{v[167:170]}"(a1), "{v[158:161]}"(b1)));
    } else {
        STRIX_HIP_RDNA3_ASM(asm("v_wmma_i32_16x16x16_iu8 %0, %1, %2, %3 neg_lo:[1,1,0]" : "={v[145:152]}"(d) : "{v[163:166]}"(a0), "{v[154:157]}"(b0), "{v[0:7]}"(magic)));
        STRIX_HIP_RDNA3_ASM(asm("v_wmma_i32_16x16x16_iu8 %0, %1, %2, %0 neg_lo:[1,1,0]" : "+{v[145:152]}"(d) : "{v[167:170]}"(a1), "{v[158:161]}"(b1)));
    }
    return d;
}

// Leaves leaf[k][t][e]: class q + 8k, output tile t, D element e (row 2e + h, output 16t + i).
template <WeightFormat Format, unsigned L>
__device__ __forceinline__ void class_wave_v2(unsigned char* lds, unsigned q, unsigned width, const unsigned* row_of, unsigned count,
                                              const signed char* codes, const float* deltas, const unsigned long long* grid,
                                              const Grid2* grid2, float (&leaf)[split_classes][2][8], const TileRows& tile_rows_src,
                                              unsigned tile_row, const uint2* signs2 = nullptr) {
    constexpr bool q2 = Format == WeightFormat::q2_k;
    constexpr bool fma = q2 && (L & v2_fma) != 0U;
    const unsigned lane = threadIdx.x % 32U, i16 = lane & 15U, h = lane >> 4U, groups = width / 32U;
    const bool upper = h != 0U;
#pragma unroll
    for (unsigned k = 0; k < split_classes; ++k)
#pragma unroll
        for (unsigned t = 0; t < 2U; ++t)
#pragma unroll
            for (unsigned e = 0; e < 8U; ++e) leaf[k][t][e] = 0.0F;
    const unsigned chunks = (groups + split_chunk_groups - 1U) / split_chunk_groups;
    unsigned char* tile = lds + v2_stage_bytes<q2, L>;
    // Q8_0 (the shared expert, whose rows stay in the L2 / MALL): no LDS tile; each lane reads its own row's groups
    // from memory one group ahead (as split_gate_up_kernel's Q8_0 form).
    constexpr bool direct = Format == WeightFormat::q8_0;
    [[maybe_unused]] TileRegs2<Format> tile_next{};
    [[maybe_unused]] const unsigned char* own_row = (tile_row < 32U ? tile_rows_src.base[0] : tile_rows_src.base[1]) + static_cast<std::uint64_t>(tile_row % 32U) * tile_rows_src.row_bytes;
    [[maybe_unused]] Raw<Format> ahead{};
    if constexpr (direct) ahead = load_raw<Format>(own_row, q);
    else tile_next = load_tile2<Format, L>(tile_rows_src, 0U, groups);
    v8i magic{dot_magic, dot_magic, dot_magic, dot_magic, dot_magic, dot_magic, dot_magic, dot_magic};
    if constexpr (STRIX_DS41_V2_MAGIC != 0) asm volatile("" : "+v"(magic));
#pragma unroll 1
    for (unsigned chunk = 0; chunk < chunks; ++chunk) {
        split_sync<7>(); // the previous chunk's readers are done (and, at chunk 0, row_of is visible)
        const StagedItems items = load_chunk_items(chunk, width, row_of, count, codes, deltas);
        if constexpr (!direct) {
            store_tile2<Format>(tile, tile_next);
            if (chunk + 1U < chunks) tile_next = load_tile2<Format, L>(tile_rows_src, chunk + 1U, groups);
        }
        v2_store_items<q2, L>(lds, items, chunk, width);
        split_sync<7>();
        const auto* a = reinterpret_cast<const v4i*>(lds);
        const auto* d = reinterpret_cast<const float*>(lds + split_code_bytes);
        constexpr bool pipe = Format == WeightFormat::iq2_xxs && (L & v2_pipe) != 0U;
        // two stages: the raw bytes of group k + 2 and the grid entries of group k + 1 are read during group k
        [[maybe_unused]] Iq2Pre pre{};
        [[maybe_unused]] Raw<Format> raw_next{};
        if constexpr (pipe) {
            pre = iq2_prefetch(load_raw<Format>(tile + tile_row * chunk_row_bytes<Format>, q), grid);
            raw_next = load_raw<Format>(tile + tile_row * chunk_row_bytes<Format>, q + 8U);
        }
#pragma unroll
        for (unsigned k = 0; k < split_classes; ++k) {
            const unsigned gg = q + 8U * k;
            // uniform per wave (a wave without groups still joins the barriers); also keeps the groups in separate blocks
            // (one straight-line body of the four groups spills)
            if (split_chunk_groups * chunk + gg >= groups) break;
            Raw<Format> cur;
            if constexpr (direct) {
                cur = ahead;
                const unsigned next = k + 1U < split_classes ? split_chunk_groups * chunk + gg + 8U : split_chunk_groups * (chunk + 1U) + q;
                if (next < groups) ahead = load_raw<Format>(own_row, next); // uniform per wave
            } else {
                cur = load_raw<Format>(tile + tile_row * chunk_row_bytes<Format>, gg);
            }
            Operands own;
            if constexpr (direct) {
                own = decode(cur, grid);
            } else if constexpr (pipe) {
                const Iq2Pre now = pre;
                if (k + 1U < split_classes) pre = iq2_prefetch(raw_next, grid);
                if (k + 2U < split_classes) raw_next = load_raw<Format>(tile + tile_row * chunk_row_bytes<Format>, gg + 16U);
                own = iq2_finish(now);
            } else if constexpr (Format == WeightFormat::iq2_xxs && (L & v2_decode) != 0U) own = decode_v2(cur, grid2);
            else if constexpr (Format == WeightFormat::iq2_xxs && (L & v2_signs16) != 0U) own = decode_signs(cur, grid, reinterpret_cast<const SignEntry*>(signs2));
            else if constexpr (Format == WeightFormat::iq2_xxs && (L & v2_signs) != 0U) own = decode_signs2(cur, grid, signs2);
            else own = decode_lean(cur, grid);
            const v4i x0 = v2_other_half(own.b0), x1 = v2_other_half(own.b1);
            const unsigned x_dm = q2 ? static_cast<unsigned>(v2_other_half(static_cast<int>(own.dm_bits))) : 0U;
            const unsigned x_sc = q2 ? static_cast<unsigned>(v2_other_half(static_cast<int>(own.sc_bits))) : 0U;
            const float x_scale = q2 ? 0.0F : v2_other_half(own.scale);
            const v4i a0 = a[(2U * gg) * split_rows + i16];
            const v4i a1 = a[(2U * gg + 1U) * split_rows + i16];
            const float* dt = d + gg * split_rows + 8U * h;
#pragma unroll
            for (unsigned t = 0; t < 2U; ++t) {
                const bool mine = (t == 1U) == upper; // this half decoded tile t
                const v4i b0 = pick(mine, own.b0, x0), b1 = pick(mine, own.b1, x1);
                v8i r;
                if constexpr ((L & v2_pin) != 0U) r = t == 0U ? v2_group_dot_pin<0U>(a0, b0, a1, b1, magic) : v2_group_dot_pin<1U>(a0, b0, a1, b1, magic);
                else r = STRIX_DS41_V2_MAGIC != 0 ? v2_group_dot(a0, b0, a1, b1, magic) : group_dot(a0, b0, a1, b1);
                if constexpr (q2) {
                    const unsigned dmb = mine ? own.dm_bits : x_dm, scb = mine ? own.sc_bits : x_sc;
                    const float dd = half_bits(static_cast<unsigned short>(dmb & 0xffffU));
                    const float dm = half_bits(static_cast<unsigned short>(dmb >> 16U));
                    const int m0 = static_cast<int>((scb & 0xffU) >> 4U), m1 = static_cast<int>((scb >> 8U) >> 4U);
                    if constexpr (fma) {
                        const auto* s0 = reinterpret_cast<const float*>(lds + split_code_bytes + split_delta_bytes) + gg * split_rows + 8U * h;
                        const float* s1 = s0 + split_chunk_groups * split_rows;
                        const float m0f = static_cast<float>(m0), m1f = static_cast<float>(m1);
#pragma unroll
                        for (unsigned e = 0; e < 8U; ++e) {
                            const float b = __builtin_fmaf(m1f, s1[e], __builtin_fmaf(m0f, s0[e], 0.0F));
                            const float scaled = dd * magic_float(r[e]);
                            const float offset = dm * b;
                            leaf[k][t][e] = leaf[k][t][e] + (scaled - offset) * dt[e];
                        }
                    } else {
                        const int* sp = reinterpret_cast<const int*>(lds + split_code_bytes + split_delta_bytes) + gg * split_rows + 8U * h;
#pragma unroll
                        for (unsigned e = 0; e < 8U; ++e) {
                            const int packed = sp[e];
                            const int s0 = static_cast<int>(static_cast<short>(packed & 0xffff)), s1 = packed >> 16;
                            const int b = __mul24(m0, s0) + __mul24(m1, s1);
                            const float scaled = dd * magic_float(r[e]);
                            const float offset = dm * static_cast<float>(b);
                            leaf[k][t][e] = leaf[k][t][e] + (scaled - offset) * dt[e];
                        }
                    }
                } else {
                    const float scale = mine ? own.scale : x_scale;
#pragma unroll
                    for (unsigned e = 0; e < 8U; ++e) leaf[k][t][e] = leaf[k][t][e] + magic_float(r[e]) * (scale * dt[e]);
                }
            }
        }
    }
}

// The butterfly over the class sums of the eight class-waves of one pair, for the class map of class_wave_v2: levels 1-2
// (class bits 4, 3) in registers, levels 3-5 (class bits 2, 1, 0 = wave bits) through LDS; class-wave 0 holds v.
__device__ __forceinline__ void split_tree_v2(unsigned char* lds, unsigned q, unsigned pair, const float (&leaf)[split_classes][2][8],
                                              float (&v)[2][8]) {
#pragma unroll
    for (unsigned t = 0; t < 2U; ++t)
#pragma unroll
        for (unsigned e = 0; e < 8U; ++e) {
            const float c0 = leaf[0][t][e] + leaf[2][t][e], c1 = leaf[1][t][e] + leaf[3][t][e]; // classes q + q+16, q+8 + q+24
            v[t][e] = c0 + c1;
        }
    auto* x = reinterpret_cast<float*>(lds);
    const unsigned lane = threadIdx.x % 32U;
#pragma unroll
    for (unsigned s = 4U; s != 0U; s /= 2U) {
        split_sync<7>();
        if (q >= s && q < 2U * s) {
            float* slot = x + (4U * pair + (q - s)) * 16U * 32U;
#pragma unroll
            for (unsigned t = 0; t < 2U; ++t)
#pragma unroll
                for (unsigned e = 0; e < 8U; ++e) slot[(8U * t + e) * 32U + lane] = v[t][e];
        }
        split_sync<7>();
        if (q < s) {
            const float* slot = x + (4U * pair + q) * 16U * 32U;
#pragma unroll
            for (unsigned t = 0; t < 2U; ++t)
#pragma unroll
                for (unsigned e = 0; e < 8U; ++e) v[t][e] = v[t][e] + slot[(8U * t + e) * 32U + lane];
        }
    }
}

// Work of this block: batch x and output block y, in bands of `band` batches (band 0: x = blockIdx.x). The dispatcher
// walks blocks x fastest; within a band the linear index runs over the band's batches first, then the output blocks, so
// the blocks in flight share both the activations (a few batches) and the weights (a few output blocks).
__device__ __forceinline__ bool v2_block(const PrefillRoutesView& routes, unsigned band, Batch& b, unsigned& y, bool halves = false) {
    unsigned x = blockIdx.x;
    y = blockIdx.y;
    if (band > 1U) {
        const unsigned X = gridDim.x, Y = gridDim.y, linear = blockIdx.x + X * blockIdx.y, full = X / band;
        if (linear < full * band * Y) {
            const unsigned in = linear % (band * Y);
            x = linear / (band * Y) * band + in % band;
            y = in / band;
        } else {
            const unsigned w = X - full * band, in = linear - full * band * Y;
            x = full * band + in % w;
            y = in / w;
        }
    }
    // halves: block x runs rows 16 (x % 2) .. of the 32-row batch x / 2
    const unsigned half = halves ? x % 2U : 0U;
    if (halves) x /= 2U;
    if (x >= *routes.batch_total) return false;
    b.expert = routes.batch_expert[x];
    const unsigned count = routes.batch_count[x];
    if (count <= split_rows * half) return false;
    b.begin = routes.batch_begin[x] + split_rows * half;
    b.count = halves ? min(split_rows, count - split_rows * half) : count;
    return true;
}

template <WeightFormat Format, unsigned L>
__global__ __launch_bounds__(split_threads) __attribute__((amdgpu_waves_per_eu(STRIX_DS41_SPLIT_WAVES))) void split_gate_up_v2_kernel(PrefillGateUpView view, unsigned band) {
    extern __shared__ int4 dynamic_lds[];
    auto* lds = reinterpret_cast<unsigned char*>(dynamic_lds);
    constexpr bool dec = Format == WeightFormat::iq2_xxs && (L & v2_decode) != 0U;
    __shared__ Grid2 grid2[dec ? 256 : 1];
    __shared__ unsigned long long grid[Format == WeightFormat::iq2_xxs && !dec ? 256 : 1];
    constexpr bool sg = Format == WeightFormat::iq2_xxs && !dec && (L & (v2_signs | v2_signs16)) != 0U;
    constexpr bool sg16 = (L & v2_signs16) != 0U;
    __shared__ uint2 signs2[sg ? (sg16 ? 256 : 128) : 1];
    __shared__ unsigned row_of[split_rows], slot_of[split_rows];
    __shared__ float values[2][split_rows][32];
    Batch batch;
    unsigned yb = 0;
    if (!v2_block(view.routes, band, batch, yb)) return;
    if constexpr (dec) load_grid2(grid2);
    else if constexpr (Format == WeightFormat::iq2_xxs) load_grid(grid);
    if constexpr (sg16) load_signs(reinterpret_cast<SignEntry*>(signs2));
    else if constexpr (sg) load_signs2(signs2);
    if (threadIdx.x < batch.count) {
        row_of[threadIdx.x] = view.routes.entry_rows[batch.begin + threadIdx.x];
        slot_of[threadIdx.x] = view.routes.entry_slots[batch.begin + threadIdx.x];
    }
    const unsigned wave = threadIdx.x / 32U, lane = threadIdx.x % 32U, m = wave & 1U, q = wave >> 1U, h = lane >> 4U;
    const std::uint64_t expert_at = batch.expert * view.expert_stride_bytes + static_cast<std::uint64_t>(yb) * 32U * view.row_bytes;
    const TileRows rows{{static_cast<const unsigned char*>(view.gate) + expert_at, static_cast<const unsigned char*>(view.up) + expert_at},
                        view.row_bytes};
    float leaf[split_classes][2][8];
    class_wave_v2<Format, L>(lds, q, view.hidden, row_of, batch.count, view.codes, view.deltas, grid, grid2, leaf, rows, 32U * m + lane, signs2);
    float v[2][8];
    split_tree_v2(lds, q, m, leaf, v);
    if (q == 0U)
#pragma unroll
        for (unsigned t = 0; t < 2U; ++t)
#pragma unroll
            for (unsigned e = 0; e < 8U; ++e) {
                const unsigned r = 2U * e + h;
                if (r < batch.count) values[m][r][16U * t + (lane & 15U)] = v[t][e];
            }
    __syncthreads();
    const unsigned groups = view.inter / 32U, at = yb * 32U + lane;
    for (unsigned e = wave; e < batch.count; e += split_threads / 32U) {
        const unsigned long long slot = slot_of[e];
        const float g = values[0][e][lane], u = values[1][e][lane];
        if (view.gate_out) { view.gate_out[slot * view.inter + at] = g; view.up_out[slot * view.inter + at] = u; }
        const float act = swiglu_clamped(g, u, view.limit);
        float amax = fabsf(act);
        for (int stride = 16; stride; stride /= 2) amax = fmaxf(amax, __shfl_xor(amax, stride, 32));
        const bool tiny = amax > 0.0F && amax < 0x1p-119F;
        const float delta = amax == 0.0F || tiny ? 1.0F : amax / 127.0F;
        const float rounded = tiny ? 0.0F : rintf(act / delta);
        const int code = static_cast<int>(fminf(127.0F, fmaxf(-127.0F, rounded)));
        int sum = code;
        for (int stride = 16; stride; stride /= 2) sum += __shfl_xor(sum, stride, 32);
        view.act_codes[slot * view.inter + at] = static_cast<signed char>(code);
        if (lane == 0U) { view.act_deltas[slot * groups + yb] = delta; view.act_sums[slot * groups + yb] = sum; }
    }
}

template <WeightFormat Format, unsigned L>
__global__ __launch_bounds__(split_threads) __attribute__((amdgpu_waves_per_eu(STRIX_DS41_SPLIT_DOWN_WAVES))) void split_down_v2_kernel(PrefillDownView view, unsigned band) {
    extern __shared__ int4 dynamic_lds[];
    auto* lds = reinterpret_cast<unsigned char*>(dynamic_lds);
    __shared__ unsigned slot_of[split_rows];
    Batch batch;
    unsigned yb = 0;
    if (!v2_block(view.routes, band, batch, yb, view.routes.batch_rows == 32U)) return;
    if (threadIdx.x < batch.count) slot_of[threadIdx.x] = view.routes.entry_slots[batch.begin + threadIdx.x];
    const unsigned wave = threadIdx.x / 32U, lane = threadIdx.x % 32U, p = wave & 1U, q = wave >> 1U, h = lane >> 4U;
    const unsigned first = yb * 64U + 32U * p;
    const auto* tile_base = static_cast<const unsigned char*>(view.weights) + batch.expert * view.expert_stride_bytes +
        static_cast<std::uint64_t>(yb) * 64U * view.row_bytes;
    const TileRows rows{{tile_base, tile_base + 32U * view.row_bytes}, view.row_bytes};
    float leaf[split_classes][2][8];
    class_wave_v2<Format, L>(lds, q, view.width, slot_of, batch.count, view.codes, view.deltas, nullptr, nullptr, leaf, rows, 32U * p + lane);
    float v[2][8];
    split_tree_v2(lds, q, p, leaf, v);
    if (q != 0U) return;
#pragma unroll
    for (unsigned t = 0; t < 2U; ++t)
#pragma unroll
        for (unsigned e = 0; e < 8U; ++e) {
            const unsigned r = 2U * e + h;
            if (r < batch.count) view.out[static_cast<unsigned long long>(slot_of[r]) * view.outputs + first + 16U * t + (lane & 15U)] = v[t][e];
        }
}

// ---- Gate/up R2 (STRIX_DS41_PREFILL_MOE_R2, opt-in): 32-row batches, each decoded weight group feeds two row tiles ----
//
// A workgroup covers a batch of up to 32 rows x 32 outputs of both matrices in two passes over the hidden width: pass p runs
// the lane classes with class bit 0 = p (the tree's LAST level), so only 16 classes are live at a time (64 accumulators
// per lane, as split_gate_up_kernel). Class-wave q (0..7) owns classes p + 2q + 16k (k = 0, 1): tree level 1 (bit 4) in
// registers, levels 2-4 (bits 3, 2, 1 = wave bits 2, 1, 0) through LDS; level 5 adds the two passes' subtrees. Each lane
// decodes its output's group once per pass and chunk (lean decoder) and both row tiles use it (half the decode per WMMA). The same
// products, chains, tree and epilogue as split_gate_up_kernel: the same bits.
#ifndef STRIX_DS41_R2_MAGIC
#define STRIX_DS41_R2_MAGIC 1
#endif
constexpr unsigned r2_rows = 32U, r2_pass_groups = 16U;
constexpr unsigned r2_code_bytes = r2_pass_groups * 2U * r2_rows * 16U;  // 16 KB
constexpr unsigned r2_delta_bytes = r2_pass_groups * r2_rows * 4U;       // 2 KB
constexpr unsigned r2_stage_bytes = r2_code_bytes + r2_delta_bytes;
constexpr unsigned r2_exchange_bytes = 2U * 4U * 32U * 32U * 4U;         // 32 KB: 2 matrices x 4 slots x 32 values x 32 lanes
template <WeightFormat Format> inline constexpr unsigned r2_lds =
    r2_stage_bytes + tile_rows * chunk_row_bytes<Format> > r2_exchange_bytes ? r2_stage_bytes + tile_rows * chunk_row_bytes<Format> : r2_exchange_bytes;

template <WeightFormat Format>
__global__ __launch_bounds__(split_threads) __attribute__((amdgpu_waves_per_eu(STRIX_DS41_SPLIT_WAVES))) void r2_gate_up_kernel(PrefillGateUpView view, unsigned band) {
    extern __shared__ int4 dynamic_lds[];
    auto* lds = reinterpret_cast<unsigned char*>(dynamic_lds);
    __shared__ unsigned long long grid[256];
    __shared__ unsigned row_of[r2_rows], slot_of[r2_rows];
    __shared__ float values[2][r2_rows][32];
    Batch batch;
    unsigned yb = 0;
    if (!v2_block(view.routes, band, batch, yb)) return;
    load_grid(grid);
    if (threadIdx.x < batch.count) {
        row_of[threadIdx.x] = view.routes.entry_rows[batch.begin + threadIdx.x];
        slot_of[threadIdx.x] = view.routes.entry_slots[batch.begin + threadIdx.x];
    }
    const unsigned wave = threadIdx.x / 32U, lane = threadIdx.x % 32U, m = wave & 1U, q = wave >> 1U, h = lane >> 4U, i16 = lane & 15U;
    const bool upper = h != 0U, two = batch.count > 16U; // uniform
    const unsigned width = view.hidden, groups = width / 32U, chunks = groups / split_chunk_groups, count = batch.count;
    const std::uint64_t expert_at = batch.expert * view.expert_stride_bytes + static_cast<std::uint64_t>(yb) * 32U * view.row_bytes;
    const TileRows src{{static_cast<const unsigned char*>(view.gate) + expert_at, static_cast<const unsigned char*>(view.up) + expert_at},
                       view.row_bytes};
    const unsigned tile_row = 32U * m + lane;
    unsigned char* tile = lds + r2_stage_bytes;
    const auto* a = reinterpret_cast<const v4i*>(lds);
    const auto* d = reinterpret_cast<const float*>(lds + r2_code_bytes);
    v8i magic{dot_magic, dot_magic, dot_magic, dot_magic, dot_magic, dot_magic, dot_magic, dot_magic};
    if constexpr (STRIX_DS41_R2_MAGIC != 0) asm volatile("" : "+v"(magic));
    TileRegs2<Format> tile_next = load_tile2<Format, v2_on>(src, 0U, groups);
#pragma unroll 1
    for (unsigned pass = 0; pass < 2U; ++pass) {
        float leaf[2][2][2][8]; // [k][output tile][row tile][e]
#pragma unroll
        for (unsigned k = 0; k < 2U; ++k)
#pragma unroll
            for (unsigned t = 0; t < 2U; ++t)
#pragma unroll
                for (unsigned rt = 0; rt < 2U; ++rt)
#pragma unroll
                    for (unsigned e = 0; e < 8U; ++e) leaf[k][t][rt][e] = 0.0F;
#pragma unroll 1
        for (unsigned chunk = 0; chunk < chunks; ++chunk) {
            split_sync<7>();
            // stage: the pass's 16 groups (pass + 2 gl) x 2 halves x 32 rows, two items per thread, read before the tile prefetch
            v4i xv[2];
            float xd[2];
#pragma unroll
            for (unsigned i = 0; i < 2U; ++i) {
                const unsigned item = threadIdx.x + split_threads * i;
                const unsigned r = item % r2_rows, kh = (item / r2_rows) % 2U, gl = item / (2U * r2_rows);
                const unsigned g = split_chunk_groups * chunk + pass + 2U * gl;
                xv[i] = v4i{0, 0, 0, 0};
                xd[i] = 0.0F;
                if (r < count) {
                    xv[i] = *reinterpret_cast<const v4i*>(view.codes + static_cast<unsigned long long>(row_of[r]) * width + 32U * g + 16U * kh);
                    if (kh == 0U) xd[i] = view.deltas[static_cast<unsigned long long>(row_of[r]) * groups + g];
                }
            }
            store_tile2<Format>(tile, tile_next);
            // the next step's tile: this pass's next chunk, else the next pass's first chunk
            const unsigned next = chunk + 1U < chunks ? chunk + 1U : pass == 0U ? 0U : chunks;
            tile_next = load_tile2<Format, v2_on>(src, next, groups);
#pragma unroll
            for (unsigned i = 0; i < 2U; ++i) {
                const unsigned item = threadIdx.x + split_threads * i;
                const unsigned r = item % r2_rows, kh = (item / r2_rows) % 2U, gl = item / (2U * r2_rows);
                reinterpret_cast<v4i*>(lds)[(2U * gl + kh) * r2_rows + r] = xv[i];
                if (kh == 0U) reinterpret_cast<float*>(lds + r2_code_bytes)[gl * r2_rows + tile_position(r)] = xd[i];
            }
            split_sync<7>();
#pragma unroll
            for (unsigned k = 0; k < 2U; ++k) {
                const unsigned gl = q + 8U * k, gg = pass + 2U * gl; // class pass + 2q + 16k
                const Raw<Format> cur = load_raw<Format>(tile + tile_row * chunk_row_bytes<Format>, gg);
                const Operands own = decode_lean(cur, grid);
                const v4i x0 = v2_other_half(own.b0), x1 = v2_other_half(own.b1);
                const float x_scale = v2_other_half(own.scale);
                v4i bt0[2], bt1[2];
                float sc[2];
#pragma unroll
                for (unsigned t = 0; t < 2U; ++t) {
                    const bool mine = (t == 1U) == upper;
                    bt0[t] = pick(mine, own.b0, x0);
                    bt1[t] = pick(mine, own.b1, x1);
                    sc[t] = mine ? own.scale : x_scale;
                }
                const auto row_tile = [&](auto rt) {
                    const v4i a0 = a[(2U * gl) * r2_rows + 16U * rt + i16];
                    const v4i a1 = a[(2U * gl + 1U) * r2_rows + 16U * rt + i16];
                    const float* dt = d + gl * r2_rows + 16U * rt + 8U * h;
#pragma unroll
                    for (unsigned t = 0; t < 2U; ++t) {
                        const v8i r = STRIX_DS41_R2_MAGIC != 0 ? v2_group_dot(a0, bt0[t], a1, bt1[t], magic) : group_dot(a0, bt0[t], a1, bt1[t]);
#pragma unroll
                        for (unsigned e = 0; e < 8U; ++e) leaf[k][t][rt][e] = leaf[k][t][rt][e] + magic_float(r[e]) * (sc[t] * dt[e]);
                    }
                };
                row_tile(std::integral_constant<unsigned, 0U>{});
                if (two) row_tile(std::integral_constant<unsigned, 1U>{}); // uniform
            }
        }
        // tree: level 1 (bit 4) in registers, levels 2-4 through LDS, class-wave 0 holds the pass's subtree
        float v[2][2][8];
#pragma unroll
        for (unsigned t = 0; t < 2U; ++t)
#pragma unroll
            for (unsigned rt = 0; rt < 2U; ++rt)
#pragma unroll
                for (unsigned e = 0; e < 8U; ++e) v[t][rt][e] = leaf[0][t][rt][e] + leaf[1][t][rt][e];
        auto* x = reinterpret_cast<float*>(lds);
        // the lane index made opaque per pass: the exchange addresses are formed here, not hoisted out of the pass loop
        // (32 live addresses through the chunk loop spill)
        unsigned xl = lane;
        asm volatile("" : "+v"(xl));
#pragma unroll
        for (unsigned s = 4U; s != 0U; s /= 2U) {
            split_sync<7>();
            if (q >= s && q < 2U * s) {
                float* slot = x + (4U * m + (q - s)) * 32U * 32U;
#pragma unroll
                for (unsigned t = 0; t < 2U; ++t)
#pragma unroll
                    for (unsigned rt = 0; rt < 2U; ++rt)
#pragma unroll
                        for (unsigned e = 0; e < 8U; ++e) slot[(16U * t + 8U * rt + e) * 32U + xl] = v[t][rt][e];
            }
            split_sync<7>();
            if (q < s) {
                const float* slot = x + (4U * m + q) * 32U * 32U;
#pragma unroll
                for (unsigned t = 0; t < 2U; ++t)
#pragma unroll
                    for (unsigned rt = 0; rt < 2U; ++rt)
#pragma unroll
                        for (unsigned e = 0; e < 8U; ++e) v[t][rt][e] = v[t][rt][e] + slot[(16U * t + 8U * rt + e) * 32U + xl];
            }
        }
        if (q == 0U)
#pragma unroll
            for (unsigned t = 0; t < 2U; ++t)
#pragma unroll
                for (unsigned rt = 0; rt < 2U; ++rt)
#pragma unroll
                    for (unsigned e = 0; e < 8U; ++e) {
                        const unsigned r = 16U * rt + 2U * e + h;
                        if (r < count) {
                            float& out = values[m][r][16U * t + (xl & 15U)];
                            out = pass == 0U ? v[t][rt][e] : out + v[t][rt][e]; // level 5: even classes + odd classes
                        }
                    }
    }
    __syncthreads();
    const unsigned at = yb * 32U + lane;
    for (unsigned e = wave; e < count; e += split_threads / 32U) {
        const unsigned long long slot = slot_of[e];
        const float g = values[0][e][lane], u = values[1][e][lane];
        if (view.gate_out) { view.gate_out[slot * view.inter + at] = g; view.up_out[slot * view.inter + at] = u; }
        const float act = swiglu_clamped(g, u, view.limit);
        float amax = fabsf(act);
        for (int stride = 16; stride; stride /= 2) amax = fmaxf(amax, __shfl_xor(amax, stride, 32));
        const bool tiny = amax > 0.0F && amax < 0x1p-119F;
        const float delta = amax == 0.0F || tiny ? 1.0F : amax / 127.0F;
        const float rounded = tiny ? 0.0F : rintf(act / delta);
        const int code = static_cast<int>(fminf(127.0F, fmaxf(-127.0F, rounded)));
        int sum = code;
        for (int stride = 16; stride; stride /= 2) sum += __shfl_xor(sum, stride, 32);
        view.act_codes[slot * view.inter + at] = static_cast<signed char>(code);
        if (lane == 0U) { view.act_deltas[slot * (view.inter / 32U) + yb] = delta; view.act_sums[slot * (view.inter / 32U) + yb] = sum; }
    }
}

// ---- Gate/up W2 (STRIX_DS41_PREFILL_MOE_R2=1 with STRIX_DS41_PREFILL_MOE_R2_GU=w, opt-in): 32-row batches in one pass ----
//
// A workgroup of 32 waves covers a batch of up to 32 rows x 32 outputs of both matrices; wave (q, m, t) = class-wave q
// (classes q + 8k, as class_wave_v2), matrix m, output tile t, and holds both row tiles (64 accumulators per lane). For
// each pair of classes (2p, 2p + 1), half-wave h decodes its output's group of class 2p + h (one decode per lane), the
// halves exchange (v_permlanex16), and both groups feed both row tiles: 8 WMMAs per decode instead of 4. Every leaf (row,
// output, class) is the chain of split_gate_up_v2_kernel (same products, epilogue and order), the tree is split_tree_v2's
// per (row, output): the same bits. Batches of up to 16 rows run the first row tile only.
constexpr unsigned w2_threads = 1024U, w2_rows = 32U;
constexpr unsigned w2_code_bytes = split_chunk_groups * 2U * w2_rows * 16U; // 32 KB
constexpr unsigned w2_delta_bytes = split_chunk_groups * w2_rows * 4U;      // 4 KB
constexpr unsigned w2_stage_bytes = w2_code_bytes + w2_delta_bytes;
constexpr unsigned w2_exchange_bytes = 4U * 4U * 16U * 32U * 4U;            // 32 KB: 4 (m, t) x 4 slots x 16 values x 32 lanes
template <WeightFormat Format> inline constexpr unsigned w2_lds =
    w2_stage_bytes + tile_rows * chunk_row_bytes<Format> > w2_exchange_bytes ? w2_stage_bytes + tile_rows * chunk_row_bytes<Format> : w2_exchange_bytes;
template <WeightFormat Format> inline constexpr unsigned w2_loads = (run_words<Format> + w2_threads - 1U) / w2_threads;
template <WeightFormat Format> struct TileRegsW { unsigned w[2][w2_loads<Format>]; };
template <WeightFormat Format>
__device__ __forceinline__ TileRegsW<Format> load_tile_w(const TileRows& t, unsigned chunk) {
    constexpr unsigned row_words = chunk_row_bytes<Format> / 4U;
    TileRegsW<Format> r{};
    const auto* b0 = t.base[0] + static_cast<std::uint64_t>(chunk) * chunk_row_bytes<Format>;
    const auto* b1 = t.base[1] + static_cast<std::uint64_t>(chunk) * chunk_row_bytes<Format>;
    const auto rb = static_cast<unsigned>(t.row_bytes);
#pragma unroll
    for (unsigned i = 0; i < w2_loads<Format>; ++i) {
        const unsigned word = threadIdx.x + w2_threads * i;
        if (word < run_words<Format>) {
            const unsigned o = word / row_words * rb + 4U * (word % row_words);
            r.w[0][i] = *reinterpret_cast<const unsigned*>(b0 + o);
            r.w[1][i] = *reinterpret_cast<const unsigned*>(b1 + o);
        }
    }
    return r;
}
template <WeightFormat Format>
__device__ __forceinline__ void store_tile_w(unsigned char* tile, const TileRegsW<Format>& r) {
#pragma unroll
    for (unsigned i = 0; i < w2_loads<Format>; ++i) {
        const unsigned word = threadIdx.x + w2_threads * i;
        if (word < run_words<Format>) {
            reinterpret_cast<unsigned*>(tile)[word] = r.w[0][i];
            reinterpret_cast<unsigned*>(tile)[run_words<Format> + word] = r.w[1][i];
        }
    }
}
template <WeightFormat Format, unsigned L>
__global__ __launch_bounds__(w2_threads) __attribute__((amdgpu_waves_per_eu(8))) void w2_gate_up_kernel(PrefillGateUpView view, unsigned band) {
    extern __shared__ int4 dynamic_lds[];
    auto* lds = reinterpret_cast<unsigned char*>(dynamic_lds);
    constexpr bool sg = (L & v2_signs) != 0U;
    __shared__ unsigned long long grid[256];
    __shared__ uint2 signs2[sg ? 128 : 1];
    __shared__ unsigned row_of[w2_rows], slot_of[w2_rows];
    __shared__ float values[2][w2_rows][32];
    Batch batch;
    unsigned yb = 0;
    if (!v2_block(view.routes, band, batch, yb)) return;
    load_grid(grid);
    if constexpr (sg) load_signs2(signs2);
    if (threadIdx.x < batch.count) {
        row_of[threadIdx.x] = view.routes.entry_rows[batch.begin + threadIdx.x];
        slot_of[threadIdx.x] = view.routes.entry_slots[batch.begin + threadIdx.x];
    }
    const unsigned wave = threadIdx.x / 32U, lane = threadIdx.x % 32U, m = wave & 1U, t = (wave >> 1U) & 1U, q = wave >> 2U;
    const unsigned h = lane >> 4U, i16 = lane & 15U;
    const bool two = batch.count > 16U; // uniform
    const unsigned width = view.hidden, groups = width / 32U, chunks = groups / split_chunk_groups, count = batch.count;
    const std::uint64_t expert_at = batch.expert * view.expert_stride_bytes + static_cast<std::uint64_t>(yb) * 32U * view.row_bytes;
    const TileRows src{{static_cast<const unsigned char*>(view.gate) + expert_at, static_cast<const unsigned char*>(view.up) + expert_at},
                       view.row_bytes};
    const unsigned tile_row = 32U * m + 16U * t + i16;
    unsigned char* tile = lds + w2_stage_bytes;
    const auto* a = reinterpret_cast<const v4i*>(lds);
    const auto* d = reinterpret_cast<const float*>(lds + w2_code_bytes);
    v8i magic{dot_magic, dot_magic, dot_magic, dot_magic, dot_magic, dot_magic, dot_magic, dot_magic};
    asm volatile("" : "+v"(magic));
    float leaf[split_classes][2][8]; // [k][row tile][e]
#pragma unroll
    for (unsigned k = 0; k < split_classes; ++k)
#pragma unroll
        for (unsigned rt = 0; rt < 2U; ++rt)
#pragma unroll
            for (unsigned e = 0; e < 8U; ++e) leaf[k][rt][e] = 0.0F;
    TileRegsW<Format> tile_next = load_tile_w<Format>(src, 0U);
#pragma unroll 1
    for (unsigned chunk = 0; chunk < chunks; ++chunk) {
        split_sync<7>(); // the previous chunk's readers are done (and, at chunk 0, row_of is visible)
        v4i xv[2];
        float xd[2];
#pragma unroll
        for (unsigned i = 0; i < 2U; ++i) {
            const unsigned item = threadIdx.x + w2_threads * i;
            const unsigned r = item % w2_rows, kh = (item / w2_rows) % 2U, gg = item / (2U * w2_rows);
            const unsigned g = split_chunk_groups * chunk + gg;
            xv[i] = v4i{0, 0, 0, 0};
            xd[i] = 0.0F;
            if (r < count) {
                xv[i] = *reinterpret_cast<const v4i*>(view.codes + static_cast<unsigned long long>(row_of[r]) * width + 32U * g + 16U * kh);
                if (kh == 0U) xd[i] = view.deltas[static_cast<unsigned long long>(row_of[r]) * groups + g];
            }
        }
        store_tile_w<Format>(tile, tile_next);
        if (chunk + 1U < chunks) tile_next = load_tile_w<Format>(src, chunk + 1U);
#pragma unroll
        for (unsigned i = 0; i < 2U; ++i) {
            const unsigned item = threadIdx.x + w2_threads * i;
            const unsigned r = item % w2_rows, kh = (item / w2_rows) % 2U, gg = item / (2U * w2_rows);
            reinterpret_cast<v4i*>(lds)[(2U * gg + kh) * w2_rows + r] = xv[i];
            if (kh == 0U) reinterpret_cast<float*>(lds + w2_code_bytes)[gg * w2_rows + tile_position(r)] = xd[i];
        }
        split_sync<7>();
#pragma unroll
        for (unsigned p = 0; p < 2U; ++p) {
            // half h decodes class 2p + h of its output (a lane-dependent group: one decode per lane)
            const Raw<Format> cur = load_raw<Format>(tile + tile_row * chunk_row_bytes<Format>, q + 8U * (2U * p + h));
            Operands own;
            if constexpr (sg) own = decode_signs2(cur, grid, signs2);
            else own = decode_lean(cur, grid);
            const v4i x0 = v2_other_half(own.b0), x1 = v2_other_half(own.b1);
            const float x_scale = v2_other_half(own.scale);
#pragma unroll
            for (unsigned kk = 0; kk < 2U; ++kk) {
                const unsigned k = 2U * p + kk, gg = q + 8U * k;
                const bool mine = h == kk; // this half decoded class k
                const v4i b0 = pick(mine, own.b0, x0), b1 = pick(mine, own.b1, x1);
                const float scale = mine ? own.scale : x_scale;
#pragma unroll
                for (unsigned rt = 0; rt < 2U; ++rt) {
                    if (rt == 1U && !two) break;
                    const v4i a0 = a[(2U * gg) * w2_rows + 16U * rt + i16];
                    const v4i a1 = a[(2U * gg + 1U) * w2_rows + 16U * rt + i16];
                    const float* dt = d + gg * w2_rows + 16U * rt + 8U * h;
                    v8i r;
                    if constexpr ((L & v2_pin) != 0U) r = rt == 0U ? v2_group_dot_pin<0U>(a0, b0, a1, b1, magic) : v2_group_dot_pin<1U>(a0, b0, a1, b1, magic);
                    else r = v2_group_dot(a0, b0, a1, b1, magic);
#pragma unroll
                    for (unsigned e = 0; e < 8U; ++e) leaf[k][rt][e] = leaf[k][rt][e] + magic_float(r[e]) * (scale * dt[e]);
                }
            }
        }
    }
    // tree (split_tree_v2 per (row, output)): classes q + q+16 and q+8 + q+24 in registers, then the wave bits through LDS
    float v[2][8];
#pragma unroll
    for (unsigned rt = 0; rt < 2U; ++rt)
#pragma unroll
        for (unsigned e = 0; e < 8U; ++e) {
            const float c0 = leaf[0][rt][e] + leaf[2][rt][e], c1 = leaf[1][rt][e] + leaf[3][rt][e];
            v[rt][e] = c0 + c1;
        }
    auto* x = reinterpret_cast<float*>(lds);
    const unsigned pair = 2U * t + m;
#pragma unroll
    for (unsigned s = 4U; s != 0U; s /= 2U) {
        split_sync<7>();
        if (q >= s && q < 2U * s) {
            float* slot = x + (4U * pair + (q - s)) * 16U * 32U;
#pragma unroll
            for (unsigned rt = 0; rt < 2U; ++rt)
#pragma unroll
                for (unsigned e = 0; e < 8U; ++e) slot[(8U * rt + e) * 32U + lane] = v[rt][e];
        }
        split_sync<7>();
        if (q < s) {
            const float* slot = x + (4U * pair + q) * 16U * 32U;
#pragma unroll
            for (unsigned rt = 0; rt < 2U; ++rt)
#pragma unroll
                for (unsigned e = 0; e < 8U; ++e) v[rt][e] = v[rt][e] + slot[(8U * rt + e) * 32U + lane];
        }
    }
    if (q == 0U)
#pragma unroll
        for (unsigned rt = 0; rt < 2U; ++rt)
#pragma unroll
            for (unsigned e = 0; e < 8U; ++e) {
                const unsigned r = 16U * rt + 2U * e + h;
                if (r < count) values[m][r][16U * t + i16] = v[rt][e];
            }
    __syncthreads();
    const unsigned at = yb * 32U + lane;
    for (unsigned e = wave; e < count; e += w2_threads / 32U) {
        const unsigned long long slot = slot_of[e];
        const float g = values[0][e][lane], u = values[1][e][lane];
        if (view.gate_out) { view.gate_out[slot * view.inter + at] = g; view.up_out[slot * view.inter + at] = u; }
        const float act = swiglu_clamped(g, u, view.limit);
        float amax = fabsf(act);
        for (int stride = 16; stride; stride /= 2) amax = fmaxf(amax, __shfl_xor(amax, stride, 32));
        const bool tiny = amax > 0.0F && amax < 0x1p-119F;
        const float delta = amax == 0.0F || tiny ? 1.0F : amax / 127.0F;
        const float rounded = tiny ? 0.0F : rintf(act / delta);
        const int code = static_cast<int>(fminf(127.0F, fmaxf(-127.0F, rounded)));
        int sum = code;
        for (int stride = 16; stride; stride /= 2) sum += __shfl_xor(sum, stride, 32);
        view.act_codes[slot * view.inter + at] = static_cast<signed char>(code);
        if (lane == 0U) { view.act_deltas[slot * (view.inter / 32U) + yb] = delta; view.act_sums[slot * (view.inter / 32U) + yb] = sum; }
    }
}

// Down R2 (Q2_K, with STRIX_DS41_PREFILL_MOE_R2): a 32-row batch x 64 outputs per workgroup in two passes over the width
// (class bit 0 = pass), class-wave q owning classes pass + 2q + 16k, output pair p = wave % 2; each decoded group feeds
// both row tiles. Q2_K epilogue of split_down_v2_kernel (V2_FMA), partial last chunk tested per group (uniform per wave)
// and loaded by its blocks only. The same products, chains, tree and epilogue as split_down_kernel: the same bits.
#ifndef STRIX_DS41_R2D_MAGIC
#define STRIX_DS41_R2D_MAGIC 1
#endif
constexpr unsigned r2d_sum_bytes = 2U * r2_pass_groups * r2_rows * 4U;  // S0, S1 as floats: 4 KB
constexpr unsigned r2d_stage_bytes = r2_code_bytes + r2_delta_bytes + r2d_sum_bytes;
constexpr unsigned r2d_lds = r2d_stage_bytes + tile_rows * chunk_row_bytes<WeightFormat::q2_k> > r2_exchange_bytes
                                 ? r2d_stage_bytes + tile_rows * chunk_row_bytes<WeightFormat::q2_k> : r2_exchange_bytes;
__global__ __launch_bounds__(split_threads) __attribute__((amdgpu_waves_per_eu(STRIX_DS41_SPLIT_DOWN_WAVES))) void r2_down_kernel(PrefillDownView view, unsigned band) {
    constexpr WeightFormat Format = WeightFormat::q2_k;
    constexpr unsigned L = v2_on | v2_fma | v2_part;
    extern __shared__ int4 dynamic_lds[];
    auto* lds = reinterpret_cast<unsigned char*>(dynamic_lds);
    __shared__ unsigned slot_of[r2_rows];
    __shared__ float values[r2_rows][64];
    Batch batch;
    unsigned yb = 0;
    if (!v2_block(view.routes, band, batch, yb)) return;
    if (threadIdx.x < batch.count) slot_of[threadIdx.x] = view.routes.entry_slots[batch.begin + threadIdx.x];
    const unsigned wave = threadIdx.x / 32U, lane = threadIdx.x % 32U, p = wave & 1U, q = wave >> 1U, h = lane >> 4U, i16 = lane & 15U;
    const bool upper = h != 0U, two = batch.count > 16U; // uniform
    const unsigned width = view.width, groups = width / 32U, chunks = (groups + split_chunk_groups - 1U) / split_chunk_groups, count = batch.count;
    const auto* tile_base = static_cast<const unsigned char*>(view.weights) + batch.expert * view.expert_stride_bytes +
        static_cast<std::uint64_t>(yb) * 64U * view.row_bytes;
    const TileRows src{{tile_base, tile_base + 32U * view.row_bytes}, view.row_bytes};
    const unsigned tile_row = 32U * p + lane;
    unsigned char* tile = lds + r2d_stage_bytes;
    const auto* a = reinterpret_cast<const v4i*>(lds);
    const auto* d = reinterpret_cast<const float*>(lds + r2_code_bytes);
    const auto* sums = reinterpret_cast<const float*>(lds + r2_code_bytes + r2_delta_bytes);
    v8i magic{dot_magic, dot_magic, dot_magic, dot_magic, dot_magic, dot_magic, dot_magic, dot_magic};
    if constexpr (STRIX_DS41_R2D_MAGIC != 0) asm volatile("" : "+v"(magic));
    TileRegs2<Format> tile_next = load_tile2<Format, L>(src, 0U, groups);
#pragma unroll 1
    for (unsigned pass = 0; pass < 2U; ++pass) {
        float leaf[2][2][2][8]; // [k][output tile][row tile][e]
#pragma unroll
        for (unsigned k = 0; k < 2U; ++k)
#pragma unroll
            for (unsigned t = 0; t < 2U; ++t)
#pragma unroll
                for (unsigned rt = 0; rt < 2U; ++rt)
#pragma unroll
                    for (unsigned e = 0; e < 8U; ++e) leaf[k][t][rt][e] = 0.0F;
#pragma unroll 1
        for (unsigned chunk = 0; chunk < chunks; ++chunk) {
            split_sync<7>();
            v4i xv[2];
            float xd[2];
#pragma unroll
            for (unsigned i = 0; i < 2U; ++i) {
                const unsigned item = threadIdx.x + split_threads * i;
                const unsigned r = item % r2_rows, kh = (item / r2_rows) % 2U, gl = item / (2U * r2_rows);
                const unsigned g = split_chunk_groups * chunk + pass + 2U * gl;
                xv[i] = v4i{0, 0, 0, 0};
                xd[i] = 0.0F;
                if (g < groups && r < count) {
                    xv[i] = *reinterpret_cast<const v4i*>(view.codes + static_cast<unsigned long long>(slot_of[r]) * width + 32U * g + 16U * kh);
                    if (kh == 0U) xd[i] = view.deltas[static_cast<unsigned long long>(slot_of[r]) * groups + g];
                }
            }
            store_tile2<Format>(tile, tile_next);
            const unsigned next = chunk + 1U < chunks ? chunk + 1U : pass == 0U ? 0U : chunks;
            tile_next = load_tile2<Format, L>(src, next, groups);
#pragma unroll
            for (unsigned i = 0; i < 2U; ++i) {
                const unsigned item = threadIdx.x + split_threads * i;
                const unsigned r = item % r2_rows, kh = (item / r2_rows) % 2U, gl = item / (2U * r2_rows);
                const v4i v = xv[i];
                reinterpret_cast<v4i*>(lds)[(2U * gl + kh) * r2_rows + r] = v;
                const unsigned at = tile_position(r);
                if (kh == 0U) reinterpret_cast<float*>(lds + r2_code_bytes)[gl * r2_rows + at] = xd[i];
                const int ones = 0x01010101;
                int sum = 0;
                sum = STRIX_HIP_SUDOT4(false, ones, true, v.x, sum, false);
                sum = STRIX_HIP_SUDOT4(false, ones, true, v.y, sum, false);
                sum = STRIX_HIP_SUDOT4(false, ones, true, v.z, sum, false);
                sum = STRIX_HIP_SUDOT4(false, ones, true, v.w, sum, false);
                reinterpret_cast<float*>(lds + r2_code_bytes + r2_delta_bytes)[kh * r2_pass_groups * r2_rows + gl * r2_rows + at] =
                    static_cast<float>(sum); // exact (|sum| <= 2032)
            }
            split_sync<7>();
#pragma unroll
            for (unsigned k = 0; k < 2U; ++k) {
                const unsigned gl = q + 8U * k, gg = pass + 2U * gl; // class pass + 2q + 16k
                if (split_chunk_groups * chunk + gg >= groups) break; // uniform per wave (partial last chunk)
                const Raw<Format> cur = load_raw<Format>(tile + tile_row * chunk_row_bytes<Format>, gg);
                const Operands own = decode_lean(cur, nullptr);
                const v4i x0 = v2_other_half(own.b0), x1 = v2_other_half(own.b1);
                const unsigned x_dm = static_cast<unsigned>(v2_other_half(static_cast<int>(own.dm_bits)));
                const unsigned x_sc = static_cast<unsigned>(v2_other_half(static_cast<int>(own.sc_bits)));
                v4i bt0[2], bt1[2];
                float dd[2], dm[2], m0f[2], m1f[2];
#pragma unroll
                for (unsigned t = 0; t < 2U; ++t) {
                    const bool mine = (t == 1U) == upper;
                    bt0[t] = pick(mine, own.b0, x0);
                    bt1[t] = pick(mine, own.b1, x1);
                    const unsigned dmb = mine ? own.dm_bits : x_dm, scb = mine ? own.sc_bits : x_sc;
                    dd[t] = half_bits(static_cast<unsigned short>(dmb & 0xffffU));
                    dm[t] = half_bits(static_cast<unsigned short>(dmb >> 16U));
                    m0f[t] = static_cast<float>(static_cast<int>((scb & 0xffU) >> 4U));
                    m1f[t] = static_cast<float>(static_cast<int>((scb >> 8U) >> 4U));
                }
                const auto row_tile = [&](auto rt) {
                    const v4i a0 = a[(2U * gl) * r2_rows + 16U * rt + i16];
                    const v4i a1 = a[(2U * gl + 1U) * r2_rows + 16U * rt + i16];
                    const float* dt = d + gl * r2_rows + 16U * rt + 8U * h;
                    const float* s0 = sums + gl * r2_rows + 16U * rt + 8U * h;
                    const float* s1 = s0 + r2_pass_groups * r2_rows;
#pragma unroll
                    for (unsigned t = 0; t < 2U; ++t) {
                        const v8i r = STRIX_DS41_R2D_MAGIC != 0 ? v2_group_dot(a0, bt0[t], a1, bt1[t], magic) : group_dot(a0, bt0[t], a1, bt1[t]);
#pragma unroll
                        for (unsigned e = 0; e < 8U; ++e) {
                            const float b = __builtin_fmaf(m1f[t], s1[e], __builtin_fmaf(m0f[t], s0[e], 0.0F));
                            const float scaled = dd[t] * magic_float(r[e]);
                            const float offset = dm[t] * b;
                            leaf[k][t][rt][e] = leaf[k][t][rt][e] + (scaled - offset) * dt[e];
                        }
                    }
                };
                row_tile(std::integral_constant<unsigned, 0U>{});
                if (two) row_tile(std::integral_constant<unsigned, 1U>{}); // uniform
            }
        }
        float v[2][2][8];
#pragma unroll
        for (unsigned t = 0; t < 2U; ++t)
#pragma unroll
            for (unsigned rt = 0; rt < 2U; ++rt)
#pragma unroll
                for (unsigned e = 0; e < 8U; ++e) v[t][rt][e] = leaf[0][t][rt][e] + leaf[1][t][rt][e];
        auto* x = reinterpret_cast<float*>(lds);
        unsigned xl = lane;
        asm volatile("" : "+v"(xl));
#pragma unroll
        for (unsigned s = 4U; s != 0U; s /= 2U) {
            split_sync<7>();
            if (q >= s && q < 2U * s) {
                float* slot = x + (4U * p + (q - s)) * 32U * 32U;
#pragma unroll
                for (unsigned t = 0; t < 2U; ++t)
#pragma unroll
                    for (unsigned rt = 0; rt < 2U; ++rt)
#pragma unroll
                        for (unsigned e = 0; e < 8U; ++e) slot[(16U * t + 8U * rt + e) * 32U + xl] = v[t][rt][e];
            }
            split_sync<7>();
            if (q < s) {
                const float* slot = x + (4U * p + q) * 32U * 32U;
#pragma unroll
                for (unsigned t = 0; t < 2U; ++t)
#pragma unroll
                    for (unsigned rt = 0; rt < 2U; ++rt)
#pragma unroll
                        for (unsigned e = 0; e < 8U; ++e) v[t][rt][e] = v[t][rt][e] + slot[(16U * t + 8U * rt + e) * 32U + xl];
            }
        }
        if (q == 0U)
#pragma unroll
            for (unsigned t = 0; t < 2U; ++t)
#pragma unroll
                for (unsigned rt = 0; rt < 2U; ++rt)
#pragma unroll
                    for (unsigned e = 0; e < 8U; ++e) {
                        const unsigned r = 16U * rt + 2U * e + h;
                        if (r < count) {
                            float& out = values[r][32U * p + 16U * t + (xl & 15U)];
                            out = pass == 0U ? v[t][rt][e] : out + v[t][rt][e]; // level 5: even classes + odd classes
                        }
                    }
    }
    __syncthreads();
    for (unsigned e = wave; e < count; e += split_threads / 32U) {
        float* o = view.out + static_cast<unsigned long long>(slot_of[e]) * view.outputs + yb * 64U;
        o[lane] = values[e][lane];
        o[32U + lane] = values[e][32U + lane];
    }
}
