// minimax-m3 "FP32 prefill identity": device code shared VERBATIM by the gfx1151 build (hipcc)
// and the RTX PRO build (nvcc), so both run the same declared order and produce the same bits (IEEE fma, RN, no
// contraction: build with -ffp-contract=off / --fmad=false).
//
// One EXL3 linear y = svh * H_out(W * H_in(suh * x)) on FP32 rows:
//   input   per 128-channel block: s = x * suh (RN32), then hadamard128 (lane + 32 j layout: xor strides 1..16 as
//           (lane & s) ? peer - v : v + peer, register strides 1, 2 as (l + r, l - r)), then * RN32(1/sqrt(128)).
//           Unlike strix_gemm / strix_gemv nothing is rounded to FP16: activations stay FP32.
//   core    per output o: acc = 0; for k = 0 .. input-1 ascending: acc = fmaf(xt[k], w[k][o], acc), w = the mul1
//           decode of the trellis tile (FP16 value, exact in FP32). One chain, no splits: every row's bits are
//           independent of the row count, the chunking and the other rows.
//   output  per 128-output block: hadamard128 of the raw sums (same form), then * svh (RN32).
// The includer defines MMX_DEV (device function qualifier), MMX_SHFL_XOR(v, s) (32-lane xor shuffle), MMX_FMA(a, b, c)
// (IEEE fma), mmx_sync(), mmx_round_half, mmx_half_to_float, mmx_rint (RNE) and mmx_u2f, and includes it inside its own
// namespace. The SwiGLU uses the portable exp above (not the platform expf), so the identity has no table-emulated math.

constexpr unsigned kF32TileRows = 64, kF32TileCols = 128, kF32Threads = 256;

MMX_DEV std::uint32_t f32_tile_state(const std::uint32_t* w, int bits, int position) {
    const int words = bits * 8;
    const int end = (position + 1) * bits + 256 * bits;
    const int start = end - 16;
    const std::uint32_t earlier = w[(start / 32) % words];
    const std::uint32_t later = w[((end - 1) / 32) % words];
    const std::uint64_t pair = (static_cast<std::uint64_t>(earlier) << 32) | later;
    return static_cast<std::uint32_t>(pair >> ((32 - (end & 31)) & 31)) & 65535U;
}
// mul1 codebook (strix-trellis codebook_mul1): RN16(fma(1024 + bytesum(state * 0x83dcd12d), h(0x1eee), h(0xc931))),
// returned as the exact FP32 value of that FP16.
MMX_DEV float f32_mul1(std::uint32_t state) {
    const std::uint32_t p = state * 0x83dcd12dU;
    const unsigned sum = (p & 255U) + ((p >> 8) & 255U) + ((p >> 16) & 255U) + (p >> 24);
    const float v = MMX_FMA(static_cast<float>(1024 + sum), 0x1.bb8p-8F /* h 0x1eee */, -0x1.4c4p+3F /* h 0xc931 */);
    return mmx_round_half(v);
}
// Lane / position -> (k, n) inside a 16 x 16 tile (the engine's wmma_item decode order; any order gives the same tile).
MMX_DEV void f32_pos_kn(int lane, int j, int& k, int& n) {
    k = 2 * (lane % 4) + (j % 2) + 8 * ((j / 2) % 2);
    n = lane / 4 + 8 * (j / 4);
}

MMX_DEV void f32_hadamard128(float (&v)[4], unsigned lane) {
#pragma unroll
    for (int stride = 1; stride < 32; stride *= 2) {
#pragma unroll
        for (int j = 0; j < 4; ++j) {
            const float peer = MMX_SHFL_XOR(v[j], stride);
            v[j] = (lane & static_cast<unsigned>(stride)) ? peer - v[j] : v[j] + peer;
        }
    }
#pragma unroll
    for (int stride = 1; stride < 4; stride *= 2) {
#pragma unroll
        for (int j = 0; j < 4; ++j)
            if (!(j & stride)) {
                const float l = v[j], r = v[j + stride];
                v[j] = l + r;
                v[j + stride] = l - r;
            }
    }
#pragma unroll
    for (int j = 0; j < 4; ++j) v[j] = v[j] * 0x1.6a09e6p-4F;
}

// Row table entry of a (grouped) call: matrix z, source row (x), destination row (y).
struct F32Job {
    const std::uint32_t* trellis;
    const std::uint16_t* suh;
    const std::uint16_t* svh;
    std::uint32_t input, output, bits;
    std::uint32_t rows;        // rows of this matrix
    std::uint32_t row_begin;   // first entry in the row map / first row of the workspace
    std::uint32_t tile_begin;  // first CTA (row tile) of this job in the flattened grid
};

// Input transform: one wave (32 lanes) per (row entry, 128-channel block). x row = xmap ? xmap[e] : e.
// Block: 256 threads = 8 waves; grid.x = ceil(entries * input/128 / 8). xt is [entries][input].
MMX_DEV void f32_transform_body(const float* x, std::uint32_t x_stride, const std::uint32_t* xmap,
                                       const std::uint16_t* suh, std::uint32_t input, std::uint32_t entries, float* xt,
                                       unsigned item, unsigned lane) {
    const unsigned blocks = input / 128;
    if (item >= entries * blocks) return;
    const unsigned e = item / blocks, b = item % blocks;
    const std::uint32_t src = xmap ? xmap[e] : e;
    const float* xr = x + static_cast<std::size_t>(src) * x_stride;
    float v[4];
#pragma unroll
    for (int j = 0; j < 4; ++j) {
        const unsigned c = b * 128 + lane + 32U * static_cast<unsigned>(j);
        v[j] = xr[c] * mmx_half_to_float(suh[c]);
    }
    f32_hadamard128(v, lane);
#pragma unroll
    for (int j = 0; j < 4; ++j) xt[static_cast<std::size_t>(e) * input + b * 128 + lane + 32U * static_cast<unsigned>(j)] = v[j];
}

// Core: CTA = TR rows (64 or 128) x 128 outputs (8 tile columns), 256 threads; thread t = rows (TR / 16) (t / 16) ..
// + TR / 16 - 1 x outputs 4 (t % 16) .. + 3 and 64 + 4 (t % 16) .. + 3 (the 16 lanes of a row group read consecutive
// float4s of the weight tile: no shared-memory bank conflicts). Each output keeps ONE fmaf chain over k ascending (the identity); the tiling
// only changes which thread runs which chain, so both tile heights give the same bits. raw is [rows][output] FP32.
// Decoder policy changes only the byte sum; the tile schedule stays fixed.
struct F32Mul1 {
    MMX_DEV float operator()(std::uint32_t state) const { return f32_mul1(state); }
};
template <unsigned TR, class Decode = F32Mul1>
MMX_DEV void f32_core_body(const std::uint32_t* trellis, std::uint32_t input, std::uint32_t output, int bits,
                           const float* xt, std::uint32_t rows, float* raw, unsigned row_tile, unsigned col_block,
                           float (*as)[TR + 4], float (*ws)[kF32TileCols + 4], std::uint32_t (*wsrc)[64], unsigned t) {
    constexpr unsigned RPT = TR / 16, LOADS = TR / 64;
    const unsigned lane = t & 31U, wave = t >> 5;
    const unsigned row0 = row_tile * TR, tn0 = col_block * (kF32TileCols / 16), tiles_n = output / 16, tiles_k = input / 16;
    const unsigned tr = t / 16, tc = t % 16;
    const unsigned lr = t / 4, lk = (t % 4) * 4;  // this thread's float4s of the activation tile: rows lr + 64 m
    const std::uint32_t* wcol = trellis + (static_cast<std::size_t>(tn0) + wave) * (static_cast<unsigned>(bits) * 8U);
    const std::size_t wstride = static_cast<std::size_t>(tiles_n) * (static_cast<unsigned>(bits) * 8U);
    float acc[RPT][8];
#pragma unroll
    for (unsigned i = 0; i < RPT; ++i)
#pragma unroll
        for (int j = 0; j < 8; ++j) acc[i][j] = 0.0F;
    // trellis words of this wave's tile (bits * 8 <= 64 words): lanes hold words lane and lane + 32 of the NEXT k-tile
    const unsigned words = static_cast<unsigned>(bits) * 8U;
    std::uint32_t wa = lane < words ? wcol[lane] : 0U, wb = lane + 32U < words ? wcol[lane + 32U] : 0U;
    float4 xa[LOADS];
#pragma unroll
    for (unsigned m = 0; m < LOADS; ++m)
        xa[m] = row0 + lr + 64U * m < rows ? *reinterpret_cast<const float4*>(xt + static_cast<std::size_t>(row0 + lr + 64U * m) * input + lk)
                                           : make_float4(0.0F, 0.0F, 0.0F, 0.0F);
    for (unsigned tk = 0; tk < tiles_k; ++tk) {
#pragma unroll
        for (unsigned m = 0; m < LOADS; ++m) {
            as[lk + 0][lr + 64U * m] = xa[m].x;
            as[lk + 1][lr + 64U * m] = xa[m].y;
            as[lk + 2][lr + 64U * m] = xa[m].z;
            as[lk + 3][lr + 64U * m] = xa[m].w;
        }
        wsrc[wave][lane] = wa;
        wsrc[wave][lane + 32U] = wb;
        mmx_sync_wave();
        {
#pragma unroll
            for (int j = 0; j < 8; ++j) {
                int k, n;
                f32_pos_kn(static_cast<int>(lane), j, k, n);
                ws[k][wave * 16U + static_cast<unsigned>(n)] = Decode{}(f32_tile_state(wsrc[wave], bits, 8 * static_cast<int>(lane) + j));
            }
        }
        mmx_sync();
        if (tk + 1 < tiles_k) {  // prefetch the next tile's trellis words
            const std::uint32_t* w = wcol + (tk + 1) * wstride;
            if (lane < words) wa = w[lane];
            if (lane + 32U < words) wb = w[lane + 32U];
        }
        if (tk + 1 < tiles_k)  // prefetch the next tile
#pragma unroll
            for (unsigned m = 0; m < LOADS; ++m)
                if (row0 + lr + 64U * m < rows)
                    xa[m] = *reinterpret_cast<const float4*>(xt + static_cast<std::size_t>(row0 + lr + 64U * m) * input + (tk + 1) * 16U + lk);
#pragma unroll
        for (int k = 0; k < 16; ++k) {
            float av[RPT];
#pragma unroll
            for (unsigned q = 0; q < RPT / 4; ++q) {
                const float4 a = *reinterpret_cast<const float4*>(&as[k][RPT * tr + 4U * q]);
                av[4 * q] = a.x;
                av[4 * q + 1] = a.y;
                av[4 * q + 2] = a.z;
                av[4 * q + 3] = a.w;
            }
            const float4 b0 = *reinterpret_cast<const float4*>(&ws[k][4 * tc]);
            const float4 b1 = *reinterpret_cast<const float4*>(&ws[k][64 + 4 * tc]);
            const float bv[8] = {b0.x, b0.y, b0.z, b0.w, b1.x, b1.y, b1.z, b1.w};
#pragma unroll
            for (unsigned i = 0; i < RPT; ++i)
#pragma unroll
                for (int j = 0; j < 8; ++j) acc[i][j] = MMX_FMA(av[i], bv[j], acc[i][j]);
        }
        mmx_sync();
    }
#pragma unroll
    for (unsigned i = 0; i < RPT; ++i) {
        const unsigned r = row0 + RPT * tr + i;
        if (r < rows) {
            float* o = raw + static_cast<std::size_t>(r) * output + tn0 * 16U + 4U * tc;
            *reinterpret_cast<float4*>(o) = make_float4(acc[i][0], acc[i][1], acc[i][2], acc[i][3]);
            *reinterpret_cast<float4*>(o + 64) = make_float4(acc[i][4], acc[i][5], acc[i][6], acc[i][7]);
        }
    }
}

// Output transform: one wave per (row entry, 128-output block): y[ymap ? ymap[e] : e][o] = svh[o] * H(raw)[o].
MMX_DEV void f32_output_body(const float* raw, const std::uint16_t* svh, std::uint32_t output, std::uint32_t entries,
                                    float* y, std::uint32_t y_stride, const std::uint32_t* ymap, unsigned item, unsigned lane) {
    const unsigned blocks = output / 128;
    if (item >= entries * blocks) return;
    const unsigned e = item / blocks, b = item % blocks;
    float v[4];
#pragma unroll
    for (int j = 0; j < 4; ++j) v[j] = raw[static_cast<std::size_t>(e) * output + b * 128 + lane + 32U * static_cast<unsigned>(j)];
    f32_hadamard128(v, lane);
    const std::uint32_t dst = ymap ? ymap[e] : e;
#pragma unroll
    for (int j = 0; j < 4; ++j) {
        const unsigned o = b * 128 + lane + 32U * static_cast<unsigned>(j);
        y[static_cast<std::size_t>(dst) * y_stride + o] = v[j] * mmx_half_to_float(svh[o]);
    }
}

// exp: the deepseek41 / MSA declared exp (strix/minimax_msa.hpp exp_f: Cody-Waite + degree-8 polynomial, all fma),
// portable bit for bit. 0 below -87, inf above 88.
MMX_DEV float f32_exp(float x) {
    if (x < -87.0F) return 0.0F;
    if (x > 88.0F) return mmx_u2f(0x7f800000U);
    const float n = mmx_rint(x * 1.44269504088896341F);
    float r = MMX_FMA(n, -0.693145751953125F, x);
    r = MMX_FMA(n, -1.42860676533018e-06F, r);
    float p = 2.48015873e-05F;
    p = MMX_FMA(p, r, 1.98412698e-04F);
    p = MMX_FMA(p, r, 1.38888889e-03F);
    p = MMX_FMA(p, r, 8.33333333e-03F);
    p = MMX_FMA(p, r, 4.16666667e-02F);
    p = MMX_FMA(p, r, 1.66666667e-01F);
    p = MMX_FMA(p, r, 0.5F);
    p = MMX_FMA(p, r, 1.0F);
    p = MMX_FMA(p, r, 1.0F);
    return p * mmx_u2f(static_cast<std::uint32_t>(static_cast<int>(n) + 127) << 23U);
}
// swigluoai of the identity: gg = min(g, 7), uu = clamp(u, -7, 7), a = (gg * (1 / (1 + exp(-1.702 gg)))) * (uu + 1).
MMX_DEV void f32_swiglu_body(const float* g, const float* u, float* a, std::size_t n, std::size_t i) {
    if (i >= n) return;
    const float gg = g[i] < 7.0F ? g[i] : 7.0F;
    const float uu = u[i] < -7.0F ? -7.0F : (u[i] > 7.0F ? 7.0F : u[i]);
    a[i] = gg * (1.0F / (1.0F + f32_exp(-1.702F * gg))) * (uu + 1.0F);
}

// MoE combine of one rank's partial (the G4 / decode order): per row, acc = 0; acc = fmaf(w_e, y_e, acc) over its 4
// experts in ascending expert id; out = acc + shared. slot[r][i] = the y_e row of route i of row r, already sorted by id
// (host), weight[r][i] its routing weight.
MMX_DEV void f32_combine_body(const float* ye, const float* ys, const std::uint32_t* slot, const float* weight,
                                     std::uint32_t rows, float* out, unsigned idx) {
    if (idx >= rows * 6144U) return;
    const unsigned r = idx / 6144U, c = idx % 6144U;
    float acc = 0.0F;
#pragma unroll
    for (unsigned i = 0; i < 4; ++i) acc = MMX_FMA(weight[r * 4U + i], ye[static_cast<std::size_t>(slot[r * 4U + i]) * 6144U + c], acc);
    out[static_cast<std::size_t>(r) * 6144U + c] = acc + ys[static_cast<std::size_t>(r) * 6144U + c];
}
