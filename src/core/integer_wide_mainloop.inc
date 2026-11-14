// Shared four-wave dense plan, including the small-shape fallback.
using Plan = typename Operand::Plan;
using Ops = typename Operand::Ops;
using v4i = typename Operand::Int4;
using v8i = typename Operand::Int8;
constexpr unsigned Outputs = NT, wide_waves = 4, wide_classes = 8;
constexpr bool Staged = Plan::staged;
const auto& v = ad.tensor();
const auto* __restrict__ weight = v.weight;
const auto* __restrict__ codes = v.codes;
const auto* __restrict__ deltas = v.deltas;
auto* __restrict__ out = v.out;
    __shared__ float part[wide_waves][Outputs][8][32];
    __shared__ float4 dsh[Staged ? 32U : 1U][2][2];  // [group of the chunk][h][e / 4] = deltas of rows 2e + h
    const unsigned w = threadIdx.x / 32U, l = threadIdx.x % 32U, h = l >> 4U, lo = l & 15U;
    const unsigned row = blockIdx.x * 16U + lo, o0 = blockIdx.y * (16U * Outputs);
    const unsigned groups = v.width / 32U, row_bytes = (v.row_width ? v.row_width : v.width) / 32U * 34U;
    const bool live = row < v.rows;
    const signed char* arow = reinterpret_cast<const signed char*>(codes) + static_cast<std::size_t>(live ? row : 0U) * v.x_stride;
    const float* drow = deltas + static_cast<std::size_t>(live ? row : 0U) * (v.x_stride / 32U);
    const std::uint8_t* wrow[Outputs];
#pragma unroll
    for (unsigned t = 0; t < Outputs; ++t) {
        const unsigned o = o0 + t * 16U + lo;
        wrow[t] = weight + v.k_offset / 32U * 34U + static_cast<std::size_t>(o < v.count ? o : v.count - 1U) * row_bytes;
    }
    float acc[wide_classes][Outputs][8];
#pragma unroll
    for (unsigned j = 0; j < wide_classes; ++j)
#pragma unroll
        for (unsigned t = 0; t < Outputs; ++t)
#pragma unroll
            for (unsigned e = 0; e < 8; ++e) acc[j][t][e] = 0.0F;
    const v8i zero{0, 0, 0, 0, 0, 0, 0, 0};
    const v8i magic{Ops::dot_magic, Ops::dot_magic, Ops::dot_magic, Ops::dot_magic, Ops::dot_magic, Ops::dot_magic, Ops::dot_magic, Ops::dot_magic};
    for (unsigned s0 = 0; s0 < groups; s0 += 32U) {
        if constexpr (Staged) {
            __syncthreads();  // the previous chunk's deltas are read
            auto* d = reinterpret_cast<float*>(dsh);
            for (unsigned i = threadIdx.x; i < 32U * 16U; i += 32U * wide_waves) {
                const unsigned gl = i / 16U, rr = i % 16U, r = blockIdx.x * 16U + rr, g = s0 + gl;
                d[(gl * 2U + (rr & 1U)) * 8U + (rr >> 1U)] =
                    r < v.rows && g < groups ? deltas[static_cast<std::size_t>(r) * (v.x_stride / 32U) + g] : 0.0F;
            }
            __syncthreads();
        }
        // Plain unrolled loop with continue, as the split-v2 wide kernel.
#pragma unroll
        for (unsigned j = 0; j < wide_classes; ++j) {
            const unsigned g = s0 + w + 4U * j;
            if (g >= groups) continue;
            v4i a0{0, 0, 0, 0}, a1{0, 0, 0, 0};
            float dl = 0.0F;
            if (live) {
                const auto* x = reinterpret_cast<const int4*>(arow + 32U * g);
                const int4 x0 = x[0], x1 = x[1];
                a0 = v4i{x0.x, x0.y, x0.z, x0.w};
                a1 = v4i{x1.x, x1.y, x1.z, x1.w};
                if constexpr (!Staged) dl = drow[g];
            }
            float rd[8];
            if constexpr (Staged) {
                const float4 d0 = dsh[g - s0][h][0], d1 = dsh[g - s0][h][1];
                rd[0] = d0.x; rd[1] = d0.y; rd[2] = d0.z; rd[3] = d0.w; rd[4] = d1.x; rd[5] = d1.y; rd[6] = d1.z; rd[7] = d1.w;
            }
#pragma unroll
            for (unsigned t = 0; t < Outputs; ++t) {
                const auto raw = Operand::load_raw(wrow[t], g);
                const v4i b0{static_cast<int>(raw.q[0]), static_cast<int>(raw.q[1]), static_cast<int>(raw.q[2]), static_cast<int>(raw.q[3])}, b1{static_cast<int>(raw.q[4]), static_cast<int>(raw.q[5]), static_cast<int>(raw.q[6]), static_cast<int>(raw.q[7])};
                const float scale = ad.half_exact(raw.d);
                v8i c = integer_wmma<8>(a0, b0, Staged ? magic : zero);
                c = integer_wmma<8>(a1, b1, c);
#pragma unroll
                for (unsigned e = 0; e < 8; ++e) {
                    if constexpr (Staged) {
                        const float dot = __fsub_rn(__int_as_float(c[e]), 0x1.8p23F);
                        acc[j][t][e] = __fadd_rn(acc[j][t][e], __fmul_rn(dot, __fmul_rn(scale, rd[e])));
                    } else {
                        const float rdl = __shfl(dl, static_cast<int>(2U * e + h), 32);
                        acc[j][t][e] = acc[j][t][e] + static_cast<float>(c[e]) * (scale * rdl);
                    }
                }
            }
        }
    }
#pragma unroll
    for (unsigned t = 0; t < Outputs; ++t)
#pragma unroll
        for (unsigned e = 0; e < 8; ++e) {
            const float s = ((acc[0][t][e] + acc[4][t][e]) + (acc[2][t][e] + acc[6][t][e])) +
                            ((acc[1][t][e] + acc[5][t][e]) + (acc[3][t][e] + acc[7][t][e]));
            part[w][t][e][l] = s;
        }
    __syncthreads();
    if constexpr (Outputs == 2U) {
        const unsigned t = w / 2U, o = o0 + t * 16U + lo;
#pragma unroll
        for (unsigned q = 0; q < 4; ++q) {
            const unsigned e = 4U * (w % 2U) + q, r = blockIdx.x * 16U + 2U * e + h;
            const float y = (part[0][t][e][l] + part[2][t][e][l]) + (part[1][t][e][l] + part[3][t][e][l]);
            if (o < v.count && r < v.rows) {
                float* dst = out + static_cast<std::size_t>(r) * v.out_stride + o;
                *dst = v.accumulate ? *dst + y : y;
            }
        }
    } else {
    // Disjoint stores of the tile. The reduction for each element is unchanged.
    for (unsigned item = threadIdx.x; item < Outputs * 8U * 32U; item += 32U * wide_waves) {
        const unsigned t = item / (8U * 32U), e = (item / 32U) % 8U, ln = item % 32U;
        const unsigned o = o0 + t * 16U + (ln & 15U), r = blockIdx.x * 16U + 2U * e + (ln >> 4U);
        const float y = (part[0][t][e][ln] + part[2][t][e][ln]) + (part[1][t][e][ln] + part[3][t][e][ln]);
        if (o < v.count && r < v.rows) {
            float* dst = out + static_cast<std::size_t>(r) * v.out_stride + o;
            *dst = v.accumulate ? *dst + y : y;
        }
    }
    }
