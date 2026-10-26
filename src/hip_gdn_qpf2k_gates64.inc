// qpf2k GDN gates, wave64 form (gdn-dense; compile ONLY in a -mwavefrontsize64 translation unit).
// gdn_qpf2k_gates_kernel's values (production gdn_prefill_gates_kernel: per (row, output) 256 chains t = 0..255, chain t
// the contracted fmaf over features t, t + 256, ... from +0, then the tree pairing t with t + 128, + 64, ..., + 1, lower
// operand first, then the sigmoid / softplus transform) with one wave64 per O outputs: lane l holds chains l + 64 m
// (m = 0..3) of each output, so a staged x value read from LDS feeds O x 1 FMAs per lane and every lane runs half the
// chains of the wave32 form. Tree: levels 128 and 64 in-lane (m with m + 2, then 0 with 1), level 32 = lane l + 32 into
// lane l (v_permlane64), levels 16..1 the wave32 helpers (permlanex16, DPP row_shl) inside lanes 0..31. Lane 0 applies
// the transform. Every add is gdn_window_add(lower, upper): the production operands in the production order.
namespace qpf2k_gdn64 {
template <unsigned O, unsigned W, unsigned RC, unsigned RB, unsigned Width = 2560U>
__global__ __launch_bounds__(W * 64U)
void gdn_qpf2k_gates64_kernel(const std::uint16_t* input, std::uint64_t input_stride, const float* beta_weights,
    const float* alpha_weights, GdnPrefillRows rows, const void* dt, SmallWeightKind dt_kind, const void* a,
    SmallWeightKind a_kind, std::uint32_t heads) {
    // wave64 translation unit only (the build of this file sets -mwavefrontsize64)
    constexpr unsigned Steps = Width / 256U, Threads = W * 64U;
    constexpr unsigned Units = RC * Width / 8U, UnitsPerThread = (Units + Threads - 1U) / Threads;  // 8 BF16 per unit
    static_assert(Width % 256U == 0U && RB % RC == 0U);
    __shared__ __align__(16) float xs[RC][Width];
    const unsigned tid = threadIdx.x, lane = tid % 64U, wave = tid / 64U;
    const unsigned g0 = (blockIdx.x * W + wave) * O, r0 = blockIdx.y * RB;
    float w[O][Steps][4];
#pragma unroll
    for (unsigned o = 0; o < O; ++o) {
        const unsigned g = g0 + o, out = g >= heads ? g - heads : g;
        const float* weights = (g >= heads ? alpha_weights : beta_weights) + static_cast<std::uint64_t>(out) * Width;
#pragma unroll
        for (unsigned k = 0; k < Steps; ++k)
#pragma unroll
            for (unsigned m = 0; m < 4U; ++m) w[o][k][m] = weights[k * 256U + m * 64U + lane];
    }
    for (unsigned first = r0; first < r0 + RB; first += RC) {
        __syncthreads();
#pragma unroll
        for (unsigned i = 0; i < UnitsPerThread; ++i) {
            const unsigned u = tid + i * Threads;
            if (Units % Threads == 0U || u < Units) {
                const uint4 staged = *reinterpret_cast<const uint4*>(input + (first + u / (Width / 8U)) * input_stride + u % (Width / 8U) * 8U);
                const unsigned r = u / (Width / 8U), f = u % (Width / 8U) * 8U;
                const unsigned words[4] = {staged.x, staged.y, staged.z, staged.w};
                float4 lo, hi;  // bf16_to_float of the even (low) and odd (high) halves
                lo.x = __uint_as_float(words[0] << 16U); lo.y = __uint_as_float(words[0] & 0xffff0000U);
                lo.z = __uint_as_float(words[1] << 16U); lo.w = __uint_as_float(words[1] & 0xffff0000U);
                hi.x = __uint_as_float(words[2] << 16U); hi.y = __uint_as_float(words[2] & 0xffff0000U);
                hi.z = __uint_as_float(words[3] << 16U); hi.w = __uint_as_float(words[3] & 0xffff0000U);
                *reinterpret_cast<float4*>(&xs[r][f]) = lo;
                *reinterpret_cast<float4*>(&xs[r][f + 4U]) = hi;
            }
        }
        __syncthreads();
#pragma unroll 1
        for (unsigned r = 0; r < RC; ++r) {
            float sum[O][4];
#pragma unroll
            for (unsigned o = 0; o < O; ++o)
#pragma unroll
                for (unsigned m = 0; m < 4U; ++m) sum[o][m] = 0.0F;
#pragma unroll
            for (unsigned k = 0; k < Steps; ++k) {
                float x[4];
#pragma unroll
                for (unsigned m = 0; m < 4U; ++m) x[m] = xs[r][k * 256U + m * 64U + lane];
#pragma unroll
                for (unsigned o = 0; o < O; ++o)
#pragma unroll
                    for (unsigned m = 0; m < 4U; ++m) sum[o][m] = __builtin_fmaf(x[m], w[o][k][m], sum[o][m]);
            }
#pragma unroll
            for (unsigned o = 0; o < O; ++o) {
                float* v = sum[o];
                v[0] = gdn_window_add(v[0], v[2]); v[1] = gdn_window_add(v[1], v[3]);   // level 128
                float total = gdn_window_add(v[0], v[1]);                                // level 64
                total = gdn_window_add(total, __uint_as_float(static_cast<unsigned>(
                    __builtin_amdgcn_permlane64(static_cast<int>(__float_as_uint(total))))));   // level 32: lane l + 32
                total = gdn_window_add(total, gdn_window_down<16U>(total));
                total = gdn_window_add(total, gdn_window_down<8U>(total));
                total = gdn_window_add(total, gdn_window_down<4U>(total));
                total = gdn_window_add(total, gdn_window_down<2U>(total));
                total = gdn_window_add(total, gdn_window_down<1U>(total));
                if (lane == 0U) {
                    const unsigned g = g0 + o, row = first + r;
                    const bool alpha = g >= heads;
                    const unsigned out = alpha ? g - heads : g;
                    const float raw = round_bf16(total);
                    const auto offset = row * rows.stride + out;
                    if (alpha) {
                        const float biased = raw + load_small(dt, dt_kind, out);
                        rows.gate[offset] = load_small(a, a_kind, out) * softplus(biased);
                    } else rows.beta[offset] = round_bf16(1.0F / (1.0F + expf(-raw)));
                }
            }
        }
    }
}
}  // namespace qpf2k_gdn64
