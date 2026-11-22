// NVFP4 register decoder and activation staging. This body is included in
// both CUDA entry points to hold the legacy kernel's lowering fixed.
    const std::uint32_t ks = J.ks, out = J.out;
    if (blockIdx.y >= ks || (blockIdx.z == 1 && J.w[1] == nullptr)) return;  // block-uniform
    const std::uint32_t bx = blockIdx.x - J.g0;
    const float gs0 = J.gs[0];
    const std::uint32_t kidx = blockIdx.y;
    const unsigned char* w = J.w[blockIdx.z];
    const std::uint32_t nch = in >> 7, ntiles = out >> 4;
    const std::uint32_t c0 = static_cast<std::uint32_t>(static_cast<std::uint64_t>(kidx) * nch / ks);
    const std::uint32_t c1 = static_cast<std::uint32_t>(static_cast<std::uint64_t>(kidx + 1) * nch / ks);
    bool on[MT];
    const unsigned char* base[MT];
#pragma unroll
    for (int mt = 0; mt < MT; ++mt) {
        const std::uint32_t m = (bx * kWarpsPerBlock + warp) * MT + mt;
        on[mt] = m < ntiles;
        base[mt] = w + static_cast<std::size_t>(on[mt] ? m : 0) * nch * kNvfp4RecordBytes;
    }
    float acc[MT][RB][4];
#pragma unroll
    for (int mt = 0; mt < MT; ++mt)
#pragma unroll
        for (int rb = 0; rb < RB; ++rb) acc[mt][rb][0] = acc[mt][rb][1] = acc[mt][rb][2] = acc[mt][rb][3] = 0.f;

    // Before pdl_wait (constant data only): record c0 of each tile goes to registers (`cur`) and record c0 + 1, when this
    // block has one, is requested into L2 (lanes 0..8 = its 9 lines; no register). A full resident wave then holds at most
    // 2 records per tile in flight (<= ~21 MB of L2 even for ks == 1 shapes).
    {
        constexpr std::uint32_t kLines = kNvfp4RecordBytes / 128;
        static_assert(kNvfp4RecordBytes % 128 == 0, "records are whole 128-byte lines");
        if (c0 + 1 < c1 && static_cast<std::uint32_t>(lane) < kLines) {
#pragma unroll
            for (int mt = 0; mt < MT; ++mt)
                if (on[mt])
                    prefetch_l2(base[mt] + static_cast<std::size_t>(c0 + 1) * kNvfp4RecordBytes +
                                static_cast<std::size_t>(lane) * 128);
        }
    }
    Nvfp4Decoder::Raw cur[MT], nxt[MT];
#pragma unroll
    for (int mt = 0; mt < MT; ++mt)
        if (on[mt] && c0 < c1) cur[mt] = Nvfp4Decoder::load(base[mt] + static_cast<std::size_t>(c0) * kNvfp4RecordBytes, lane);
    const PdlWaited waited = pdl_wait();
    pdl_trigger();
    const float* const x = xa(waited);
    float* const part = J.part(waited);
    float* const y = J.y(waited);
    const float* const resid = J.resid(waited);
    for (std::uint32_t c = c0; c < c1; ++c) {
        const std::uint32_t cs = (c - c0) % kSlab;
        if (cs == 0) {
            __syncthreads();
            const std::uint32_t nc = min(static_cast<std::uint32_t>(kSlab), c1 - c);
            const std::uint32_t total = nc * RB * 8 * 32;
            for (std::uint32_t e = threadIdx.x; e < total; e += kWarpsPerBlock * 32) {
                const std::uint32_t ln = e & 31, j = (e >> 5) & 7, rb = (e >> 8) % RB, cc_ = e / (256 * RB);
                const std::uint32_t n = rb * 8 + (ln >> 2), k0 = ((c + cc_) << 7) + j * 16 + (ln & 3) * 2;
                float2 p0 = make_float2(0.f, 0.f), p1 = p0;
                if (n < rows) {
                    const float* xr = x + static_cast<std::size_t>(n) * in + k0;
                    p0 = *reinterpret_cast<const float2*>(xr);
                    p1 = *reinterpret_cast<const float2*>(xr + 8);
                }
                xs_dyn[(cc_ * RB + rb) * 256 + j * 32 + ln] = make_uint2(pack_f16x2(p0.x, p0.y), pack_f16x2(p1.x, p1.y));
            }
            __syncthreads();
        }
        if (c + 1 < c1) {
#pragma unroll
            for (int mt = 0; mt < MT; ++mt)
                if (on[mt]) nxt[mt] = Nvfp4Decoder::load(base[mt] + static_cast<std::size_t>(c + 1) * kNvfp4RecordBytes, lane);
        }
        Nvfp4Decoder::compute_chunk<RB, MT>(cur, on, xs_dyn + cs * RB * 256, lane, acc);
#pragma unroll
        for (int mt = 0; mt < MT; ++mt) cur[mt] = nxt[mt];
    }
#include "rows_fp4_store.cuh"
