// Shared scalar MMA tile schedule and descriptor-row stripe.
    constexpr int U = F == WeightFormat::bf16 ? 4 : 8;
    constexpr std::size_t kRed = (FUSED ? 2 : 1) * 8 * NT * 128 * sizeof(float), kStg = F == WeightFormat::q8_0 ? 8 * 272 * 16 : 0;
    __shared__ __align__(16) unsigned char smem[kRed > kStg ? kRed : kStg];
    float(*red)[8][NT * 128] = reinterpret_cast<float(*)[8][NT * 128]>(smem);
    uint4* stg = reinterpret_cast<uint4*>(smem) + warp_id_of(threadIdx.x) * 272;
    pdl_trigger();
    const std::uint32_t tile = blockIdx.x;
    const void* wsel = ma.w[0];
    float* y = ma.y[0];
    std::uint32_t start = 0, out = ma.out[0];
#pragma unroll
    for (int j = 0; j < kMmaMaxMulti - 1; ++j) {
        if (tile >= ma.tile_end[j]) {
            start = ma.tile_end[j];
            wsel = ma.w[j + 1];
            y = ma.y[j + 1];
            out = ma.out[j + 1];
        }
    }
    const std::uint32_t o0 = (tile - start) * 16;
    const int warp = threadIdx.x >> 5, lane = threadIdx.x & 31, g = lane >> 2, qd = lane & 3;
    STRIX_MMA_TILE_BODY
