// Frozen from src/cuda/models/minimax/mm_rtx_kernels.cu:115-136 at 453b063b1.
// The launch and row bounds live in the fixture so the row body can also run on the host.
namespace frozen {

inline constexpr unsigned kExperts = 128U, kTopE = 4U;
inline constexpr float kSumEps = 1e-20F, kRouteScale = 2.0F;

#if defined(__CUDACC__)
__host__ __device__
#endif
inline void router_top4_row(const float* scratch, std::size_t r, std::uint32_t* ids, float* weights) {
    const float* sc = scratch + r * kExperts * 2;
    std::uint32_t pick[kTopE];
    for (unsigned k = 0; k < kTopE; ++k) {  // (c desc, id asc), excluding the earlier picks
        std::uint32_t best = 0xffffffffU;
        for (unsigned e = 0; e < kExperts; ++e) {
            bool taken = false;
            for (unsigned j = 0; j < k; ++j) taken |= pick[j] == e;
            if (taken) continue;
            if (best == 0xffffffffU || sc[e * 2 + 1] > sc[best * 2 + 1]) best = e;
        }
        pick[k] = best;
    }
    float sum = 0.0F;
    for (unsigned k = 0; k < kTopE; ++k) sum = sum + sc[pick[k] * 2];
    for (unsigned k = 0; k < kTopE; ++k) {
        ids[r * kTopE + k] = pick[k];
        weights[r * kTopE + k] = (sc[pick[k] * 2] / (sum + kSumEps)) * kRouteScale;
    }
}

}  // namespace frozen
