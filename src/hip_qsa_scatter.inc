template <unsigned D> __device__ __forceinline__ float qsa_lane_xor(float v) {
    // Distance 16: permlanex16 with the identity selects swaps the two 16-lane rows (lane L ^ 16).
    if constexpr (D == 16U) return __uint_as_float(__builtin_amdgcn_permlanex16(__float_as_uint(v), __float_as_uint(v),
        0x76543210U, 0xfedcba98U, false, false));
    else return __int_as_float(__builtin_amdgcn_mov_dpp(__float_as_int(v), 0x160 | static_cast<int>(D), 0xf, 0xf, false));
}
// Reduce-scatter of P per-lane partial sums (P a power of two <= 16): lane step D (16, 8, ...)
// halves the values a lane carries, the lower lane keeping the lower half. Every head's sum
// is still the 16..1 lane tree of its partials (each step adds the partials of lanes L and
// L ^ D; one addition commutes), at a fraction of the exchanges. Afterwards lane L holds
// head qsa_scatter_head<P>(L) (all lanes of a pair hold the same value).
template <unsigned P> __device__ __forceinline__ unsigned qsa_scatter_head(unsigned lane) {
    unsigned head = 0;
    unsigned weight = P / 2U;
#pragma unroll
    for (unsigned D = 16U; D > 0U && weight > 0U; D >>= 1U, weight >>= 1U) if (lane & D) head += weight;
    return head;
}
template <unsigned P, unsigned Step = 0U> __device__ __forceinline__ void qsa_scatter(float (&v)[P], unsigned lane) {
    if constexpr (Step < 5U) {
        constexpr unsigned D = 16U >> Step, Count = (P >> Step) > 0U ? (P >> Step) : 1U;
        if constexpr (Count > 1U) {
            constexpr unsigned H = Count / 2U;
            const bool upper = (lane & D) != 0U;
#pragma unroll
            for (unsigned j = 0; j < H; ++j) {
                const float low = qsa_lane_xor<D>(v[j]), high = qsa_lane_xor<D>(v[j + H]);
                v[j] = upper ? v[j + H] + high : v[j] + low;
            }
        } else v[0] += qsa_lane_xor<D>(v[0]);
        qsa_scatter<P, Step + 1U>(v, lane);
    }
}
