#pragma once

namespace strix::moe_core::microscaled_r1 {

// A CUDA warp fetches a 128-byte code run per output as 32 consecutive words.
// Each FMA lane receives its original pair for every stripe in K order.
template <unsigned CodeBytesPerPair, unsigned ScaleColumns>
struct CudaRun {
    static constexpr unsigned kLanes = 32U;
    static constexpr unsigned kWordBytes = 4U;
    static constexpr unsigned kRunBytes = kLanes * kWordBytes;
    static constexpr unsigned kPairsPerWord = kWordBytes / CodeBytesPerPair;
    static constexpr unsigned kStripeColumns = kLanes * 2U;
    static constexpr unsigned kColumns = kPairsPerWord * kStripeColumns;
    static constexpr unsigned kScales = kColumns / ScaleColumns;
    static_assert(kScales <= kLanes && (kScales & (kScales - 1U)) == 0U);

#if defined(__CUDACC__)
    __host__ __device__
#endif
    static constexpr unsigned code_lane(unsigned stripe, unsigned lane) {
        return (stripe * kLanes + lane) / kPairsPerWord;
    }

#if defined(__CUDACC__)
    __host__ __device__
#endif
    static constexpr unsigned code_shift(unsigned lane) {
        return lane % kPairsPerWord * CodeBytesPerPair * 8U;
    }

#if defined(__CUDACC__)
    __host__ __device__
#endif
    static constexpr unsigned scale_lane(unsigned stripe, unsigned lane) {
        return (stripe * kStripeColumns + 2U * lane) / ScaleColumns;
    }
};

} // namespace strix::moe_core::microscaled_r1
