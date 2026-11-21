#include "strix/cuda/token_sampling_device.hpp"

#include <cuda_runtime.h>

#include <stdexcept>
#include <string>

namespace strix {
namespace {

using u64 = unsigned long long;
using Stream = cudaStream_t;
using Event = cudaEvent_t;
constexpr auto host_alloc_flags = cudaHostAllocMapped;

// Windows nvcc cannot use __int128 with its MSVC host compiler. The two words
// carry the same unsigned fixed-point bits across all device sums and comparisons.
struct alignas(16) u128 {
    u64 lo, hi;

    u128() = default;
    __device__ __forceinline__ u128(u64 low, u64 high = 0) : lo(low), hi(high) {}
    __device__ __forceinline__ explicit operator u64() const { return lo; }

    __device__ __forceinline__ u128& operator+=(u128 other) {
        lo += other.lo;
        hi += other.hi + static_cast<u64>(lo < other.lo);
        return *this;
    }
    __device__ __forceinline__ u128& operator-=(u128 other) {
        hi -= other.hi + static_cast<u64>(lo < other.lo);
        lo -= other.lo;
        return *this;
    }
};

__device__ __forceinline__ u128 operator+(u128 lhs, u128 rhs) { return lhs += rhs; }
__device__ __forceinline__ u128 operator-(u128 lhs, u128 rhs) { return lhs -= rhs; }
__device__ __forceinline__ bool operator==(u128 lhs, u128 rhs) { return lhs.hi == rhs.hi && lhs.lo == rhs.lo; }
__device__ __forceinline__ bool operator>(u128 lhs, u128 rhs) {
    return lhs.hi > rhs.hi || (lhs.hi == rhs.hi && lhs.lo > rhs.lo);
}
__device__ __forceinline__ bool operator>=(u128 lhs, u128 rhs) { return !(rhs > lhs); }
__device__ __forceinline__ u128 operator<<(u128 x, unsigned bits) {
    if (bits == 0U) return x;
    if (bits >= 64U) return {0, x.lo << (bits - 64U)};
    return {x.lo << bits, (x.hi << bits) | (x.lo >> (64U - bits))};
}
__device__ __forceinline__ u128 operator>>(u128 x, unsigned bits) {
    if (bits == 0U) return x;
    if (bits >= 64U) return {x.hi >> (bits - 64U), 0};
    return {(x.lo >> bits) | (x.hi << (64U - bits)), x.hi >> bits};
}
__device__ __forceinline__ u128 operator|(u128 lhs, u128 rhs) { return {lhs.lo | rhs.lo, lhs.hi | rhs.hi}; }

void check(cudaError_t error, const char* message) {
    if (error != cudaSuccess) throw std::runtime_error(std::string(message) + ": " + cudaGetErrorString(error));
}

template <typename T>
T* mapped_device_pointer(T* pointer) {
    if (!pointer) return nullptr;
    void* mapped = nullptr;
    check(cudaHostGetDevicePointer(&mapped, pointer, 0U), "map device sampler host buffer");
    return static_cast<T*>(mapped);
}

void read_logits_row(const ExecutionBufferAddress& row, std::uint32_t vocabulary, void* stream, std::vector<float>& out) {
    out.resize(vocabulary);
    const auto s = static_cast<Stream>(stream);
    check(cudaMemcpyAsync(out.data(), reinterpret_cast<const void*>(row.address),
        out.size() * sizeof(float), cudaMemcpyDeviceToHost, s), "sampled logits readback");
    check(cudaStreamSynchronize(s), "sampled logits readback completion");
}

} // namespace
} // namespace strix

#define STRIX_SAMPLE_TARGET CudaTokenSamplerDevice
#define STRIX_SAMPLE_DRAFT CudaTokenDraftSamplerDevice
#define STRIX_SAMPLE_HOST_DRAFT cuda_host_token_draft
#define STRIX_SAMPLE_TEST_DECIDE cuda_decide_token_for_testing
#define STRIX_SAMPLE_TEST_DRAFT cuda_draft_for_testing
#define STRIX_SAMPLE_RT(name) cuda##name
#define STRIX_SAMPLE_HOST_ALLOC cudaHostAlloc
#define STRIX_SAMPLE_HOST_FREE cudaFreeHost
#define STRIX_SAMPLE_LAUNCH(kernel, grid, block, shared, stream, ...) \
    kernel<<<grid, block, shared, stream>>>(__VA_ARGS__)

#include "../../core/token_sampling_device_body.inl"
