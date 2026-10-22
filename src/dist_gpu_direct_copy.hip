// Scatter / gather of one GPU-direct exchange between the caller's device buffer and the per-stream arena slots,
// one launch each, with the in-band header: identity of the exchange and a checksum of every 4-byte word.
#include "strix/dist/gpu_direct_copy.hpp"

#include <hip/hip_runtime.h>

#include <stdexcept>
#include <string>

namespace strix::dist {
namespace {

// The RX arena is read through a DMA-BUF mapping that is slow per access (the gather ran at ~1.2 GB/s with one
// 4-byte load per thread in 64 blocks): 16-byte accesses, about one per thread, up to 256 blocks per part.
constexpr unsigned threads = 256, max_blocks = 256;

// Word i of the message contributes (w ^ golden(i)) * (2i + 1): order independent, sensitive to shifts.
__device__ __forceinline__ unsigned long long mix(unsigned w, unsigned long long i) {
    return static_cast<unsigned long long>(w ^ static_cast<unsigned>(i * 0x9E3779B1ULL)) * (2ULL * i + 1ULL);
}

// Blocks x parts; the last block to finish (over the whole grid) handles the header.
template <bool Scatter, bool Vector>
__global__ __launch_bounds__(threads) void exchange_kernel(GpuDirectParts parts, unsigned* __restrict__ message, GpuDirectIdentity id,
                                                           GpuDirectScratch* __restrict__ scratch, const unsigned long long* flag,
                                                           GpuDirectError* error, unsigned long long* produced) {
    __shared__ unsigned long long block_sum[threads];
    const unsigned j = blockIdx.y;
    unsigned long long sum = 0;
    if constexpr (!Scatter) {
        // One load of the (fine-grained, uncached) flag per block, not per thread.
        __shared__ bool failed;
        if (threadIdx.x == 0) failed = *reinterpret_cast<const volatile unsigned long long*>(flag) == ~0ULL;
        __syncthreads();
        if (failed) {
            if (threadIdx.x == 0 && blockIdx.x == 0 && j == 0) {
                error->sequence = id.sequence;
                error->tag = id.tag;
                error->layer = id.layer;
                __threadfence_system();
                atomicCAS(&error->code, 0U, 1U);
                __threadfence_system();
            }
            __builtin_trap();
        }
    }
    if (j < parts.count) {
        auto* slot = static_cast<unsigned*>(parts.slot[j]);
        // Only the last part may end inside a word: its last word travels zero-padded, and only its real bytes are
        // read from / written to the message.
        const unsigned long long first = parts.offset[j] / 4U, words = (parts.bytes[j] + 3U) / 4U;
        const unsigned tail = static_cast<unsigned>(parts.bytes[j] % 4U);
        // Whole 16-byte groups first (offsets are multiples of 64 B; the message is 16-byte aligned on this path),
        // then the last 0-3 words (the final one possibly partial) one by one.
        unsigned long long scalar_from = 0;
        if constexpr (Vector) {
            const unsigned long long groups = (tail ? words - 1U : words) / 4U;
            scalar_from = groups * 4U;
            auto* slot4 = reinterpret_cast<uint4*>(slot);
            auto* message4 = reinterpret_cast<uint4*>(message + first);
            for (unsigned long long g = static_cast<unsigned long long>(blockIdx.x) * threads + threadIdx.x; g < groups;
                 g += static_cast<unsigned long long>(gridDim.x) * threads) {
                uint4 v;
                if constexpr (Scatter) { v = message4[g]; slot4[g] = v; }
                else { v = slot4[g]; message4[g] = v; }
                const unsigned long long i = first + 4U * g;
                sum += mix(v.x, i) + mix(v.y, i + 1U) + mix(v.z, i + 2U) + mix(v.w, i + 3U);
            }
        }
        for (unsigned long long i = scalar_from + static_cast<unsigned long long>(blockIdx.x) * threads + threadIdx.x; i < words;
             i += static_cast<unsigned long long>(gridDim.x) * threads) {
            unsigned w;
            if (tail && i + 1U == words) {
                auto* m = reinterpret_cast<unsigned char*>(message + first + i);
                if constexpr (Scatter) {
                    w = 0U;
                    for (unsigned b = 0; b < tail; ++b) w |= static_cast<unsigned>(m[b]) << (8U * b);
                    slot[i] = w;
                } else {
                    w = slot[i];
                    for (unsigned b = 0; b < tail; ++b) m[b] = static_cast<unsigned char>(w >> (8U * b));
                }
            } else if constexpr (Scatter) { w = message[first + i]; slot[i] = w; }
            else { w = slot[i]; message[first + i] = w; }
            sum += mix(w, first + i);
        }
    }
    block_sum[threadIdx.x] = sum;
    __syncthreads();
    for (unsigned s = threads / 2U; s; s /= 2U) {
        if (threadIdx.x < s) block_sum[threadIdx.x] += block_sum[threadIdx.x + s];
        __syncthreads();
    }
    if (threadIdx.x != 0) return;
    atomicAdd(&scratch->sum, block_sum[0]);
    // System scope on the scatter: the last block's `produced` store then implies that every block's slot writes are
    // visible outside the GPU (the controller reads the slots).
    if constexpr (Scatter) __threadfence_system();
    else __threadfence();
    const unsigned total = gridDim.x * gridDim.y;
    if (atomicAdd(&scratch->blocks_done, 1U) + 1U != total) return;
    // Last block: every word is accounted for.
    __threadfence();
    const unsigned long long checksum = atomicAdd(&scratch->sum, 0ULL);
    scratch->sum = 0;
    scratch->blocks_done = 0;
    auto* header = reinterpret_cast<GpuDirectHeader*>(static_cast<unsigned char*>(parts.slot[parts.count - 1U]) +
                                                      (parts.bytes[parts.count - 1U] + 3U) / 4U * 4U);
    if constexpr (Scatter) {
        header->magic = gpu_direct_header_magic;
        header->sequence = id.sequence;
        header->plan = id.plan;
        header->request = id.request;
        header->bytes = id.bytes;
        header->checksum = checksum;
        header->tag = id.tag;
        header->layer = id.layer;
        header->reserved = 0;
        __threadfence_system();
        // The CPU polls this (host-mapped) instead of querying the producer event in a loop: no HIP call while the
        // caller's thread launches work. The event is still checked once before SUBMIT.
        if (produced) *reinterpret_cast<volatile unsigned long long*>(produced) = id.sequence;
    } else {
        const GpuDirectHeader h = *header;
        const unsigned code = h.magic != gpu_direct_header_magic ? 2U : h.sequence != id.sequence ? 3U : h.plan != id.plan ? 4U
            : h.request != id.request ? 5U : h.tag != id.tag ? 6U : h.layer != id.layer ? 7U : h.bytes != id.bytes ? 8U
            : h.checksum != checksum ? 9U : 0U;
        if (code) {
            error->sequence = id.sequence;
            error->tag = id.tag;
            error->layer = id.layer;
            error->peer_tag = h.tag;
            error->peer_layer = h.layer;
            __threadfence_system();
            atomicCAS(&error->code, 0U, code);
            __threadfence_system();
            __builtin_trap();
        }
    }
}

template <bool Scatter, bool Vector>
void launch_kernel(const GpuDirectParts& parts, void* message, const GpuDirectIdentity& id, GpuDirectScratch* scratch, const std::uint64_t* flag,
                   GpuDirectError* error, std::uint64_t* produced, hipStream_t stream, unsigned blocks) {
    hipLaunchKernelGGL(HIP_KERNEL_NAME(exchange_kernel<Scatter, Vector>), dim3(blocks, parts.count), dim3(threads), 0, stream, parts,
                       static_cast<unsigned*>(message), id, scratch, reinterpret_cast<const unsigned long long*>(flag), error,
                       reinterpret_cast<unsigned long long*>(produced));
}
template <bool Scatter>
void launch(const GpuDirectParts& parts, void* message, const GpuDirectIdentity& id, GpuDirectScratch* scratch, const std::uint64_t* flag,
            GpuDirectError* error, std::uint64_t* produced, hipStream_t stream) {
    if (!parts.count || parts.count > 8U || reinterpret_cast<std::uintptr_t>(message) % 4U) throw std::invalid_argument("gpu-direct copy: parts");
    std::size_t largest = 0;
    for (unsigned j = 0; j < parts.count; ++j) {
        if (parts.offset[j] % 4U || (j + 1U < parts.count && parts.bytes[j] % 4U) || reinterpret_cast<std::uintptr_t>(parts.slot[j]) % 4U)
            throw std::invalid_argument("gpu-direct copy: every part but the last must be whole 4-byte words");
        largest = parts.bytes[j] > largest ? parts.bytes[j] : largest;
    }
    const bool vector = reinterpret_cast<std::uintptr_t>(message) % 16U == 0U;
    const std::size_t units = vector ? (largest + 15U) / 16U : (largest + 3U) / 4U; // accesses per part
    const std::size_t want = (units + threads - 1U) / threads;
    const unsigned blocks = static_cast<unsigned>(want < 1U ? 1U : want > max_blocks ? max_blocks : want);
    if (vector) launch_kernel<Scatter, true>(parts, message, id, scratch, flag, error, produced, stream, blocks);
    else launch_kernel<Scatter, false>(parts, message, id, scratch, flag, error, produced, stream, blocks);
    const auto status = hipGetLastError();
    if (status != hipSuccess) throw std::runtime_error(std::string("gpu-direct copy launch: ") + hipGetErrorString(status));
}

} // namespace

void enqueue_scatter(const GpuDirectParts& parts, const void* message, const GpuDirectIdentity& identity, GpuDirectScratch* scratch,
                     std::uint64_t* produced, hipStream_t stream) {
    launch<true>(parts, const_cast<void*>(message), identity, scratch, nullptr, nullptr, produced, stream);
}
void enqueue_gather(const GpuDirectParts& parts, void* message, const GpuDirectIdentity& identity, GpuDirectScratch* scratch,
                    const std::uint64_t* flag, GpuDirectError* error, hipStream_t stream) {
    launch<false>(parts, message, identity, scratch, flag, error, nullptr, stream);
}
namespace {
// Read-path diagnostics: mode 0 copies (16-byte accesses), mode 1 only reads and sums, mode 2 copies and sums (the
// gather's work without the header).
__global__ __launch_bounds__(threads) void read_test_kernel(const uint4* __restrict__ src, uint4* __restrict__ dst, unsigned long long groups,
                                                            int mode, unsigned long long* sink) {
    unsigned long long sum = 0;
    for (unsigned long long g = static_cast<unsigned long long>(blockIdx.x) * threads + threadIdx.x; g < groups;
         g += static_cast<unsigned long long>(gridDim.x) * threads) {
        const uint4 v = src[g];
        if (mode != 1) dst[g] = v;
        if (mode != 0) {
            const unsigned long long i = 4U * g;
            sum += mix(v.x, i) + mix(v.y, i + 1U) + mix(v.z, i + 2U) + mix(v.w, i + 3U);
        }
    }
    if (mode != 0 && sum == 0x123456789ULL) *sink = sum; // keeps the reads live
}
} // namespace

void enqueue_read_test(const void* src, void* dst, std::size_t bytes, int mode, unsigned blocks, void* sink, hipStream_t stream) {
    hipLaunchKernelGGL(read_test_kernel, dim3(blocks), dim3(threads), 0, stream, static_cast<const uint4*>(src), static_cast<uint4*>(dst),
                       static_cast<unsigned long long>(bytes / 16U), mode, static_cast<unsigned long long*>(sink));
    const auto status = hipGetLastError();
    if (status != hipSuccess) throw std::runtime_error(std::string("gpu-direct read test launch: ") + hipGetErrorString(status));
}

const char* gpu_direct_error_text(std::uint32_t code) noexcept {
    switch (code) {
    case 1: return "the exchange had already failed on the host side";
    case 2: return "the received header is not a GPU-direct header (magic)";
    case 3: return "the peer sent a different sequence";
    case 4: return "the peer runs a different plan";
    case 5: return "the peer is on a different request";
    case 6: return "the peer sent a different collective (tag)";
    case 7: return "the peer sent a different layer";
    case 8: return "the peer sent a different size";
    case 9: return "the payload checksum differs (data corrupted in transit or not visible to the GPU)";
    default: return "unknown";
    }
}

} // namespace strix::dist
