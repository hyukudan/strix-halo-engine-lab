#pragma once

#include <hip/hip_runtime.h>

#include <cstddef>
#include <cstdint>

namespace strix::dist {

// The parts of one exchange: part j is bytes [offset[j], offset[j] + bytes[j]) of the message and lives at
// slot[j] (a device pointer into stream j's arena). Every part but the last is whole 4-byte words; the last one's
// final word travels zero-padded, and the 64-byte header follows it (at the last part's bytes rounded up to 4).
struct GpuDirectParts {
    unsigned count = 0;
    void* slot[8] = {};
    std::size_t offset[8] = {};
    std::size_t bytes[8] = {};
};

// The identity of one exchange, carried in-band after the payload and checked by the receiver's GPU.
struct GpuDirectIdentity {
    std::uint64_t sequence = 0, plan = 0, request = 0, bytes = 0;
    std::uint32_t tag = 0, layer = 0;
};
inline constexpr std::uint64_t gpu_direct_header_magic = 0x31444750434e5953ULL; // "SYNCPGD1"
inline constexpr std::size_t gpu_direct_header_bytes = 64;
struct GpuDirectHeader {
    std::uint64_t magic, sequence, plan, request, bytes, checksum;
    std::uint32_t tag, layer;
    std::uint64_t reserved;
};
static_assert(sizeof(GpuDirectHeader) == gpu_direct_header_bytes);

// Why a received exchange was refused, written by the gather kernel (host-mapped coherent memory) before it traps.
struct GpuDirectError {
    std::uint32_t code;      // 0 none; 1 the host side failed; 2 magic, 3 sequence, 4 plan, 5 request, 6 tag, 7 layer, 8 size, 9 checksum
    std::uint32_t tag, layer, peer_tag, peer_layer;
    std::uint32_t pad;
    std::uint64_t sequence;
};
[[nodiscard]] const char* gpu_direct_error_text(std::uint32_t code) noexcept;

// Device scratch of one direction (checksum accumulation; reset by the kernel that consumes it).
struct GpuDirectScratch {
    unsigned long long sum;
    unsigned int blocks_done;
    unsigned int pad;
};

// One launch: message -> slots, then the header (identity + checksum of every 4-byte word) after the last part.
// `produced` (host-mapped, may be null): the scatter's last block stores identity.sequence there after a system-scope
// fence, once every slot write is visible outside the GPU.
void enqueue_scatter(const GpuDirectParts& parts, const void* message, const GpuDirectIdentity& identity, GpuDirectScratch* scratch,
                     std::uint64_t* produced, hipStream_t stream);
// One launch: slots -> message, then the received header is checked against `identity`, the checksum of the
// received words and the failure flag; any mismatch records the reason in `error` and traps (the stream's queue
// faults, so no later work runs on it and later HIP calls fail). `flag` is the exchange's 64-bit completion flag (failure
// value 0xffffffff).
void enqueue_gather(const GpuDirectParts& parts, void* message, const GpuDirectIdentity& identity, GpuDirectScratch* scratch,
                    const std::uint64_t* flag, GpuDirectError* error, hipStream_t stream);

// Diagnostics of the read path (not used by exchanges): mode 0 copy, 1 read + checksum only, 2 copy + checksum.
void enqueue_read_test(const void* src, void* dst, std::size_t bytes, int mode, unsigned blocks, void* sink, hipStream_t stream);

} // namespace strix::dist
