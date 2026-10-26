#pragma once
// bp-v1: lossless codec of the RTX state hand-off chunks (docs/rtx-state-import.md, "Chunk encoding").
//
// The plain bytes are a concatenation of regions, each with an element width (1, 2 or 4 bytes: raw headers, BF16
// rows, FP32 matrices). Each region is cut into blocks; a block splits its elements into `width` byte planes (plane j
// holds byte j of every element) and stores each plane raw, as one constant byte, or canonical-Huffman coded (code
// lengths <= 11, four interleaved bit streams). The trailing bytes of a block that do not fill an element are stored
// raw. Every block carries the XXH64 (seed 0) of its plain bytes, verified by the decoder, so a codec defect can never
// hand over different bytes silently. The decoder needs no knowledge of the regions: it reproduces the plain bytes.
//
// Frame (little-endian):
//   "BPV1" | u32 block_count | u64 plain_bytes | block_count x {u32 plain_bytes, u32 encoded_bytes, u64 xxh64} | blocks
// Block:
//   u8 width (1, 2, 4) | u8 mode[width] (0 raw, 1 constant, 2 huffman) | plane payloads in plane order | tail bytes
//   (plain_bytes % width, raw). Plane of n = plain_bytes / width symbols:
//   raw: n bytes; constant: 1 byte; huffman: 128 bytes of code lengths (4 bits per symbol, symbol 2k in the low nibble
//   of byte k; 0 = absent; the code is complete), u32 size[4] of the four streams, then the streams. Stream s codes
//   symbols [s*q, min((s+1)*q, n)) with q = ceil(n / 4), LSB-first, codes canonical (shorter first, then by symbol,
//   as in deflate) and bit-reversed; each stream is padded with zero bits to a whole byte.
#include <cstddef>
#include <cstdint>
#include <functional>
#include <memory>
#include <span>
#include <string_view>
#include <vector>

namespace strix::bp_codec {

inline constexpr std::string_view codec_name = "bp-v1";

struct Region {
    const std::uint8_t* data = nullptr;
    std::size_t bytes = 0;
    std::uint32_t width = 1;  // element bytes: 1, 2 or 4 (a hint for the ratio only; any value decodes exactly)
};

// Runs indexed tasks on background workers; the thread that calls run() works too. run() may be called from several
// threads at once (each call waits only for its own tasks). threads == 0: everything runs on the caller.
class WorkerPool {
public:
    explicit WorkerPool(unsigned threads);
    ~WorkerPool();
    WorkerPool(const WorkerPool&) = delete;
    WorkerPool& operator=(const WorkerPool&) = delete;
    // Calls fn(i) for every i < tasks; rethrows the first exception after every task finished.
    void run(std::size_t tasks, const std::function<void(std::size_t)>& fn);
    [[nodiscard]] unsigned threads() const noexcept;

private:
    struct State;
    std::unique_ptr<State> state_;
};

struct EncodeOptions {
    std::size_t block_bytes = std::size_t{1} << 20;  // plain bytes per block (rounded down to the width)
    double huffman_min_gain = 0.015;                 // code a plane only when it saves at least this fraction
};

struct EncodeStats {
    std::uint64_t plain_bytes = 0, encoded_bytes = 0, blocks = 0, raw_planes = 0, constant_planes = 0, huffman_planes = 0;
};

[[nodiscard]] std::vector<std::uint8_t> encode(std::span<const Region> regions, WorkerPool* pool = nullptr,
    const EncodeOptions& options = {}, EncodeStats* stats = nullptr);
// The plain size of a frame; validates the frame header and block directory (throws std::invalid_argument).
[[nodiscard]] std::uint64_t decoded_size(std::span<const std::uint8_t> frame);
// Decodes into out (out.size() == decoded_size(frame)). Throws std::invalid_argument on any malformed frame or block
// checksum mismatch; never reads or writes outside the spans.
void decode(std::span<const std::uint8_t> frame, std::span<std::uint8_t> out, WorkerPool* pool = nullptr);

[[nodiscard]] std::uint64_t xxh64(const std::uint8_t* data, std::size_t bytes, std::uint64_t seed = 0) noexcept;

}  // namespace strix::bp_codec
