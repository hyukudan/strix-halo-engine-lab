#pragma once
// How a resident copy reads its source: straight from the read-only model mapping, or through temporary
// registrations of that mapping (host-only planning; the HIP calls live in src/hip_model_residency.cpp).
//
// Why: HIP 7.17 copies pageable host memory to the device of an integrated GPU through a CPU staging buffer. From a
// page-cache-warm GGUF mapping on gfx1151 (3 GiB regions, each read for the first time) a direct copy runs at
// 24-27 GB/s there, against 74-79 GB/s on HIP 7.16, while a registered source copies at 50-57 GB/s on both runtimes
// including its first registration (and 82-85 GB/s when re-registered). So the copy registers its source on runtimes
// from 7.17 on integrated GPUs and keeps the direct copy elsewhere (on 7.16 the direct copy is the faster form). The
// registrations are released as soon as their bytes are copied: after setup nothing of the model is registered, as
// before. Same bytes copied to the same device addresses.
#include <algorithm>
#include <cstdint>
#include <stdexcept>
#include <vector>

namespace strix::model_copy {

// First HIP runtime (HIP_VERSION encoding: major * 10000000 + minor * 100000 + patch) that stages pageable copies.
inline constexpr int kStagedPageableRuntime = 71700000;

// The measured rule: register the source exactly where the runtime stages pageable copies through the CPU.
[[nodiscard]] constexpr bool register_source(int hip_runtime_version, bool integrated_gpu) noexcept {
    return integrated_gpu && hip_runtime_version >= kStagedPageableRuntime;
}

// One copy piece: copy bytes [begin, end) of the mapping, registering the whole pages [register_begin, register_end).
struct SourcePiece {
    std::uint64_t begin = 0, end = 0;
    std::uint64_t register_begin = 0, register_end = 0;
};

// Splits the mapping range [begin, end) into pieces of at most `piece` bytes (0: one piece) whose inner boundaries are
// page aligned, so consecutive registrations never share a page. The last registration ends at the page that holds
// `end`, inside the mapping's last page (mapped_bytes).
[[nodiscard]] inline std::vector<SourcePiece> source_pieces(std::uint64_t begin, std::uint64_t end, std::uint64_t piece,
                                                            std::uint64_t page, std::uint64_t mapped_bytes) {
    if (!page || (page & (page - 1U)) || begin > end || end > mapped_bytes)
        throw std::invalid_argument("model copy source pieces need a power-of-two page and a range inside the mapping");
    const auto down = [page](std::uint64_t value) { return value & ~(page - 1U); };
    const auto up = [page, &down](std::uint64_t value) { return down(value + page - 1U); };
    const std::uint64_t step = piece ? std::max(down(piece), page) : 0U;
    std::vector<SourcePiece> pieces;
    for (std::uint64_t at = begin; at < end;) {
        const std::uint64_t next = step && end - down(at) > step ? down(at) + step : end;
        pieces.push_back({at, next, down(at), std::min(up(next), up(mapped_bytes))});
        at = next;
    }
    return pieces;
}

}  // namespace strix::model_copy
