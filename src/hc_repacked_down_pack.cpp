#include "strix/hc_repacked_down_pack.hpp"
#include "strix/model_content_hash.hpp"

#include <algorithm>
#include <cstring>
#include <limits>

namespace strix::hc_repacked_down {
namespace {
void validate(const Geometry& g, std::size_t original, std::size_t plane) {
    if (g != geometry(g.width, g.rank)) {
        throw std::invalid_argument("HC repack geometry is not canonical");
    }
    if (original != g.original_bytes || plane != g.plane_bytes) {
        throw std::invalid_argument("HC repack span size does not match geometry");
    }
}

void reject_overlap(std::span<const std::uint8_t> source,
                    std::span<std::uint8_t> destination) {
    const auto a = reinterpret_cast<std::uintptr_t>(source.data());
    const auto b = reinterpret_cast<std::uintptr_t>(destination.data());
    constexpr auto limit = std::numeric_limits<std::uintptr_t>::max();
    if (source.size() > limit - a || destination.size() > limit - b) {
        throw std::invalid_argument("HC repack address range overflow");
    }
    if (a < b + destination.size() && b < a + source.size()) {
        throw std::invalid_argument("HC repack source and destination overlap");
    }
}

void transfer(const Geometry& g, const std::uint8_t* source,
              std::uint8_t* destination, bool packing) {
    for (std::uint32_t output = 0; output < g.rank; ++output) {
        for (std::uint32_t block = 0; block < g.blocks; ++block) {
            const auto original = (std::uint64_t{output} * g.blocks + block) * 34;
            const auto scale = g.scales_offset +
                4 * scale_word_index(block, output, g.rank_pad) + 2 * (block & 1U);
            if (packing) {
                std::memcpy(destination + scale, source + original, 2);
            } else {
                std::memcpy(destination + original, source + scale, 2);
            }
            for (std::uint32_t half = 0; half != 2; ++half) {
                const auto code = 16 * code_unit_index(block, half, output, g.rank_pad);
                const auto old_code = original + 2 + 16 * half;
                if (packing) {
                    std::memcpy(destination + code, source + old_code, 16);
                } else {
                    std::memcpy(destination + old_code, source + code, 16);
                }
            }
        }
    }
}
}  // namespace

void pack(const Geometry& g, std::span<const std::uint8_t> original,
          std::span<std::uint8_t> planes) {
    validate(g, original.size(), planes.size());
    reject_overlap(original, planes);
    std::fill(planes.begin(), planes.end(), std::uint8_t{0});
    transfer(g, original.data(), planes.data(), true);
}

void unpack(const Geometry& g, std::span<const std::uint8_t> planes,
            std::span<std::uint8_t> original) {
    validate(g, original.size(), planes.size());
    reject_overlap(planes, original);
    transfer(g, planes.data(), original.data(), false);
}

bool padding_zero(const Geometry& g, std::span<const std::uint8_t> planes) {
    validate(g, static_cast<std::size_t>(g.original_bytes), planes.size());
    const auto zeros = [&](std::uint64_t start, std::uint64_t count) {
        return std::all_of(planes.begin() + static_cast<std::ptrdiff_t>(start),
                           planes.begin() + static_cast<std::ptrdiff_t>(start + count),
                           [](std::uint8_t x) { return x == 0; });
    };
    for (std::uint32_t block = 0; g.rank < g.rank_pad && block < g.blocks; ++block) {
        for (std::uint32_t half = 0; half != 2; ++half) {
            if (!zeros(16 * code_unit_index(block, half, g.rank, g.rank_pad),
                       16ULL * (g.rank_pad - g.rank))) {
                return false;
            }
        }
    }
    for (std::uint32_t pair = 0; pair < g.blocks / 2; ++pair) {
        const auto start = g.scales_offset + 4ULL * (std::uint64_t{pair} * g.rank_pad + g.rank);
        if (!zeros(start, 4ULL * (g.rank_pad - g.rank))) {
            return false;
        }
    }
    const auto payload = g.codes_bytes + g.scales_bytes;
    return zeros(payload, g.plane_bytes - payload);
}

std::string sha256_hex(std::span<const std::uint8_t> bytes) {
    return model_content_identity_hex(sha256_bytes(std::as_bytes(bytes)));
}

}  // namespace strix::hc_repacked_down
