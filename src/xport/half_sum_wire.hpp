#pragma once

#include <cstddef>
#include <cstdint>
#include <string_view>

namespace strix::xport {

// The two FP32 half-sums per layer and the vocabulary argmax share one schedule.
// A model keeps its kind numbers, tag type and trailer magic.
template<unsigned Layers, std::size_t Width, class Kind, class Tag>
struct HalfSumSchedule {
    static constexpr unsigned kLayers = Layers;
    static constexpr unsigned kPerLayer = 2;
    static constexpr unsigned kPerStep = Layers * kPerLayer + 1;
    static constexpr unsigned kArgmaxBytes = 8;

    [[nodiscard]] static constexpr std::size_t row_bytes(unsigned rows) {
        return std::size_t{rows} * Width * sizeof(float);
    }
    [[nodiscard]] static constexpr Tag tag_at(unsigned site, std::uint64_t step) {
        return site == kPerStep - 1
            ? Tag{Kind::argmax, kLayers, step}
            : Tag{site % kPerLayer == 0 ? Kind::attn_sum : Kind::ffn_sum, site / kPerLayer, step};
    }
    [[nodiscard]] static constexpr std::size_t payload_bytes(unsigned site, unsigned rows) {
        return site == kPerStep - 1 ? std::size_t{rows} * kArgmaxBytes : row_bytes(rows);
    }
    [[nodiscard]] static constexpr int memory_kind(std::string_view value) {
        return value == "host" ? 0 : value == "device" ? 1 : value == "uncached" ? 2 : -1;
    }
};

template<std::uint64_t Magic>
struct HalfSumTrailer {
    std::uint64_t magic = Magic;
    std::uint64_t epoch = 0;
    std::uint64_t plan = 0;
    std::uint64_t sequence = 0;
    std::uint64_t step = 0;
    std::uint32_t bytes = 0;
    std::uint16_t kind = 0;
    std::uint16_t flags = 0;
    std::uint32_t layer = 0;
    std::uint32_t rank = 0;
    std::uint32_t crc32c = 0;
    std::uint32_t prev_crc32c = 0;
};
static_assert(sizeof(HalfSumTrailer<0>) == 64);
inline constexpr std::uint16_t kHalfSumTrailerDeferred = 1, kHalfSumTrailerPrevious = 2;
// Small control and argmax frames always carry their own CRC.
inline constexpr std::size_t kHalfSumInlineCrcMax = 1024;

} // namespace strix::xport
