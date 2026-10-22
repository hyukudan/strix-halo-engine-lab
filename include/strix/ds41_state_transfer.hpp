#pragma once

#include <cstdint>
#include <span>
#include <stdexcept>
#include <string>
#include <vector>

namespace strix::ds41 {

struct StateTransferSection { std::string name; std::uint64_t bytes = 0; };
struct StateTransferSlice {
    std::string section;
    std::uint64_t section_offset = 0, file_offset = 0, bytes = 0;
};
struct StateTransferPlan {
    std::uint32_t position = 0;
    std::vector<StateTransferSection> sections;
    std::vector<StateTransferSlice> slices;  // in original v1 file order
};

// Structural framing only: callers must use the native v1 reader for semantic validation.
// No bytes are normalized, omitted, or regenerated. Slices concatenate to the original file.
inline StateTransferPlan plan_state_transfer_v1(std::span<const std::uint8_t> data) {
    const auto check = [](bool good) { if (!good) throw std::invalid_argument("invalid v1 state transfer framing"); };
    check(data.size() >= 48U && data.size() <= (1536ULL << 20U));
    constexpr char magic[] = "DS41ATST";
    for (unsigned i = 0; i < 8U; ++i) check(data[i] == static_cast<std::uint8_t>(magic[i]));
    std::size_t at = 8U;
    const auto read = [&](unsigned bytes) {
        check(bytes <= data.size() - at);
        std::uint64_t value = 0;
        for (unsigned i = 0; i < bytes; ++i) value |= std::uint64_t{data[at++]} << (i * 8U);
        return value;
    };
    check(read(4) == 1U);
    StateTransferPlan plan;
    plan.position = static_cast<std::uint32_t>(read(4));
    check(plan.position > 0U && plan.position <= 2097152U);
    (void)read(4);  // next token
    const auto layers = read(4);
    check(layers > 0U && layers <= 64U);
    std::uint64_t previous = 0, final_bytes = 0;
    const auto final_to = [&](std::uint64_t end) {
        if (end > previous) {
            plan.slices.push_back({"final", final_bytes, previous, end - previous});
            final_bytes += end - previous;
        }
        previous = end;
    };
    for (std::uint64_t layer = 0; layer < layers; ++layer) {
        (void)read(1); (void)read(4);  // flags and compression ratio
        for (unsigned field = 0; field < 5U; ++field) {
            const auto count = read(8);
            const std::uint64_t width = field >= 3U ? 4U : 1U;
            check(count <= (data.size() - at) / width);
            const auto bytes = count * width;
            if ((field == 1U || field == 2U) && bytes != 0U) {
                final_to(at);
                const auto name = std::string(field == 1U ? "kv." : "idx.") + std::to_string(layer);
                plan.sections.push_back({name, bytes});
                plan.slices.push_back({name, 0, at, bytes});
                previous = at + bytes;
            }
            at += static_cast<std::size_t>(bytes);
        }
    }
    check(read(8) == plan.position);
    check(std::uint64_t{plan.position} * 4U + 16U == data.size() - at);
    final_to(data.size());
    plan.sections.push_back({"final", final_bytes});
    return plan;
}

}  // namespace strix::ds41
