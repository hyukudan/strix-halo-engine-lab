#include "inkling2/inkling2_mini_moe_dispatch.hpp"

#include <array>
#include <cstdint>
#include <iostream>
#include <memory>
#include <span>
#include <stdexcept>
#include <vector>

namespace {

namespace dispatch = strix::moe_core::grouped_dispatch;
namespace packed = strix::weight::microscaled;

struct WiderExpert {
    enum class Projection { gate_up, down };
    static constexpr unsigned hidden = 128, intermediate = 256;
    static constexpr unsigned experts = 4, top_k = 2, layers = 1, max_rows = 256;

    static constexpr packed::MatrixFormat format(unsigned) { return packed::MatrixFormat::mxfp8; }
    static constexpr packed::RowOrder row_order(Projection) { return packed::RowOrder::contiguous; }
};

template<class Config, class Backend>
bool check(unsigned rows, std::size_t expected_bf16, std::size_t expected_fp32) {
    using Projection = typename Config::Projection;
    const auto elements = dispatch::workspace_elements<Config>(rows);
    if (elements.bf16 != expected_bf16 || elements.fp32 != expected_fp32) return false;

    const std::size_t assignments = std::size_t{rows} * Config::top_k;
    std::vector<std::uint32_t> ids(assignments);
    for (std::size_t i = 0; i < assignments; ++i) {
        ids[i] = Config::top_k - 1U - static_cast<unsigned>(i % Config::top_k);
    }
    const std::vector<float> weights(assignments, 1.0F);
    // Only descriptor metadata is read during host planning.
    alignas(32) const std::array<std::uint8_t, 32> packed_storage{};
    const auto* packed_data = packed_storage.data();
    const dispatch::ExpertWeights expert{
        {packed_data, packed_data, 2U * Config::intermediate, Config::hidden, 1.0F,
         Config::row_order(Projection::gate_up)},
        {packed_data, packed_data, Config::hidden, Config::intermediate, 1.0F, Config::row_order(Projection::down)},
    };
    const std::vector<dispatch::ExpertWeights> matrices(Config::experts, expert);
    std::vector<std::uint16_t> bf16(elements.bf16 + 7U);
    std::vector<float> fp32(elements.fp32);
    void* aligned = bf16.data();
    auto space = bf16.size() * sizeof(std::uint16_t);
    if (!std::align(16U, elements.bf16 * sizeof(std::uint16_t), aligned, space)) return false;
    const dispatch::Workspace workspace{{static_cast<std::uint16_t*>(aligned), elements.bf16}, fp32};

    const auto plan = dispatch::prepare<Config, Backend>(0U, rows, ids, weights, matrices, workspace);
    if (plan.gate_up.size() != 2U * Config::top_k || plan.down.size() != plan.gate_up.size()) return false;
    for (unsigned id = 0; id < Config::experts; ++id) {
        const auto first = std::size_t{id} * rows;
        const auto offset = plan.routes.grouping.expert_offsets[id];
        const auto next = plan.routes.grouping.expert_offsets[id + 1U];
        if (id >= Config::top_k) {
            if (offset != assignments || next != assignments) return false;
            continue;
        }
        if (offset != first || next != first + rows || plan.routes.combine_rows[id] != first ||
            plan.routes.combine_rows[assignments - Config::top_k + id] != first + rows - 1U) return false;
        for (unsigned part = 0; part < 2U; ++part) {
            const auto tile = std::size_t{id} * 2U + part;
            const auto base = first + part * Backend::rows_per_tile;
            const auto count = part == 0U ? Backend::rows_per_tile : 1U;
            const auto& gate = plan.gate_up[tile];
            const auto& down = plan.down[tile];
            if (gate.rows != count || down.rows != count ||
                gate.input != workspace.bf16.data() + base * Config::hidden ||
                gate.output != workspace.fp32.data() + base * 2U * Config::intermediate ||
                down.input != workspace.bf16.data() + base * Config::intermediate ||
                down.output != workspace.fp32.data() + base * Config::hidden) return false;
        }
    }
    try {
        const dispatch::Workspace short_scratch{workspace.bf16, workspace.fp32.first(elements.fp32 - 1U)};
        (void)dispatch::prepare<Config, Backend>(0U, rows, ids, weights, matrices, short_scratch);
    } catch (const std::invalid_argument&) {
        return true;
    }
    return false;
}

}  // namespace

int main() {
    if (!check<strix::inkling2::MiniMoeConfig, dispatch::CudaBackend>(
            17U, 313344U, 313344U)) {
        std::cerr << "Mini grouped planning failed\n";
        return 1;
    }
    if (!check<WiderExpert, dispatch::HipBackend>(129U, 66048U, 132096U)) {
        std::cerr << "Wider expert grouped planning failed\n";
        return 1;
    }
    return 0;
}
