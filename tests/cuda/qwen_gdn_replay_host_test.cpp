#include "qwen_gdn_replay.hpp"

#include <array>
#include <cstdio>
#include <cstdlib>
#include <stdexcept>
#include <string_view>
#include <vector>

namespace g = strix::cuda::qwen_gdn;
namespace r = strix::ops::recur;
namespace e = strix::exec;
using strix::RecurrentStateType;

static void require(bool ok, const char* field) {
    if (ok) return;
    std::fprintf(stderr, "Qwen GDN replay staging: %s\n", field);
    std::exit(1);
}

int main() {
    constexpr auto shape = g::replay_shape_for({2560, 16, 48, 128, 4, 256});
    static_assert(shape == r::shape::gdn_2560_16x48x128);
    auto unsupported = shape;
    unsupported.value_heads = 32;
    require(unsupported.valid(), "alternate shared shape is valid");
    bool refused = false;
    try {
        (void)g::replay_shape_for(unsupported);
    } catch (const std::invalid_argument& e) {
        refused = std::string_view(e.what()).find("gdn_2560_16x48x128") != std::string_view::npos;
    }
    require(refused, "unsupported Qwen shape names the required preset");
    using L = r::GdnDecodeLayout<shape>;
    constexpr std::array<std::uint32_t, 8> tokens{3, 9, 7, 5, 11, 13, 21, 99};
    constexpr std::array<unsigned, 8> widths{1, 2, 4, 4, 8, 8, 8, 8};
    std::vector<std::uint16_t> history(4U * L::history_elements);
    std::vector<float> matrices(4U * L::matrix_elements);
    auto* alternate_history = history.data() + 2U * L::history_elements;
    auto* alternate_matrix = matrices.data() + 2U * L::matrix_elements;
    g::ReplayWindow window{};
    const auto step = g::bind_replay_layer(shape, window, {}, {}, 1, RecurrentStateType::f32);
    for (unsigned rows = 1; rows <= tokens.size(); ++rows) {
        constexpr unsigned capacity = 100;
        const unsigned position = capacity - rows;
        const auto input = std::span(tokens).first(rows);
        const bool odd = rows % 2U != 0;
        auto* current = odd ? matrices.data() : alternate_matrix;
        auto* spare = odd ? alternate_matrix : matrices.data();
        window = g::stage_replay_window(input, position, capacity, history.data(), current, alternate_history, spare);
        const unsigned width = widths[rows - 1U];
        require(window.descriptor.h.rows == width, "width");
        for (unsigned i = 0; i < width; ++i) {
            require(e::step_row_valid(window.descriptor, i) == (i < rows), "padding");
            if (i >= rows) continue;
            const auto& row = window.descriptor.row[i];
            require(row.token == tokens[i] && row.position == position + i && row.slot == 0, "row mapping");
            require(row.parent == (i == 0 ? e::kStepNoParent : i - 1U), "parent");
            require(row.ancestors == ((2U << i) - 1U), "ancestors");
        }
        const auto selected = r::decode_replay_images<shape>(width, step);
        require(selected.rows == rows, "frontier and capacity");
        require(selected.history == history.data() + L::history_elements &&
                selected.pending_history == alternate_history + L::history_elements, "tap layer offset");
        require(selected.matrix == current + L::matrix_elements &&
                selected.pending_matrix == spare + L::matrix_elements, "matrix layer offset and parity");
    }
    std::puts("Qwen GDN replay staging PASS (R=1..8, padding, frontier, offsets, parity)");
}
