// Host-only route ownership, byte accounting, admission and missing-layout gates.
#include "minimax/engine/route_identity.hpp"
#include "minimax/engine/mm_common.hpp"

#include <array>
#include <iostream>
#include <stdexcept>
#include <string_view>

namespace r = strix::minimax::route;
namespace d = strix::minimax::engine::detail;
namespace {
void check(bool ok) { if (!ok) throw std::runtime_error("MiniMax index layout test failed"); }
template<class Exception, class F> void refuses(F action, std::string_view message) {
    try { action(); } catch (const Exception& e) {
        check(std::string_view(e.what()).find(message) != std::string_view::npos);
        return;
    }
    throw std::runtime_error("MiniMax index layout gate did not refuse");
}
constexpr auto bytes(unsigned mask) {
    return r::index_weight_bytes(mask, d::kHidden, d::kIdxQ / 2, 128, d::kLayers);
}
static_assert(bytes(8) == 283115520ULL);
static_assert(bytes(15) - bytes(8) == 1132462080ULL);  // 1.0546875 GiB/rank
static_assert(bytes(1) == bytes(6));
static_assert(bytes(2) == bytes(8));
}

int main() {
    struct Case { const char* env; bool decode; unsigned mask, small, large; };
    const std::array cases = {
        Case{nullptr, true, 8, 8, 8}, Case{nullptr, false, 8, 8, 8},
        Case{"attn", true, 8, 8, 8}, Case{"attn", false, 8, 8, 8},
        Case{"tiled", true, 6, 4, 2}, Case{"tiled", false, 2, 2, 2},
        Case{"off", true, 1, 1, 1}, Case{"off", false, 1, 1, 1}};
    for (const auto& c : cases) {
        const auto policy = r::resolve_index_projection(c.env, c.decode);
        check(r::index_projection_layouts(policy) == c.mask);
        for (std::size_t rows : {1U, 2U, 3U, 8U, 9U, 16U, 4096U, 8192U}) {
            const auto layout = r::index_layout_bit(r::index_projection_route(policy, rows));
            check(layout == (rows <= 8 ? c.small : c.large));
            r::require_index_layouts(c.mask, layout);
        }
        r::admit_index_weights(bytes(c.mask), bytes(c.mask));
        refuses<std::runtime_error>([&] { r::admit_index_weights(bytes(c.mask) - 1, bytes(c.mask)); },
                                    "insufficient device memory");
        // Every runtime flag combination, including flags masked by attn/off.
        for (bool attn : {false, true}) for (bool tiled : {false, true}) for (bool decode : {false, true}) {
            const auto needed = r::index_projection_layouts({attn, tiled, decode});
            if ((c.mask & needed) == needed) r::require_index_layouts(c.mask, needed);
            else refuses<std::logic_error>([&] { r::require_index_layouts(c.mask, needed); }, "reload the engine");
        }
    }
    for (const char* invalid : {"", "typo", "ATTN"})
        refuses<std::invalid_argument>([&] { (void)r::resolve_index_projection(invalid, true); },
                                      "STRIX_MM_INDEX_PROJ must be attn, tiled or off");
    // All packed kernel diagnostics (ring, LDS, split and prefill tile) keep wip.
    for (unsigned rows : {1U, 2U, 8U, 9U, 8192U}) for (unsigned ring : {0U, 8U, 16U, 32U, 48U})
        for (unsigned lds : {0U, 1U}) for (unsigned split : {0U, 2U}) {
            const auto kernel = r::index_route(rows, ring, split, lds);
            check(kernel.tiled == (rows > 8));
            check(r::index_projection_route({}, rows) == r::IndexLayout::packed);
            // Key-only short-context form: ring 48 where measured faster (1 / 2 rows), an explicit ring still wins.
            const auto key = r::index_route(rows, ring, split, lds, true);
            check(key.ring == (ring != 0 ? ring : rows <= 2 ? 48U : kernel.ring) && key.tiled == kernel.tiled &&
                  key.lds == kernel.lds && key.rows_per_wave == kernel.rows_per_wave);
        }
    std::cout << "MiniMax index layouts: routes, bytes, admission and refusal gates passed\n";
}
