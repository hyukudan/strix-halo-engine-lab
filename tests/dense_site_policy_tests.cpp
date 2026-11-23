// Host test: the geometry-derived dense site decisions equal the literals the engine compiled in before (copied
// below verbatim as the reference), for the geometry taken from the generated constants of the model TOML.

#include "strix/core/dense_site_policy.hpp"
#include "strix/model/bind.hpp"
#include "strix/model_data/deepseek_v4_1_flash.hpp"

#include <cstdint>
#include <cstdio>
#include <initializer_list>

namespace ds = strix::core::dense_sites;
namespace d = strix::model_data::deepseek_v4_1_flash;
using strix::model::narrow;

namespace {

int failures = 0;
void check(bool ok, const char* what) {
    if (!ok) {
        std::fprintf(stderr, "FAIL: %s\n", what);
        ++failures;
    }
}

constexpr std::uint32_t kTpWorld = 2U;  // the pair contract of the tensor-parallel dense route

constexpr ds::Geometry geometry() {
    return {narrow<std::uint32_t>(d::dims::hidden),
            narrow<std::uint32_t>(d::attention::q_lora_rank),
            narrow<std::uint32_t>(d::dims::n_kv_heads * d::dims::head_dim),
            narrow<std::uint32_t>(d::dims::n_heads),
            narrow<std::uint32_t>(d::dims::head_dim),
            narrow<std::uint32_t>(d::attention::o_groups),
            narrow<std::uint32_t>(d::attention::o_lora_rank),
            narrow<std::uint32_t>(d::dims::intermediate),
            narrow<std::uint32_t>(d::dims::vocab),
            kTpWorld,
            narrow<std::uint32_t>((d::extras::engram_max_ngram_size - 1) * d::extras::engram_n_heads * d::extras::engram_head_dim),
            narrow<std::uint32_t>(d::dims::hidden * (d::extras::hc_mult + 1))};
}

// ---- the previous literals, copied from the engine sources ----
bool old_tp_shape(std::uint32_t c, std::uint32_t r, std::uint32_t bits, bool head) {
    if (bits != 4U && bits != 5U) return false;
    return (c == 1280U && r == 16384U) || (c == 4096U && r == 5120U) ||
           (c == 5120U && (r == 1152U || r == 1280U || r == 512U)) || (c == 1152U && r == 5120U) ||
           (head && c == 5120U && (r == 64640U || r == 129280U));
}
std::uint32_t old_sizes(std::uint32_t columns, std::uint32_t rows, std::uint32_t bits) {
    const auto set = [](std::initializer_list<std::uint32_t> counts) {
        std::uint32_t m = 0;
        for (const auto c : counts) m |= 1U << (c - 1U);
        return m;
    };
    if (bits == 5U && columns == 4096U && rows == 1024U) return set({1, 4, 8});
    if (bits == 5U && columns == 5120U && rows == 2304U) return set({2, 3, 4, 5, 6, 7});
    if (bits == 5U && columns == 2304U && rows == 5120U) return set({2, 3, 4, 5, 6, 7, 8});
    return 0xffU;
}
unsigned old_lean(std::uint32_t rows, std::uint32_t width, std::uint32_t count, std::uint32_t row_width) {
    unsigned pick = 0U;
    if (rows >= 256U) {
        if (width == 5120U && count <= 1536U) pick = rows >= 6144U ? 914U : 814U;
        else if (width <= 1536U && count >= 4096U) pick = 814U;
        else if (width == 4096U && count == 1024U) pick = rows >= 512U ? 914U : 822U;
        else if (row_width == 8192U && width == 4096U) pick = 822U;
    }
    return pick;
}
unsigned old_earlier(std::uint32_t rows, std::uint32_t width, std::uint32_t count) {
    return rows >= 3072U || (rows >= 2048U && count >= 1024U) ? 14U : rows >= 256U && count >= 4096U && width >= 4096U ? 12U : 0U;
}

}  // namespace

int main() {
    constexpr ds::Geometry g = geometry();
    static_assert(ds::valid(g));
    constexpr ds::DensePolicy p{};

    check(g.hidden == 5120U && g.q_lora == 1280U && g.kv_rows == 512U && g.intermediate == 2304U && g.vocab == 129280U,
          "geometry values");
    check(ds::q_b_rows_tp(g) == 16384U && ds::o_a_group_width(g) == 4096U && ds::o_a_row_width(g) == 8192U &&
              ds::o_b_width_tp(g) == 4096U && ds::ffn_half(g) == 1152U && ds::vocab_half(g) == 64640U,
          "derived dims");
    check(g.engram_columns == 6144U && g.engram_rows == 25600U, "engram dims");
    check(p.dense_min_rows == 256U && p.shared_fast_rows == 2048U, "policy thresholds");

    // exhaustive over the dimension values the engine and the other sites use
    const std::uint32_t dims[] = {0U,    1U,    32U,   512U,  1024U, 1152U,  1280U,  1536U,  2048U,  2304U,
                                  4096U, 5120U, 6144U, 8192U, 16384U, 25600U, 64640U, 129280U, 32768U, 256U};
    for (const auto c : dims)
        for (const auto r : dims)
            for (std::uint32_t bits = 2U; bits <= 8U; ++bits) {
                check(ds::tp_shape(g, c, r, bits, true) == old_tp_shape(c, r, bits, true), "tp_shape head");
                check(ds::tp_shape(g, c, r, bits, false) == old_tp_shape(c, r, bits, false), "tp_shape");
                check(ds::trellis_submit_sizes(g, c, r, bits) == old_sizes(c, r, bits), "submit sizes");
            }
    check(ds::q_b_rank_rows(g, 1280U, 16384U) && !ds::q_b_rank_rows(g, 1280U, 16385U), "q_b rank rows");
    check(ds::qa_kv_pair(g, 5120U, 5120U, 1280U, 512U) && !ds::qa_kv_pair(g, 5120U, 5120U, 1280U, 511U), "q_a kv pair");

    const std::uint32_t rowsv[] = {0U, 1U, 31U, 32U, 255U, 256U, 511U, 512U, 1023U, 2047U, 2048U, 3071U, 3072U, 6143U, 6144U, 8192U, 8193U};
    const std::uint32_t widths[] = {0U, 1024U, 1280U, 1536U, 1537U, 4096U, 5120U, 6144U, 2304U, 1152U};
    const std::uint32_t counts[] = {0U, 512U, 1024U, 1280U, 1536U, 1537U, 4095U, 4096U, 5120U, 16384U, 25600U};
    const std::uint32_t row_widths[] = {0U, 4096U, 5120U, 8192U};
    for (const auto rows : rowsv)
        for (const auto w : widths)
            for (const auto n : counts) {
                check(ds::tall_earlier_pick(p, rows, w, n) == old_earlier(rows, w, n), "tall earlier");
                for (const auto rw : row_widths)
                    check(ds::tall_lean_pick(g, p, rows, w, n, rw) == old_lean(rows, w, n, rw), "tall lean");
                for (const auto first : {0U, 1U})
                    check(ds::engram_prefill_shape(g, p, w, n, first, rows) ==
                              (w == 6144U && n == 25600U && first == 0U && rows >= 32U && rows <= 8192U),
                          "engram shape");
            }

    // Q2_K half layout of the 1152-wide expert half: 756 / 420 / 336 / 380
    constexpr auto q = ds::q2k_half_layout(ds::ffn_half(g));
    check(q.full_row == 756U && q.part_row == 420U && q.last == 336U && q.packed_row == 380U && q.group_width == 32U,
          "q2k layout");
    check(ds::ffn_half(g) / q.group_width == 36U, "tp down groups");

    // other two-rank-split geometries stay consistent (smaller hidden, odd world refused by valid())
    ds::Geometry other = g;
    other.hidden = 4096U;
    check(ds::valid(other) && !ds::tp_shape(other, 5120U, 1152U, 5U, false) && ds::tp_shape(other, 4096U, 1152U, 5U, false),
          "other geometry follows data");
    other.tp_world = 3U;
    check(!ds::valid(other), "invalid world");

    if (failures != 0) {
        std::fprintf(stderr, "%d failure(s)\n", failures);
        return 1;
    }
    std::puts("dense site policy tests passed");
    return 0;
}
