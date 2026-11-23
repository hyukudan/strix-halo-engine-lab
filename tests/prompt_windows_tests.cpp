// Shared prompt windows (strix/session/prompt_windows.hpp) and the two-stage window picker (strix/pair/layer_pipeline.hpp):
// (1) a plan covers [pos, pos + count) exactly: lead steps to an aligned position, aligned windows within [min_rows, rmax],
//     a tail shorter than min_rows; an inadmissible rmax makes every position a step;
// (2) the pipeline invariant: the layer pipeline's windows, each planned as a stage window, concatenate to the one-rank plan
//     from position 0 (same windows, same single steps, in order), so a stage runs the one-rank prefill's work in order;
// (3) the picker minimises (windows + 1) x half-window time over a calibration and returns 0 without a usable cost.
#include "strix/pair/layer_pipeline.hpp"
#include "strix/session/prompt_windows.hpp"

#include <algorithm>
#include <cstdint>
#include <cstdio>
#include <stdexcept>
#include <vector>

namespace pw = strix::session::prompt_windows;
namespace layers = strix::pair::layers;

namespace {

int failures = 0;
#define CHECK(c)                                                       \
    do {                                                               \
        if (!(c)) {                                                    \
            ++failures;                                                \
            std::fprintf(stderr, "FAIL %s:%d %s\n", __FILE__, __LINE__, #c); \
        }                                                              \
    } while (0)

void plan_covers(const pw::Rule& r) {
    for (const std::uint32_t rmax : {r.min_rows, r.min_rows + r.align, 512U, 2048U, 4096U}) {
        for (const std::uint32_t pos : {0U, 3U, 8U, 2045U}) {
            for (const std::uint32_t count : {1U, 7U, 15U, 16U, 17U, 31U, 2048U, 4096U, 4100U, 4111U}) {
                const auto p = pw::plan(r, pos, count, rmax);
                std::uint32_t at = pos + p.lead, covered = p.lead + p.tail;
                CHECK(p.lead < r.align || p.lead == count);
                for (const auto& w : p.windows) {
                    CHECK(w.p0 == at && w.p0 % r.align == 0U && pw::window_ok(r, w.rows) && w.rows <= rmax);
                    at += w.rows;
                    covered += w.rows;
                }
                CHECK(covered == count && p.tail < r.min_rows);
            }
        }
    }
    const auto off = pw::plan(r, 0U, 100U, r.min_rows - 1U);
    CHECK(off.lead == 100U && off.windows.empty() && off.tail == 0U);
}

void stage_windows(const pw::Rule& r) {
    for (const std::uint32_t rmax : {r.min_rows, r.min_rows + r.align, 512U, 1024U, 2048U}) {
        for (const std::uint32_t count : {1U, 7U, 15U, 16U, 17U, 31U, 32U, 1000U, 2048U, 4096U, 4100U, 4111U, 4119U, 16384U}) {
            const auto one = pw::plan(r, 0U, count, rmax);
            std::vector<pw::Window> wins;
            std::uint32_t steps_mid = 0, tail = 0, at = 0;
            const auto ws = layers::windows(count, rmax);
            for (std::size_t k = 0; k < ws.size(); ++k) {
                CHECK(ws[k].start == at);
                const auto p = pw::stage_plan(r, ws[k].start, ws[k].rows, rmax);
                CHECK(p.lead == 0U && p.windows.size() <= 1U);
                for (const auto& w : p.windows) wins.push_back(w);
                (k + 1U == ws.size() ? tail : steps_mid) += p.tail;
                at += ws[k].rows;
            }
            CHECK(at == count && steps_mid == 0U && tail == one.tail && one.lead == 0U);
            CHECK(wins.size() == one.windows.size());
            for (std::size_t k = 0; k < std::min(wins.size(), one.windows.size()); ++k)
                CHECK(wins[k].p0 == one.windows[k].p0 && wins[k].rows == one.windows[k].rows);
        }
    }
    bool refused = false;
    try {
        (void)pw::stage_plan(r, r.align, r.min_rows, r.min_rows);  // a window off the shared grid
    } catch (const std::invalid_argument&) {
        refused = true;
    }
    CHECK(refused);
}

void picker() {
    // Synthetic calibration: the wide window 2.3 % cheaper per row.
    const layers::WindowCost costs[] = {{1024U, 1000.0}, {2048U, 1954.0}};
    CHECK(layers::two_stage_window_rows(16U, costs) == 1024U && layers::two_stage_window_rows(4096U, costs) == 1024U &&
          layers::two_stage_window_rows(32768U, costs) == 1024U);
    CHECK(layers::two_stage_window_rows(65536U, costs) == 2048U);
    CHECK(layers::two_stage_window_rows(4096U, {}) == 0U);
    const layers::WindowCost unusable[] = {{0U, 5.0}, {1024U, 0.0}};
    CHECK(layers::two_stage_window_rows(4096U, unusable) == 0U);
}

// (4) append_chunks tail absorption: the chunks cover the prompt in order, only the last carries the head, each is at
//     most max_rows + tail_rows, only the last may differ from max_rows, and tail_rows = 0 is the greedy plan.
std::vector<std::size_t> chunk_sizes(std::size_t count, std::size_t max_rows, std::size_t tail_rows) {
    std::vector<unsigned> prompt(count);
    for (std::size_t i = 0; i < count; ++i) prompt[i] = static_cast<unsigned>(i);
    std::vector<std::size_t> sizes;
    std::vector<unsigned> ordered;
    pw::append_chunks(std::span<const unsigned>(prompt), max_rows, [&](std::span<const unsigned> chunk, bool last) {
        sizes.push_back(chunk.size());
        ordered.insert(ordered.end(), chunk.begin(), chunk.end());
        CHECK(last == (ordered.size() == prompt.size()));
        CHECK(!chunk.empty() && chunk.size() <= max_rows + tail_rows);
        CHECK(last || chunk.size() == max_rows);
    }, tail_rows);
    CHECK(ordered == prompt);
    return sizes;
}

void tail_absorption() {
    using Sizes = std::vector<std::size_t>;
    CHECK(chunk_sizes(8236U, 8192U, 0U) == (Sizes{8192U, 44U}));
    CHECK(chunk_sizes(8236U, 8192U, 44U) == (Sizes{8236U}));
    CHECK(chunk_sizes(8236U, 8192U, 1024U) == (Sizes{8236U}));
    CHECK(chunk_sizes(8236U, 8192U, 43U) == (Sizes{8192U, 44U}));
    CHECK(chunk_sizes(16428U, 8192U, 44U) == (Sizes{8192U, 8236U}));
    CHECK(chunk_sizes(8192U, 8192U, 1024U) == (Sizes{8192U}));
    CHECK(chunk_sizes(100U, 8192U, 1024U) == (Sizes{100U}));
    CHECK(chunk_sizes(5U, 2U, 0U) == (Sizes{2U, 2U, 1U}));
    CHECK(chunk_sizes(5U, 2U, 1U) == (Sizes{2U, 3U}));
    // Exhaustive small cases: full chunks while more than max_rows + tail remain, then one final chunk; with
    // tail < max_rows that is exactly one chunk fewer than greedy when the greedy final chunk fits the tail.
    for (std::size_t count = 1; count <= 40U; ++count)
        for (std::size_t max_rows = 1; max_rows <= 9U; ++max_rows)
            for (std::size_t tail = 0; tail <= 12U; ++tail) {
                const auto sizes = chunk_sizes(count, max_rows, tail);
                const auto expected = count <= max_rows + tail ? 1U : (count - tail - 1U) / max_rows + 1U;
                CHECK(sizes.size() == expected);
                if (tail < max_rows) {
                    const auto greedy = chunk_sizes(count, max_rows, 0U);
                    const auto last = count % max_rows == 0U ? max_rows : count % max_rows;
                    const bool absorbed = count > max_rows && last <= tail;
                    CHECK(sizes.size() + (absorbed ? 1U : 0U) == greedy.size());
                }
            }
    bool threw = false;
    try {
        pw::append_chunks(std::span<const unsigned>(), 1U, [](std::span<const unsigned>, bool) {}, 0U);
    } catch (const std::invalid_argument&) {
        threw = true;
    }
    CHECK(threw);
}

}  // namespace

int main() {
    static_assert(pw::Rule{8U, 16U}.valid() && pw::Rule{4U, 8U}.valid() && !pw::Rule{8U, 12U}.valid() && !pw::Rule{0U, 0U}.valid());
    for (const pw::Rule r : {pw::Rule{8U, 16U}, pw::Rule{4U, 8U}}) {
        plan_covers(r);
        stage_windows(r);
    }
    picker();
    tail_absorption();
    if (failures != 0) {
        std::fprintf(stderr, "prompt windows: %d failure(s)\n", failures);
        return 1;
    }
    std::puts("prompt windows PASS");
    return 0;
}
