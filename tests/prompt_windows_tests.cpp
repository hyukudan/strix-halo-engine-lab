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

}  // namespace

int main() {
    static_assert(pw::Rule{8U, 16U}.valid() && pw::Rule{4U, 8U}.valid() && !pw::Rule{8U, 12U}.valid() && !pw::Rule{0U, 0U}.valid());
    for (const pw::Rule r : {pw::Rule{8U, 16U}, pw::Rule{4U, 8U}}) {
        plan_covers(r);
        stage_windows(r);
    }
    picker();
    if (failures != 0) {
        std::fprintf(stderr, "prompt windows: %d failure(s)\n", failures);
        return 1;
    }
    std::puts("prompt windows PASS");
    return 0;
}
