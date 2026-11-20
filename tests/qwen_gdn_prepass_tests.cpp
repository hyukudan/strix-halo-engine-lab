// Conv prepass split of the CUDA Qwen GDN prefill (include/strix/cuda/qwen_gdn_prepass.hpp): one block per chunk writes the
// same boundary taps and the same new history as the serial prepass (one thread walking every chunk), in any block order.
#include "strix/cuda/qwen_gdn_prepass.hpp"

#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <vector>

namespace {

namespace pp = strix::cuda::qwen_gdn::prepass;

constexpr std::uint32_t kChannels = 7;
int failures = 0;

void check(bool ok, const char* what, std::uint32_t count) {
    if (!ok) {
        std::fprintf(stderr, "FAIL count %u: %s\n", count, what);
        ++failures;
    }
}

struct Case {
    std::vector<std::uint16_t> rows;     // [count][kChannels] raw rows of the workspace
    std::vector<std::uint16_t> history;  // [kChannels][kHistory]
};

Case make_case(std::uint32_t count, std::uint32_t seed) {
    Case c;
    std::uint32_t x = seed * 2654435761U + 1U;
    const auto next = [&] {
        x = x * 1664525U + 1013904223U;
        return static_cast<std::uint16_t>((x >> 16U) | 1U);  // never 0: a zero-filled read cannot pass for a history word
    };
    c.rows.resize(std::size_t{count} * kChannels);
    for (auto& v : c.rows) v = next();
    c.history.resize(std::size_t{kChannels} * pp::kHistory);
    for (auto& v : c.history) v = next();
    return c;
}

// The body of one prepass thread over chunks [first, last); `owns` = reads the old history and writes the new one.
void body(const Case& c, std::uint32_t count, std::uint32_t ch, std::vector<std::uint16_t>& state, std::vector<std::uint16_t>& boundary,
          std::uint32_t first, std::uint32_t last, bool owns) {
    std::uint16_t old[pp::kHistory] = {};
    if (owns)
        for (std::uint32_t i = 0; i < pp::kHistory; ++i) old[i] = state[std::size_t{ch} * pp::kHistory + i];
    const auto raw = [&](std::int64_t row) -> std::uint16_t {
        return row >= 0 ? c.rows[static_cast<std::size_t>(row) * kChannels + ch] : old[row + static_cast<std::int64_t>(pp::kHistory)];
    };
    for (std::uint32_t k = first; k < last; ++k)
        for (std::uint32_t j = 0; j < pp::kTaps; ++j) boundary[(std::size_t{k} * pp::kTaps + j) * kChannels + ch] = raw(pp::tap_row(k, j));
    if (!owns) return;
    for (std::uint32_t i = 0; i < pp::kHistory; ++i) state[std::size_t{ch} * pp::kHistory + i] = raw(pp::history_row(count, i));
}

void run_case(std::uint32_t count, std::uint32_t seed) {
    const Case c = make_case(count, seed);
    const std::uint32_t chunks = pp::chunks(count);
    check(chunks * pp::kChunk >= count && (chunks - 1U) * pp::kChunk < count, "chunk count covers the rows", count);
    // Rule: only chunk kHistoryChunk's taps and the history of fewer than kHistory rows reach the old history.
    for (std::uint32_t k = 0; k < chunks; ++k)
        for (std::uint32_t j = 0; j < pp::kTaps; ++j) {
            const std::int64_t r = pp::tap_row(k, j);
            check(r >= -static_cast<std::int64_t>(pp::kTaps) && r < static_cast<std::int64_t>(count), "tap row in range", count);
            check(k == pp::kHistoryChunk || r >= 0, "only the history chunk reads the old history", count);
        }
    for (std::uint32_t i = 0; i < pp::kHistory; ++i)
        check(pp::history_row(count, i) >= 0 || count < pp::kHistory, "history reads old slots only below kHistory rows", count);

    // Serial reference: one owner over every chunk.
    std::vector<std::uint16_t> ref_state = c.history, ref_boundary(std::size_t{chunks} * pp::kTaps * kChannels, 0);
    for (std::uint32_t ch = 0; ch < kChannels; ++ch) body(c, count, ch, ref_state, ref_boundary, 0, chunks, true);

    // Split: block y saves chunk y, the history chunk owns the history; forward and reverse block order.
    for (int order = 0; order < 2; ++order) {
        std::vector<std::uint16_t> state = c.history, boundary(ref_boundary.size(), 0);
        for (std::uint32_t n = 0; n < chunks; ++n) {
            const std::uint32_t y = order == 0 ? n : chunks - 1U - n;
            for (std::uint32_t ch = 0; ch < kChannels; ++ch) body(c, count, ch, state, boundary, y, y + 1U, y == pp::kHistoryChunk);
        }
        check(boundary == ref_boundary, order == 0 ? "boundary equal (forward blocks)" : "boundary equal (reverse blocks)", count);
        check(state == ref_state, order == 0 ? "history equal (forward blocks)" : "history equal (reverse blocks)", count);
    }
}

}  // namespace

int main() {
    static_assert(pp::chunks(1) == 1U && pp::chunks(32) == 1U && pp::chunks(33) == 2U && pp::chunks(8163) == 256U);
    static_assert(pp::tap_row(0, 0) == -3 && pp::tap_row(1, 2) == 31 && pp::history_row(2, 0) == -2);
    const std::uint32_t counts[] = {1, 2, 3, 4, 5, 16, 31, 32, 33, 35, 63, 64, 65, 100, 257, 1000, 8163};
    std::uint32_t seed = 1;
    for (const std::uint32_t count : counts) run_case(count, seed++);
    if (failures) return EXIT_FAILURE;
    std::printf("qwen_gdn_prepass_tests: PASS (%zu row counts)\n", sizeof(counts) / sizeof(counts[0]));
    return EXIT_SUCCESS;
}
