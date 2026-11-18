// core::WindowStateBuffers (src/core/window_state_buffers.hpp): verify windows over a recurrent state without snapshot copies.
// A CPU model of the GDN recurrent kernel (load state once, run rows, store once; store address may differ from the load)
// runs random windows (1..16 rows) and commits (0..rows accepted, plus discarded windows) through three protocols:
//   reference : only the committed rows, applied in order, in place;
//   incumbent : snapshot copy -> window in place -> on partial accept restore copy + replay accepted rows;
//   buffers   : window reads committed(), writes window_target(); accept_all() swaps; partial -> replay in place.
// After every resolution all three committed states must be bitwise equal (memcmp), per layer, and the out-of-place
// window must equal the in-place window bitwise.
#include "core/window_state_buffers.hpp"

#include <cmath>
#include <cstdint>
#include <cstdio>
#include <cstring>
#include <random>
#include <vector>

namespace {
constexpr std::size_t kLayers = 3, kState = 96, kMaxRows = 16;
int failures = 0;

struct Row {
    float decay, beta, x[kState];
};
// Same shape as recurrent_rows_kernel: every element is loaded once, rows run on the retained copy, stored once.
void window(const float* in, float* out, const Row* rows, unsigned n) {
    float retained[kState];
    for (std::size_t i = 0; i < kState; ++i) retained[i] = in[i];
    for (unsigned r = 0; r < n; ++r) {
        float dot = 0.0F;
        for (std::size_t i = 0; i < kState; ++i) dot += retained[i] * rows[r].x[i];
        for (std::size_t i = 0; i < kState; ++i) retained[i] = retained[i] * rows[r].decay + rows[r].beta * (rows[r].x[i] - dot);
    }
    for (std::size_t i = 0; i < kState; ++i) out[i] = retained[i];
}
void check(bool ok, const char* what, int step) {
    if (!ok) {
        ++failures;
        std::fprintf(stderr, "FAIL step %d: %s\n", step, what);
    }
}
}  // namespace

int main() {
    std::mt19937 rng(1234U);
    std::uniform_real_distribution<float> u(-1.0F, 1.0F);
    const std::size_t total = kLayers * kState;
    std::vector<float> ref(total), inc(total), inc_bak(total), a(total), b(total);
    for (std::size_t i = 0; i < total; ++i) ref[i] = inc[i] = a[i] = 0.25F * u(rng);
    std::fill(b.begin(), b.end(), std::nanf(""));  // the spare's prior contents must never be read
    strix::core::WindowStateBuffers<float> buffers(a.data(), b.data());
    std::vector<Row> rows(kLayers * kMaxRows);
    unsigned swaps = 0, partial = 0, discards = 0;
    for (int step = 0; step < 4000; ++step) {
        const unsigned n = 1U + unsigned(rng() % kMaxRows);
        for (auto& r : rows) {
            r.decay = 0.9F + 0.1F * std::fabs(u(rng));
            r.beta = 0.5F * std::fabs(u(rng));
            for (float& v : r.x) v = u(rng);
        }
        const unsigned roll = unsigned(rng() % 8U);
        const bool discard = roll == 0U;  // window dropped before commit (engine: next run with a pending window)
        const unsigned k = discard ? 0U : roll == 1U ? n : unsigned(rng() % (n + 1U));
        // incumbent: snapshot, window in place, then restore + replay on partial accept
        inc_bak = inc;
        for (std::size_t l = 0; l < kLayers; ++l) window(&inc[l * kState], &inc[l * kState], &rows[l * kMaxRows], n);
        if (k < n) {
            inc = inc_bak;
            for (std::size_t l = 0; l < kLayers; ++l) window(&inc[l * kState], &inc[l * kState], &rows[l * kMaxRows], k);
        }
        // buffers: window out of place; the in-place result of the same window is the bitwise oracle for the target
        for (std::size_t l = 0; l < kLayers; ++l) {
            float in_place[kState];
            std::memcpy(in_place, buffers.committed_at(l * kState), sizeof(in_place));
            window(in_place, in_place, &rows[l * kMaxRows], n);
            window(buffers.committed_at(l * kState), buffers.target_at(l * kState), &rows[l * kMaxRows], n);
            check(std::memcmp(in_place, buffers.target_at(l * kState), sizeof(in_place)) == 0, "out-of-place window != in place", step);
        }
        if (k == n) {
            buffers.accept_all();
            ++swaps;
        } else {
            discards += discard ? 1U : 0U;
            partial += discard ? 0U : 1U;
            for (std::size_t l = 0; l < kLayers; ++l)
                window(buffers.committed_at(l * kState), buffers.committed_at(l * kState), &rows[l * kMaxRows], k);
        }
        // reference: committed rows only
        for (std::size_t l = 0; l < kLayers; ++l) window(&ref[l * kState], &ref[l * kState], &rows[l * kMaxRows], k);
        check(std::memcmp(ref.data(), inc.data(), total * sizeof(float)) == 0, "incumbent != reference", step);
        check(std::memcmp(ref.data(), buffers.committed(), total * sizeof(float)) == 0, "buffers != reference", step);
        check(buffers.committed() != buffers.window_target(), "roles alias", step);
        if (failures > 8) break;
    }
    check(swaps > 100U && partial > 100U && discards > 100U, "every resolution kind exercised", -1);
    std::printf("window-state-buffers: %s (swaps %u, partial %u, discards %u)\n", failures ? "FAIL" : "PASS", swaps, partial, discards);
    return failures ? 1 : 0;
}
