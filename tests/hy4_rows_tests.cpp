// Hy4-flash verify rows, host side (src/hy4/hip/hy4_hip_rows.hpp):
//  1. rewind replay: with the rings sized by rows::cache_rows(prompt_rows, 8) every read after a verify pass of R <= 8
//     rows and any commit k finds its own key (sliding latent ring and indexer token ring, tiny and window-512 geometry);
//     negative control: rings sized for one row (the decode-only allocation before hy4_rows) are clobbered.
//  2. position words of a row == the decode step's {p, p/4*4, p/4, 0}.
//  3. speculative decode through the shared strix::spec layer over a history-exact fake forward: ids == plain greedy,
//     speculation engages (accepted drafts > 0) and a commit that keeps rejected rows is detected (ids differ).
#include "hy4/hip/hy4_hip_rows.hpp"

#include <cstdio>
#include <cstdlib>
#include <filesystem>
#include <span>
#include <string>
#include <vector>

namespace {
namespace hb = strix::hy4::hipbind;
int failures = 0;
#define CHECK(cond)                                                                 \
    do {                                                                            \
        if (!(cond)) {                                                              \
            ++failures;                                                             \
            std::fprintf(stderr, "%s:%d: CHECK failed: %s\n", __FILE__, __LINE__, #cond); \
        }                                                                           \
    } while (0)

void check_rings(const strix::hy4::Config& c, const char* name) {
    const std::uint32_t context = 8192;
    std::uint32_t clobbered = 0;
    for (std::uint32_t l = 0; l < c.layers; ++l) {
        for (const std::uint32_t prompt_rows : {0U, 16U, 1024U}) {
            const auto e = hb::rows::check_rewind(c, l, context, hb::rows::cache_rows(prompt_rows, hb::rows::kMaxVerifyRows),
                                                  hb::rows::kMaxVerifyRows);
            if (!e.empty()) std::fprintf(stderr, "%s: %s\n", name, e.c_str());
            CHECK(e.empty());
        }
        const auto broken = hb::rows::check_rewind(c, l, context, hb::rows::cache_rows(0U, 0U), hb::rows::kMaxVerifyRows);
        clobbered += broken.empty() ? 0U : 1U;
    }
    CHECK(clobbered == c.layers);  // every layer keeps a ring: one-row rings fail the replay
    std::printf("%s: rewind replay R 1..8 PASS on %u layers, one-row rings clobbered on %u\n", name, c.layers, clobbered);
}

void check_words() {
    for (std::uint32_t p = 0; p < 100; ++p) {
        const auto w = hb::rows::position_words(p, 4U);
        CHECK(w[0] == p && w[1] == p / 4U * 4U && w[2] == p / 4U && w[3] == 0U);
    }
    CHECK(hb::rows::cache_rows(0U, 0U) == 1U && hb::rows::cache_rows(0U, 8U) == 8U && hb::rows::cache_rows(1024U, 8U) == 1024U);
}

// A target whose greedy id is a pure function of its whole token history: next = the pattern successor of the last
// token, except at every 11th position (a function of the history length) -> copy-friendly with rejections.
struct FakeForward {
    std::vector<std::uint32_t> hist, greedy;
    std::uint32_t p0 = 0, rows = 0;
    bool broken_commit = false;
    static std::uint32_t next(const std::vector<std::uint32_t>& h) {
        const auto n = static_cast<std::uint32_t>(h.size());
        if (n % 11U == 0U) return 100U + n % 7U;
        const std::uint32_t last = h.back();
        return last < 100U ? (last + 1U) % 9U : 0U;
    }
    void forward_rows(std::span<const std::uint32_t> toks, bool) {
        if (rows != 0U) throw std::logic_error("fake: pass before commit");
        p0 = static_cast<std::uint32_t>(hist.size());
        rows = static_cast<std::uint32_t>(toks.size());
        greedy.clear();
        for (const auto t : toks) {
            hist.push_back(t);
            greedy.push_back(next(hist));
        }
    }
    [[nodiscard]] std::span<const std::uint32_t> rows_greedy() const { return greedy; }
    void commit_rows(std::uint32_t k) {
        if (k > rows) throw std::logic_error("fake: commit beyond the pass");
        if (!broken_commit) hist.resize(p0 + k);
        rows = 0;
    }
    [[nodiscard]] std::uint32_t position() const { return static_cast<std::uint32_t>(hist.size()); }
};

void check_spec() {
    const std::vector<std::uint32_t> prompt{0, 1, 2, 3, 4, 5, 6, 7, 8, 0, 1, 2};
    const std::uint32_t n = 300;
    std::vector<std::uint32_t> plain{FakeForward::next(prompt)};
    {
        auto h = prompt;
        while (plain.size() < n) {
            h.push_back(plain.back());
            plain.push_back(FakeForward::next(h));
        }
    }
    hb::rows::SpecOptions o;
    o.max_rows = hb::rows::kMaxVerifyRows;
    for (std::uint32_t r = 1; r <= o.max_rows; ++r) o.verify_ms.push_back(57.7 + 18.0 * (r - 1U));  // the predicted curve
    FakeForward f;
    f.hist = prompt;
    const auto res = hb::rows::spec_decode(f, prompt, plain[0], n, o);
    CHECK(res.ids == plain);
    CHECK(res.accepted > 0U);
    CHECK(res.steps.size() < n - 1U);  // fewer passes than ids: rows > 1 were verified and kept
    std::uint64_t multi = 0;
    for (std::size_t r = 2; r < res.rows_hist.size(); ++r) multi += res.rows_hist[r];
    CHECK(multi > 0U);
    std::printf("spec: ids == plain over %u ids, %zu passes, %llu multi-row, accepted %llu of %llu proposed\n", n, res.steps.size(),
                static_cast<unsigned long long>(multi), static_cast<unsigned long long>(res.accepted),
                static_cast<unsigned long long>(res.proposed));
    FakeForward bad;
    bad.hist = prompt;
    bad.broken_commit = true;
    const auto wrong = hb::rows::spec_decode(bad, prompt, plain[0], n, o);
    CHECK(wrong.ids != plain);  // negative control: rejected rows left in the history change the ids
    std::printf("spec negative control (commit keeps rejected rows): ids differ = %d\n", wrong.ids != plain ? 1 : 0);
}
}  // namespace

int main(int argc, char** argv) {
    try {
        const auto tiny = strix::hy4::Config::tiny();
        check_rings(tiny, "tiny");
        auto wide = tiny;
        wide.window = 512;  // the pack's sliding window: ring 1024 with 8 verify rows, 512 (clobbered) with one
        check_rings(wide, "window 512");
        if (argc > 1 && std::filesystem::exists(argv[1])) check_rings(strix::hy4::Config::from_gguf(argv[1]), "pack");
        check_words();
        check_spec();
    } catch (const std::exception& e) {
        std::fprintf(stderr, "exception: %s\n", e.what());
        return 1;
    }
    if (failures) {
        std::fprintf(stderr, "%d check(s) failed\n", failures);
        return 1;
    }
    std::printf("hy4 rows tests PASS\n");
    return 0;
}
