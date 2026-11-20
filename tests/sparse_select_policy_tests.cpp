// Short-context rule of indexed sparse attention (strix/sparse_select_policy.hpp): the shared predicate equals the
// two engine forms it replaces at every position (GLM DSA: floor((pos + 1) / 4) <= 512, the old `p_bound > kIdxPools`
// gate; MiniMax MSA: pos < 16 * 128, the old route::step_dense), a step is identity only when every row is, and the
// off switch parses fail-closed.
#include "strix/sparse_select_policy.hpp"

#include <algorithm>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <span>
#include <stdexcept>
#include <vector>

namespace ss = strix::sparse_select;

namespace {
int g_fail = 0;
void check(bool ok, const char* what) {
    if (!ok) {
        ++g_fail;
        std::fprintf(stderr, "FAIL: %s\n", what);
    }
}
struct Row {
    std::uint64_t pos;
};
void set_env(const char* value) {
#if defined(_WIN32)
    (void)_putenv_s("STRIX_SELECT_SHORT_SKIP", value == nullptr ? "" : value);
#else
    if (value == nullptr) (void)unsetenv("STRIX_SELECT_SHORT_SKIP");
    else (void)setenv("STRIX_SELECT_SHORT_SKIP", value, 1);
#endif
}
}  // namespace

int main() {
    constexpr ss::Geometry glm{4, 512, false};    // pools of 4 tokens, 512 kept, completed pools ranked
    constexpr ss::Geometry mm{128, 16, true};     // blocks of 128, 16 kept, the newest block ranked
    static_assert(ss::selection_is_identity(glm, 2050) && !ss::selection_is_identity(glm, 2051));
    static_assert(ss::selection_is_identity(mm, 2047) && !ss::selection_is_identity(mm, 2048));
    static_assert(ss::scored_units(mm, 0) == 1 && ss::scored_units(glm, 0) == 0 && ss::scored_units(glm, 3) == 1);

    std::uint64_t glm_bad = 0, mm_bad = 0;
    for (std::uint64_t pos = 0; pos < 70000; ++pos) {
        const unsigned p_bound = std::max(1U, static_cast<unsigned>((pos + 1) / 4));  // glm_fast_attn.hip dsa_run
        glm_bad += ss::selection_is_identity(glm, pos) != !(p_bound > 512U);
        mm_bad += ss::selection_is_identity(mm, pos) != (pos < 16U * 128U);
    }
    check(glm_bad == 0, "GLM predicate equals the p_bound <= kIdxPools gate");
    check(mm_bad == 0, "MiniMax predicate equals pos < kTopK * kBlock");

    auto pos_of = [](const Row& r) { return r.pos; };
    const std::vector<Row> short_rows{{0}, {100}, {2047}}, mixed{{5}, {2048}, {7}}, none{};
    check(ss::step_selection_is_identity(mm, std::span<const Row>(short_rows), pos_of), "step: all rows short");
    check(!ss::step_selection_is_identity(mm, std::span<const Row>(mixed), pos_of), "step: one long row decides");
    check(ss::step_selection_is_identity(glm, std::span<const Row>(mixed), pos_of), "step: GLM keeps 2048 (512 pools)");
    check(ss::step_selection_is_identity(mm, std::span<const Row>(none), pos_of), "step: no rows");

    set_env(nullptr);
    check(ss::query_skip_enabled(), "switch: unset = on");
    set_env("1");
    check(ss::query_skip_enabled(), "switch: 1 = on");
    set_env("0");
    check(!ss::query_skip_enabled(), "switch: 0 = off");
    set_env("yes");
    bool threw = false;
    try {
        (void)ss::query_skip_enabled();
    } catch (const std::invalid_argument&) {
        threw = true;
    }
    check(threw, "switch: other values fail closed");
    const auto recorded = strix::envcfg::registered();
    const auto entry = std::find_if(recorded.begin(), recorded.end(), [](const auto& e) {
        return e.name == "STRIX_SELECT_SHORT_SKIP";
    });
    check(entry != recorded.end() && entry->set && entry->value == "yes",
          "switch: invalid user setting is recorded for diagnosis");
    set_env(nullptr);
    check(ss::query_skip_enabled(), "switch: unset restores default");
    const auto defaults = strix::envcfg::registered();
    const auto unconfigured = std::find_if(defaults.begin(), defaults.end(), [](const auto& e) {
        return e.name == "STRIX_SELECT_SHORT_SKIP";
    });
    check(unconfigured != defaults.end() && !unconfigured->set && unconfigured->value.empty(),
          "switch: default is recorded as unset");

    std::printf("sparse_select_policy: %s\n", g_fail == 0 ? "PASS" : "FAIL");
    return g_fail == 0 ? EXIT_SUCCESS : EXIT_FAILURE;
}
