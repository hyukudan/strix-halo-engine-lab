#include "cuda/core/head_emit.hpp"

#include <array>
#include <bit>
#include <cstdio>
#include <limits>

namespace {
namespace head = strix::cuda::core::head;
namespace ha = strix::head_argmax;

bool ties_and_order() {
    constexpr float inf = std::numeric_limits<float>::infinity();
    constexpr float nan = std::numeric_limits<float>::quiet_NaN();
    constexpr std::array logits{nan, -0.0F, 5.0F, -3.0F, 5.0F, 0.0F, inf, -inf, 4.0F};
    constexpr std::array<std::uint32_t, 7> ids{12, 14, 18, 11, 15, 13, ha::kNone};
    std::array<head::Candidate, ids.size()> top;
    head::reference_candidates(logits, 10, top);
    for (std::size_t i = 0; i < top.size(); ++i) {
        if (top[i].id != ids[i]) return false;
        if (ids[i] != ha::kNone && std::bit_cast<std::uint32_t>(top[i].value) !=
                std::bit_cast<std::uint32_t>(logits[ids[i] - 10U])) return false;
    }
    const auto pick = ha::reference(logits, 10U);
    return pick.index == top[0].id && pick.nonfinite == 3U && std::signbit(top[3].value) && !std::signbit(top[4].value);
}

bool wide_candidates() {
    constexpr std::uint32_t first = ha::kNone - 129U;
    std::array<float, 129> logits;
    logits.fill(-8.0F);
    logits.back() = 9.0F;
    std::array<head::Candidate, head::kMaxCandidates> top;
    head::reference_candidates(logits, first, top);
    if (top[0].id != ha::kNone - 1U || top[0].value != 9.0F) return false;
    for (std::uint32_t i = 1; i < top.size(); ++i)
        if (top[i].id != first + i - 1U || top[i].value != -8.0F) return false;
    head::reference_candidates(logits, first, {});
    return true;
}

bool lse() {
    constexpr float inf = std::numeric_limits<float>::infinity();
    constexpr std::array logits{1000.0F, inf, 1000.0F, -inf};
    const auto out = head::reference_lse(logits);
    const auto empty = head::reference_lse(std::span(logits).subspan(1, 1));
    return out.maximum == 1000.0F && out.sum == 2.0F && out.lse == 1000.0F + std::log(2.0F) && out.nonfinite == 2U &&
           empty.maximum == -inf && empty.sum == 0.0F && empty.lse == -inf && empty.nonfinite == 1U;
}
}  // namespace

int main() {
    if (!ties_and_order() || !wide_candidates() || !lse()) {
        std::fputs("k11c host FAIL\n", stderr);
        return 1;
    }
    std::puts("k11c host PASS");
}
