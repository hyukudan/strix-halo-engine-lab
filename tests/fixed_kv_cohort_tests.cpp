// FixedKvCohort: a resident decoder with preallocated KV slots bound to the greedy cohort contract.
// Invariants: geometry refusal, one decode row per lane forwarded in window order, final logits of a sampled
// lane read from its row of the last window (never another lane's), decoders without logits stay greedy-only,
// an emits-first decoder serves the first id and final scores of its own last prefill call.
#include "strix/serve/fixed_kv_cohort.hpp"

#include <cstdio>
#include <cstdlib>
#include <memory>
#include <stdexcept>
#include <string>
#include <vector>

using strix::serve::CohortGroup;
using strix::serve::FixedKvCohort;

namespace {

int failures = 0;

void check(bool value, const std::string& label) {
    if (!value) {
        std::fprintf(stderr, "FAIL %s\n", label.c_str());
        ++failures;
    }
}

template <class F>
bool throws(F&& f) {
    try {
        f();
    } catch (const std::exception&) {
        return true;
    }
    return false;
}

// Greedy-only decoder: top[row] = 1000 * lane + token, records every call.
struct GreedyDecoder {
    std::uint32_t allocated = 0;
    std::vector<std::uint32_t> prefilled;
    std::vector<std::uint32_t> last_slots;
    void allocate_slots(std::uint32_t lanes) { allocated = lanes; }
    void prefill_slot(std::uint32_t lane, std::span<const std::uint32_t>, bool first) {
        prefilled.push_back(lane * 2U + (first ? 1U : 0U));
    }
    void decode_slots(std::span<const std::uint32_t> slots, std::span<const std::uint32_t> tokens,
                      std::span<std::uint32_t> top) {
        last_slots.assign(slots.begin(), slots.end());
        for (std::size_t row = 0; row < slots.size(); ++row) top[row] = 1000U * slots[row] + tokens[row];
    }
};

// Sampling-capable decoder: row r of the last window holds logits {r, lane, r, lane}.
struct LogitsDecoder : GreedyDecoder {
    bool supported = true;
    [[nodiscard]] bool final_logits_supported() const { return supported; }
    [[nodiscard]] std::uint32_t logit_width() const { return 4U; }
    void read_slot_logits(std::uint32_t row, std::span<float> out) const {
        const auto lane = static_cast<float>(last_slots.at(row));
        const auto r = static_cast<float>(row);
        out[0] = r;
        out[1] = lane;
        out[2] = r;
        out[3] = lane;
    }
};

void test_geometry() {
    check(throws([] { FixedKvCohort<GreedyDecoder>(nullptr, 64U, 4U, 2U); }), "null decoder refused");
    check(throws([] { FixedKvCohort<GreedyDecoder>(std::make_unique<GreedyDecoder>(), 0U, 4U, 2U); }),
          "zero context refused");
    check(throws([] { FixedKvCohort<GreedyDecoder>(std::make_unique<GreedyDecoder>(), 64U, 4U, 0U); }),
          "zero lanes refused");
    check(throws([] { FixedKvCohort<GreedyDecoder>(std::make_unique<GreedyDecoder>(), 64U, 2U, 3U); }),
          "lanes above rows refused");
    auto decoder = std::make_unique<GreedyDecoder>();
    auto* raw = decoder.get();
    FixedKvCohort<GreedyDecoder> cohort(std::move(decoder), 64U, 4U, 3U);
    check(raw->allocated == 3U, "slots allocated once for every lane");
    check(cohort.lanes() == 3U && cohort.rows_per_forward() == 4U && cohort.max_positions() == 64U,
          "geometry reported");
}

void test_greedy_window() {
    auto decoder = std::make_unique<GreedyDecoder>();
    auto* raw = decoder.get();
    FixedKvCohort<GreedyDecoder> cohort(std::move(decoder), 64U, 4U, 4U);
    const std::uint32_t prompt[] = {7U, 8U};
    cohort.prefill(2U, prompt, true);
    cohort.prefill(2U, prompt, false);
    check(raw->prefilled == std::vector<std::uint32_t>{5U, 4U}, "prefill forwards lane and first flag");

    // Window order is the group order, not lane order.
    const CohortGroup groups[] = {{3U, 1U}, {0U, 1U}, {2U, 1U}};
    const std::uint32_t tokens[] = {11U, 12U, 13U};
    std::uint32_t top[3] = {};
    cohort.verify(groups, tokens, top);
    check(raw->last_slots == std::vector<std::uint32_t>{3U, 0U, 2U}, "decode slots in group order");
    check(top[0] == 3011U && top[1] == 12U && top[2] == 2013U, "top ids per row");
    check(!cohort.supports_final_logits(), "greedy-only decoder declines final logits");
    const std::uint32_t keep_ok[] = {1U, 1U, 1U};
    check(!throws([&] { cohort.commit(keep_ok); }), "keep one row per lane accepted");
    const std::uint32_t keep_short[] = {1U, 1U};
    check(throws([&] { cohort.commit(keep_short); }), "keep count != window refused");
    const std::uint32_t keep_zero[] = {1U, 0U, 1U};
    check(throws([&] { cohort.commit(keep_zero); }), "dropping a guaranteed row refused");

    const CohortGroup multi[] = {{1U, 2U}};
    const std::uint32_t two[] = {1U, 2U};
    std::uint32_t top2[2] = {};
    check(throws([&] { cohort.verify(multi, two, top2); }), "multi-row group refused");
}

void test_sampled_logits() {
    auto decoder = std::make_unique<LogitsDecoder>();
    auto* raw = decoder.get();
    FixedKvCohort<LogitsDecoder> cohort(std::move(decoder), 64U, 4U, 4U);
    check(cohort.supports_final_logits(), "logits decoder serves sampled lanes");
    const CohortGroup groups[] = {{2U, 1U}, {0U, 1U}};
    const std::uint32_t tokens[] = {1U, 2U};
    std::uint32_t top[2] = {};
    cohort.verify(groups, tokens, top);
    const auto lane0 = cohort.read_final_logits(0U);
    check(lane0.size() == 4U && lane0[0] == 1.0F && lane0[1] == 0.0F, "lane 0 reads its own row (row 1)");
    const auto lane2 = cohort.read_final_logits(2U);
    check(lane2.size() == 4U && lane2[0] == 0.0F && lane2[1] == 2.0F, "lane 2 reads its own row (row 0)");
    check(throws([&] { (void)cohort.read_final_logits(3U); }), "lane outside the window refused");
    raw->supported = false;
    check(!cohort.supports_final_logits(), "decoder may decline at run time");
}

// Emits-first decoder: each prefill call returns id 7000 + 10 * lane + rows; its final scores are row 0 of the window
// (LogitsDecoder: {0, lane, 0, lane}).
struct EmitsDecoder : LogitsDecoder {
    std::vector<std::uint32_t> first_rows;
    std::uint32_t prefill_slot_first(std::uint32_t lane, std::span<const std::uint32_t> tokens, bool first) {
        prefilled.push_back(lane * 2U + (first ? 1U : 0U));
        first_rows.push_back(static_cast<std::uint32_t>(tokens.size()));
        last_slots.assign(1U, lane);
        return 7000U + 10U * lane + static_cast<std::uint32_t>(tokens.size());
    }
};
// Emits-first without final logits: greedy ids only.
struct EmitsGreedyDecoder : GreedyDecoder {
    std::uint32_t prefill_slot_first(std::uint32_t lane, std::span<const std::uint32_t> tokens, bool) {
        return 7000U + 10U * lane + static_cast<std::uint32_t>(tokens.size());
    }
};

// The lane reads its whole prompt by prefill; the first id and the final scores are the prefill call's own and are
// valid until the next backend call. A decoder without the hook keeps the decode-row contract.
void test_prefill_emits_first() {
    FixedKvCohort<GreedyDecoder> plain(std::make_unique<GreedyDecoder>(), 64U, 4U, 2U);
    check(!plain.prefill_emits_first(), "decoder without prefill_slot_first keeps the decode-row first id");
    check(throws([&] { (void)plain.prefill_top(0U); }), "plain cohort has no prefill_top");

    auto decoder = std::make_unique<EmitsDecoder>();
    auto* raw = decoder.get();
    FixedKvCohort<EmitsDecoder> cohort(std::move(decoder), 64U, 4U, 4U);
    check(cohort.prefill_emits_first(), "decoder with prefill_slot_first emits the first id");
    const std::uint32_t prompt[] = {5U, 6U, 7U};
    cohort.prefill(1U, prompt, true);
    check(raw->prefilled.size() == 1U && raw->prefilled[0] == 3U && raw->first_rows[0] == 3U,
          "whole prompt reaches prefill_slot_first with first = true");
    check(cohort.prefill_top(1U) == 7013U, "first id = the prefill's own");
    const auto scores = cohort.read_final_logits(1U);
    check(scores.size() == 4U && scores[0] == 0.0F && scores[1] == 1.0F, "final logits = the prefill's row 0");
    check(throws([&] { (void)cohort.prefill_top(2U); }), "first id of a lane not just prefilled refused");

    // a decode window invalidates the prefill's first id; a later prefill invalidates the window rows
    const CohortGroup groups[] = {{2U, 1U}, {1U, 1U}};
    const std::uint32_t tokens[] = {4U, 4U};
    std::uint32_t top[2] = {};
    cohort.verify(groups, tokens, top);
    check(throws([&] { (void)cohort.prefill_top(1U); }), "first id not readable after a decode window");
    const auto window = cohort.read_final_logits(1U);
    check(window.size() == 4U && window[0] == 1.0F && window[1] == 1.0F, "window row read after the decode window");
    cohort.prefill(3U, std::span<const std::uint32_t>(prompt).first(2), true);
    check(throws([&] { (void)cohort.read_final_logits(1U); }), "window rows not readable after a prefill call");
    check(cohort.prefill_top(3U) == 7032U, "second lane's own first id");

    FixedKvCohort<EmitsGreedyDecoder> greedy(std::make_unique<EmitsGreedyDecoder>(), 64U, 4U, 2U);
    greedy.prefill(0U, prompt, true);
    check(greedy.prefill_emits_first() && greedy.prefill_top(0U) == 7003U && !greedy.supports_final_logits(),
          "greedy-only emits-first decoder");
}

}  // namespace

// A decoder that admits a prefill tail: the cohort reports it, a decoder without the accessor reports none.
struct TailDecoder : GreedyDecoder {
    [[nodiscard]] std::uint32_t prefill_tail_rows() const noexcept { return 1024U; }
};

void test_prompt_tail() {
    using strix::serve::prompt_slice_rows;
    FixedKvCohort<TailDecoder> tail(std::make_unique<TailDecoder>(), 64U, 4U, 2U);
    FixedKvCohort<GreedyDecoder> plain(std::make_unique<GreedyDecoder>(), 64U, 4U, 2U);
    check(tail.prompt_tail_rows() == 1024U && plain.prompt_tail_rows() == 0U, "tail rows forwarded from the decoder");
    static_assert(prompt_slice_rows(8192U, 8192U, 8235U, 1024U) == 8235U);  // 8192 + 43 -> one pass
    static_assert(prompt_slice_rows(8192U, 8192U, 9216U, 1024U) == 9216U);  // remainder == tail
    static_assert(prompt_slice_rows(8192U, 8192U, 9300U, 1024U) == 8192U);  // remainder > tail: exact slice
    static_assert(prompt_slice_rows(8192U, 8192U, 8235U, 0U) == 8192U);     // no tail: greedy slices
    static_assert(prompt_slice_rows(4096U, 8192U, 8235U, 1024U) == 4096U);  // the planner granted a part: unchanged
    static_assert(prompt_slice_rows(8192U, 8192U, 100U, 1024U) == 100U);    // short prompt: all of it
}

int main() {
    test_prompt_tail();
    test_geometry();
    test_greedy_window();
    test_sampled_logits();
    test_prefill_emits_first();
    if (failures != 0) {
        std::fprintf(stderr, "fixed-kv-cohort: %d failures\n", failures);
        return EXIT_FAILURE;
    }
    std::printf("fixed-kv-cohort: PASS\n");
    return EXIT_SUCCESS;
}
