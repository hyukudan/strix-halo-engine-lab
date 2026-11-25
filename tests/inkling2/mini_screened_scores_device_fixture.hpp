#pragma once

// A sampled serving lane that steps alone reads the final scores of its own step (read_slot_logits), bit for bit
// those of the full head, also when that step's pick was screened (the screened greedy pick does not write the
// score row). Reference: the full head through decode(token, host logits), which never screens. Sampling is the
// shared sample_token_row with a fixed policy and fixed uniforms, so both runs sample the same ids exactly when
// their scores agree. Covers a lane sampled alone from the first step, and a lane sampled alone after a packed
// cohort step and after another lane's greedy step (whose scores are never read and never projected).
// Verify rows level 2 (STRIX_INK_VERIFY_ROWS=2, set_verify_rows(2)): cohorts of 2..4 distinct lanes whose groups declare
// no score read pick through the lanes screened head up to its capacity (lanes_picks counts it; scores deferred, then read
// bit for bit against each lane's own full-head rows), wider cohorts and announced reads keep the full head, and deferred
// rows read after a verify window or a held-back prompt on another slot still return the cohort's rows. Verify windows
// at level 2 (partial commits, round fold, continuation) are the verify rows fixture's --lanes-pick arm.

#include "inkling2/inkling2_mini_resident_decoder.hpp"
#include "strix/token_sampling.hpp"

#include <array>
#include <cstdint>
#include <cstring>
#include <exception>
#include <iostream>
#include <optional>
#include <span>
#include <stdexcept>
#include <string>
#include <string_view>
#include <vector>

namespace strix::inkling2::test {

inline constexpr unsigned kScreenPromptRows = 37U;
inline constexpr unsigned kScreenSteps = 6U;

inline double screen_fixed_uniform(unsigned step) {
    const auto bits = (std::uint64_t{step} + 1U) * 0x9E3779B97F4A7C15ULL;
    return static_cast<double>(bits >> 11U) * 0x1.0p-53;
}

inline unsigned screen_sample(std::span<const float> scores, unsigned step) {
    TokenSamplingPolicy policy{};
    policy.temperature = 0.8F;
    policy.seed = 7U;
    return sample_token_row(scores, {}, policy, screen_fixed_uniform(step));
}

struct ScreenReference {
    std::vector<unsigned> ids;   // input of step s; ids[s + 1] is sampled from the scores of step s
    std::vector<unsigned> top;   // full-head greedy pick of step s
    std::vector<float> scores;   // final scores of step s
    [[nodiscard]] std::span<const float> row(unsigned step) const {
        return std::span(scores).subspan(std::size_t{step} * kOutputVocabulary, kOutputVocabulary);
    }
};

inline void check_step(const char* where, unsigned step, std::span<const float> actual, unsigned top,
                       const ScreenReference& reference) {
    if (std::memcmp(actual.data(), reference.row(step).data(), actual.size_bytes()) != 0) {
        std::string stale = "no reference step";
        for (unsigned other = 0; other < kScreenSteps; ++other)
            if (std::memcmp(actual.data(), reference.row(other).data(), actual.size_bytes()) == 0)
                stale = "the scores of step " + std::to_string(other);
        throw std::runtime_error(std::string(where) + ": step " + std::to_string(step) +
                                 " read scores that differ from the full head (they equal " + stale + ")");
    }
    if (top != reference.top[step])
        throw std::runtime_error(std::string(where) + ": step " + std::to_string(step) + " greedy pick differs");
    if (screen_sample(actual, step) != reference.ids[step + 1U])
        throw std::runtime_error(std::string(where) + ": step " + std::to_string(step) + " sampled id differs");
}

// Whether the last pick left its score row to the first score read. The same file builds on a decoder without the
// deferred read (the defect this test catches), where only the scores are checked (no value).
template<class Decoder>
std::optional<bool> scores_deferred(const Decoder& decoder) {
    if constexpr (requires { decoder.scores_deferred(); }) return decoder.scores_deferred();
    else return std::nullopt;
}

inline unsigned step_lane(MiniResidentDecoder& decoder, unsigned slot, unsigned id) {
    const std::array<unsigned, 1> slots{slot};
    const std::array<unsigned, 1> ids{id};
    std::array<unsigned, 1> top{};
    decoder.decode_slots(slots, ids, top);
    return top[0];
}

inline ScreenReference screened_scores(const char* checkpoint, weight::FormatId storage) {
    // The serving construction (inkling2_mini_serve_device.hpp): grouped decode, default step graphs, no verify
    // rows, no DSpark taps, one decode row per slot.
    constexpr unsigned lanes = 2U;
    MiniResidentDecoder decoder(checkpoint, 128U, 64U, false, MiniResidentBackend::default_step_graphs, storage, 0U,
                                false, lanes);
    decoder.allocate_slots(lanes);
    // The screened-head lever selected, as configure_mini_decoder passes it: CUDA builds the screened head over the BF16
    // head (a compact Q8_0 head keeps the full head); HIP keeps its own binding (BF16 head, unless STRIX_SCREENED_HEAD=0).
    const auto head = decoder.set_screened_head(true);
    const bool screening = head.screened;
#if defined(__HIPCC__)
    if (screening != (storage == weight::kBf16WeightStorage && core::hip_screened_head::enabled()))
        throw std::logic_error("the HIP head is not its own binding's");
    // Every lone lane steps through append (no cohort programs on HIP).
    const auto screened_lane = [&](unsigned) { return screening; };
#else
    if (screening != (storage == weight::kBf16WeightStorage))
        throw std::logic_error("the CUDA screened-head lever did not select the screened head on the BF16 head");
    // A lone lane on a slot other than 0 runs its width-1 cohort program (the full head) while cohort programs are on.
    const auto screened_lane = [&](unsigned slot) {
        return screening && (slot == 0U || decoder.cohort_programs().width(1U, lanes) == 0U);
    };
#endif
    std::cout << "INFO screened head " << (screening ? "on" : "off") << " (head " << mini_head_field(head) << ")\n";
    std::vector<unsigned> prompt(kScreenPromptRows);
    for (std::size_t i = 0; i < prompt.size(); ++i)
        prompt[i] = 1000U + static_cast<unsigned>((i * 173U + i / 13U) % 20000U);

    ScreenReference reference;
    reference.scores.resize(std::size_t{kScreenSteps} * kOutputVocabulary);
    reference.ids.push_back(1000U + (kScreenPromptRows * 173U + kScreenPromptRows / 13U) % 20000U);
    (void)decoder.prefill_slot_first(0U, prompt, true);
    for (unsigned step = 0; step < kScreenSteps; ++step) {
        const auto row = std::span(reference.scores).subspan(std::size_t{step} * kOutputVocabulary, kOutputVocabulary);
        reference.top.push_back(decoder.decode(reference.ids[step], row));  // host logits: the full head
        reference.ids.push_back(screen_sample(row, step));
    }

    std::vector<float> scores(kOutputVocabulary);
    // A: one sampled lane stepping alone from its first step. Its greedy picks are those of the full head (check_step)
    // and, with the screened head, every one of them ran screened (the decoder's count of screened picks).
    (void)decoder.prefill_slot_first(0U, prompt, true);
    const auto picks_before = decoder.screened_picks();
    for (unsigned step = 0; step < kScreenSteps; ++step) {
        const auto top = step_lane(decoder, 0U, reference.ids[step]);
        decoder.read_slot_logits(0U, scores);
        check_step("lone sampled lane", step, scores, top, reference);
    }
    const auto screened_steps = decoder.screened_picks() - picks_before;
    if (screened_steps != (screened_lane(0U) ? kScreenSteps : 0U))
        throw std::logic_error("lone lane: " + std::to_string(screened_steps) + " of " + std::to_string(kScreenSteps) +
                               " greedy picks ran screened with the screened head " + (screening ? "on" : "off"));
    std::cout << "PASS Inkling Mini screened scores: lone sampled lane, " << kScreenSteps << " steps, screened picks "
              << screened_steps << '\n';

    // B: a packed two-lane step, then lane 0 greedy alone (scores never read), then lane 1 sampled alone.
    (void)decoder.prefill_slot_first(0U, prompt, true);
    (void)decoder.prefill_slot_first(1U, prompt, true);
    {
        const std::array<unsigned, lanes> slots{0U, 1U};
        const std::array<unsigned, lanes> ids{reference.ids[0], reference.ids[0]};
        std::array<unsigned, lanes> top{};
        decoder.decode_slots(slots, ids, top);
        for (unsigned lane = 0; lane < lanes; ++lane) {
            decoder.read_slot_logits(lane, scores);
            check_step("packed lanes", 0U, scores, top[lane], reference);
        }
    }
    for (unsigned step = 1; step < kScreenSteps; ++step) {
        const unsigned greedy = step % 2U == 1U ? 0U : 1U;
        const unsigned sampled = 1U - greedy;
        if (step_lane(decoder, greedy, reference.ids[step]) != reference.top[step])
            throw std::runtime_error("greedy lane: step " + std::to_string(step) + " pick differs from the full head");
        // A screened greedy pick projects no score row (screened_lane: the lanes the screened head picks).
        const auto deferred = scores_deferred(decoder);
        if (deferred && *deferred != screened_lane(greedy))
            throw std::logic_error(std::string("greedy lane: score row ") + (*deferred ? "deferred" : "projected") +
                                   " with the screened head " + (screening ? "on" : "off"));
        const auto top = step_lane(decoder, sampled, reference.ids[step]);
        decoder.read_slot_logits(0U, scores);
        check_step("sampled lane after a greedy lane", step, scores, top, reference);
        if (scores_deferred(decoder).value_or(false)) throw std::logic_error("scores still deferred after a score read");
    }
    std::cout << "PASS Inkling Mini screened scores: sampled lane after packed and greedy lanes\n";
    return reference;
}

// Verify rows level 2 on a decoder built like the command line's verifying route plus four cohort rows.
// Verify rows level 2 over distinct lanes: lane l has its own prompt (length and ids), its own greedy continuation and
// its own full-head score rows, taken from one-lane steps (a one-row pick takes the same head at every level, and a read
// gets the per-row head every level-1 cohort row runs), so a swapped or stale deferred row fails against its own lane.
inline constexpr unsigned kLaneSteps = 4U;

struct LaneReference {
    std::vector<unsigned> prompt;
    std::vector<unsigned> ids;  // input of step s (ids[0] = the prefill's pick); ids[s + 1] = top[s]
    std::vector<unsigned> top;
    std::vector<float> scores;
    [[nodiscard]] std::span<const float> row(unsigned step) const {
        return std::span(scores).subspan(std::size_t{step} * kOutputVocabulary, kOutputVocabulary);
    }
};

inline std::vector<unsigned> lane_prompt(unsigned lane) {
    std::vector<unsigned> prompt(kScreenPromptRows - 3U * lane);
    for (std::size_t i = 0; i < prompt.size(); ++i)
        prompt[i] = 1000U + static_cast<unsigned>((i * 173U + i / 13U + 7919U * lane) % 20000U);
    return prompt;
}

inline void check_lane(const std::string& where, const std::vector<LaneReference>& lanes, unsigned lane, unsigned step,
                       std::span<const float> actual, std::optional<unsigned> top) {
    const auto same = [&](const LaneReference& other, unsigned s) {
        return std::memcmp(actual.data(), other.row(s).data(), actual.size_bytes()) == 0;
    };
    if (!same(lanes[lane], step)) {
        std::string stale = "no reference row";
        for (unsigned other = 0; other < lanes.size(); ++other)
            for (unsigned s = 0; s < kLaneSteps; ++s)
                if (same(lanes[other], s)) stale = "lane " + std::to_string(other) + " step " + std::to_string(s);
        throw std::runtime_error(where + ": lane " + std::to_string(lane) + " step " + std::to_string(step) +
                                 " read scores that differ from its full-head row (they equal " + stale + ")");
    }
    if (top && *top != lanes[lane].top[step])
        throw std::runtime_error(where + ": lane " + std::to_string(lane) + " step " + std::to_string(step) +
                                 " greedy pick differs from the full head");
}

inline void lanes_pick_scores(const char* checkpoint, weight::FormatId storage) {
    constexpr unsigned slots = 4U;
    MiniResidentDecoder decoder(checkpoint, 128U, 64U, false, MiniResidentBackend::default_step_graphs, storage,
                                MiniResidentDecoder::kMaxVerifyRows, false, slots);
    decoder.allocate_slots(slots);
    const auto admitted = decoder.set_verify_rows(2U);
#if defined(__HIPCC__)
    // Level 1 always (verify windows exist); the lanes pick only over the BF16 screened head.
    const unsigned expected =
        storage == weight::kBf16WeightStorage && core::hip_screened_head::enabled() ? 2U : 1U;
#else
    const unsigned expected = 0U;
#endif
    const unsigned capacity = decoder.lanes_pick_capacity();
    std::cout << "INFO verify rows level " << admitted << " (2 selected), lanes pick capacity " << capacity << '\n';
    if (admitted != expected)
        throw std::logic_error("verify rows level 2 must be admitted exactly when the BF16 screened head exists");
    if (admitted != 2U) {
        if (capacity != 0U) throw std::logic_error("a declined lanes pick reports a capacity");
        std::cout << "PASS Inkling Mini lanes pick: declined without the BF16 screened head (level " << admitted << ")\n";
        return;
    }
    if (capacity < 2U) throw std::logic_error("level 2 admitted without two screened lanes");

    std::vector<LaneReference> lanes(slots);
    std::vector<float> scores(kOutputVocabulary);
    for (unsigned lane = 0; lane < slots; ++lane) {
        auto& reference = lanes[lane];
        reference.prompt = lane_prompt(lane);
        reference.scores.resize(std::size_t{kLaneSteps} * kOutputVocabulary);
        reference.ids.push_back(decoder.prefill_slot_first(lane, reference.prompt, true));
        for (unsigned step = 0; step < kLaneSteps; ++step) {
            reference.top.push_back(step_lane(decoder, lane, reference.ids[step]));
            decoder.read_slot_logits(0U, std::span(reference.scores).subspan(std::size_t{step} * kOutputVocabulary,
                                                                              kOutputVocabulary));
            reference.ids.push_back(reference.top[step]);
        }
    }
    for (unsigned lane = 1; lane < slots; ++lane)
        if (lanes[lane].ids == lanes[0].ids) throw std::logic_error("the lanes are not distinct");

    // Cohorts of 2..4 distinct lanes: unread steps take the lanes pick up to the capacity (scores deferred, then read),
    // wider cohorts and announced reads keep the full head; every lane against its own rows.
    unsigned admitted_steps = 0U, declined_steps = 0U;
    for (unsigned width = 2U; width <= slots; ++width) {
        for (unsigned lane = 0; lane < width; ++lane)
            if (decoder.prefill_slot_first(lane, lanes[lane].prompt, true) != lanes[lane].ids[0])
                throw std::runtime_error("prefill pick differs on lane " + std::to_string(lane));
        std::array<unsigned, slots> slot_ids{0U, 1U, 2U, 3U}, ids{}, top{};
        for (unsigned step = 0; step < kLaneSteps; ++step) {
            const bool announced = step % 2U == 1U;
            for (unsigned lane = 0; lane < width; ++lane) ids[lane] = lanes[lane].ids[step];
            const auto before = decoder.lanes_picks();
            decoder.decode_slots(std::span(slot_ids).first(width), std::span(ids).first(width),
                                 std::span(top).first(width), announced);
            const bool expect_lanes = !announced && width <= capacity;
            if (decoder.scores_deferred() != expect_lanes || decoder.lanes_picks() - before != (expect_lanes ? 1U : 0U))
                throw std::logic_error("cohort C" + std::to_string(width) + " step " + std::to_string(step) +
                                       (expect_lanes ? ": the lanes pick did not run" : ": the lanes pick ran"));
            (expect_lanes ? admitted_steps : declined_steps) += announced ? 0U : 1U;
            for (unsigned lane = 0; lane < width; ++lane) {
                decoder.read_slot_logits(lane, scores);
                check_lane("cohort C" + std::to_string(width) + (expect_lanes ? " (lanes pick, deferred read)"
                                                                               : announced ? " (announced read)"
                                                                                           : " (full head)"),
                           lanes, lane, step, scores, top[lane]);
            }
        }
    }
    std::cout << "PASS Inkling Mini lanes pick: cohorts C2..C" << slots << " of distinct lanes, unread steps "
              << admitted_steps << " lanes / " << declined_steps << " above capacity " << capacity << '\n';

    // Another operation between a deferred cohort and its read: a verify window on slot 0 (its scores go to the verify
    // buffer) and a held-back prompt on slot 0 overwrite the cohort's head inputs; the late reads still return the
    // cohort's own rows, as level 1's eager rows do.
    for (unsigned lane = 0; lane < 3U; ++lane) (void)decoder.prefill_slot_first(lane, lanes[lane].prompt, true);
    const std::array<unsigned, 2> pair{1U, 2U};
    std::array<unsigned, 2> pair_top{};
    for (unsigned step = 0; step < 2U; ++step) {
        const std::array<unsigned, 2> pair_ids{lanes[1].ids[step], lanes[2].ids[step]};
        decoder.decode_slots(pair, pair_ids, pair_top, false);
        if (!decoder.scores_deferred()) throw std::logic_error("the C2 cohort did not defer its scores");
        const char* between = step == 0U ? "verify window" : "held-back prompt";
        if (step == 0U) {
            const auto result =
                decoder.forward_rows(lanes[0].ids.data(), 2U, MiniResidentDecoder::LogitRows::all);
            std::array<unsigned, 2> verify_top{};
            decoder.read_top(result, 2U, verify_top.data());
            if (verify_top[0] != lanes[0].top[0] || verify_top[1] != lanes[0].top[1])
                throw std::runtime_error("the verify window on slot 0 picks differ from the full head");
            decoder.commit(2U);
        } else {
            decoder.prefill_slot(0U, lanes[0].prompt, true);
        }
        for (unsigned row = 0; row < 2U; ++row) {
            decoder.read_slot_logits(row, scores);
            check_lane(std::string("deferred C2 read after a ") + between + " on slot 0", lanes, pair[row], step, scores,
                       pair_top[row]);
        }
    }
    std::cout << "PASS Inkling Mini lanes pick: deferred rows read after a verify window and a prompt on another slot\n";
}

}  // namespace strix::inkling2::test

int main(int argc, char** argv) {
    using namespace strix;
    try {
        if (argc < 2) throw std::invalid_argument("expected MODEL.safetensors [--qualify-q8]");
        auto storage = weight::kBf16WeightStorage;
        for (int i = 2; i < argc; ++i) {
            const std::string_view option{argv[i]};
            if (option == "--qualify-q8") storage = weight::kQ8WeightStorage;
            else throw std::invalid_argument("unknown option: " + std::string(option));
        }
        const auto reference = inkling2::test::screened_scores(argv[1], storage);
        static_cast<void>(reference);
        inkling2::test::lanes_pick_scores(argv[1], storage);
        return 0;
    } catch (const std::exception& error) {
        std::cerr << "FAIL Inkling Mini screened scores: " << error.what() << '\n';
        return 1;
    }
}
