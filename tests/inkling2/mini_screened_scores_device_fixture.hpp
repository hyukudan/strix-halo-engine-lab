#pragma once

// A sampled serving lane that steps alone reads the final scores of its own step (read_slot_logits), bit for bit
// those of the full head, also when that step's pick was screened (the screened greedy pick does not write the
// score row). Reference: the full head through decode(token, host logits), which never screens. Sampling is the
// shared sample_token_row with a fixed policy and fixed uniforms, so both runs sample the same ids exactly when
// their scores agree. Covers a lane sampled alone from the first step, and a lane sampled alone after a packed
// cohort step and after another lane's greedy step (whose scores are never read and never projected); verify windows
// of 2-4 rows read greedily, read lazily (unannounced score read) and announced (LogitRows::scores), each followed by
// a partial commit, then a decode step; packed cohorts of 2, 4 and 8 lanes with and without an announced score read
// (8 lanes exceed one pass-1 group: the rows head). Announced reads never defer a score row.

#include "inkling2/inkling2_mini_resident_decoder.hpp"
#include "strix/token_sampling.hpp"

#include <algorithm>
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
inline constexpr unsigned kScreenSteps = 12U;

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

inline unsigned step_lane(MiniResidentDecoder& decoder, unsigned slot, unsigned id, bool reads_scores = false) {
    const std::array<unsigned, 1> slots{slot};
    const std::array<unsigned, 1> ids{id};
    std::array<unsigned, 1> top{};
    decoder.decode_slots(slots, ids, top, reads_scores);
    if (reads_scores && scores_deferred(decoder).value_or(false))
        throw std::logic_error("an announced score read deferred its score row");
    return top[0];
}

inline void screened_scores(const char* checkpoint, weight::FormatId storage) {
    // Grouped decode, default step graphs, no DSpark taps (the serving construction), plus verify rows and eight
    // decode rows for the verify windows and the C8 cohort.
    constexpr unsigned lanes = 2U, max_lanes = 8U;
    MiniResidentDecoder decoder(checkpoint, 128U, 64U, false, MiniResidentBackend::default_step_graphs, storage,
                                MiniResidentDecoder::kMaxVerifyRows, false, max_lanes);
    decoder.allocate_slots(max_lanes);
#if defined(__HIPCC__)
    constexpr bool kHipScreen = true;
    const bool screening = storage == weight::kBf16WeightStorage && core::hip_screened_head::enabled();
#else
    constexpr bool kHipScreen = false;
    // The CUDA BF16 head screens one-row greedy picks unless STRIX_SCREENED_HEAD=0; the compact (Q8_0) head never does.
    const bool screening = decoder.step_program_screen().get() != nullptr;
#endif
    std::cout << "INFO screened head " << (screening ? "on" : "off") << '\n';
    std::vector<unsigned> prompt(kScreenPromptRows);
    for (std::size_t i = 0; i < prompt.size(); ++i)
        prompt[i] = 1000U + static_cast<unsigned>((i * 173U + i / 13U) % 20000U);

    ScreenReference reference;
    reference.scores.resize(std::size_t{kScreenSteps} * kOutputVocabulary);
    reference.ids.push_back(1000U + (kScreenPromptRows * 173U + kScreenPromptRows / 13U) % 20000U);
    decoder.prefill_slot(0U, prompt, true);
    for (unsigned step = 0; step < kScreenSteps; ++step) {
        const auto row = std::span(reference.scores).subspan(std::size_t{step} * kOutputVocabulary, kOutputVocabulary);
        reference.top.push_back(decoder.decode(reference.ids[step], row));  // host logits: the full head
        reference.ids.push_back(screen_sample(row, step));
    }

    std::vector<float> scores(kOutputVocabulary);
    // A: one sampled lane stepping alone from its first step, its score read announced on odd steps only.
    decoder.prefill_slot(0U, prompt, true);
    for (unsigned step = 0; step < kScreenSteps; ++step) {
        const auto top = step_lane(decoder, 0U, reference.ids[step], step % 2U == 1U);
        decoder.read_slot_logits(0U, scores);
        check_step("lone sampled lane", step, scores, top, reference);
    }
    std::cout << "PASS Inkling Mini screened scores: lone sampled lane, " << kScreenSteps << " steps\n";

    // B: a packed two-lane step, then lane 0 greedy alone (scores never read), then lane 1 sampled alone.
    decoder.prefill_slot(0U, prompt, true);
    decoder.prefill_slot(1U, prompt, true);
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
        // A screened greedy pick projects no score row (the BF16 head screens unless STRIX_SCREENED_HEAD=0).
        const auto deferred = scores_deferred(decoder);
        if (step == 1U && deferred) std::cout << "INFO greedy lane deferred its score row: " << *deferred << '\n';
        // HIP: the switch decides; the CUDA screen also needs its own admission, so only the scores are checked there.
        if constexpr (kHipScreen) {
            if (deferred && *deferred != screening)
                throw std::logic_error(std::string("greedy lane: score row ") + (*deferred ? "deferred" : "projected") +
                                       " with the screened head " + (screening ? "on" : "off"));
        }
        const auto top = step_lane(decoder, sampled, reference.ids[step]);
        decoder.read_slot_logits(0U, scores);
        check_step("sampled lane after a greedy lane", step, scores, top, reference);
        if (scores_deferred(decoder).value_or(false)) throw std::logic_error("scores still deferred after a score read");
    }
    std::cout << "PASS Inkling Mini screened scores: sampled lane after packed and greedy lanes\n";

    // C: verify windows on slot 0 (greedy picks, then the scores read lazily or announced), partial commits, decode.
    decoder.prefill_slot(0U, prompt, true);
    struct Window {
        unsigned width, keep;
        MiniResidentDecoder::LogitRows read;
    };
    constexpr std::array<Window, 4> windows{{{2U, 1U, MiniResidentDecoder::LogitRows::all},
                                             {3U, 3U, MiniResidentDecoder::LogitRows::scores},
                                             {4U, 2U, MiniResidentDecoder::LogitRows::all},
                                             {4U, 4U, MiniResidentDecoder::LogitRows::scores}}};
    std::vector<float> rows(std::size_t{4} * kOutputVocabulary);
    unsigned cursor = 0U;
    for (const auto& window : windows) {
        const auto result = decoder.forward_rows(reference.ids.data() + cursor, window.width, window.read);
        const bool announced = window.read == MiniResidentDecoder::LogitRows::scores;
        if (announced && scores_deferred(decoder).value_or(false))
            throw std::logic_error("verify: an announced score read deferred its rows");
        std::array<unsigned, MiniResidentDecoder::kMaxVerifyRows> top{};
        decoder.read_top(result, window.width, top.data());
        decoder.read_logits_rows(result, window.width, rows.data());
        for (unsigned row = 0; row < window.width; ++row) {
            const auto actual = std::span<const float>(rows).subspan(std::size_t{row} * kOutputVocabulary, kOutputVocabulary);
            check_step(announced ? "verify window (announced read)" : "verify window (lazy read)", cursor + row, actual,
                       top[row], reference);
        }
        decoder.commit(window.keep);
        cursor += window.keep;
    }
    {
        const auto top = step_lane(decoder, 0U, reference.ids[cursor], true);
        decoder.read_slot_logits(0U, scores);
        check_step("decode after verify commits", cursor, scores, top, reference);
    }
    std::cout << "PASS Inkling Mini screened scores: verify windows 2-4, lazy and announced reads, partial commits\n";

    // D: packed cohorts of 2, 4 and 8 lanes, every lane on the same sequence; odd steps announce a score read.
    std::array<unsigned, max_lanes> slots{}, ids{}, top{};
    for (unsigned lane = 0; lane < max_lanes; ++lane) slots[lane] = lane;
    for (const unsigned width : {2U, 4U, 8U}) {
        for (unsigned lane = 0; lane < width; ++lane) decoder.prefill_slot(lane, prompt, true);
        for (unsigned step = 0; step < 3U; ++step) {
            const bool announced = step % 2U == 1U;
            std::fill_n(ids.begin(), width, reference.ids[step]);
            decoder.decode_slots(std::span(slots).first(width), std::span(ids).first(width), std::span(top).first(width),
                                 announced);
            if (announced && scores_deferred(decoder).value_or(false))
                throw std::logic_error("cohort: an announced score read deferred its rows");
            for (unsigned lane = 0; lane < width; ++lane) {
                decoder.read_slot_logits(lane, scores);
                check_step(width == 8U ? "cohort C8" : width == 4U ? "cohort C4" : "cohort C2", step, scores, top[lane],
                           reference);
            }
        }
    }
    std::cout << "PASS Inkling Mini screened scores: cohorts C2 C4 C8, lazy and announced reads\n";
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
        inkling2::test::screened_scores(argv[1], storage);
        return 0;
    } catch (const std::exception& error) {
        std::cerr << "FAIL Inkling Mini screened scores: " << error.what() << '\n';
        return 1;
    }
}
