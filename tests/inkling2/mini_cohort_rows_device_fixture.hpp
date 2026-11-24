#pragma once

#include "inkling2/inkling2_mini_resident_decoder.hpp"

#include <array>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <exception>
#include <iostream>
#include <span>
#include <stdexcept>
#include <string>
#include <string_view>
#include <vector>

// Cohort decode lanes against their lone decode (review of the packed cohort serving route).
// Widths 17 (packed: passes 9 + 8) and 34 (packed: passes 12 + 11 + 11). Every lane has its own prompt length (distinct
// positions) and the cohort calls list the slots in a nonsequential order. For each of `kSteps` greedy steps the test
// compares every lane's full score row (read_slot_logits, the pitched row the pick saw), its pick and, through the
// fed picks, its continuation with the lane decoded alone.
// - packed: every lane must equal its lone decode bit for bit (the packed contract) -> PASS lines, else FAIL.
// - grouped: no batch-invariance contract; the test prints how many lanes equal lone and a hash of all cohort score
//   rows and picks, compared byte for byte against the same test built on main (grouped serving keeps its bits).
// A cohort that repeats a slot (16 lanes + slot 0 again: a second pass on packed) is refused before any lane moves:
// the cohort steps that follow would differ from lone otherwise.
namespace strix::inkling2::test {

inline constexpr unsigned kCohortSteps = 4U;
inline constexpr unsigned kCohortMaxLanes = 34U;

inline std::uint64_t fnv1a(std::uint64_t hash, const void* data, std::size_t bytes) {
    const auto* bytes_in = static_cast<const unsigned char*>(data);
    for (std::size_t i = 0; i < bytes; ++i) {
        hash ^= bytes_in[i];
        hash *= 1099511628211ULL;
    }
    return hash;
}

inline std::vector<unsigned> cohort_prompt(unsigned slot) {
    std::vector<unsigned> ids(7U + 3U * slot);
    for (std::size_t i = 0; i < ids.size(); ++i)
        ids[i] = 1000U + static_cast<unsigned>((slot * 389U + i * 173U + i / 5U) % 20000U);
    return ids;
}

inline unsigned cohort_first_feed(unsigned slot) { return 2000U + 97U * slot; }

inline void cohort_rows(const char* checkpoint, bool packed) {
    constexpr unsigned context = 160U;
    constexpr unsigned prefill_chunk = 64U;
    const auto moe = packed ? "packed" : "grouped";
    // The EXACT screened greedy head (on by default) picks a lone lane's token without writing its score row, so the
    // lone reference would read a stale row 0. Full head rows on every route: the comparison is of real score rows.
    ::setenv("STRIX_SCREENED_HEAD", "0", 1);
    MiniResidentDecoder decoder(checkpoint, context, prefill_chunk, packed, MiniResidentBackend::default_step_graphs,
                                weight::kBf16WeightStorage, 0U, false, kCohortMaxLanes);
    decoder.allocate_slots(kCohortMaxLanes);
    const std::size_t width = kOutputVocabulary;

    // Lone reference: each slot decoded alone (one lane per call).
    std::vector<float> lone(std::size_t{kCohortMaxLanes} * kCohortSteps * width);
    std::vector<unsigned> lone_top(std::size_t{kCohortMaxLanes} * kCohortSteps);
    const auto lone_row = [&](unsigned slot, unsigned step) {
        return std::span(lone).subspan((std::size_t{slot} * kCohortSteps + step) * width, width);
    };
    std::uint64_t lone_hash = 1469598103934665603ULL;
    for (unsigned slot = 0; slot < kCohortMaxLanes; ++slot) {
        decoder.prefill_slot(slot, cohort_prompt(slot), true);
        unsigned feed = cohort_first_feed(slot);
        for (unsigned step = 0; step < kCohortSteps; ++step) {
            const std::array slots{slot};
            const std::array ids{feed};
            std::array<unsigned, 1> top{};
            decoder.decode_slots(slots, ids, top);
            decoder.read_slot_logits(0U, lone_row(slot, step));
            lone_top[std::size_t{slot} * kCohortSteps + step] = top[0];
            feed = top[0];
        }
    }
    lone_hash = fnv1a(lone_hash, lone.data(), lone.size() * sizeof(float));
    lone_hash = fnv1a(lone_hash, lone_top.data(), lone_top.size() * sizeof(unsigned));
    std::printf("HASH Inkling Mini cohort %s lone %016llx\n", moe, static_cast<unsigned long long>(lone_hash));

    std::vector<float> row(width);
    for (const auto [lanes, stride] : {std::array{17U, 7U}, std::array{34U, 9U}}) {
        for (unsigned slot = 0; slot < lanes; ++slot) decoder.prefill_slot(slot, cohort_prompt(slot), true);
        if (lanes == 17U) {
            std::array<unsigned, 17> repeated{};
            for (unsigned r = 0; r < 16U; ++r) repeated[r] = r;
            repeated[16] = 0U;
            std::array<unsigned, 17> feeds{}, top{};
            for (unsigned r = 0; r < 17U; ++r) feeds[r] = cohort_first_feed(repeated[r]);
            bool refused = false;
            try {
                decoder.decode_slots(repeated, feeds, top);
            } catch (const std::invalid_argument&) {
                refused = true;
            }
            if (!refused) throw std::runtime_error("a cohort that repeats slot 0 was not refused");
            std::printf("PASS Inkling Mini cohort %s repeated slot refused\n", moe);
        }
        // Row r of every call is slot (r * stride + 3) % lanes: a permutation (stride coprime to lanes).
        std::vector<unsigned> order(lanes), feeds(lanes), top(lanes);
        for (unsigned r = 0; r < lanes; ++r) {
            order[r] = (r * stride + 3U) % lanes;
            feeds[r] = cohort_first_feed(order[r]);
        }
        std::uint64_t hash = 1469598103934665603ULL;
        unsigned equal_lanes = 0U;
        std::vector<bool> lane_equal(lanes, true);
        for (unsigned step = 0; step < kCohortSteps; ++step) {
            decoder.decode_slots(order, feeds, top);
            for (unsigned r = 0; r < lanes; ++r) {
                decoder.read_slot_logits(r, row);
                hash = fnv1a(hash, row.data(), row.size() * sizeof(float));
                const auto slot = order[r];
                const bool same = std::memcmp(row.data(), lone_row(slot, step).data(), width * sizeof(float)) == 0 &&
                                  top[r] == lone_top[std::size_t{slot} * kCohortSteps + step];
                if (!same) lane_equal[r] = false;
                if (!same && packed)
                    throw std::runtime_error("packed cohort C" + std::to_string(lanes) + " row " + std::to_string(r) +
                                             " (slot " + std::to_string(slot) + ", step " + std::to_string(step) +
                                             ") differs from its lone decode");
                feeds[r] = top[r];
            }
            hash = fnv1a(hash, top.data(), top.size() * sizeof(unsigned));
        }
        for (const bool equal : lane_equal) equal_lanes += equal ? 1U : 0U;
        std::printf("HASH Inkling Mini cohort %s C%u %016llx\n", moe, lanes, static_cast<unsigned long long>(hash));
        if (packed)
            std::printf("PASS Inkling Mini cohort packed C%u lanes == lone %u/%u (scores, picks, %u steps)\n", lanes,
                        equal_lanes, lanes, kCohortSteps);
        else
            std::printf("INFO Inkling Mini cohort grouped C%u lanes == lone %u/%u (no batch-invariance contract)\n",
                        lanes, equal_lanes, lanes);
    }
}

}  // namespace strix::inkling2::test

int main(int argc, char** argv) {
    using namespace strix;
    try {
        if (argc < 2) throw std::invalid_argument("expected MODEL.safetensors [--grouped-decode]");
        bool packed = true;
        for (int i = 2; i < argc; ++i) {
            const std::string_view option{argv[i]};
            if (option == "--grouped-decode") packed = false;
            else throw std::invalid_argument("unknown option: " + std::string(option));
        }
        inkling2::test::cohort_rows(argv[1], packed);
        return 0;
    } catch (const std::exception& error) {
        std::cerr << "FAIL Inkling Mini cohort rows: " << error.what() << '\n';
        return 1;
    }
}
