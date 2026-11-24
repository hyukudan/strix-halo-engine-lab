#pragma once

// Serving cohort lanes against each lane decoded alone, bit for bit, plus canonical digests for build-to-build
// identity. Backend-neutral body: on CUDA decode_slots() runs the width-W cohort program (captured at the first
// step, replayed afterwards); on HIP it runs the lane-batched append. Every row's final logits, the K/V rows each
// lane appended at every visit and the lane's recurrent histories must equal those of the same sequence stepped
// alone through slot 0. The DIGEST lines (FNV-1a 64) let two builds be compared without logits leaving the process.

#include "inkling2/inkling2_mini_resident_decoder.hpp"

#include <algorithm>
#include <array>
#include <cstdint>
#include <cstdio>
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

struct Fnv64 {
    std::uint64_t value = 0xcbf29ce484222325ULL;
    void add(const void* data, std::size_t bytes) {
        const auto* p = static_cast<const unsigned char*>(data);
        for (std::size_t i = 0; i < bytes; ++i) value = (value ^ p[i]) * 0x100000001b3ULL;
    }
    void add(std::uint64_t word) { add(&word, sizeof(word)); }
};

// One lane step: what the greedy pick saw and what the step left in the lane's state.
struct LaneStep {
    unsigned position = 0;
    unsigned top = 0;
    std::uint64_t logits = 0, kv = 0, history = 0;
    bool operator==(const LaneStep&) const = default;
};

inline constexpr unsigned kCohortSlots = 17U;
inline constexpr unsigned kCohortSteps = 12U;
inline constexpr unsigned kCohortContext = 2560U;
inline constexpr unsigned kCohortChunk = 512U;
// Every lane at its own position: below and past the 1024-key window / bias extent, and slot 7 crosses the
// 2048-row sliding ring (window 1024 + chunk 512 -> ring 2048) during its steps.
// Prompt lengths around the prefill chunk edges (512 / 1024 / 1536 / 2048).
inline constexpr std::array<unsigned, kCohortSlots> kPromptRows{517U, 301U, 1029U, 763U, 1250U, 640U, 1500U, 2040U,
                                                                 129U, 511U, 513U, 1023U, 1025U, 2047U, 777U, 64U, 1536U};
// 2..8 lanes run the verify route; 9 and 16 one pass of the decode route; 17 two passes (9 + 8) on HIP.
inline constexpr std::array<unsigned, 7> kCohortWidths{2U, 3U, 4U, 8U, 9U, 16U, 17U};

inline unsigned cohort_token(unsigned slot, unsigned index) {
    return 1000U + (index * 173U + slot * 7919U + index / 13U) % 20000U;
}

inline std::vector<unsigned> cohort_prompt(unsigned slot) {
    std::vector<unsigned> ids(kPromptRows[slot]);
    for (unsigned i = 0; i < ids.size(); ++i) ids[i] = cohort_token(slot, i);
    return ids;
}

// The K and V rows the last step appended at every visit (position committed - 1), and the whole histories.
inline void digest_state(MiniResidentDecoder& decoder, unsigned slot, LaneStep& step, std::vector<std::uint16_t>& row,
                         std::vector<float>& history) {
    const auto view = decoder.slot_view(slot);
    if (view.committed == 0U) throw std::logic_error("lane has no committed row");
    step.position = view.committed - 1U;
    Fnv64 kv;
    for (const auto visit : MiniSessionStatePlan::visits()) {
        const auto& layout = view.plan.visit(visit);
        const auto offset = static_cast<std::size_t>(layout.kv_offset + attn::cache_rows::row_offset(layout.kv, step.position));
        row.resize(layout.kv.width);
        for (const auto plane : {view.kv.keys, view.kv.values}) {
            mini_gpu::copy(row.data(), plane.data() + offset, row.size() * sizeof(std::uint16_t), mini_gpu::device_to_host,
                           nullptr);
            mini_gpu::synchronize(nullptr);
            kv.add(row.data(), row.size() * sizeof(std::uint16_t));
        }
    }
    step.kv = kv.value;
    history.resize(view.history.size());
    mini_gpu::copy(history.data(), view.history.data(), view.history.size_bytes(), mini_gpu::device_to_host, nullptr);
    mini_gpu::synchronize(nullptr);
    Fnv64 h;
    h.add(history.data(), history.size() * sizeof(float));
    step.history = h.value;
}

inline void print_digest(std::string_view route, unsigned width, unsigned step, unsigned slot, const LaneStep& s,
                         Fnv64& all) {
    char line[256];
    std::snprintf(line, sizeof(line), "DIGEST %.*s w=%u step=%u slot=%u pos=%u top=%u logits=%016llx kv=%016llx hist=%016llx",
                  static_cast<int>(route.size()), route.data(), width, step, slot, s.position, s.top,
                  static_cast<unsigned long long>(s.logits), static_cast<unsigned long long>(s.kv),
                  static_cast<unsigned long long>(s.history));
    all.add(line, std::strlen(line));
    std::cout << line << '\n';
}

inline int cohort_lanes(const char* checkpoint, weight::FormatId storage, bool packed_decode) {
    // The serving construction (inkling2_mini_serve_device.hpp): the resolved decode form (packed by default, as on
    // the command line), default step graphs, no verify rows, no DSpark taps, one decode row per slot.
    std::cout << "ROUTE moe_decode " << mini_decode_form_name(packed_decode) << " slots " << kCohortSlots << '\n';
    MiniResidentDecoder decoder(checkpoint, kCohortContext, kCohortChunk, packed_decode,
                                MiniResidentBackend::default_step_graphs, storage, 0U, false, kCohortSlots);
    decoder.allocate_slots(kCohortSlots);
    const std::size_t vocab = kOutputVocabulary;
    std::vector<std::uint16_t> row;
    std::vector<float> history, scores(vocab);
    Fnv64 all;

    // Reference: every slot's sequence stepped alone through slot 0 (the lone-lane decode route).
    std::vector<float> lone_logits(std::size_t{kCohortSlots} * kCohortSteps * vocab);
    std::array<std::array<LaneStep, kCohortSteps>, kCohortSlots> lone{};
    for (unsigned slot = 0; slot < kCohortSlots; ++slot) {
        const auto prompt = cohort_prompt(slot);
        decoder.prefill_slot(0U, prompt, true);
        for (unsigned step = 0; step < kCohortSteps; ++step) {
            const std::array<unsigned, 1> slots{0U};
            const std::array<unsigned, 1> ids{cohort_token(slot, kPromptRows[slot] + step)};
            std::array<unsigned, 1> top{};
            decoder.decode_slots(slots, ids, top);
            const auto out = std::span(lone_logits).subspan((std::size_t{slot} * kCohortSteps + step) * vocab, vocab);
            decoder.read_slot_logits(0U, out);
            auto& s = lone[slot][step];
            s.top = top[0];
            Fnv64 h;
            h.add(out.data(), out.size_bytes());
            s.logits = h.value;
            digest_state(decoder, 0U, s, row, history);
            if (s.position != kPromptRows[slot] + step) throw std::logic_error("lone lane position");
            print_digest("lone", 1U, step, slot, s, all);
        }
    }

    unsigned failures = 0;
    const auto fail = [&](unsigned width, unsigned step, unsigned slot, const char* what) {
        if (++failures <= 20U)
            std::cout << "FAIL lanes != lone w=" << width << " step=" << step << " slot=" << slot << ' ' << what << '\n';
    };
    for (const auto width : kCohortWidths) {
        // Lane row r is slot (r + 1) % width: lane 0 of the cohort is not slot 0.
        std::vector<unsigned> slots(width), ids(width), top(width);
        for (unsigned r = 0; r < width; ++r) slots[r] = (r + 1U) % width;
        for (const auto slot : slots) decoder.prefill_slot(slot, cohort_prompt(slot), true);
        for (unsigned step = 0; step < kCohortSteps; ++step) {
            for (unsigned r = 0; r < width; ++r) ids[r] = cohort_token(slots[r], kPromptRows[slots[r]] + step);
            decoder.decode_slots(slots, ids, top);
            for (unsigned r = 0; r < width; ++r) {
                const auto slot = slots[r];
                decoder.read_slot_logits(r, scores);
                LaneStep s;
                s.top = top[r];
                Fnv64 h;
                h.add(scores.data(), scores.size() * sizeof(float));
                s.logits = h.value;
                digest_state(decoder, slot, s, row, history);
                print_digest("lanes", width, step, slot, s, all);
                const auto expected = std::span(lone_logits).subspan((std::size_t{slot} * kCohortSteps + step) * vocab, vocab);
                if (std::memcmp(scores.data(), expected.data(), expected.size_bytes()) != 0) fail(width, step, slot, "logits");
                if (s.top != lone[slot][step].top) fail(width, step, slot, "top");
                if (s.position != lone[slot][step].position) fail(width, step, slot, "position");
                if (s.kv != lone[slot][step].kv) fail(width, step, slot, "kv rows");
                if (s.history != lone[slot][step].history) fail(width, step, slot, "histories");
            }
        }
#if !defined(__HIPCC__)
        const bool program = decoder.cohort_graphs().policy().ready(width);
        std::cout << "COHORT w=" << width << " program=" << (program ? "replayed" : "eager") << '\n';
        if (decoder.cohort_programs().enabled && !program) fail(width, 0U, 0U, "no cohort program at this width");
#endif
    }
    char line[64];
    std::snprintf(line, sizeof(line), "%016llx", static_cast<unsigned long long>(all.value));
    std::cout << "DIGEST ALL " << line << '\n';
    if (failures != 0U) {
        std::cout << "FAIL Inkling Mini cohort lanes: " << failures << " mismatches\n";
        return 1;
    }
    std::cout << "PASS Inkling Mini cohort lanes == lone (" << mini_decode_form_name(packed_decode)
              << ", widths 2 3 4 8 9 16 17, " << kCohortSteps
              << " steps, logits + KV rows + histories bitwise)\n";
    return 0;
}

}  // namespace strix::inkling2::test

int main(int argc, char** argv) {
    using namespace strix;
    try {
        if (argc < 2)
            throw std::invalid_argument("expected MODEL.safetensors [--qualify-q8] [--packed-decode | --grouped-decode]");
        auto storage = weight::kBf16WeightStorage;
        std::optional<bool> decode_form;
        for (int i = 2; i < argc; ++i) {
            const std::string_view option{argv[i]};
            if (option == "--qualify-q8") storage = weight::kQ8WeightStorage;
            else if (option == "--packed-decode") decode_form = true;
            else if (option == "--grouped-decode") decode_form = false;
            else throw std::invalid_argument("unknown option: " + std::string(option));
        }
        return inkling2::test::cohort_lanes(argv[1], storage, inkling2::mini_packed_decode(decode_form));
    } catch (const std::exception& error) {
        std::cerr << "FAIL Inkling Mini cohort lanes: " << error.what() << '\n';
        return 1;
    }
}
