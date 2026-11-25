#pragma once

// Serving cohort lanes against each lane decoded alone, bit for bit, plus canonical digests for build-to-build
// identity. Backend-neutral body: on CUDA decode_slots() runs the width-W cohort program (captured at the first
// step, replayed afterwards); on HIP it runs the lane-batched append. Every row's final logits, the K/V rows each
// lane appended at every visit and the lane's recurrent histories must equal those of the same sequence stepped
// alone through slot 0. The DIGEST lines (FNV-1a 64) let two builds be compared without logits leaving the process.

#include "inkling2/inkling2_mini_resident_decoder.hpp"
#include "strix/serve/cohort_decode.hpp"
#include "strix/serve/prompt_piece.hpp"

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

// The serving construction (make_mini_device_cohort, inkling2_mini_serve_device.hpp) for the backend's admission of
// the served route lever: with it the command line's decode form and packed cohort passes; without it (lever off, or a
// backend that declines it) the incumbent served decoder (grouped decode).
inline std::unique_ptr<MiniResidentDecoder> served_decoder(const char* checkpoint, unsigned context, unsigned chunk,
                                                           weight::FormatId storage, unsigned lanes, unsigned tail,
                                                           bool served_route) {
    return served_route
        ? std::make_unique<MiniResidentDecoder>(checkpoint, context, chunk, kMiniPackedDecodeDefault,
              MiniResidentBackend::default_step_graphs, storage, 0U, false, lanes, tail, true)
        : std::make_unique<MiniResidentDecoder>(checkpoint, context, chunk, false,
              MiniResidentBackend::default_step_graphs, storage, 0U, false, lanes);
}

inline int cohort_lanes(const char* checkpoint, weight::FormatId storage, bool served_route) {
    const auto owned = served_decoder(checkpoint, kCohortContext, kCohortChunk, storage, kCohortSlots, 0U, served_route);
    auto& decoder = *owned;
    const bool packed_decode = decoder.prefill_emits_first();  // the admitted served route runs packed
    std::cout << "ROUTE served_route " << (packed_decode ? "on" : served_route ? "declined" : "off") << " moe_decode "
              << mini_decode_form_name(packed_decode) << " slots " << kCohortSlots << '\n';
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
        (void)decoder.prefill_slot_first(0U, prompt, true);
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
        for (const auto slot : slots) (void)decoder.prefill_slot_first(slot, cohort_prompt(slot), true);
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
                // The incumbent served decoder (grouped, the whole cohort in one forward) keeps the lone bits up to the
                // verify rows (each row on the decode arithmetic) and makes no lane contract past them: those rows are
                // printed as UNCHECKED; their identity with the base build is the base/head digest comparison.
                if (!packed_decode && width > kMiniMaxVerifyRows) {
                    std::cout << "UNCHECKED w=" << width << " step=" << step << " slot=" << slot << '\n';
                    continue;
                }
                if (std::memcmp(scores.data(), expected.data(), expected.size_bytes()) != 0) fail(width, step, slot, "logits");
                if (s.top != lone[slot][step].top) fail(width, step, slot, "top");
                if (s.position != lone[slot][step].position) fail(width, step, slot, "position");
                if (s.kv != lone[slot][step].kv) fail(width, step, slot, "kv rows");
                if (s.history != lone[slot][step].history) fail(width, step, slot, "histories");
            }
        }
#if !defined(__HIPCC__)
        // The program width the policy resolves for these lanes (STRIX_COHORT_WIDTHS=classes: 3 lanes replay the width-4
        // program, 9 lanes the width-16 one); readiness and a declined capture are read at that width.
        const auto graph_width = decoder.cohort_programs().width(width, kCohortSlots);
        const bool program = graph_width != 0U && decoder.cohort_graphs().policy().ready(graph_width) &&
                             !decoder.cohort_graphs().policy().failed(graph_width);
        std::cout << "COHORT w=" << width << " graph_width=" << graph_width
                  << " program=" << (program ? "ready" : "eager") << '\n';
        // A width whose class cannot round up inside the slot capacity (exec::cohort_width_class: 17 lanes with classes
        // and capacity 17) runs eagerly: that is a valid outcome, not a miss. Both decode forms run the width programs.
        const bool expect_program = graph_width != 0U;
        if (expect_program && !program) fail(width, 0U, 0U, "no cohort program at this width");
        if (!expect_program && program) fail(width, 0U, 0U, "cohort program at a width that must run eagerly");
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
              << (packed_decode ? ", widths 2 3 4 8 9 16 17, " : ", widths 2 3 4 8 (9 16 17 digests only), ") << kCohortSteps
              << " steps, logits + KV rows + histories bitwise)\n";
    return 0;
}

// Served prefill == the command line's prefill (--served-prefill). One decoder in the served construction (chunk =
// the command line's, tail capacity, two lanes). For each prompt length around the chunk, tail and key-split edges:
//   cli    : prefill() of slot 0 (the single-session call of the command line), host logits = the full head;
//   served : prefill_slot_first() of the whole prompt as the cohort issues it with no other lane decoding (regions of
//            prompt_slice_rows with the decoder's tail capacity), the first id from it, scores read_slot_logits(0);
//   pieces : the same prompt in the cohort's pieces beside a decoding lane (prompt_piece_rows inside each chunk of a
//            region, prefill_call_chunk_rows / prefill_piece_rule), one decode step of lane 1 between pieces;
//   repeat : served again (two identical served requests).
// First id, raw final scores (memcmp), the K/V rows of every committed position and the histories after the prompt, then four decode
// steps (ids, scores, last K/V rows, histories) must be equal across the four arms. The command line builds its
// decoder per prompt (chunk clamped to the prompt, tail = the actual remainder); the host test
// inkling2-mini-schedule-test proves those calls give every row the same chunk forms as this construction.
inline constexpr unsigned kServedContext = 8320U;
inline constexpr unsigned kServedSteps = 4U;
inline constexpr std::array<unsigned, 8> kServedPrompt{1U, 17U, 127U, 128U, 129U, 4113U, 8193U, 8209U};

// --served-prompts LIST: prompt lengths instead of kServedPrompt, comma-separated, each N or A-B (inclusive), every
// length in 1..kServedContext - kServedSteps - 1 (e.g. 8193-8208,8234: one row past the 8192 chunk up to a 16-row
// remainder, and a remainder the final chunk absorbs).
inline std::vector<unsigned> parse_served_prompts(std::string_view list) {
    const auto number = [](std::string_view text) {
        unsigned value = 0;
        if (text.empty() || text.size() > 6U) throw std::invalid_argument("bad --served-prompts length");
        for (const char c : text) {
            if (c < '0' || c > '9') throw std::invalid_argument("bad --served-prompts length");
            value = value * 10U + static_cast<unsigned>(c - '0');
        }
        if (value == 0U || value + kServedSteps + 1U > kServedContext)
            throw std::invalid_argument("--served-prompts length outside 1.." + std::to_string(kServedContext - kServedSteps - 1U));
        return value;
    };
    std::vector<unsigned> lengths;
    while (!list.empty()) {
        const auto comma = list.find(',');
        const auto item = list.substr(0, comma);
        list = comma == std::string_view::npos ? std::string_view{} : list.substr(comma + 1U);
        const auto dash = item.find('-');
        const unsigned first = number(item.substr(0, dash));
        const unsigned last = dash == std::string_view::npos ? first : number(item.substr(dash + 1U));
        if (last < first) throw std::invalid_argument("bad --served-prompts range");
        for (unsigned rows = first; rows <= last; ++rows) lengths.push_back(rows);
    }
    if (lengths.empty()) throw std::invalid_argument("empty --served-prompts");
    return lengths;
}

struct PrefillArm {
    unsigned first = 0;
    std::vector<float> scores;
    std::uint64_t kv = 0, history = 0;
    std::array<LaneStep, kServedSteps> steps{};
};

inline std::uint64_t digest_planes(MiniResidentDecoder& decoder, unsigned slot, std::vector<std::uint16_t>& plane,
                                   std::vector<float>& history, std::uint64_t& history_hash) {
    // The K and V rows of every committed position at every visit (rows past the frontier are not state).
    const auto view = decoder.slot_view(slot);
    Fnv64 kv;
    for (const auto device : {view.kv.keys, view.kv.values}) {
        plane.resize(device.size());
        mini_gpu::copy(plane.data(), device.data(), device.size_bytes(), mini_gpu::device_to_host, nullptr);
        mini_gpu::synchronize(nullptr);
        for (const auto visit : MiniSessionStatePlan::visits()) {
            const auto& layout = view.plan.visit(visit);
            for (unsigned position = 0; position < view.committed; ++position) {
                const auto offset =
                    static_cast<std::size_t>(layout.kv_offset + attn::cache_rows::row_offset(layout.kv, position));
                kv.add(plane.data() + offset, std::size_t{layout.kv.width} * sizeof(std::uint16_t));
            }
        }
    }
    history.resize(view.history.size());
    mini_gpu::copy(history.data(), view.history.data(), view.history.size_bytes(), mini_gpu::device_to_host, nullptr);
    mini_gpu::synchronize(nullptr);
    Fnv64 h;
    h.add(history.data(), history.size() * sizeof(float));
    history_hash = h.value;
    return kv.value;
}

inline int served_prefill(const char* checkpoint, weight::FormatId storage, std::span<const unsigned> prompt_lengths) {
#if defined(__HIPCC__)
    constexpr bool kHip = true;
#else
    constexpr bool kHip = false;
#endif
    const auto schedule = mini_prefill_schedule(kHip);
    const unsigned tail = mini_prefill_tail_capacity(schedule.chunk, schedule.tail_divisor, kServedContext);
    const auto owned = served_decoder(checkpoint, kServedContext, schedule.chunk, storage, 2U, tail, true);
    auto& decoder = *owned;
    // The served route under test: the command line's (the served route lever; a CUDA decoder declines it, and then
    // this arm has nothing to run: the incumbent held-back route never calls the emits-first prefill).
    if (decoder.prefill_emits_first() != kHip) throw std::logic_error("served route admission");
    if (!decoder.prefill_emits_first()) {
        std::cout << "SKIP served_prefill: the backend declines the served route lever\n";
        return 0;
    }
    decoder.allocate_slots(2U);
    const bool packed_decode = kMiniPackedDecodeDefault;
    const auto capacity = decoder.prefill_tail_rows();
    const auto rule = decoder.prefill_piece_rule();
    std::cout << "ROUTE served_prefill chunk " << schedule.chunk << " tail " << capacity << " pieces "
              << (rule.cuttable() ? kMiniServeDecodingRows : 0U) << " moe_decode " << mini_decode_form_name(packed_decode)
              << '\n';
    const std::size_t vocab = kOutputVocabulary;
    std::vector<std::uint16_t> plane, row;
    std::vector<float> history;
    Fnv64 all;
    unsigned failures = 0;
    // Lane 1: a short decoding request beside the pieces arm.
    std::vector<unsigned> other(40U);
    for (unsigned i = 0; i < other.size(); ++i) other[i] = cohort_token(3U, i);

    const auto finish = [&](PrefillArm& arm) {
        arm.kv = digest_planes(decoder, 0U, plane, history, arm.history);
        unsigned token = arm.first;
        for (unsigned step = 0; step < kServedSteps; ++step) {
            const std::array<unsigned, 1> slots{0U};
            const std::array<unsigned, 1> ids{token};
            std::array<unsigned, 1> top{};
            decoder.decode_slots(slots, ids, top);
            std::vector<float> scores(vocab);
            decoder.read_slot_logits(0U, scores);
            auto& s = arm.steps[step];
            s.top = top[0];
            Fnv64 h;
            h.add(scores.data(), scores.size() * sizeof(float));
            s.logits = h.value;
            digest_state(decoder, 0U, s, row, history);
            token = top[0];
        }
    };
    for (const auto rows : prompt_lengths) {
        std::vector<unsigned> prompt(rows);
        for (unsigned i = 0; i < rows; ++i) prompt[i] = cohort_token(5U, i);
        const std::span<const unsigned> ids(prompt);
        std::array<PrefillArm, 4> arms;
        // cli
        arms[0].scores.resize(vocab);
        decoder.reset_gate_session();  // a fresh single session, as the command line starts
        arms[0].first = decoder.prefill(ids, arms[0].scores);
        finish(arms[0]);
        // served (whole regions) and repeat
        for (const unsigned a : {1U, 3U}) {
            bool first_call = true;
            for (std::size_t at = 0; at < rows;) {
                const auto region = static_cast<std::size_t>(
                    serve::prompt_slice_rows(schedule.chunk, schedule.chunk, rows - at, capacity));
                arms[a].first = decoder.prefill_slot_first(0U, ids.subspan(at, region), first_call);
                first_call = false;
                at += region;
            }
            arms[a].scores.resize(vocab);
            decoder.read_slot_logits(0U, arms[a].scores);
            finish(arms[a]);
        }
        // pieces beside a decoding lane
        {
            unsigned other_token = decoder.prefill_slot_first(1U, other, true);
            bool first_call = true;
            for (std::size_t at = 0; at < rows;) {
                const auto region = serve::prompt_slice_rows(schedule.chunk, schedule.chunk, rows - at, capacity);
                const auto head = decoder.prefill_call_chunk_rows(region);
                for (const auto part : {head, region - head}) {
                    for (std::uint64_t left = part; left != 0U;) {
                        const auto piece = static_cast<std::size_t>(
                            serve::prompt_piece_rows(left, kMiniServeDecodingRows, schedule.chunk, rule));
                        arms[2].first = decoder.prefill_slot_first(0U, ids.subspan(at, piece), first_call);
                        first_call = false;
                        at += piece;
                        left -= piece;
                        if (at < rows) {
                            const std::array<unsigned, 1> slots{1U};
                            const std::array<unsigned, 1> next{other_token};
                            std::array<unsigned, 1> top{};
                            decoder.decode_slots(slots, next, top);
                            other_token = top[0];
                        }
                    }
                }
            }
            arms[2].scores.resize(vocab);
            decoder.read_slot_logits(0U, arms[2].scores);
            finish(arms[2]);
        }
        static constexpr std::array<const char*, 4> kNames{"cli", "served", "pieces", "repeat"};
        for (unsigned a = 0; a < 4U; ++a) {
            Fnv64 h;
            h.add(arms[a].scores.data(), arms[a].scores.size() * sizeof(float));
            char line[256];
            std::snprintf(line, sizeof(line), "DIGEST prefill rows=%u arm=%s first=%u scores=%016llx kv=%016llx hist=%016llx",
                          rows, kNames[a], arms[a].first, static_cast<unsigned long long>(h.value),
                          static_cast<unsigned long long>(arms[a].kv), static_cast<unsigned long long>(arms[a].history));
            all.add(line, std::strlen(line));
            std::cout << line << '\n';
            if (a == 0U) continue;
            const auto fail = [&](const char* what) {
                if (++failures <= 20U) std::cout << "FAIL " << kNames[a] << " != cli rows=" << rows << ' ' << what << '\n';
            };
            if (arms[a].first != arms[0].first) fail("first id");
            if (std::memcmp(arms[a].scores.data(), arms[0].scores.data(), vocab * sizeof(float)) != 0) fail("scores");
            if (arms[a].kv != arms[0].kv) fail("kv planes");
            if (arms[a].history != arms[0].history) fail("histories");
            for (unsigned step = 0; step < kServedSteps; ++step)
                if (!(arms[a].steps[step] == arms[0].steps[step])) fail("decode step");
        }
    }
    char line[64];
    std::snprintf(line, sizeof(line), "%016llx", static_cast<unsigned long long>(all.value));
    std::cout << "DIGEST ALL " << line << '\n';
    if (failures != 0U) {
        std::cout << "FAIL Inkling Mini served prefill: " << failures << " mismatches\n";
        return 1;
    }
    std::cout << "PASS Inkling Mini served prefill == CLI prefill (" << prompt_lengths.size()
              << " prompt lengths, whole / pieces beside a decoding lane / repeat; first id, scores, KV planes, "
                 "histories, " << kServedSteps << " decode steps bitwise)\n";
    return 0;
}

}  // namespace strix::inkling2::test

int main(int argc, char** argv) {
    using namespace strix;
    try {
        if (argc < 2)
            throw std::invalid_argument("expected MODEL.safetensors [--qualify-q8] [--served-route] [--served-prefill [--served-prompts LIST]]");
        auto storage = weight::kBf16WeightStorage;
        bool served_route = false;  // the served route lever (off: the incumbent served decoder)
        bool served_prefill = false;
        std::vector<unsigned> prompt_lengths(inkling2::test::kServedPrompt.begin(), inkling2::test::kServedPrompt.end());
        for (int i = 2; i < argc; ++i) {
            const std::string_view option{argv[i]};
            if (option == "--qualify-q8") storage = weight::kQ8WeightStorage;
            else if (option == "--served-route") served_route = true;
            else if (option == "--served-prefill") served_prefill = true;
            else if (option == "--served-prompts" && i + 1 < argc)
                prompt_lengths = inkling2::test::parse_served_prompts(argv[++i]);
            else throw std::invalid_argument("unknown option: " + std::string(option));
        }
        if (served_prefill)
            return inkling2::test::served_prefill(argv[1], storage, prompt_lengths);
#if defined(__HIPCC__)
        const bool admitted = served_route;
#else
        const bool admitted = false;  // a CUDA server declines the lever
#endif
        if (served_route && !admitted) std::cout << "NOTE served route lever declined by this backend\n";
        return inkling2::test::cohort_lanes(argv[1], storage, served_route);
    } catch (const std::exception& error) {
        std::cerr << "FAIL Inkling Mini cohort lanes: " << error.what() << '\n';
        return 1;
    }
}
