#pragma once

// Serving cohort lanes against each lane decoded alone, bit for bit, plus canonical digests for build-to-build
// identity. Backend-neutral body: on CUDA decode_slots() runs the width-W cohort program (captured at the first
// step, replayed afterwards); on HIP it runs the lane-batched append. Every row's final logits, the K/V rows each
// lane appended at every visit and the lane's recurrent histories must equal those of the same sequence stepped
// alone through slot 0. The DIGEST lines (FNV-1a 64) let two builds be compared without logits leaving the process.

#include "inkling2/inkling2_mini_resident_decoder.hpp"
#include "strix/serve/cohort_decode.hpp"
#include "strix/serve/fixed_kv_cohort.hpp"
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

// The serving construction (make_mini_device_cohort, inkling2_mini_serve_device.hpp) for an ADMITTED served route: with
// it the command line's decode form and packed cohort passes; without it (lever off, or a backend that declines it: the
// caller passes the production admission, mini_served_prefill_emits_first) the incumbent served decoder (HIP grouped
// decode; CUDA packed lone lanes and grouped cohorts, mini_served_packed_decode).
#if defined(__HIPCC__)
inline constexpr bool kFixtureHip = true;
#else
inline constexpr bool kFixtureHip = false;
#endif

inline std::unique_ptr<MiniResidentDecoder> served_decoder(const char* checkpoint, unsigned context, unsigned chunk,
                                                           weight::FormatId storage, unsigned lanes, unsigned tail,
                                                           bool admitted) {
    return admitted
        ? std::make_unique<MiniResidentDecoder>(checkpoint, context, chunk, kMiniPackedDecodeDefault,
              MiniResidentBackend::default_step_graphs, storage, 0U, false, lanes, tail, false, false, true)
        : std::make_unique<MiniResidentDecoder>(checkpoint, context, chunk, mini_served_packed_decode(kFixtureHip, false),
              MiniResidentBackend::default_step_graphs, storage, 0U, false, lanes);
}

using MiniFixedCohort = serve::FixedKvCohort<MiniResidentDecoder>;

// One lane's prompt through the production adapter, as the server ingests it: emits-first prefills the whole prompt
// and takes the first id from its final row; the incumbent route prefills all but the last prompt token, which the
// first generated step feeds (decoded with the cohort).
inline void ingest_prompt(MiniFixedCohort& cohort, unsigned slot) {
    const auto prompt = cohort_prompt(slot);
    const auto tokens = cohort.prefill_emits_first() ? std::span<const unsigned>(prompt)
                                                     : std::span<const unsigned>(prompt).first(prompt.size() - 1U);
    cohort.prefill(slot, tokens, true);
}

// The id a lane feeds at generated step `step` (step 0 = the first generated id: on the incumbent route that step
// decodes the held-back prompt token; an emits-first lane takes step 0 from its prefill and calls this from step 1).
// Both routes feed input index prompt - 1 + step, so step 1 consumes index `prompt` on either route.
inline unsigned step_token(unsigned slot, unsigned step) { return cohort_token(slot, kPromptRows[slot] - 1U + step); }

// The last committed position after generated step `step` on either route: the held-back route commits prompt - 1 rows
// in prefill and one at step 0, emits-first commits the whole prompt in prefill; each later step commits one row.
inline unsigned step_position(unsigned slot, unsigned step) { return kPromptRows[slot] - 1U + step; }

inline int cohort_lanes(const char* checkpoint, weight::FormatId storage, bool requested, bool admitted) {
    auto owned = served_decoder(checkpoint, kCohortContext, kCohortChunk, storage, kCohortSlots, 0U, admitted);
    auto& decoder = *owned;
    MiniFixedCohort cohort(std::move(owned), kCohortContext, kCohortSlots, kCohortSlots);
    const bool emits_first = cohort.prefill_emits_first();
    const bool packed_decode = decoder.packed_decode();
    std::cout << "ROUTE served_route " << (admitted ? "on" : requested ? "declined" : "off") << " prompt "
              << (emits_first ? "emits-first" : "held-back") << " moe_decode " << mini_decode_form_name(packed_decode)
              << " slots " << kCohortSlots << '\n';
    // The forms the server runs: admitted = the command line's route (emits-first, packed); otherwise the held-back
    // route with the served decode form (CUDA: packed lone lanes, grouped cohorts: lanes == lone tests both forms).
    if (emits_first != admitted || packed_decode != mini_served_packed_decode(kFixtureHip, admitted))
        throw std::logic_error("served decoder runs a form its admission did not select");
    const std::size_t vocab = kOutputVocabulary;
    std::vector<std::uint16_t> row;
    std::vector<float> history;
    Fnv64 all;
    std::vector<serve::CohortGroup> groups;
    std::vector<std::uint32_t> keep, ids, top;

    // Generated step `step` of the given lanes through the adapter: step 0 of an emits-first lane is its prefill's
    // first id (prefill_top + its final row); every other step is one verify window of one row per lane + commit.
    // Calls `seen(r, top, logits)` per lane row before the commit.
    const auto run_step = [&](std::span<const unsigned> slots, unsigned step, auto&& seen) {
        if (step == 0U && emits_first) {
            for (unsigned r = 0; r < slots.size(); ++r) {
                ingest_prompt(cohort, slots[r]);
                const auto first = cohort.prefill_top(slots[r]);
                seen(r, first, cohort.read_final_logits(slots[r]));
            }
            return;
        }
        if (step == 0U)
            for (const auto slot : slots) ingest_prompt(cohort, slot);
        groups.clear();
        ids.resize(slots.size());
        top.resize(slots.size());
        for (unsigned r = 0; r < slots.size(); ++r) {
            groups.push_back({slots[r], 1U});
            ids[r] = step_token(slots[r], step);
        }
        cohort.verify(groups, ids, top);
        for (unsigned r = 0; r < slots.size(); ++r) seen(r, top[r], cohort.read_final_logits(slots[r]));
        keep.assign(slots.size(), 1U);
        cohort.commit(keep);
    };

    // Reference: every slot's sequence alone (the lone-lane route), the first generated id included.
    std::vector<float> lone_logits(std::size_t{kCohortSlots} * kCohortSteps * vocab);
    std::array<std::array<LaneStep, kCohortSteps>, kCohortSlots> lone{};
    for (unsigned slot = 0; slot < kCohortSlots; ++slot) {
        const std::array<unsigned, 1> one{slot};
        for (unsigned step = 0; step < kCohortSteps; ++step)
            run_step(one, step, [&](unsigned, unsigned pick, std::span<const float> logits) {
                const auto out = std::span(lone_logits).subspan((std::size_t{slot} * kCohortSteps + step) * vocab, vocab);
                if (logits.size() != out.size()) throw std::logic_error("lone lane logit width");
                std::memcpy(out.data(), logits.data(), out.size_bytes());
                auto& s = lone[slot][step];
                s.top = pick;
                Fnv64 h;
                h.add(out.data(), out.size_bytes());
                s.logits = h.value;
                digest_state(decoder, slot, s, row, history);
                if (s.position != step_position(slot, step))
                    throw std::logic_error("lone lane position: slot " + std::to_string(slot) + " step " +
                                           std::to_string(step) + " committed " + std::to_string(s.position) +
                                           " expected " + std::to_string(step_position(slot, step)));
                print_digest("lone", 1U, step, slot, s, all);
            });
    }

    unsigned failures = 0;
    const auto fail = [&](unsigned width, unsigned step, unsigned slot, const char* what) {
        if (++failures <= 20U)
            std::cout << "FAIL lanes != lone w=" << width << " step=" << step << " slot=" << slot << ' ' << what << '\n';
    };
    for (const auto width : kCohortWidths) {
        // Lane row r is slot (r + 1) % width: lane 0 of the cohort is not slot 0.
        std::vector<unsigned> slots(width);
        for (unsigned r = 0; r < width; ++r) slots[r] = (r + 1U) % width;
#if !defined(__HIPCC__)
        // Replay counters before this lane count: the witness below credits only replays of THIS arm (a lifetime
        // per-width flag could have been set by a smaller lane count padded to the same program width).
        const std::array<std::uint64_t, 2> replays_before{decoder.cohort_replay_count(false),
                                                          decoder.cohort_replay_count(true)};
#endif
        for (unsigned step = 0; step < kCohortSteps; ++step)
            run_step(slots, step, [&](unsigned r, unsigned pick, std::span<const float> scores) {
                const auto slot = slots[r];
                LaneStep s;
                s.top = pick;
                Fnv64 h;
                h.add(scores.data(), scores.size_bytes());
                s.logits = h.value;
                digest_state(decoder, slot, s, row, history);
                print_digest("lanes", width, step, slot, s, all);
                const auto expected = std::span(lone_logits).subspan((std::size_t{slot} * kCohortSteps + step) * vocab, vocab);
                // The incumbent served decoder (grouped, the whole cohort in one forward) keeps the lone bits up to the
                // verify rows (each row on the decode arithmetic) and makes no lane contract past them: those rows are
                // printed as UNCHECKED; their identity with the base build is the served base/head identity ticket
                // (complete ids + raw logits of the real server at 9 lanes), not this fixture.
                if (!packed_decode && width > kMiniMaxVerifyRows) {
                    std::cout << "UNCHECKED w=" << width << " step=" << step << " slot=" << slot << '\n';
                    return;
                }
                if (scores.size() != expected.size() ||
                    std::memcmp(scores.data(), expected.data(), expected.size_bytes()) != 0)
                    fail(width, step, slot, "logits");
                if (s.top != lone[slot][step].top) fail(width, step, slot, "top");
                if (s.position != lone[slot][step].position) fail(width, step, slot, "position");
                if (s.kv != lone[slot][step].kv) fail(width, step, slot, "kv rows");
                if (s.history != lone[slot][step].history) fail(width, step, slot, "histories");
            });
#if !defined(__HIPCC__)
        // The program width the policy resolves for these lanes (STRIX_COHORT_WIDTHS=classes: 3 lanes replay the width-4
        // program, 9 lanes the width-16 one); readiness, a declined capture and a COMPLETED replay are read at that width.
        // A width whose class cannot round up inside the slot capacity (exec::cohort_width_class: 17 lanes with classes
        // and capacity 17) resolves no program and runs eagerly: a valid outcome, not a miss.
        const auto graph_width = decoder.cohort_programs().width(width, kCohortSlots);
        const bool program = graph_width != 0U && decoder.cohort_graphs().policy().ready(graph_width) &&
                             !decoder.cohort_graphs().policy().failed(graph_width);
        // The mode this lane count implies: padded when the program is wider than the lanes, else full-width. This arm
        // must have completed at least one replay in that mode and none in the other.
        const bool padded = graph_width > width;
        const auto replays = decoder.cohort_replay_count(padded) - replays_before[padded ? 1U : 0U];
        const auto other = decoder.cohort_replay_count(!padded) - replays_before[padded ? 0U : 1U];
        const bool replayed = graph_width != 0U && replays != 0U;
        std::cout << "COHORT w=" << width << " graph_width=" << graph_width
                  << " program=" << (program ? "ready" : graph_width != 0U ? "declined" : "eager")
                  << " replay=" << (replayed ? "complete" : "none") << " mode=" << (padded ? "padded" : "full")
                  << " replays=" << replays << " other_mode=" << other << '\n';
        if (graph_width != 0U && !program) fail(width, 0U, 0U, "no cohort program at this width");
        if (graph_width != 0U && !replayed) fail(width, 0U, 0U, "this lane count completed no replay of its program");
        if (other != 0U) fail(width, 0U, 0U, "this lane count replayed a program in the other padding mode");
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
              << " steps from the first id through the serving adapter, logits + KV rows + histories bitwise)\n";
    return 0;
}

// Served prefill call forms (--served-prefill), lever ON. One decoder in the served construction (chunk = the command
// line's, tail capacity, two lanes). For each prompt length around the chunk, tail and key-split edges:
//   single : prefill() of slot 0 (the single-session call form) on THIS construction, host logits = the full head;
//   served : prefill_slot_first() of the whole prompt as the cohort issues it with no other lane decoding (regions of
//            prompt_slice_rows with the decoder's tail capacity), the first id from it, scores read_slot_logits(0);
//   pieces : the same prompt in the cohort's pieces beside a decoding lane (prompt_piece_rows inside each chunk of a
//            region, prefill_call_chunk_rows / prefill_piece_rule), one decode step of lane 1 between pieces;
//   repeat : served again (two identical served requests).
// First id, raw final scores (memcmp), the K/V rows of every committed position and the histories after the prompt, then four decode
// steps (ids, scores, last K/V rows, histories) must be equal across the four arms.
// SCOPE: this is call-form invariance of one served decoder, not the native command line's decoder (built per prompt:
// chunk clamped to the prompt, tail = the actual remainder) and not the served scheduler; the host test
// inkling2-mini-schedule-test proves those calls give every row the same chunk forms, and native CLI == real server
// (seeded and greedy, prompts 8192..8209 + an absorbed tail, a decode lane overlapping the prefill) is the HTTP
// identity kit's lever-on arm.
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
    const auto schedule = mini_prefill_schedule(kHip, profile::Name::standard);  // the EXACT command line
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
        // single
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
        static constexpr std::array<const char*, 4> kNames{"single", "served", "pieces", "repeat"};
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
    std::cout << "PASS Inkling Mini served prefill call forms single == served == pieces == repeat (" << prompt_lengths.size()
              << " prompt lengths, whole / pieces beside a decoding lane / repeat; first id, scores, KV planes, "
                 "histories, " << kServedSteps << " decode steps bitwise)\n";
    return 0;
}

// --served-held-back CHUNK [--held-back-levers] [--held-back-incumbent]: the held-back served route of a
// chunk-invariant backend (CUDA, mini_prefill_chunk_invariant) on the serving decoder built at CHUNK rows (2 lanes, no
// tail). Arms per prompt length (prompt tokens; the server holds the last one back and decodes it as the lane's first
// decode row):
//   incumbent  128-row calls (kMiniServedHeldBackChunk: every served prompt before the CUDA chunk change);
//   whole      calls of up to CHUNK rows (a prompt alone on the server);
//   fallback   2048-row calls (the chunk ladder's middle candidate);
//   pieces     calls of up to CHUNK rows cut by the decoder's piece rule into kMiniServeDecodingRows pieces, with a
//              neighbour lane decoding one step between pieces (mixed service, serve::prompt_piece_rows);
//   staggered  two pieces beside the neighbour, then the rest of the call and whole calls (the neighbour finished).
// Each arm must equal the incumbent bitwise: the first id and its scores, the KV planes of every committed position and
// the histories, then kServedSteps decode steps (top, logits, appended KV, histories). The neighbour's tokens and
// logits must equal its solo run. --held-back-incumbent runs the incumbent arm only (a decoder at 128 rows: the
// incumbent's own KV ring layout); the ticket compares the DIGEST fields across decoders (8192 / 128 rows) and lever
// settings.
inline int served_held_back(const char* checkpoint, weight::FormatId storage, std::span<const unsigned> prompt_lengths,
                            unsigned chunk, bool levers, bool incumbent_only) {
#if defined(__HIPCC__)
    (void)checkpoint;
    (void)storage;
    (void)prompt_lengths;
    (void)chunk;
    (void)levers;
    (void)incumbent_only;
    std::cout << "SKIP served_held_back: HIP keeps its 128-row held-back route\n";
    return 0;
#else
    const auto owned = served_decoder(checkpoint, kServedContext, chunk, storage, 2U, 0U, false);
    auto& decoder = *owned;
    if (decoder.prefill_emits_first()) throw std::logic_error("held-back route admission");
    decoder.set_attention_projected_bias(levers);
    (void)decoder.set_screened_head(levers);
    decoder.allocate_slots(2U);
    const auto rule = decoder.prefill_piece_rule();
    std::cout << "ROUTE served_held_back chunk " << decoder.prefill_chunk_rows() << " rule " << rule.align << '/'
              << rule.short_rows << " pieces " << kMiniServeDecodingRows << " levers " << (levers ? "on" : "off")
              << " arms " << (incumbent_only ? "incumbent" : "all") << '\n';
    if (!rule.cuttable()) throw std::logic_error("the CUDA held-back route states no piece rule");
    const std::size_t vocab = kOutputVocabulary;
    std::vector<std::uint16_t> plane, row;
    std::vector<float> history;
    Fnv64 all;
    unsigned failures = 0;

    // The neighbour lane (slot 1): its held-back prefill, then one width-1 decode step per call.
    struct Neighbour {
        unsigned token = 0;
        std::vector<unsigned> tops;
        std::vector<std::uint64_t> logits;
    };
    std::vector<unsigned> other(40U);
    for (unsigned i = 0; i < other.size(); ++i) other[i] = cohort_token(3U, i);
    const auto neighbour_start = [&](Neighbour& n) {
        decoder.prefill_slot(1U, std::span<const unsigned>(other).first(other.size() - 1U), true);
        n = Neighbour{other.back(), {}, {}};
    };
    std::vector<float> scratch(vocab);
    const auto neighbour_step = [&](Neighbour& n) {
        const std::array<unsigned, 1> slots{1U};
        const std::array<unsigned, 1> ids{n.token};
        std::array<unsigned, 1> top{};
        decoder.decode_slots(slots, ids, top);
        decoder.read_slot_logits(0U, scratch);
        Fnv64 h;
        h.add(scratch.data(), scratch.size() * sizeof(float));
        n.tops.push_back(top[0]);
        n.logits.push_back(h.value);
        n.token = top[0];
    };
    Neighbour solo;
    neighbour_start(solo);
    for (unsigned step = 0; step < 72U; ++step) neighbour_step(solo);

    enum Kind : unsigned { incumbent, whole, fallback, pieces, staggered, kinds };
    static constexpr std::array<const char*, kinds> kNames{"incumbent", "whole", "fallback", "pieces", "staggered"};
    const unsigned arm_count = incumbent_only ? 1U : unsigned{kinds};
    for (const auto length : prompt_lengths) {
        if (length < 2U) throw std::invalid_argument("--served-held-back needs prompts of at least 2 tokens");
        std::vector<unsigned> prompt(length);
        for (unsigned i = 0; i < length; ++i) prompt[i] = cohort_token(5U, i);
        const std::span<const unsigned> ids(prompt);
        const std::size_t rows = length - 1U;  // the held-back prefill rows
        std::array<PrefillArm, kinds> arms;
        for (unsigned kind = 0; kind < arm_count; ++kind) {
            auto& arm = arms[kind];
            Neighbour n;
            const bool mixed = kind == pieces || kind == staggered;
            if (mixed) neighbour_start(n);
            unsigned pieces_left = kind == pieces ? ~0U : 2U;
            bool first_call = true;
            const auto call = [&](std::size_t at, std::size_t count) {
                decoder.prefill_slot(0U, ids.subspan(at, count), first_call);
                first_call = false;
            };
            for (std::size_t at = 0; at < rows;) {
                const std::size_t left = rows - at;
                if (!mixed) {
                    const std::size_t count = std::min<std::size_t>(
                        left, kind == incumbent ? kMiniServedHeldBackChunk : kind == whole ? chunk : 2048U);
                    call(at, count);
                    at += count;
                    continue;
                }
                // One region (the call a lone prompt runs: prompt_slice_rows, no tail), cut while the neighbour
                // decodes.
                const std::size_t end = at + std::min<std::size_t>(left, chunk);
                while (at < end) {
                    std::size_t piece = end - at;
                    const bool beside = pieces_left != 0U;
                    if (beside) {
                        piece = static_cast<std::size_t>(
                            serve::prompt_piece_rows(end - at, kMiniServeDecodingRows, chunk, rule));
                        if (pieces_left != ~0U) --pieces_left;
                    }
                    call(at, piece);
                    at += piece;
                    if (beside && at < rows) neighbour_step(n);
                }
            }
            const std::array<unsigned, 1> slots{0U};
            const std::array<unsigned, 1> held{ids[rows]};
            std::array<unsigned, 1> top{};
            decoder.decode_slots(slots, held, top);
            arm.first = top[0];
            arm.scores.resize(vocab);
            decoder.read_slot_logits(0U, arm.scores);
            arm.kv = digest_planes(decoder, 0U, plane, history, arm.history);
            unsigned token = arm.first;
            for (unsigned step = 0; step < kServedSteps; ++step) {
                const std::array<unsigned, 1> next{token};
                decoder.decode_slots(slots, next, top);
                decoder.read_slot_logits(0U, scratch);
                auto& s = arm.steps[step];
                s.top = top[0];
                Fnv64 h;
                h.add(scratch.data(), scratch.size() * sizeof(float));
                s.logits = h.value;
                digest_state(decoder, 0U, s, row, history);
                token = top[0];
            }
            if (mixed) {
                const bool same = n.tops.size() <= solo.tops.size() &&
                                  std::equal(n.tops.begin(), n.tops.end(), solo.tops.begin()) &&
                                  std::equal(n.logits.begin(), n.logits.end(), solo.logits.begin());
                if (!same && ++failures <= 20U)
                    std::cout << "FAIL neighbour != solo rows=" << rows << " arm=" << kNames[kind] << " steps "
                              << n.tops.size() << '\n';
            }
        }
        for (unsigned kind = 0; kind < arm_count; ++kind) {
            const auto& arm = arms[kind];
            Fnv64 scores, steps;
            scores.add(arm.scores.data(), arm.scores.size() * sizeof(float));
            for (const auto& s : arm.steps) {
                steps.add((std::uint64_t{s.position} << 32U) | s.top);
                steps.add(s.logits);
                steps.add(s.kv);
                steps.add(s.history);
            }
            char line[256];
            std::snprintf(line, sizeof(line),
                          "DIGEST held_back rows=%zu first=%u scores=%016llx kv=%016llx hist=%016llx steps=%016llx",
                          rows, arm.first, static_cast<unsigned long long>(scores.value),
                          static_cast<unsigned long long>(arm.kv), static_cast<unsigned long long>(arm.history),
                          static_cast<unsigned long long>(steps.value));
            if (kind == incumbent) all.add(line, std::strlen(line));
            std::cout << line << " arm=" << kNames[kind] << '\n';
            if (kind == incumbent) continue;
            const auto fail = [&](const char* what) {
                if (++failures <= 20U)
                    std::cout << "FAIL " << kNames[kind] << " != incumbent rows=" << rows << ' ' << what << '\n';
            };
            const auto& base = arms[incumbent];
            if (arm.first != base.first) fail("first id");
            if (std::memcmp(arm.scores.data(), base.scores.data(), vocab * sizeof(float)) != 0) fail("scores");
            if (arm.kv != base.kv) fail("kv planes");
            if (arm.history != base.history) fail("histories");
            for (unsigned step = 0; step < kServedSteps; ++step)
                if (!(arm.steps[step] == base.steps[step])) fail("decode step");
        }
    }
    char line[64];
    std::snprintf(line, sizeof(line), "%016llx", static_cast<unsigned long long>(all.value));
    std::cout << "DIGEST HELD_BACK " << line << '\n';
    if (failures != 0U) {
        std::cout << "FAIL Inkling Mini served held-back route: " << failures << " mismatches\n";
        return 1;
    }
    std::cout << "PASS Inkling Mini served held-back route chunk " << chunk << " levers " << (levers ? "on" : "off")
              << ": "
              << (incumbent_only ? "incumbent arm digests" : "whole == fallback == pieces == staggered == incumbent")
              << " (" << prompt_lengths.size() << " prompt lengths; first id, scores, KV planes, histories, "
              << kServedSteps << " decode steps bitwise; neighbour == solo)\n";
    return 0;
#endif
}

#if !defined(__HIPCC__)
// Fill the resident image and compare the CLI's resident session with the production serving cohort.
inline int w4a4_prompt_parity(const char* checkpoint, weight::FormatId storage,
                              moe_core::grouped_w4a4::Activations activations) {
    constexpr unsigned chunk = 128U, generated = 4U, context = chunk + 1U + generated;
    constexpr std::array lengths{2U, chunk - 1U, chunk, chunk + 1U};
    std::array<std::vector<unsigned>, lengths.size()> prompts, expected_ids;
    std::array<std::vector<float>, lengths.size()> expected_scores;
    for (std::size_t i = 0; i < lengths.size(); ++i)
        for (unsigned row = 0; row < lengths[i]; ++row) prompts[i].push_back(cohort_token(0U, row));
    {
        // Match the server's grouped decode arithmetic and use a singleton tail at chunk + 1.
        MiniResidentDecoder decoder(checkpoint, context, chunk, false, MiniResidentBackend::default_step_graphs, storage);
        std::vector<float> bf16_scores(kOutputVocabulary);
        (void)decoder.prefill(prompts[2U], bf16_scores);
        decoder.set_moe_w4a4(activations);
        static_assert(MiniMoeConfig::format(0U) == weight::microscaled::MatrixFormat::nvfp4);
        if (decoder.step_program_backend().moe_w4a4_class(0U) != activations)
            throw std::logic_error("resident NVFP4 layer did not bind the selected W4A4 class");
        for (std::size_t i = 0; i < lengths.size(); ++i) {
            decoder.reset_gate_session();
            auto& scores = expected_scores[i];
            auto& ids = expected_ids[i];
            scores.resize(kOutputVocabulary);
            ids.push_back(decoder.prefill(prompts[i], scores));
            while (ids.size() < generated && ids.back() != kMiniEosToken) ids.push_back(decoder.decode(ids.back()));
        }
        if (std::memcmp(expected_scores[2U].data(), bf16_scores.data(), bf16_scores.size() * sizeof(float)) == 0)
            throw std::logic_error("W4A4 matched the BF16 control at full resident capacity");
    }

    auto decoder = std::make_unique<MiniResidentDecoder>(checkpoint, context, chunk, false,
        MiniResidentBackend::default_step_graphs, storage);
    decoder->set_moe_w4a4(activations);
    serve::FixedKvCohort<MiniResidentDecoder> backend(std::move(decoder), context, 1U, 1U);
    serve::CohortConfig config;
    config.context_draft = false;
    config.max_draft = 0U;
    config.prompt_slice = chunk;
    serve::CohortDecoder cohort(backend, config);
    for (std::size_t i = 0; i < lengths.size(); ++i) {
        bool saw_final_scores = false;
        serve::CohortRequest request;
        request.id = i + 1U;
        request.prompt = prompts[i];
        request.max_tokens = generated;
        request.stop = {kMiniEosToken};
        request.prefilled = [&](std::uint64_t rows) {
            if (rows != lengths[i]) return;
            const auto scores = backend.read_final_logits(0U);
            if (scores.size() != expected_scores[i].size() ||
                std::memcmp(scores.data(), expected_scores[i].data(), scores.size_bytes()) != 0)
                throw std::logic_error("W4A4 CLI and served prompt logits differ");
            saw_final_scores = true;
        };
        cohort.submit(std::move(request));
        cohort.run();
        const auto results = cohort.take_finished();
        if (results.size() != 1U || !results[0].error.empty() || !results[0].first_from_prefill ||
            !saw_final_scores || results[0].ids != expected_ids[i])
            throw std::logic_error("W4A4 CLI and served generation differ");
        std::cout << "PASS " << moe_core::grouped_w4a4::identity(activations) << " prompt rows " << lengths[i]
                  << " (raw logits and all generated IDs)\n";
    }
    return 0;
}
#endif

}  // namespace strix::inkling2::test

int main(int argc, char** argv) {
    using namespace strix;
    try {
        if (argc < 2)
            throw std::invalid_argument("expected MODEL.safetensors [--qualify-q8] [--served-route] [--served-prefill | --served-held-back CHUNK "
                "[--held-back-levers] [--held-back-incumbent]] [--served-prompts LIST]");
        auto storage = weight::kBf16WeightStorage;
        bool served_route = false;  // the served route lever (off: the incumbent served decoder)
        bool served_prefill = false;
        unsigned held_back = 0;  // --served-held-back CHUNK
        bool held_back_levers = false, held_back_incumbent = false;
        std::vector<unsigned> prompt_lengths(inkling2::test::kServedPrompt.begin(), inkling2::test::kServedPrompt.end());
        for (int i = 2; i < argc; ++i) {
            const std::string_view option{argv[i]};
            if (option == "--qualify-q8") storage = weight::kQ8WeightStorage;
            else if (option == "--served-route") served_route = true;
            else if (option == "--served-prefill") served_prefill = true;
            else if (option == "--served-held-back" && i + 1 < argc)
                held_back = static_cast<unsigned>(std::stoul(argv[++i]));
            else if (option == "--held-back-levers") held_back_levers = true;
            else if (option == "--held-back-incumbent") held_back_incumbent = true;
            else if (option == "--served-prompts" && i + 1 < argc)
                prompt_lengths = inkling2::test::parse_served_prompts(argv[++i]);
            else throw std::invalid_argument("unknown option: " + std::string(option));
        }
        if (served_prefill)
            return inkling2::test::served_prefill(argv[1], storage, prompt_lengths);
        if (held_back != 0U)
            return inkling2::test::served_held_back(argv[1], storage, prompt_lengths, held_back, held_back_levers,
                                                    held_back_incumbent);
#if !defined(__HIPCC__)
        inkling2::MiniWeightOptions options;
        options.tier.profile = profile::Name::fast;
        const auto activations = inkling2::admit_mini_moe_w4a4(inkling2::resolve_mini_profile(options, true), "cuda");
        if (moe_core::grouped_w4a4::w4a4_gate_up(activations))
            return inkling2::test::w4a4_prompt_parity(argv[1], storage, activations);
#endif
        // The production admission (the server's, inkling2_mini_serve.cpp), decided before the decoder is built.
#if defined(__HIPCC__)
        constexpr bool hip_backend = true;
#else
        constexpr bool hip_backend = false;
#endif
        const bool admitted = inkling2::mini_served_prefill_emits_first(hip_backend, served_route);
        if (served_route && !admitted) std::cout << "NOTE served route lever declined by this backend\n";
        return inkling2::test::cohort_lanes(argv[1], storage, served_route, admitted);
    } catch (const std::exception& error) {
        std::cerr << "FAIL Inkling Mini cohort lanes: " << error.what() << '\n';
        return 1;
    }
}
