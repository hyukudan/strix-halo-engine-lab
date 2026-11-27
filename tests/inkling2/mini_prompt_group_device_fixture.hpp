#pragma once

// Grouped prompt prefill (serve::FixedKvCohort::prefill_group over MiniResidentDecoder::prefill_group_first) against each
// member's solo prefill (prefill_slot_first through the same adapter), bit for bit: first id, the last row's scores,
// committed frontier, the live K/V rows of every visit and the recurrent histories of the member's slot, then two decode
// steps (ids and scores) of every lane. The prompt list may end with (or contain) a member of at most
// kMiniGroupSoloRows rows: the scheduler's plan refuses it into a solo prefill and the decoder refuses it when a group
// carries it. Needs the model checkpoint; HIP only (the grouped route is a HIP route).
//
//   strix-hip-inkling2-mini-prompt-group-test MODEL.safetensors [--chunk N] [--context N] [--prompts LIST]
//
// Defaults: the served route's 8192-row chunk clamped to the context (4096), prompts 666,1885,40,10 (one pass of 2591
// rows, the 10-row prompt alone). With --chunk 2048 the same prompts run as two passes (666 + ... never splitting a
// member).

#include "inkling2/inkling2_mini_resident_decoder.hpp"
#include "inkling2/inkling2_mini_serve_options.hpp"
#include "strix/serve/cohort_decode.hpp"
#include "strix/serve/fixed_kv_cohort.hpp"
#include "strix/serve/prefill_group.hpp"

#include <algorithm>
#include <array>
#include <cstdint>
#include <cstring>
#include <exception>
#include <iostream>
#include <limits>
#include <memory>
#include <span>
#include <stdexcept>
#include <string>
#include <string_view>
#include <vector>

namespace strix::inkling2::test {

using GroupFixtureCohort = serve::FixedKvCohort<MiniResidentDecoder>;

// What one slot holds after its prompt: everything the next steps read.
struct MemberImage {
    unsigned top = 0;
    unsigned committed = 0;
    std::vector<float> scores;
    std::vector<std::uint16_t> kv;  // the live K / V rows of every visit, keys then values
    std::vector<float> history;
};

inline unsigned group_token(unsigned member, unsigned index) {
    return 1000U + (index * 173U + member * 7919U + index / 13U) % 20000U;
}

inline std::vector<unsigned> group_prompt(unsigned member, unsigned rows) {
    std::vector<unsigned> ids(rows);
    for (unsigned i = 0; i < rows; ++i) ids[i] = group_token(member, i);
    return ids;
}

inline void copy_image(MiniResidentDecoder& decoder, unsigned slot, MemberImage& image) {
    const auto view = decoder.slot_view(slot);
    image.committed = view.committed;
    image.kv.clear();
    std::vector<std::uint16_t> plane;
    for (const auto device : {view.kv.keys, view.kv.values}) {
        plane.resize(device.size());
        mini_gpu::copy(plane.data(), device.data(), device.size_bytes(), mini_gpu::device_to_host, nullptr);
        mini_gpu::synchronize(nullptr);
        for (const auto visit : MiniSessionStatePlan::visits()) {
            const auto& layout = view.plan.visit(visit);
            const unsigned live =
                uses_sliding_attention(visit) ? std::min(view.committed, kSlidingWindow) : view.committed;
            for (unsigned position = view.committed - live; position < view.committed; ++position) {
                const auto offset =
                    static_cast<std::size_t>(layout.kv_offset + attn::cache_rows::row_offset(layout.kv, position));
                image.kv.insert(image.kv.end(), plane.begin() + static_cast<std::ptrdiff_t>(offset),
                                plane.begin() + static_cast<std::ptrdiff_t>(offset + layout.kv.width));
            }
        }
    }
    image.history.resize(view.history.size());
    mini_gpu::copy(image.history.data(), view.history.data(), view.history.size_bytes(), mini_gpu::device_to_host, nullptr);
    mini_gpu::synchronize(nullptr);
}

inline MemberImage image_of(GroupFixtureCohort& cohort, MiniResidentDecoder& decoder, unsigned slot) {
    MemberImage image;
    image.top = cohort.prefill_top(slot);
    const auto scores = cohort.read_final_logits(slot);
    image.scores.assign(scores.begin(), scores.end());
    copy_image(decoder, slot, image);
    return image;
}

// Empty: identical; else the first difference.
inline std::string difference(const MemberImage& a, const MemberImage& b) {
    if (a.top != b.top) return "first id " + std::to_string(a.top) + " vs " + std::to_string(b.top);
    if (a.committed != b.committed) return "committed rows";
    if (a.scores.size() != b.scores.size() ||
        std::memcmp(a.scores.data(), b.scores.data(), a.scores.size() * sizeof(float)) != 0)
        return "last-row scores";
    if (a.kv.size() != b.kv.size() || std::memcmp(a.kv.data(), b.kv.data(), a.kv.size() * sizeof(std::uint16_t)) != 0)
        return "K/V rows";
    if (a.history.size() != b.history.size() ||
        std::memcmp(a.history.data(), b.history.data(), a.history.size() * sizeof(float)) != 0)
        return "convolution histories";
    return {};
}

inline std::vector<unsigned> parse_prompt_list(std::string_view list) {
    std::vector<unsigned> rows;
    unsigned value = 0;
    bool digits = false;
    for (const char c : list) {
        if (c >= '0' && c <= '9') {
            value = value * 10U + static_cast<unsigned>(c - '0');
            digits = true;
        } else if (c == ',' && digits) {
            rows.push_back(value);
            value = 0;
            digits = false;
        } else {
            throw std::invalid_argument("bad --prompts list");
        }
    }
    if (!digits) throw std::invalid_argument("bad --prompts list");
    rows.push_back(value);
    for (const auto n : rows)
        if (n == 0U) throw std::invalid_argument("bad --prompts length");
    return rows;
}

inline int prompt_group(const char* checkpoint, unsigned context, unsigned chunk, const std::vector<unsigned>& lengths) {
    const unsigned n = static_cast<unsigned>(lengths.size());
    if (n < 2U || 2U * n > 16U) throw std::invalid_argument("expected 2..8 prompts");
    chunk = std::min(chunk, context);
    const auto schedule = mini_prefill_schedule(true, profile::Name::standard);
    const unsigned tail = mini_prefill_tail_capacity(chunk, schedule.tail_divisor, context);
    const unsigned lanes = 2U * n;  // solo arm on slots 0..n-1, group arm on slots n..2n-1
    auto owned = std::make_unique<MiniResidentDecoder>(checkpoint, context, chunk, kMiniPackedDecodeDefault,
        MiniResidentBackend::default_step_graphs, weight::kBf16WeightStorage, 0U, false, lanes, tail, false, false, true);
    auto& decoder = *owned;
    GroupFixtureCohort cohort(std::move(owned), context, lanes, lanes);
    if (!cohort.prefill_emits_first()) throw std::logic_error("the served route is not admitted");
    const auto capacity = cohort.prompt_group_capacity();
    std::cout << "ROUTE prompt_group chunk " << chunk << " tail " << tail << " capacity members " << capacity.members
              << " rows " << capacity.rows << " member_rows " << capacity.member_rows << " passes " << capacity.passes
              << " reason " << capacity.reason << '\n';
    if (capacity.members < 2U) throw std::logic_error(std::string("grouped prefill refused: ") + capacity.reason);

    std::vector<std::vector<unsigned>> prompts;
    for (unsigned i = 0; i < n; ++i) prompts.push_back(group_prompt(i, lengths[i]));

    // Solo arm: every prompt alone through the adapter.
    std::vector<MemberImage> solo(n);
    for (unsigned i = 0; i < n; ++i) {
        cohort.prefill(i, prompts[i], true);
        solo[i] = image_of(cohort, decoder, i);
    }

    // The scheduler's plan over the arrival-ordered prompts: the oldest candidate is the head; what it refuses runs alone.
    const auto policy = mini_serve_prompt_group(chunk);
    std::vector<serve::GroupCandidate> candidates;
    for (unsigned i = 0; i < n; ++i) {
        serve::GroupCandidate c;
        c.id = i;
        c.rows = lengths[i];
        c.arrival_seq = i;
        c.route_class = cohort.prompt_route_class(lengths[i]);
        candidates.push_back(c);
    }
    const auto grouped = serve::plan_prefill_group(candidates, policy, serve::PrefillGroupCost{},
                                                    std::numeric_limits<double>::infinity(), capacity);
    std::vector<serve::CohortPrompt> members;
    for (const auto id : grouped) members.push_back({static_cast<std::uint32_t>(n + id), prompts[id], true});
    std::cout << "PLAN group members " << grouped.size() << " of " << n << " prompts, rows";
    std::uint64_t rows = 0;
    for (const auto id : grouped) {
        std::cout << ' ' << lengths[id];
        rows += lengths[id];
    }
    std::cout << " (total " << rows << ")\n";
    if (grouped.size() < 2U) throw std::logic_error("the plan grouped fewer than two prompts");

    // The decoder refuses a member the plan would never send (tight rows) before any write.
    for (unsigned i = 0; i < n; ++i) {
        if (lengths[i] > kMiniGroupSoloRows) continue;
        bool refused = false;
        const serve::CohortPrompt pair[] = {{n + i, prompts[i], true}, {n + (i + 1U) % n, prompts[(i + 1U) % n], true}};
        try {
            cohort.prefill_group(pair);
        } catch (const std::logic_error&) {
            refused = true;
        }
        if (!refused) throw std::logic_error("a tight-rows member was not refused by the decoder");
        std::cout << "prompt-group member rows " << lengths[i] << ": refused into solo\n";
    }

    // Group arm.
    cohort.prefill_group(members);
    std::vector<MemberImage> group(n);
    for (const auto id : grouped) group[id] = image_of(cohort, decoder, static_cast<unsigned>(n + id));
    // The refused prompts run alone on the group arm's slots (the scheduler's solo prefill).
    for (unsigned i = 0; i < n; ++i) {
        if (std::ranges::find(grouped, i) != grouped.end()) continue;
        cohort.prefill(n + i, prompts[i], true);
        group[i] = image_of(cohort, decoder, n + i);
    }
    int failures = 0;
    for (unsigned i = 0; i < n; ++i) {
        if (const auto why = difference(solo[i], group[i]); !why.empty()) {
            std::cout << "FAIL prompt-group member " << i << " rows " << lengths[i] << ": " << why << '\n';
            ++failures;
        }
    }

    // Two decode steps of every lane (ids and scores): the state a group left is the state the next steps read.
    struct Step {
        std::vector<unsigned> top;
        std::vector<std::vector<float>> scores;
    };
    const auto steps = [&](unsigned arm, const std::vector<MemberImage>& images) {
        std::vector<unsigned> slots(n), ids(n), top(n);
        for (unsigned i = 0; i < n; ++i) {
            slots[i] = arm * n + i;
            ids[i] = images[i].top;
        }
        std::vector<Step> out(2U);
        for (auto& step : out) {
            decoder.decode_slots(slots, ids, top, true);
            step.top = top;
            step.scores.assign(n, std::vector<float>(decoder.logit_width()));
            for (unsigned i = 0; i < n; ++i) decoder.read_slot_logits(i, step.scores[i]);
            ids = top;
        }
        return out;
    };
    const auto solo_steps = steps(0U, solo);
    const auto group_steps = steps(1U, group);
    for (unsigned step = 0; step < 2U; ++step) {
        for (unsigned i = 0; i < n; ++i) {
            const auto& a = solo_steps[step].scores[i];
            const auto& b = group_steps[step].scores[i];
            if (solo_steps[step].top[i] != group_steps[step].top[i] ||
                std::memcmp(a.data(), b.data(), a.size() * sizeof(float)) != 0) {
                std::cout << "FAIL prompt-group decode step " << step << " lane " << i << ": ids or scores differ" << std::endl;
                ++failures;
            }
        }
    }

    if (failures != 0) return 1;
    std::cout << "prompt-group members " << grouped.size() << ": identical (first ids, last-row scores, K/V rows, "
              << "histories; " << n << " prompts, " << (n - grouped.size()) << " refused into solo)\n";
    return 0;
}

}  // namespace strix::inkling2::test

int main(int argc, char** argv) {
    try {
        if (argc < 2)
            throw std::invalid_argument("expected MODEL.safetensors [--chunk N] [--context N] [--prompts LIST]");
        unsigned chunk = 8192U, context = 4096U;
        std::vector<unsigned> lengths{666U, 1885U, 40U, 10U};
        for (int i = 2; i < argc; ++i) {
            const std::string_view option{argv[i]};
            if (option == "--chunk" && i + 1 < argc) chunk = static_cast<unsigned>(std::stoul(argv[++i]));
            else if (option == "--context" && i + 1 < argc) context = static_cast<unsigned>(std::stoul(argv[++i]));
            else if (option == "--prompts" && i + 1 < argc)
                lengths = strix::inkling2::test::parse_prompt_list(argv[++i]);
            else throw std::invalid_argument("unknown option: " + std::string(option));
        }
        return strix::inkling2::test::prompt_group(argv[1], context, chunk, lengths);
    } catch (const std::exception& error) {
        std::cerr << "FAIL Inkling Mini prompt group: " << error.what() << '\n';
        return 1;
    }
}
