#pragma once

#include "inkling2_mini_weight_options.hpp"
#include "strix/serve/cohort_decode.hpp"
#include "strix/serve/row_budget.hpp"

#include <algorithm>
#include <cstdint>
#include <memory>
#include <string>

namespace strix::inkling2 {

struct MiniServeOptions {
    std::string checkpoint;
    std::string tokenizer;
    std::string devices_path;
    std::string device_id;
    std::string device_profile;
    unsigned context = 8192;
    unsigned prefill_rows = 128;
    unsigned slots = 0;
    std::uint64_t kv_bytes = 0;
    MiniWeightOptions weight_options;
    profile::Name tier = profile::Name::standard;
    // Resolved operand class of the FAST W4A4 expert lever (mini_moe_w4a4_class; bf16 = off): the served prefill runs
    // the class the profile line reports, as the CLI and the gate do.
    moe_core::grouped_w4a4::Activations moe_w4a4 = moe_core::grouped_w4a4::Activations::bf16;
    bool packed_decode = false;  // resolved decode expert form (mini_packed_decode, entry serve: grouped)
};

// Served KV slots, and whether the screened greedy head plane stays resident. The plane is an EXACT decode speed lever
// (same picks as the full head) allocated with the decoder, before the slot budget is read: it stays only when it costs
// no KV slot, so screening never lowers the slot capacity the full head admits.
struct MiniServeSlots {
    std::uint32_t lanes;
    bool keep_screen;
};

inline MiniServeSlots mini_serve_slots(std::uint64_t free, std::uint64_t total, std::uint64_t device_limit,
                                       std::uint64_t kv_limit, std::uint64_t slot_bytes, std::uint32_t rows,
                                       std::uint64_t screen_bytes) {
    const auto with_screen = serve::resident_kv_slots(free, total, device_limit, kv_limit, slot_bytes, rows);
    if (screen_bytes == 0U) return {with_screen, false};
    const auto freed = free + std::min(screen_bytes, total - std::min(free, total));
    const auto full_head = serve::resident_kv_slots(freed, total, device_limit, kv_limit, slot_bytes, rows);
    return full_head > with_screen ? MiniServeSlots{full_head, false} : MiniServeSlots{with_screen, true};
}

struct MiniServeCohort {
    std::unique_ptr<serve::CohortBackend> backend;
    weight::WeightLoadPlan weights;
};

MiniServeCohort make_mini_hip_cohort(const MiniServeOptions&);
MiniServeCohort make_mini_cuda_cohort(const MiniServeOptions&);

} // namespace strix::inkling2
