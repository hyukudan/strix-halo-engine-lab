#pragma once

#include "inkling2_mini_weight_options.hpp"
#include "strix/serve/cohort_decode.hpp"

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
    unsigned prefill_rows = 0;        // 0 = the served data's chunk (mini_serve_prefill), resolved by the server resolver
    unsigned prefill_tail_divisor = 0; // resolved: final-chunk tail absorption (0 = greedy chunks)
    unsigned prefill_rows_decoding = 0; // piece rows while another lane decodes (0 = the data's); resolved, 0 = off
    double prefill_gap_ms = -1.0;     // lane gap target ms (< 0 = the device class's data at launch, 0 = fixed pieces)
    double prompt_share = -1.0;       // decode ms owed per prompt ms while lanes decode (< 0 = the data's)
    unsigned slots = 0;
    std::uint64_t kv_bytes = 0;
    MiniWeightOptions weight_options;
    profile::Name tier = profile::Name::standard;
    // Resolved operand class of the FAST W4A4 expert lever (mini_moe_w4a4_class; bf16 = off): the served prefill runs
    // the class the profile line reports, as the CLI and the gate do.
    moe_core::grouped_w4a4::Activations moe_w4a4 = moe_core::grouped_w4a4::Activations::bf16;
};

struct MiniServeCohort {
    std::unique_ptr<serve::CohortBackend> backend;
    weight::WeightLoadPlan weights;
    bool hip = true;
    std::string device_identity;  // strix/device/cost_identity.hpp key of the device (empty = not described)
};

MiniServeCohort make_mini_hip_cohort(const MiniServeOptions&);
MiniServeCohort make_mini_cuda_cohort(const MiniServeOptions&);

} // namespace strix::inkling2
