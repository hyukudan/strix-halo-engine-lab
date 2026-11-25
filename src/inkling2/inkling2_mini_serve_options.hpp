#pragma once

#include "inkling2_mini_resident_route.hpp"
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
    // Prefill chunk rows and tail capacity: --prefill-rows, else the command line's chunk for the backend
    // (mini_prefill_schedule, resolved with the backend); the tail by the command line's rule
    // (mini_prefill_tail_capacity, STRIX_INKLING_PREFILL_TAIL). The whole prompt is prefilled (emits-first).
    unsigned prefill_rows = 0;
    unsigned prefill_tail = 0;
    unsigned slots = 0;
    std::uint64_t kv_bytes = 0;
    MiniWeightOptions weight_options;
    profile::Name tier = profile::Name::standard;
    // Resolved operand class of the FAST W4A4 expert lever (mini_moe_w4a4_class; bf16 = off): the served prefill runs
    // the class the profile line reports, as the CLI and the gate do.
    moe_core::grouped_w4a4::Activations moe_w4a4 = moe_core::grouped_w4a4::Activations::bf16;
    // Decode expert form: --packed-decode / --grouped-decode, else the one default of every entry point
    // (mini_packed_decode, the command line's default too).
    bool packed_decode = kMiniPackedDecodeDefault;
    // Resolved prepared-attention round fold (mini_prepared_round_fold), the CLI's setting from the same resolution.
    bool prepared_round_fold = false;
    // Resolved exact down-image lever (mini_combine_image): the HIP resident prefill stores the BF16 down image where the
    // target is qualified.
    bool combine_image = false;
};

struct MiniServeCohort {
    std::unique_ptr<serve::CohortBackend> backend;
    weight::WeightLoadPlan weights;
};

MiniServeCohort make_mini_hip_cohort(const MiniServeOptions&);
MiniServeCohort make_mini_cuda_cohort(const MiniServeOptions&);

} // namespace strix::inkling2
