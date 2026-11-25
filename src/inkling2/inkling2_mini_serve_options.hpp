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
    // Prefill chunk rows and tail capacity: --prefill-rows, else the served route's chunk
    // (mini_served_prefill_schedule_from_env: the command line's chunk and tail rule on its route, emits-first; the
    // incumbent 128-row held-back chunks without a tail otherwise).
    unsigned prefill_rows = 0;
    unsigned prefill_tail = 0;
    unsigned slots = 0;
    // Lane cap of an explicit --prefill-rows given without --slots (0: kMiniServeLaneRows), the cap it set before.
    unsigned lane_rows = 0;
    // Resolved served route (mini_served_cli_route, admitted on HIP only): the command line's route; false = the
    // incumbent served route (held-back prompt token, grouped cohort decode without a flag).
    bool served_cli_route = false;
    std::uint64_t kv_bytes = 0;
    MiniWeightOptions weight_options;
    profile::Name tier = profile::Name::standard;
    // Resolved operand class of the FAST W4A4 expert lever (mini_moe_w4a4_class; bf16 = off): the served prefill runs
    // the class the profile line reports, as the CLI and the gate do.
    moe_core::grouped_w4a4::Activations moe_w4a4 = moe_core::grouped_w4a4::Activations::bf16;
    // Decode expert form: --packed-decode / --grouped-decode, else the served route's default
    // (mini_served_packed_default: packed on the command line's route, grouped on the incumbent served route).
    bool packed_decode = false;
    // Resolved prepared-attention round fold (mini_prepared_round_fold), the CLI's setting from the same resolution.
    bool prepared_round_fold = false;
    // Resolved exact down-image lever (mini_combine_image): the HIP resident prefill stores the BF16 down image where the
    // target is qualified.
    bool combine_image = false;
};

// The prepared_round_fold field of the served startup line: the requested bit from the options and what the decoder's backend
// admitted of it (MiniDecoderLevers::prepared_round_fold), so a fold the backend declines reads "declined", never "on".
inline const char* mini_round_fold_startup_field(const MiniServeOptions& opt, const MiniDecoderLevers& levers) {
    return mini_round_fold_name(levers.prepared_round_fold, opt.prepared_round_fold);
}

struct MiniServeCohort {
    std::unique_ptr<serve::CohortBackend> backend;
    weight::WeightLoadPlan weights;
};

MiniServeCohort make_mini_hip_cohort(const MiniServeOptions&);
MiniServeCohort make_mini_cuda_cohort(const MiniServeOptions&);

} // namespace strix::inkling2
