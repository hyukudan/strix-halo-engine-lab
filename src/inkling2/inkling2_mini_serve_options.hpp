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
    unsigned prefill_rows = 128;
    unsigned slots = 0;
    std::uint64_t kv_bytes = 0;
    MiniWeightOptions weight_options;
    profile::Name tier = profile::Name::standard;
    // Resolved operand class of the FAST W4A4 expert lever (mini_moe_w4a4_class; bf16 = off): the served prefill runs
    // the class the profile line reports, as the CLI and the gate do.
    moe_core::grouped_w4a4::Activations moe_w4a4 = moe_core::grouped_w4a4::Activations::bf16;
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
