#pragma once

#include "inkling2_mini_resident_decoder.hpp"
#include "inkling2_mini_serve_options.hpp"
#include "inkling2_mini_weight_device.hpp"
#include "strix/device/gate.hpp"
#include "strix/serve/fixed_kv_cohort.hpp"

#include <iostream>
#include <limits>

namespace strix::inkling2 {

inline device::Request mini_serve_device_request(const MiniServeOptions& opt) {
    device::Request request;
    request.select = opt.device_id;
    request.profile = opt.device_profile;
    return request;
}

inline MiniServeCohort make_mini_device_cohort(
    const MiniServeOptions& opt, const device::MemoryBudget& budget, std::string_view backend,
    unsigned row_limit = std::numeric_limits<unsigned>::max()) {
    const auto rows = std::min(mini_serve_lane_rows(opt), row_limit);
    // The instance's extent for the weight rows: its configured context and its slot cap (0 = left to the memory
    // budget: lane-banded gains do not select).
    const weight::WeightLoadExtent extent{opt.context, opt.slots};
    const auto scope = mini_weight_scope(opt.checkpoint, opt.tier, opt.weight_options, extent);
    auto weights = resolve_mini_resident_weight_load(opt.tier, opt.weight_options, scope.view(), extent);
    if (!(scope.checkpoint == weight::Digest256{})) std::cerr << weight::weight_load_scope_receipt(scope.view()) << '\n';
    std::uint64_t slot_bytes = 0;
    unsigned lanes = 0;
    MiniDecoderLevers levers;
#if !defined(__HIPCC__)
    exec::CohortProgramPolicy programs;
#endif
    auto cohort = load_mini_weights(weights, opt.tier, opt.weight_options, scope.view(), extent, [&](MiniLowBitRows lowbit) {
        const bool calibrated = lowbit.admitted != nullptr;
        // The served route is admitted before constructing the decoder. The same load transaction
        // checks capacity and allocates the slots before publishing the cohort.
        auto decoder = std::make_unique<MiniResidentDecoder>(opt.checkpoint, opt.context, opt.prefill_rows,
            opt.served_cli_route && kMiniPackedDecodeDefault, MiniResidentBackend::default_step_graphs,
            weights.roles(), 0U, false, rows, opt.served_cli_route ? opt.prefill_tail : 0U,
            opt.experts_nvfp4, opt.controller_one_plane, opt.served_cli_route, std::move(lowbit));
        levers = configure_mini_decoder(*decoder, opt.moe_w4a4, opt.combine_image, opt.prepared_round_fold,
                                        opt.attn_projected_bias);
        const auto memory = mini_gpu::memory_info();
        slot_bytes = decoder->slot_bytes();
        lanes = serve::resident_kv_slots(memory.free, memory.total, budget.device_bytes, opt.kv_bytes, slot_bytes, rows);
        if (!lanes) throw std::runtime_error("Inkling KV budget holds no session slots");
        if (calibrated && lanes < rows)
            throw std::runtime_error("calibrated image cannot retain the requested slot capacity");
#if !defined(__HIPCC__)
        programs = decoder->cohort_programs();
#endif
        return std::make_unique<serve::FixedKvCohort<MiniResidentDecoder>>(std::move(decoder), opt.context, rows, lanes);
    });
    std::cerr << "Inkling " << backend << " slots " << lanes << " context " << opt.context
              << " KV and history bytes per slot " << slot_bytes
              << " profile " << mini_profile_name(opt.tier)
              << " moe_prefill " << moe_core::grouped_w4a4::identity(opt.moe_w4a4)
              << " served_route " << mini_served_route_startup_field(opt, cohort->prompt_tail_rows())
              << " prepared_round_fold " << mini_round_fold_startup_field(opt, levers)
              << " down_image " << mini_down_image_name(levers.down_image, opt.combine_image)
              << " experts " << (opt.experts_nvfp4 ? "nvfp4" : "checkpoint")
              << " prefill_route " << (cohort->prefill_emits_first() ? "emits-first" : "plain")
              << " weight_storage " << weights.storage.name << '@' << weights.storage.revision
              << (weights.head_storage != weights.storage ? " weight_head_storage " : "")
              << (weights.head_storage != weights.storage ? weights.head_storage.name : "")
#if !defined(__HIPCC__)
              << " cohort_programs " << (programs.enabled ? "on" : "off")
              << " widths " << exec::cohort_widths_name(programs.widths)
#endif
              << " weight_status " << profile::status_name(weights.status) << " (" << weights.reason << ")"
              << (weights.turbo_quality.empty() ? "" : " turbo_quality ") << weights.turbo_quality << '\n'
              << weight::weight_load_regime_receipt(weights) << '\n';
    return {std::move(cohort), weights};
}

} // namespace strix::inkling2
