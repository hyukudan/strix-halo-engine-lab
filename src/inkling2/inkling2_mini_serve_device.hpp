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
    // Lanes per forward: --slots, else the served lane cap (independent of the prefill chunk).
    const auto rows = std::min(opt.slots ? opt.slots : kMiniServeLaneRows, row_limit);
    const auto scope = mini_weight_scope(opt.checkpoint, opt.tier, opt.weight_options);
    const auto weights = resolve_mini_resident_weight_load(opt.tier, opt.weight_options, scope.view());
    // The served prefill chunk and tail capacity (inkling2_mini_serve.cpp resolve: the command line's on HIP).
    auto decoder = std::make_unique<MiniResidentDecoder>(opt.checkpoint, opt.context, opt.prefill_rows,
        opt.packed_decode, MiniResidentBackend::default_step_graphs, weights.storage, 0U, false, rows, opt.prefill_tail);
    const auto levers = configure_mini_decoder(*decoder, opt.moe_w4a4, opt.combine_image, opt.prepared_round_fold);
    const auto memory = mini_gpu::memory_info();
    const auto slot_bytes = decoder->slot_bytes();
    const auto lanes = serve::resident_kv_slots(memory.free, memory.total, budget.device_bytes, opt.kv_bytes,
                                               slot_bytes, rows);
    if (!lanes) throw std::runtime_error("Inkling KV budget holds no session slots");
#if !defined(__HIPCC__)
    const auto programs = decoder->cohort_programs();
#endif
    const auto decoder_tail = decoder->prefill_tail_rows();
    auto cohort = std::make_unique<serve::FixedKvCohort<MiniResidentDecoder>>(std::move(decoder), opt.context, rows, lanes);
    std::cerr << "Inkling " << backend << " slots " << lanes << " context " << opt.context
              << " KV and history bytes per slot " << slot_bytes
              << " profile " << mini_profile_name(opt.tier)
              << " moe_prefill " << moe_core::grouped_w4a4::identity(opt.moe_w4a4)
              << " moe_decode " << mini_decode_form_name(opt.packed_decode)
              << " prefill_chunk " << opt.prefill_rows << " prefill_tail " << decoder_tail
              << " first_id " << (cohort->prefill_emits_first() ? "prefill" : "decode")
              << " prepared_round_fold " << mini_round_fold_name(levers.prepared_round_fold, opt.prepared_round_fold)
              << " down_image " << mini_down_image_name(levers.down_image, opt.combine_image)
              << " weight_storage " << weights.storage.name << '@' << weights.storage.revision
#if !defined(__HIPCC__)
              << " cohort_programs " << (programs.enabled ? "on" : "off")
              << " widths " << exec::cohort_widths_name(programs.widths)
#endif
              << " weight_status " << profile::status_name(weights.status) << " (" << weights.reason << ")\n";
    return {std::move(cohort), weights};
}

} // namespace strix::inkling2
