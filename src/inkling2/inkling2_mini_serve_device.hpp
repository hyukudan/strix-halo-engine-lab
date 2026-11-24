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
    const auto rows = std::min(opt.slots ? opt.slots : opt.prefill_rows, row_limit);
    const auto scope = mini_weight_scope(opt.checkpoint, opt.tier, opt.weight_options);
    std::string pack_note;
    std::shared_ptr<const weight::LowbitPack> admitted_pack;  // the verified open pack the load reads
    const auto weights = resolve_mini_resident_weight_load(opt.tier, opt.weight_options, scope.view(), opt.checkpoint,
                                                           &pack_note, &admitted_pack);
    if (!pack_note.empty()) std::cerr << "strix-lowbit-pack/v2 " << pack_note << '\n';
    // The scope a decision row of this load must match (computed only when a row or an arm can apply).
    if (!(scope.checkpoint == weight::Digest256{})) std::cerr << weight::weight_load_scope_receipt(scope.view()) << '\n';
    auto decoder = std::make_unique<MiniResidentDecoder>(opt.checkpoint, opt.context, opt.prefill_rows,
        false, MiniResidentBackend::default_step_graphs, weights.roles(), 0U, false, rows, 0U,
        mini_lowbit_rows(weights, opt.weight_options, scope.checkpoint, admitted_pack));
    decoder->set_moe_w4a4(opt.moe_w4a4);
    const auto memory = mini_gpu::memory_info();
    const auto slot_bytes = decoder->slot_bytes();
    const auto lanes = serve::resident_kv_slots(memory.free, memory.total, budget.device_bytes, opt.kv_bytes,
                                               slot_bytes, rows);
    if (!lanes) throw std::runtime_error("Inkling KV budget holds no session slots");
#if !defined(__HIPCC__)
    const auto programs = decoder->cohort_programs();
#endif
    auto cohort = std::make_unique<serve::FixedKvCohort<MiniResidentDecoder>>(std::move(decoder), opt.context, rows, lanes);
    std::cerr << "Inkling " << backend << " slots " << lanes << " context " << opt.context
              << " KV and history bytes per slot " << slot_bytes
              << " profile " << mini_profile_name(opt.tier)
              << " moe_prefill " << moe_core::grouped_w4a4::identity(opt.moe_w4a4)
              << " weight_storage " << weights.storage.name << '@' << weights.storage.revision
              << (weights.head_storage != weights.storage ? " weight_head_storage " : "")
              << (weights.head_storage != weights.storage ? weights.head_storage.name : "")
#if !defined(__HIPCC__)
              << " cohort_programs " << (programs.enabled ? "on" : "off")
              << " widths " << exec::cohort_widths_name(programs.widths)
#endif
              << " weight_status " << profile::status_name(weights.status) << " (" << weights.reason << ")"
              << (weights.turbo_quality.empty() ? "" : " turbo_quality ") << weights.turbo_quality << '\n';
    return {std::move(cohort), weights};
}

} // namespace strix::inkling2
