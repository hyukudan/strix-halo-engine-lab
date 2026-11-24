#pragma once

#include "inkling2_mini_resident_decoder.hpp"
#include "inkling2_mini_serve_options.hpp"
#include "inkling2_mini_weight_device.hpp"
#include "strix/device/gate.hpp"
#include "strix/serve/fixed_kv_cohort.hpp"
#if defined(__HIPCC__)
#include "device/hip_cost_identity.hpp"
#else
#include "device/cuda_cost_identity.hpp"
#endif

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
    // Slot rows: --slots, else main's implicit cap (the requested --prefill-rows or 128, clipped to the context).
    const auto rows = std::min(opt.slots ? opt.slots : mini_serve_slot_cap(opt.prefill_rows_requested, opt.context),
                               row_limit);
    const auto scope = mini_weight_scope(opt.checkpoint, opt.tier, opt.weight_options);
    const auto weights = resolve_mini_resident_weight_load(opt.tier, opt.weight_options, scope.view());
    std::unique_ptr<MiniResidentDecoder> decoder;
    try {
        decoder = std::make_unique<MiniResidentDecoder>(opt.checkpoint, opt.context, opt.prefill_rows,
            false, MiniResidentBackend::default_step_graphs, weights.storage, 0U, false, rows,
            mini_prefill_tail_capacity(opt.prefill_rows, opt.prefill_tail_divisor, opt.context));
    } catch (const std::exception& e) {
        throw std::runtime_error("Inkling decoder with a prefill chunk of " + std::to_string(opt.prefill_rows) +
                                 " rows did not fit the device (" + e.what() + "); pass --prefill-rows " +
                                 std::to_string(kMiniServeDefaultSlotRows) + " for the incumbent served chunk");
    }
    decoder->set_served_identity(opt.served_identity, opt.form_rows);
    decoder->set_moe_w4a4(opt.moe_w4a4);
    // Sized before any slot is committed: the slots the budget holds with this chunk, and with the incumbent chunk
    // (its smaller rows-proportional arenas handed back); explicit --slots that do not fit and a default chunk that
    // costs slots the incumbent chunk admits are refused (mini_slot_refusal), never silently fewer.
    const auto memory = mini_gpu::memory_info();
    const auto slot_bytes = decoder->slot_bytes();
    const auto lanes = serve::resident_kv_slots(memory.free, memory.total, budget.device_bytes, opt.kv_bytes,
                                               slot_bytes, rows);
    const auto incumbent_rows = std::min(kMiniServeDefaultSlotRows, opt.prefill_rows);
    const auto handed_back = mini_row_arena_bytes(opt.prefill_rows, kHiddenWidth, kMiniResidentImages) -
                             mini_row_arena_bytes(incumbent_rows, kHiddenWidth, kMiniResidentImages);
    const auto lanes_incumbent = serve::resident_kv_slots(std::min(memory.total, memory.free + handed_back),
                                                         memory.total, budget.device_bytes, opt.kv_bytes, slot_bytes,
                                                         rows);
    if (const auto refusal = mini_slot_refusal(lanes, lanes_incumbent, opt.slots, opt.prefill_rows_requested != 0U,
                                               opt.prefill_rows, incumbent_rows, slot_bytes);
        !refusal.empty())
        throw std::runtime_error(refusal);
#if !defined(__HIPCC__)
    const auto programs = decoder->cohort_programs();
#endif
    auto cohort = std::make_unique<serve::FixedKvCohort<MiniResidentDecoder>>(std::move(decoder), opt.context, rows, lanes);
    std::cerr << "Inkling " << backend << " slots " << lanes << " context " << opt.context
              << " KV and history bytes per slot " << slot_bytes
              << " profile " << mini_profile_name(opt.tier)
              << " moe_prefill " << moe_core::grouped_w4a4::identity(opt.moe_w4a4)
              << " weight_storage " << weights.storage.name << '@' << weights.storage.revision
#if !defined(__HIPCC__)
              << " cohort_programs " << (programs.enabled ? "on" : "off")
              << " widths " << exec::cohort_widths_name(programs.widths)
#endif
              << " weight_status " << profile::status_name(weights.status) << " (" << weights.reason << ")\n";
#if defined(__HIPCC__)
    auto identity = device::hip_cost_device_identity();
#else
    auto identity = device::cuda_cost_device_identity();
#endif
    return {std::move(cohort), weights, std::string_view(backend) == "HIP", std::move(identity)};
}

} // namespace strix::inkling2
