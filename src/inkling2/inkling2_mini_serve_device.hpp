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
#include <optional>

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
    const auto build = [&](unsigned chunk, unsigned tail_capacity) {
        std::unique_ptr<MiniResidentDecoder> built;
        try {
            built = std::make_unique<MiniResidentDecoder>(opt.checkpoint, opt.context, chunk,
                false, MiniResidentBackend::default_step_graphs, weights.storage, 0U, false, rows, tail_capacity);
        } catch (const std::exception& e) {
            throw std::runtime_error("Inkling decoder with a prefill chunk of " + std::to_string(chunk) +
                                     " rows did not fit the device (" + e.what() + "); pass --prefill-rows " +
                                     std::to_string(kMiniServeDefaultSlotRows) + " for the incumbent served chunk");
        }
        built->set_served_identity(opt.served_identity, opt.form_rows);
        built->set_moe_w4a4(opt.moe_w4a4);
        return built;
    };
    unsigned chunk = opt.prefill_rows;
    auto decoder = build(chunk, mini_prefill_tail_capacity(chunk, opt.prefill_tail_divisor, opt.context));
    // Sized before any slot is committed: the slots the budget holds with this chunk, and the slots it would hold with
    // the decoder at the incumbent chunk (both complete plans: the rows-proportional reservation, first slot included,
    // and the per-slot KV layout of each chunk). Explicit --slots that do not fit and a default chunk that costs slots
    // the incumbent chunk admits (or that cannot be compared) are refused (mini_slot_refusal), never silently fewer.
    // The incumbent identity keeps its bits at any chunk that is a multiple of its form chunk, so a default chunk that
    // costs slots is replaced by the incumbent chunk (the decoder is rebuilt; mini_slot_keep_incumbent) instead of
    // refused; another identity, explicit pieces or an unpriced plan are refused below.
    auto memory = mini_gpu::memory_info();
    auto slot_bytes = decoder->slot_bytes();
    auto lanes = serve::resident_kv_slots(memory.free, memory.total, budget.device_bytes, opt.kv_bytes, slot_bytes, rows);
    const auto incumbent_rows = std::min(kMiniServeDefaultSlotRows, opt.prefill_rows);
    std::optional<unsigned> lanes_incumbent;
    if (chunk > incumbent_rows) {
        const auto capacity = MiniResidentDecoder::served_row_capacity(incumbent_rows, rows);
        if (const auto reservation = MiniResidentDecoder::row_reservation_bytes(opt.context, capacity, false))
            lanes_incumbent = mini_counterfactual_slots(memory.free, memory.total, budget.device_bytes, opt.kv_bytes,
                                                        rows, decoder->allocated_row_reservation_bytes(), *reservation,
                                                        MiniResidentDecoder::slot_bytes(opt.context, capacity));
    }
    if (mini_slot_keep_incumbent(lanes, lanes_incumbent, opt.slots, opt.prefill_rows_requested != 0U, chunk,
                                 incumbent_rows) &&
        opt.served_identity == MiniServedIdentity::incumbent && opt.prefill_rows_decoding == 0U) {
        std::cerr << "Inkling: the default prefill chunk of " << chunk << " rows would leave " << lanes
                  << " session slots where the incumbent " << incumbent_rows << "-row chunk leaves " << *lanes_incumbent
                  << ": serving the incumbent chunk (same bits; one-pass prefill: --prefill-rows " << chunk
                  << " --slots " << lanes << ")\n";
        decoder.reset();
        chunk = incumbent_rows;
        decoder = build(chunk, 0U);
        memory = mini_gpu::memory_info();
        slot_bytes = decoder->slot_bytes();
        lanes = serve::resident_kv_slots(memory.free, memory.total, budget.device_bytes, opt.kv_bytes, slot_bytes, rows);
        lanes_incumbent.reset();
    }
    if (const auto refusal = mini_slot_refusal(lanes, lanes_incumbent, opt.slots, opt.prefill_rows_requested != 0U,
                                               chunk, incumbent_rows, slot_bytes);
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
    return {std::move(cohort), weights, std::string_view(backend) == "HIP", std::move(identity), chunk};
}

} // namespace strix::inkling2
