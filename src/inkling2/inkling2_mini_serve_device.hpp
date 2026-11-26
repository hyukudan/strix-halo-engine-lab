#pragma once

#include "inkling2_mini_resident_decoder.hpp"
#include "inkling2_mini_serve_options.hpp"
#include "inkling2_mini_weight_device.hpp"
#include "strix/device/gate.hpp"
#include "strix/serve/fixed_kv_cohort.hpp"
#include "strix/serve/prefill_chunk_fallback.hpp"
#include "strix/serve/row_budget.hpp"

#include <functional>
#include <stdexcept>
#include <string>
#include <iostream>
#include <limits>

namespace strix::inkling2 {

inline device::Request mini_serve_device_request(const MiniServeOptions& opt) {
    device::Request request;
    request.select = opt.device_id;
    request.profile = opt.device_profile;
    return request;
}

#if defined(__HIPCC__)
inline constexpr bool kServeHip = true;
#else
inline constexpr bool kServeHip = false;
#endif

inline MiniServeCohort make_mini_device_cohort(
    const MiniServeOptions& opt, const device::MemoryBudget& budget, std::string_view backend,
    unsigned row_limit = std::numeric_limits<unsigned>::max()) {
    const auto rows = std::min(mini_serve_lane_rows(opt), row_limit);
    // The instance's extent for the weight rows: its configured context and its slot cap (0 = left to the memory
    // budget: lane-banded gains do not select).
    const weight::WeightLoadExtent extent{opt.context, opt.slots ? rows : 0U};
    auto weight_options = opt.weight_options;
    const bool calibrated_row = mini_server_calibrated_row_allowed(weight_options.lowbit_turbo, opt.served_cli_route);
    if (weight_options.lowbit_turbo && !calibrated_row) {
        weight_options.lowbit_turbo = false;
        std::cerr << "strix-lowbit-pack/v2 declined (held-back served prompt route); using incumbent weights\n";
    }
    const auto scope = mini_weight_scope(opt.checkpoint, opt.tier, weight_options, extent);
    auto weights = resolve_mini_resident_weight_load(opt.tier, weight_options, scope.view(), extent);
    if (!(scope.checkpoint == weight::Digest256{})) std::cerr << weight::weight_load_scope_receipt(scope.view()) << '\n';
    // Decode expert form, the CLI's on CUDA (flags included), fixed for every candidate load.
    const bool packed_decode =
        mini_served_packed_decode(kServeHip, opt.served_cli_route, opt.packed_decode_flag, opt.grouped_decode_flag);
    std::uint64_t slot_bytes = 0;
    unsigned lanes = 0;
    MiniDecoderLevers levers;
    std::function<std::uint64_t()> projected_bias_dispatches;
#if !defined(__HIPCC__)
    exec::CohortProgramPolicy programs;
#endif
    // The admitted prompt chunk (serve::load_with_chunk_ladder) and the free device bytes the slot plan read.
    unsigned chunk_rows = opt.prefill_rows;
    std::uint64_t admission_free = 0;
    auto cohort = load_mini_weights(opt.checkpoint, weights, opt.tier, weight_options, scope.view(), extent, [&](MiniLowBitRows lowbit) {
        const bool calibrated = lowbit.admitted != nullptr;
        using Cohort = serve::FixedKvCohort<MiniResidentDecoder>;
        using Admission = serve::ChunkAdmission<std::unique_ptr<Cohort>>;
        // The served route is admitted before constructing the decoder. ONE load transaction per candidate chunk builds
        // the decoder, checks capacity and allocates the slots before publishing the cohort. An implicit held-back
        // chunk on a chunk-invariant backend (CUDA) has candidates (mini_served_chunk_ladder): the largest whose whole
        // load keeps the slot capacity of the incumbent 128-row chunk is kept (the chunk changes no output there).
        std::uint64_t incumbent_free = 0, incumbent_slot_bytes = 0, incumbent_plane = 0;
        const auto ladder = mini_served_chunk_ladder(opt.prefill_rows, opt.prefill_fallback_rows);
        const auto load = [&](std::uint32_t chunk, bool last) {
            auto decoder = std::make_unique<MiniResidentDecoder>(opt.checkpoint, opt.context, chunk,
                packed_decode, MiniResidentBackend::default_step_graphs,
                weights.roles(), 0U, false, rows, opt.served_cli_route ? opt.prefill_tail : 0U,
                opt.experts_nvfp4, opt.controller_one_plane, opt.served_cli_route, lowbit);
            levers = configure_mini_decoder(*decoder, opt.moe_w4a4, opt.combine_image, opt.prepared_round_fold,
                                            opt.attn_projected_bias, opt.screened_head, opt.verify_rows);
            slot_bytes = decoder->slot_bytes();
            const auto plane = decoder->screen_device_bytes();
            // Keep the CUDA screened plane only when it costs no slot, then enforce the calibrated load's slot promise.
            const auto memory = mini_gpu::memory_info();
            admission_free = memory.free;
            lanes = mini_serve_kv_slots(*decoder, memory.free, memory.total, budget.device_bytes, opt.kv_bytes, rows);
            if (last) {
                incumbent_free = memory.free;
                incumbent_slot_bytes = slot_bytes;
                incumbent_plane = plane;
            } else if (lanes == 0U) {
                return Admission{};  // a larger candidate holds no slot: the ladder declines it
            }
            if (lanes == 0U)
                throw std::runtime_error("Inkling KV budget holds no session slots (prefill chunk " +
                                         std::to_string(chunk) + " rows, KV and history bytes per slot " +
                                         std::to_string(slot_bytes) + ", free device bytes " +
                                         std::to_string(memory.free) + ")");
            if (last || lanes >= rows) require_mini_weight_slot_capacity(lanes, rows, calibrated);
            levers.head = decoder->head();
#if !defined(__HIPCC__)
            programs = decoder->cohort_programs();
#endif
            // The cohort owns the decoder while the server observes its projected-bias dispatches.
            auto* observed = decoder.get();
            Cohort::PrefillPolicy policy;
            if (!kMiniProfile.ladder.empty())
                policy = [options = weight_options, device = scope.device_class](MiniResidentDecoder& resident,
                                                                                std::uint64_t tokens, bool first) {
                    resident.set_moe_w4a4(mini_prompt_moe(options, device, tokens, first));
                };
            auto candidate = std::make_unique<Cohort>(std::move(decoder), opt.context, rows, lanes, std::move(policy));
            projected_bias_dispatches = nullptr;
            if (opt.attn_projected_bias)
                projected_bias_dispatches = [observed] { return observed->projected_attention_dispatches(); };
            return Admission{std::move(candidate), lanes};
        };
        // A candidate's slots cannot exceed the incumbent's budget with its slot 0 returned and the candidate's own
        // slot 0 taken (its row-scaled workspaces are never smaller than the incumbent's): host arithmetic only.
        const auto bound = [&](std::uint32_t chunk) -> std::uint32_t {
            const auto bytes = MiniResidentDecoder::held_back_slot_bytes(opt.context, chunk, rows);
            const std::uint64_t returned = incumbent_free + incumbent_slot_bytes;
            const std::uint64_t room = returned > bytes ? returned - bytes : 0U;
            const auto memory = mini_gpu::memory_info();
            return serve::resident_kv_slots_with_plane(room, memory.total, budget.device_bytes, opt.kv_bytes, bytes,
                                                       rows, incumbent_plane).slots;
        };
        auto admitted = serve::load_with_chunk_ladder(ladder.view(), rows, std::cerr, load, bound);
        chunk_rows = admitted.chunk;
        return std::move(admitted.service);
    });
    std::cerr << "Inkling " << backend << " slots " << lanes << " context " << opt.context
              << " prefill_chunk " << chunk_rows
              << " moe_decode " << mini_decode_form_name(packed_decode)
              << " KV and history bytes per slot " << slot_bytes
              << " slot budget free bytes " << admission_free
              << " profile " << mini_profile_name(opt.tier)
              << " moe_prefill " << moe_core::grouped_w4a4::identity(opt.moe_w4a4)
              << " served_route " << mini_served_route_startup_field(opt, cohort->prompt_tail_rows())
              << " prepared_round_fold " << mini_round_fold_startup_field(opt, levers)
              << " verify_rows "
              << mini_verify_rows_name(mini_served_verify_rows(levers.verify_rows, opt.served_cli_route, lanes),
                                       opt.verify_rows)
              << " down_image " << mini_down_image_name(levers.down_image, opt.combine_image)
              << " head " << mini_head_field(levers.head)
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
    return {std::move(cohort), weights, std::move(projected_bias_dispatches), chunk_rows};
}

} // namespace strix::inkling2
