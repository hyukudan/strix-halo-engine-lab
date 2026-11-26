#pragma once

#include "inkling2_mini_resident_decoder.hpp"
#include "inkling2_mini_serve_options.hpp"
#include "inkling2_mini_weight_device.hpp"
#include "strix/device/gate.hpp"
#include "strix/serve/fixed_kv_cohort.hpp"
#include "strix/serve/prefill_chunk_fallback.hpp"
#include "strix/serve/row_budget.hpp"
#include "strix/spec/drafter_source.hpp"

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
    // Any served route admits the calibrated row: its decoder serves prompts emits-first (mini_served_prompt_emits_first),
    // so the final prompt token stays a BF16 prefill row. Outside the row's measured regime the incumbent loads.
    if (const auto decline = mini_lowbit_row_decline(opt.tier, weight_options, kMiniWeightBackend, extent);
        !decline.empty())
        std::cerr << "strix-lowbit-pack/v2 declined (" << decline << "); using incumbent weights\n";
    const auto scope = mini_weight_scope(opt.checkpoint, opt.tier, weight_options, extent);
    auto weights = resolve_mini_resident_weight_load(opt.tier, weight_options, scope.view(), extent);
    if (!(scope.checkpoint == weight::Digest256{})) std::cerr << weight::weight_load_scope_receipt(scope.view()) << '\n';
    // Decode expert form, the CLI's on CUDA (flags included), fixed for every candidate load.
    const bool packed_decode =
        mini_served_packed_decode(kServeHip, opt.served_cli_route, opt.packed_decode_flag, opt.grouped_decode_flag);
    std::uint64_t slot_bytes = 0;
    unsigned lanes = 0;
    MiniDecoderLevers levers;
    bool windows = false;
    std::string_view windows_declined = "decoder declined verify windows";
    std::function<std::uint64_t()> projected_bias_dispatches;
#if !defined(__HIPCC__)
    exec::CohortProgramPolicy programs;
#endif
    // The admitted prompt chunk (serve::load_with_chunk_ladder) and the free device bytes the slot plan read.
    unsigned chunk_rows = opt.prefill_rows;
    std::uint64_t admission_free = 0;
    auto cohort = load_mini_weights(opt.checkpoint, weights, opt.tier, weight_options, scope.view(), extent, [&](MiniLowBitRows lowbit) {
        const bool calibrated = lowbit.admitted != nullptr;
        const bool decode_only_pack = lowbit.decode_only;
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
            if (!mini_decode_only_pack_route_ok(calibrated, decode_only_pack, decoder->prefill_emits_first()))
                throw std::logic_error("calibrated decode image on the held-back served prompt route");
            // Held-back route: every prompt row on the prefill form whatever chunk or piece cuts it (CUDA; HIP no-op).
            if (!opt.served_cli_route) decoder->keep_prefill_grouped();
#if defined(__HIPCC__)
            // A TURBO load whose prompts may select the rotated integer class builds its images now, not in the first
            // prompt.
            if (mini_prompt_moe_may_select(weight_options, scope.device_class,
                                           moe_core::grouped_w4a4::Activations::iu4_rot,
                                           MiniPromptExecution{scope.cost_device, opt.context, chunk}))
                decoder->set_moe_w4a4(moe_core::grouped_w4a4::Activations::iu4_rot);
#endif
            levers = configure_mini_decoder(*decoder, opt.moe_w4a4, opt.combine_image, opt.prepared_round_fold,
                                            opt.attn_projected_bias, opt.screened_head, opt.verify_rows);
            // Admitted speculation: one-lane verify windows on any slot (per-slot tapes count in the slot's bytes). The
            // decoder declines windows on a calibrated low-bit target (mini_speculation_policy never requests both).
            windows = decoder->set_slot_windows(opt.speculation);
            windows_declined = "decoder declined verify windows";
            slot_bytes = decoder->slot_bytes();
            const auto plane = decoder->screen_device_bytes();
            // Keep the CUDA screened plane only when it costs no slot, then enforce the calibrated load's slot promise.
            auto memory = mini_gpu::memory_info();
            lanes = mini_serve_kv_slots(*decoder, memory.free, memory.total, budget.device_bytes, opt.kv_bytes, rows);
            if (windows && opt.slots != 0U && lanes < rows) {
                // The windows' tapes and scratch would cost a configured slot: the slots win, the lanes decode one row.
                windows = decoder->set_slot_windows(false);
                windows_declined = "slot capacity";
                slot_bytes = decoder->slot_bytes();
                memory = mini_gpu::memory_info();
                lanes = mini_serve_kv_slots(*decoder, memory.free, memory.total, budget.device_bytes, opt.kv_bytes, rows);
            }
            admission_free = memory.free;
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
                policy = [options = weight_options, device = scope.device_class, product = scope.cost_device,
                          context = std::uint64_t{opt.context}, chunk](MiniResidentDecoder& resident,
                                                                      std::uint64_t tokens, bool first) {
                    resident.set_moe_w4a4(mini_prompt_moe(options, device, tokens, first, nullptr,
                                                          MiniPromptExecution{product, context, chunk}));
                };
            // A forward's row budget holds a lone lane's whole window even under --slots 1 (the verify width is not the
            // slot count): the window is its own forward, the one-row lanes stay within the decoder's rows.
            const auto forward_rows = std::max<unsigned>(rows, windows ? decoder->max_slot_window() : 1U);
            auto candidate =
                std::make_unique<Cohort>(std::move(decoder), opt.context, forward_rows, lanes, std::move(policy));
            projected_bias_dispatches = nullptr;
            if (opt.attn_projected_bias)
                projected_bias_dispatches = [observed] { return observed->projected_attention_dispatches(); };
            return Admission{std::move(candidate), lanes};
        };
        // A candidate's slots cannot exceed the incumbent's budget with its slot 0 returned and the candidate's own
        // slot 0 taken (its row-scaled workspaces are never smaller than the incumbent's): host arithmetic only. Only
        // an implicit CUDA chunk has more than one candidate, and a CUDA decoder declines verify windows (no tapes).
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
              << " speculation " << (windows ? "auto" : opt.speculation_requested ? "declined" : "off")
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
    // The command line's speculation line (mini_speculation_line), with the decoder's admission; a served lane drafts
    // from its own context (no sidecar drafter on this route).
    auto speculation = opt.speculation_policy;
    if (speculation.admitted && !windows) speculation.reason = windows_declined;
    speculation.admitted = windows;
    std::cerr << mini_speculation_line(speculation, opt.verify_rows) << '\n';
    if (windows) std::cerr << spec::drafter_receipt("served", {{}, spec::DrafterSource::context, {}}) << '\n';
    // CUDA, either prompt route: the shared long pieces beside decoding lanes, cut by the CUDA rule
    // (mini_prefill_piece_rule); HIP keeps the short ones.
    const auto pieces = !kServeHip ? kMiniCudaServePieces : serve::kMixedServiceShortPieces;
    return {std::move(cohort), weights, std::move(projected_bias_dispatches), chunk_rows, pieces};
}

} // namespace strix::inkling2
