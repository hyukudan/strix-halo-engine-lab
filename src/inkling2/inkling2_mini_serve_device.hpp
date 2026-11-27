#pragma once

#include "inkling2_mini_resident_decoder.hpp"
#include "inkling2_mini_serve_lanes.hpp"
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
    // The SP4 attention + head column in this instance's decode regime (the command line's rule with its own extent).
    {
        const auto probe = mini_weight_scope(opt.checkpoint, opt.tier, weight_options, extent);
        const MiniPromptExecution run{probe.cost_device, opt.context, opt.prefill_rows};
        const auto lowbit_head = mini_decode_lowbit_head(weight_options, probe.device_class, extent, run, kServeHip);
        if (!lowbit_head.ladder.empty()) std::cerr << lowbit_head.ladder << '\n';
        // A lowered column is a recorded decline (startup line, /v1/capabilities), with the extent's residency.
        if (auto decline = mini_lowbit_head_decline(opt.weight_options, lowbit_head, probe.device_class, extent, run))
            profile::record_decline(std::move(*decline), std::cerr);
    }
    // Any served route admits the calibrated row: its decoder serves prompts emits-first (mini_served_prompt_emits_first),
    // so the final prompt token stays a BF16 prefill row. Outside the row's measured regime the incumbent loads.
    if (const auto decline = mini_lowbit_row_decline(opt.tier, weight_options, kMiniWeightBackend, extent);
        !decline.empty()) {
        std::cerr << "strix-lowbit-pack/v2 declined (" << decline << "); using incumbent weights\n";
        profile::record_decline({std::string(kMiniLowBitTurbo), "1", "0", profile::DeclineOrigin::load,
                                 profile::decline_regime("decode", extent.context, extent.lanes), decline},
                                std::cerr);
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
    bool windows = false;
    std::string_view windows_declined = "decoder declined verify windows";
    std::function<std::uint64_t()> projected_bias_dispatches;
#if !defined(__HIPCC__)
    exec::CohortProgramPolicy programs;
#endif
    // Lone-lane speculation (either backend): the model's DSpark sidecar beside the checkpoint, validated like the
    // command line's (absent or declined: the lone lane drafts lookup-only). Found, the decoder captures slot 0's DSpark
    // taps and the cohort binds the drafter (MiniServeLaneBinding) before the slots are budgeted.
    spec::ResolvedDrafter drafter{{}, spec::DrafterSource::context, {}};
    if (opt.speculation) {
        drafter = spec::resolve_drafter_file({{}, false, true}, opt.checkpoint,
                                             model_data::inkling2_mini::speculation::drafter);
        if (drafter.source == spec::DrafterSource::beside_model)
            if (auto problem = MiniDSparkDeviceWeights::sidecar_problem(drafter.path); !problem.empty())
                drafter = spec::decline_drafter(drafter, std::move(problem));
    }
    const bool draft_taps = opt.speculation && !drafter.path.empty();
    bool lanes_bound = false;
    [[maybe_unused]] bool window_programs = false, window_programs_declined = false;
    [[maybe_unused]] unsigned window_prewarmed = 0U;
    // The admitted prompt chunk (serve::load_with_chunk_ladder) and the free device bytes the slot plan read.
    unsigned chunk_rows = opt.prefill_rows;
    std::uint64_t admission_free = 0;
    auto cohort = load_mini_weights(opt.checkpoint, weights, opt.tier, weight_options, scope.view(), extent, [&](MiniLowBitRows lowbit) {
        const bool calibrated = lowbit.admitted != nullptr;
        const bool decode_only_pack = lowbit.decode_only;
        using Cohort = serve::FixedKvCohort<MiniResidentDecoder>;
        using Admission = serve::ChunkAdmission<std::unique_ptr<Cohort>>;
        // The served route is admitted before constructing the decoder. ONE load transaction per candidate chunk builds
        // the decoder, checks capacity and allocates the slots before publishing the cohort. An implicit chunk on a
        // chunk-invariant backend (CUDA) has candidates (mini_served_chunk_ladder): the largest whose whole load keeps
        // the slot capacity of the incumbent 128-row chunk is kept. The served route lever runs at the first candidate
        // only (the command line's chunk); a smaller candidate runs the held-back route (whose bits do not depend on
        // the chunk), so the command line's chunk never costs a session slot.
        std::uint64_t incumbent_free = 0, incumbent_slot_bytes = 0, incumbent_plane = 0;
        const auto ladder = mini_served_chunk_ladder(opt.prefill_rows, opt.prefill_fallback_rows);
        const auto load = [&](std::uint32_t chunk, bool last) {
            const bool route = opt.served_cli_route && chunk == ladder.rows[0];
            // CUDA: slot 0 binds the command line's verify window tape (the constructor's verify rows, as the command
            // line's load.verify_rows) when windows may be served: a staged window is then taped, so it replays its
            // window program and a partial commit rebuilds by copies. Without it (q63) every served window ran eagerly
            // and every partial commit replayed its kept rows: served code 100.9 tok/s vs 180.6 on the command line.
#if defined(__HIPCC__)
            const unsigned tape_rows = 0U;  // HIP served windows: unchanged
#else
            const unsigned tape_rows = opt.speculation ? kMiniMaxVerifyRows : 0U;
#endif
            auto decoder = std::make_unique<MiniResidentDecoder>(opt.checkpoint, opt.context, chunk,
                packed_decode, MiniResidentBackend::default_step_graphs,
                weights.roles(), tape_rows, draft_taps, rows, route ? opt.prefill_tail : 0U,
                opt.experts_nvfp4, opt.controller_one_plane, route, lowbit);
            if (!mini_decode_only_pack_route_ok(calibrated, decode_only_pack, decoder->prefill_emits_first()))
                throw std::logic_error("calibrated decode image on the held-back served prompt route");
            // Held-back route: every prompt row on the prefill form whatever chunk or piece cuts it (CUDA; HIP no-op).
            if (!route) decoder->keep_prefill_grouped();
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
#if !defined(__HIPCC__)
            // Verify window programs for the staged windows: the command line's rule (on by default,
            // STRIX_INKLING_WINDOW_PROGRAMS=0 turns them off) and its proven levers (identity fixture q52r: EXACT levers only), so a served window runs the command
            // line's window arithmetic and wall.
            window_programs = windows && mini_window_programs_requested();
            // The fold the backend runs (levers), not the request: CUDA declines the default-on fold.
            if (window_programs && !mini_window_programs_proven(opt.combine_image, levers.prepared_round_fold,
                                                                opt.experts_nvfp4, opt.controller_one_plane,
                                                                opt.moe_w4a4, opt.attn_projected_bias, weights.roles())) {
                window_programs = false;
                window_programs_declined = true;
            }
            decoder->set_window_programs(window_programs);
#endif
            // The drafter's device weights and context exist before the slots read the free memory.
            std::unique_ptr<MiniServeLaneBinding> binding;
            if (windows && draft_taps) binding = std::make_unique<MiniServeLaneBinding>(*decoder, opt.context, drafter.path);
            slot_bytes = decoder->slot_bytes();
            const auto plane = decoder->screen_device_bytes();
            // Keep the CUDA screened plane only when it costs no slot, then enforce the calibrated load's slot promise.
            auto memory = mini_gpu::memory_info();
            lanes = mini_serve_kv_slots(*decoder, memory.free, memory.total, budget.device_bytes, opt.kv_bytes, rows);
            if (windows && opt.slots != 0U && lanes < rows) {
                // The windows' tapes and scratch would cost a configured slot: the slots win, the lanes decode one row.
                binding.reset();
                windows = decoder->set_slot_windows(false);
                windows_declined = "slot capacity";
#if !defined(__HIPCC__)
                window_programs = false;
                decoder->set_window_programs(false);
#endif
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
            if (!kMiniProfile.ladder.empty()) {
                policy = [options = weight_options, device = scope.device_class, product = scope.cost_device,
                          context = std::uint64_t{opt.context}, chunk](MiniResidentDecoder& resident,
                                                                      std::uint64_t tokens, bool first) {
                    resident.set_moe_w4a4(mini_prompt_moe(options, device, tokens, first, nullptr,
                                                          MiniPromptExecution{product, context, chunk}));
                };
                // The same rule for a whole waiting prompt: one the policy would run in the integer class stays alone.
                decoder->set_prompt_integer_rows([options = weight_options, device = scope.device_class,
                                                  product = scope.cost_device, context = std::uint64_t{opt.context},
                                                  chunk](std::uint64_t prompt_rows) {
                    return mini_prompt_moe(options, device, prompt_rows, true, nullptr,
                                           MiniPromptExecution{product, context, chunk}) ==
                           moe_core::grouped_w4a4::Activations::iu4_rot;
                });
            }
            // A forward's row budget holds a lone lane's whole window even under --slots 1 (the verify width is not the
            // slot count): the window is its own forward, the one-row lanes stay within the decoder's rows.
            const auto forward_rows = std::max<unsigned>(rows, windows ? decoder->max_slot_window() : 1U);
            auto candidate =
                std::make_unique<Cohort>(std::move(decoder), opt.context, forward_rows, lanes, std::move(policy));
            lanes_bound = binding != nullptr;
            if (binding) candidate->bind_lanes(std::move(binding));
#if !defined(__HIPCC__)
            // Every window width captured now, slots allocated and slot 0 empty (MiniResidentDecoder::
            // prewarm_window_programs): no capture wall enters a served decode; prompts keep the programs.
            window_prewarmed = window_programs ? observed->prewarm_window_programs() : 0U;
#endif
            projected_bias_dispatches = nullptr;
            if (opt.attn_projected_bias)
                projected_bias_dispatches = [observed] { return observed->projected_attention_dispatches(); };
            return Admission{std::move(candidate), lanes};
        };
        // A candidate's slots cannot exceed the incumbent's budget with its slot 0 returned and the candidate's own
        // slot 0 taken (its row-scaled workspaces are never smaller than the incumbent's): host arithmetic only. An
        // implicit CUDA chunk (and the explicit HIP tall-chunk option) has more than one candidate; the bound leaves
        // verify tapes out, so it can only over-estimate and the candidate's own load decides.
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
    // The served route lever as admitted: at the command line's chunk only (a smaller candidate runs held-back).
    const bool served_route = opt.served_cli_route && chunk_rows == opt.prefill_rows;
    auto shown = opt;
    shown.served_cli_route = served_route;
    if (opt.served_cli_route && !served_route)
        std::cerr << "Inkling served_route declined: the command line's chunk " << opt.prefill_rows
                  << " rows would cost session slots; held-back route at chunk " << chunk_rows << " rows\n";
    std::cerr << "Inkling " << backend << " slots " << lanes << " context " << opt.context
              << " prefill_chunk " << chunk_rows
              << " moe_decode " << mini_decode_form_name(packed_decode)
              << " KV and history bytes per slot " << slot_bytes
              << " slot budget free bytes " << admission_free
              << " profile " << mini_profile_name(opt.tier)
              << " moe_prefill " << moe_core::grouped_w4a4::identity(opt.moe_w4a4)
              << " served_route " << mini_served_route_startup_field(shown, cohort->prompt_tail_rows())
              << " prepared_round_fold " << mini_round_fold_startup_field(opt, levers)
              << " verify_rows "
              << mini_verify_rows_name(mini_served_verify_rows(levers.verify_rows, lanes), opt.verify_rows)
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
    // The command line's speculation line (mini_speculation_line), with the decoder's admission.
    auto speculation = opt.speculation_policy;
    if (speculation.admitted && !windows) speculation.reason = windows_declined;
    speculation.admitted = windows;
    std::cerr << mini_speculation_line(speculation, opt.verify_rows) << '\n';
    if (windows) {
        // Lone-lane speculation: the DSpark drafter bound on slot 0 (else lookup-only drafts).
        if (!lanes_bound && !drafter.path.empty()) drafter = spec::decline_drafter(drafter, "verify windows declined");
        std::cerr << spec::drafter_receipt("served", drafter) << '\n';
        std::cerr << "Inkling lane_speculation on max_live 1 drafter " << (lanes_bound ? "dspark" : "lookup") << '\n';
#if !defined(__HIPCC__)
        std::cerr << "Inkling verify window programs "
                  << (window_programs ? "on prewarmed " + std::to_string(window_prewarmed)
                                      : window_programs_declined ? std::string("declined (unproven for this profile)")
                                                                 : std::string("off"))
                  << '\n';
#endif
    }
    // Prompt pieces beside decoding lanes (mini_serve_pieces): the shared long pieces where the admitted route cuts
    // prompt rows (CUDA either route, HIP the command line's route), the short ones on the HIP held-back route.
    const auto pieces = mini_serve_pieces(kServeHip, served_route);
    return {std::move(cohort), weights, std::move(projected_bias_dispatches), chunk_rows, pieces, served_route};
}

} // namespace strix::inkling2
