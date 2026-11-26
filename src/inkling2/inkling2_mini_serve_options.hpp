#pragma once

#include "inkling2_mini_resident_route.hpp"
#include "inkling2_mini_schedule.hpp"
#include "inkling2_mini_weight_options.hpp"
#include "strix/serve/cohort_decode.hpp"
#include "strix/serve/row_budget.hpp"

#include <cstdint>
#include <functional>
#include <memory>
#include <optional>
#include <string>

namespace strix::inkling2 {

struct MiniServeOptions {
    std::string checkpoint;
    std::string tokenizer;
    std::string devices_path;
    std::string device_id;
    std::string device_profile;
    unsigned context = 8192;
    // Prefill chunk rows (resolve: --prefill-rows, else the route's chunk, clamped to the context).
    unsigned prefill_rows = 128;
    unsigned slots = 0;
    // Served route lever (kMiniServedCliRoute): requested by the profile resolution, and admitted by the backend (HIP;
    // a CUDA server declines). Only the admitted route reads the three fields below it.
    bool served_route_requested = false;
    bool served_cli_route = false;
    // Tail capacity of the admitted route (the command line's tail rule).
    unsigned prefill_tail = 0;
    // Lane cap without --slots. Admitted route: an explicit --prefill-rows (0: kMiniServeLaneRows). Held-back route
    // (resolve): the incumbent cap, an explicit --prefill-rows else kMiniServeLaneRows, clamped (0: the chunk).
    unsigned lane_rows = 0;
    // Held-back route: rows the decoder load retries at when the implicit chunk's device workspaces are refused
    // (mini_served_prefill_fallback_rows, serve::load_with_chunk_fallback); 0 = no retry.
    unsigned prefill_fallback_rows = 0;
    // Decode form flags of a CUDA server (--packed-decode / --grouped-decode, mini_served_packed_decode); a HIP server
    // refuses them.
    bool packed_decode_flag = false;
    bool grouped_decode_flag = false;
    std::uint64_t kv_bytes = 0;
    MiniWeightOptions weight_options;
    profile::Name tier = profile::Name::standard;
    // Resolved operand class of the FAST W4A4 expert lever (mini_moe_w4a4_class; bf16 = off): the served prefill runs
    // the class the profile line reports, as the CLI and the gate do.
    moe_core::grouped_w4a4::Activations moe_w4a4 = moe_core::grouped_w4a4::Activations::bf16;
    // Resolved prepared-attention round fold (mini_prepared_round_fold), the CLI's setting from the same resolution.
    bool prepared_round_fold = false;
    // Resolved narrow expert image lever (mini_experts_nvfp4): the MXFP8 expert layers load as NVFP4.
    bool experts_nvfp4 = false;
    // Resolved exact down-image lever (mini_combine_image): the HIP resident prefill stores the BF16 down image where the
    // target is qualified.
    bool combine_image = false;
    // Resolved projected-bias attention lever (mini_attn_projected_bias), the value the CLI and the gate read.
    bool attn_projected_bias = false;
    // Resolved verify rows level (mini_verify_rows), the value the CLI and the gate read. A served lane decodes one row
    // per step (no verify window), so level 1 runs the same steps; level 2 also lets greedy cohorts whose groups declare
    // no score read pick through the lanes screened head (same tokens, distinct identity).
    unsigned verify_rows = 0U;
    bool controller_one_plane = false;
    // Resolved screened-head lever (mini_screened_head), the value the CLI and the gate read.
    bool screened_head = false;
    // Speculation (kMiniSpeculation, or an explicit --speculation off|auto): requested, and admitted (HIP: lanes draft
    // from their context through one-lane verify windows while one lane is live; a CUDA server declines).
    bool speculation_requested = false;
    bool speculation = false;
    MiniSpeculationPolicy speculation_policy{};  // the resolved rule (mini_speculation_line)
};

// The served speculation field of the startup and model lines: off, auto, or declined (requested, not admitted).
inline const char* mini_served_speculation_name(const MiniServeOptions& opt) noexcept {
    return opt.speculation ? "auto" : opt.speculation_requested ? "declined" : "off";
}

// Served KV slots of a resident decoder whose screened-head lever may have built a plane (screen_device_bytes; 0 on HIP,
// whose own binding keeps its incumbent budget, and without the lever): the plane stays only when it costs no KV slot
// (serve::resident_kv_slots_with_plane); otherwise the decoder releases it before the cohort commits its slots and its
// head reports "full kv-slot". Same tokens either way.
template <class Decoder>
std::uint32_t mini_serve_kv_slots(Decoder& decoder, std::uint64_t free, std::uint64_t total, std::uint64_t device_limit,
                                  std::uint64_t kv_limit, std::uint32_t rows) {
    const auto plane = decoder.screen_device_bytes();
    const auto plan = ::strix::serve::resident_kv_slots_with_plane(free, total, device_limit, kv_limit,
                                                                  decoder.slot_bytes(), rows, plane);
    if (plane != 0U && !plan.keep_plane) decoder.release_screen();
    return plan.slots;
}

// Lanes per batched forward: --slots, else, on the held-back served route, the cap it always had (an explicit
// --prefill-rows, else 128 rows, context-clamped: lane_rows from resolve, the chunk for options built without it);
// on the admitted served route lever an explicit --prefill-rows (clamped) or kMiniServeLaneRows.
inline unsigned mini_serve_lane_rows(const MiniServeOptions& opt) noexcept {
    if (opt.slots) return opt.slots;
    if (!opt.served_cli_route) return opt.lane_rows ? opt.lane_rows : opt.prefill_rows;
    return opt.lane_rows ? opt.lane_rows : kMiniServeLaneRows;
}

// The served_route field of the startup and inspection lines: off, declined (requested, the backend does not run it)
// or on, which states its qualification (the lever row is not qualified in any profile column) and its route.
// `effective_tail`: the tail capacity the constructed decoder admitted (startup line); without a decoder (inspection)
// the field states the offered capacity as prefill_tail_offered.
inline std::string mini_served_route_startup_field(const MiniServeOptions& opt,
                                                   std::optional<unsigned> effective_tail = std::nullopt) {
    if (!opt.served_cli_route) return mini_down_image_name(false, opt.served_route_requested);
    return "on qualification unqualified moe_decode " + std::string(mini_decode_form_name(kMiniPackedDecodeDefault)) +
           " prefill_chunk " + std::to_string(opt.prefill_rows) +
           (effective_tail ? " prefill_tail " + std::to_string(*effective_tail)
                           : " prefill_tail_offered " + std::to_string(opt.prefill_tail)) +
           " first_id prefill";
}

// The prepared_round_fold field of the served startup line: the requested bit from the options and what the decoder's backend
// admitted of it (MiniDecoderLevers::prepared_round_fold), so a fold the backend declines reads "declined", never "on".
inline const char* mini_round_fold_startup_field(const MiniServeOptions& opt, const MiniDecoderLevers& levers) {
    return mini_round_fold_name(levers.prepared_round_fold, opt.prepared_round_fold);
}

struct MiniServeCohort {
    std::unique_ptr<serve::CohortBackend> backend;
    weight::WeightLoadPlan weights;
    // With the projected-bias lever selected: the decoder's projected-bias submissions so far (the counter the command
    // line reports), read once when the server stops (mini_projected_bias_dispatches_line). Empty: lever off.
    std::function<std::uint64_t()> projected_bias_dispatches;
    // The prompt chunk the load admitted (serve::load_with_chunk_ladder; 0 = MiniServeOptions::prefill_rows): the
    // cohort's prompt slice.
    unsigned prefill_rows = 0;
    // Prompt pieces beside decoding lanes (shared mixed-service data): the CUDA held-back route's long pieces.
    serve::MixedServicePieces mixed_service = serve::kMixedServiceShortPieces;
};

// The served cohort's prompt service: alone, a prompt runs the admitted chunk per call (prompt_slice); beside a
// decoding lane, pieces of kMiniServeDecodingRows (the incumbent 128-row mixed-service bound) cut inside those chunks
// where the decoder's piece rule keeps the bits (CohortBackend::prompt_piece_rule: the HIP command line's route, the
// CUDA held-back route, whose pieces are the shared long pieces); a decoder without a rule runs whole chunks.
inline serve::CohortConfig mini_serve_scheduling(unsigned chunk, serve::PromptPieceRule rule,
                                                 serve::MixedServicePieces pieces = serve::kMixedServiceShortPieces) {
    serve::CohortConfig config;
    config.context_draft = false;
    config.max_draft = 0;
    config.prompt_slice = chunk;
    config.prompt_slice_decoding = serve::mixed_service_piece_rows(pieces, rule, chunk);
    return config;
}

MiniServeCohort make_mini_hip_cohort(const MiniServeOptions&);
MiniServeCohort make_mini_cuda_cohort(const MiniServeOptions&);

} // namespace strix::inkling2
