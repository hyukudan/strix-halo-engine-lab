#pragma once

#include "strix/engine_profile.hpp"
#include "strix/spec/block_draft_policy.hpp"
#include "strix/weight/levers.hpp"
#include "strix/weight/registry.hpp"

#include <array>
#include <cstdint>
#include <string_view>

namespace strix::inkling2 {

// The resident binding supplies its consumer formats and complete-model receipts. The lever is the shared
// generic policy weights.attention_head.q8_0@1; records under its former model-named id still resolve to it.
inline constexpr weight::WeightLoadPolicy kMiniWeightLoadPolicy = weight::with_legacy(
    weight::kAttentionHeadQ8Policy, {"inkling2-mini-attention-head-bf16-q8_0", 1});

// One resident image serves both regimes (prompt chunks and one-row steps share it, in the CLI and in a served
// cohort step), so its storage changes only when every regime has a gate v2 PASS row and one of them a measured gain.
inline constexpr std::array<std::string_view, 2> kMiniWeightLoadRegimes{weight::kPrefillRegime, weight::kDecodeRegime};

// Gate v2 rows of the shared policy (receipt ids name the evidence register entry; detailed receipts stay outside
// the repository). PREFILL, gfx1151, HIP device compiler clang 24.0.0git d6f6cb691863: PASS, 39 windows (prose,
// code, structured at 8K, 32K and 64K), 2,457 pooled rows; no measured gain in this regime (prompt projections are
// WMMA-bound: 8K 4,684 -> 4,673 ms, 32K 20,650 -> 20,732 ms), so the row is gate-only. Decode qualification
// is pending. FAST and TURBO use BF16 by default; --profile fast --qualify-q8 selects the qualification arm.
inline constexpr weight::WeightLoadDecision kMiniWeightLoadRows[] = {
    {kMiniWeightLoadPolicy.id,
     {weight::Backend::hip, "gfx1151", "hip-clang-24.0.0git-d6f6cb691863", weight::kPrefillRegime,
      weight::digest_from_hex("06b924b4f4116787df43a3668795417070ca22669e6314fd3bb8250eb29bd9c6"),
      weight::digest_from_hex("65e60f51aa1284d4fb022584908c315270334d991828bb774af60584523539d1")},
     weight::digest_from_hex("d64b562553020001310a9ad9831d4197fb17ae47ae8cc49fa3fec80fdc51518c"), {},
     {"docs/decisions/evidence-register.md#inkling-2-mini-q8_0-attention-and-head-on-gfx1151", "2026-11-23", 2457,
      65536}},
};
inline constexpr std::span<const weight::WeightLoadDecision> kMiniWeightLoadDecisions{kMiniWeightLoadRows};
// FAST weight mix (data): the head alone in q4_g64 from the calibrated pack, every other role BF16. It loads only
// through a FAST row, and there is none: on the held-out decode gate (39 windows, prose / code / structured, 8K-64K) the
// head row passed every class and the pooled bars but failed dNLL on two prose windows, so FAST keeps the BF16 image
// and the q4 head is part of the TURBO mix only.
inline constexpr weight::WeightLoadPolicy kMiniFastWeightPolicy = weight::kHeadQ4Policy;
// TURBO weight mix (data): the shared low-bit policy --profile turbo resolves when no explicit arm is named. It loads
// only through a TURBO row of kMiniWeightLoadDecisions (measured gain, quality against BF16 stated in the row);
// without one TURBO keeps the BF16 image.
inline constexpr weight::WeightLoadPolicy kMiniTurboWeightPolicy = weight::kAttentionHeadQ4Policy;
// The regime of these rows: decode steps (one row per lane, and the rows of a verify step), CLI and server alike. A
// load that takes a row keeps the BF16 image resident and adds the low-bit copies of the row's roles; prompt-prefill
// chunks, and steps wider than the row's band (evidence max_rows), read the BF16 image.
inline constexpr std::string_view kMiniWeightRegime = weight::kDecodeRegime;

// The calibrated pack the TURBO row was measured with (strix-lowbit-calib-pack, technique both, damp 0.1, q4 for the
// attention projections and the head; docs/engine/lowbit-calibration.md). The evidence belongs to this
// identity. Qualification accepts an explicit pack via STRIX_LOWBIT_PACK.
inline constexpr weight::Digest256 kMiniLowBitPackIdentity =
    weight::digest_from_hex("b0bd0103dcdd9ffd89e65ae06a3d86663d516c5eb09f263346555d2376bf658e");

// Served receipt (gfx1151, 16 slots, context 16384, 256 tokens per request): EXACT (= FAST) B=1 decode 23.36 tok/s,
// C8 53.76, C16 66.74; TURBO row, q4 attention and head, 27.03 / 68.48 / 86.60 tok/s. 8K prefill 14.15 s EXACT,
// 14.38 s TURBO (neutral to slower), so prompt chunks keep the BF16 image and the row is decode-only, banded to the
// step widths the receipt measured a gain at (kMiniLowBitStepBand rows). Its quality against BF16 is stated in the
// row: decode gate, 39 held-out windows, 2457 rows, scored on the production route (the BF16 image prefills the
// teacher prefix, the q4 copies run the decode steps). The q4 head alone fails the FAST bar on two prose windows.
inline constexpr weight::Digest256 kMiniLowBitBuild =
    weight::digest_from_hex("89665a6147dd8ca5eefcf4fbc4306f56a4c9e5e35bea774943d22ea08dd60f03");
inline constexpr weight::Digest256 kMiniCheckpoint =
    weight::digest_from_hex("65e60f51aa1284d4fb022584908c315270334d991828bb774af60584523539d1");
inline constexpr std::uint32_t kMiniLowBitStepBand = 16U;
inline constexpr weight::WeightLoadExtent kMiniLowBitMeasuredExtent{16384U, 16U};

// The served receipt measured these decode widths at this configured capacity. A receipt for one
// width does not establish a gain for the other widths, or for a verify or prefill call.
inline constexpr bool mini_lowbit_receipt_width(std::uint32_t rows) noexcept {
    return rows == 1U || rows == 8U || rows == kMiniLowBitStepBand;
}

inline constexpr bool mini_lowbit_decode_receipt(weight::WeightLoadExtent extent, std::uint32_t rows) noexcept {
    return extent.context == kMiniLowBitMeasuredExtent.context && extent.lanes == kMiniLowBitMeasuredExtent.lanes &&
           mini_lowbit_receipt_width(rows);
}

inline constexpr std::array<weight::WeightLoadDecision, 1> kMiniWeightLoadDecisionRows{{
    {kMiniTurboWeightPolicy.id,
     {weight::Backend::hip, "gfx1151", "hip-clang-24.0", kMiniWeightRegime, kMiniLowBitBuild, kMiniCheckpoint},
     {},
     weight::digest_from_hex("54182047338baf7005882cf2062d334aea4b8bb662d6cb484f1ccfce6926508b"),
     {"docs/engine/lowbit-calibration.md", "2026-11-24", 2457, 16384U, 16U, kMiniLowBitStepBand},
     "vs BF16 decode (bf16 prefill, q4 decode) kl=0.0033 top1=0.980 dnll=+0.0027 flips=0.00x windows_failed=0/39 "
     "pack=b0bd0103dcdd9ffd"},
}};
inline constexpr std::span<const weight::WeightLoadDecision> kMiniLowBitDecisions{kMiniWeightLoadDecisionRows};

// `extent`: what the instance holds, the same rule for the CLI and the server: context = the resident positions
// (serial CLI: prompt + generated tokens of its one sequence; server: its configured context), lanes = concurrent
// decode lanes (serial CLI: 1; server: its slot cap, 0 when the slot count is left to the memory budget). A banded
// row applies only inside its bands.
inline weight::WeightLoadPlan resolve_mini_weight_load(
    profile::Name tier, const weight::WeightLoadScope& scope,
    std::span<const weight::FormatId> consumer_formats, bool enabled, bool qualify,
    std::span<const weight::WeightLoadDecision> decisions = kMiniWeightLoadDecisions, weight::WeightLoadExtent extent = {}) {
    return weight::resolve_weight_load(weight::builtin_registry(), kMiniWeightLoadPolicy,
        {tier, scope, consumer_formats, enabled, qualify, kMiniWeightLoadRegimes, extent}, decisions);
}

// FAST lever: W4A4 experts. Multi-row prompt-prefill chunks run the NVFP4 expert layers through the shared grouped W4A4
// operand class (moe_core/grouped_w4a4.hpp, identity grouped-nvfp4-w4a4-f32-v1: activations quantised once per projection
// input by the family NVFP4 rule, sm_120 block-scaled FP4 MMA, weights used as stored). MXFP8 layers, decode, verify and
// one-row calls keep the BF16 tile. CUDA sm_120 only (HIP refuses a selected class). Runnable, not ready: opt in with
// STRIX_INK_MOE_W4A4=1 until the FAST gate (prefill regime) and a prefill receipt exist. Off switch: STRIX_INK_MOE_W4A4=0.
inline constexpr std::string_view kMiniMoeW4a4 = "inkling2-mini-moe-w4a4@1";
// Narrow expert image: the checkpoint's MXFP8 expert layers are transcoded to NVFP4 at load (shared
// weight/formats/microscaled_encode.hpp, mse@2 block scales, one outer scale per expert) and every expert consumer
// reads them as NVFP4 (4.5 instead of 8.25 bits per weight). The checkpoint's NVFP4 layers are untouched. Bit-changing,
// every regime (prefill, decode, verify, served), HIP and the CUDA resident route (the CUDA step program refuses it).
// Pending: gate v2 quality against BF16 and a served receipt decide FAST or TURBO. Off switch: STRIX_MOE_EXPERTS_NVFP4=0.
inline constexpr std::string_view kMiniExpertsNvfp4 = "moe.experts.mxfp8-nvfp4@1";

// Exact lever, opt-in: the grouped MoE down projection of the HIP resident prefill stores a BF16 image of its sums and the
// combine reads it (moe_core grouped_rows combine, ImageOperand), with a sticky NaN word sending NaN sums to the FP32 sum:
// the bits of the FP32 down rows for every sum. Off in every profile column (all cells empty, builtin 0): select it with
// STRIX_INK_COMBINE_IMAGE=1 (any profile). Runs only on the qualified HIP target (mini_combine_image_arch).
inline constexpr std::string_view kMiniCombineImage = "inkling2-mini-combine-down-image@1";
// Exact schedule trait of the HIP prepared attention (identity: the bits of the separate BF16 round passes): the stage's
// plain BF16 rounds fold into their producers, the K/V convolutions round their reads and stores and, at prefill
// heights, the output projection rounds each completed sum in its store with the separate pass's own function
// (core::RoundStore::separate). Opt-in: off in every profile column (all cells empty, builtin 0); select it with
// STRIX_INK_PREPARED_ROUND_FOLD=1 (any profile), through the one resolver the CLI, the gate and the server share.
inline constexpr std::string_view kMiniPreparedRoundFold = "inkling2-mini-prepared-round-fold@1";
// Exact lever (opt-in): the CUDA prompt flash projects the relative bias itself (cuda_core::PreparedBf16Projection)
// instead of reading the [row][head][distance] image the prepare stage produces. Same FP32 FMA chain in feature order,
// same BF16 round per distance, same failure contract as the image route for every launch it admits (single prompt
// chunk of two or more rows, host positions, no null-order count). A launch it cannot admit keeps the image route; real
// errors stay visible. CUDA only (HIP selects its projected route in its own binding). Off in every tier (builtin "0":
// the image route); an explicit 1 selects it and is a distinct identity until device identity on the landing head
// makes it a default. Switch: STRIX_INK_ATTN_PROJECTED_BIAS=0/1.
inline constexpr std::string_view kMiniAttnProjectedBias = "inkling2-mini-attn-projected-bias@1";
// Served route lever, opt-in: a HIP server runs the command line's route (packed decode unless a flag selects grouped,
// the whole prompt by the command line's chunks and tail rule, the first id from the final prefill row), so a served
// lane carries the bits of the same request on the command line at any cohort width. Off in every profile column (all
// cells empty, builtin 0): the served default keeps its incumbent route (the last prompt token held back and decoded,
// grouped cohort decode) until the migration is qualified against it. Select it with STRIX_INK_SERVED_CLI_ROUTE=1 (any
// profile); a CUDA server declines it. The command line always runs this route and does not read the switch.
inline constexpr std::string_view kMiniServedCliRoute = "inkling2-mini-served-cli-route@1";
// Draft CUDA controller schedule. One plane requires explicit selection and a resident weight certificate.
inline constexpr std::string_view kMiniControllerPlanes = "inkling2-mini-controller-planes@1";
inline constexpr std::array<profile::Lever, 8> kMiniLevers{{
    {kMiniMoeW4a4, "STRIX_INK_MOE_W4A4", "", {"", "1", "", ""}, "0", profile::Kind::bit_changing,
     {}, {false, true, false, true}, "1", "0",
     "waits for the FAST gate (prefill regime, routes replayed) and an RTX prefill receipt",
     false, {}, true},
    {kMiniCombineImage, "STRIX_INK_COMBINE_IMAGE", "", {"", "", "", ""}, "0", profile::Kind::exact,
     {true, true, true, true}, {true, true, true, true}, "", "0",
     "BF16 down image + combine: output bits of the FP32 down rows (device A/B through the grouped down producer, NaN "
     "operand-position fixture); gfx1151 only",
     false, {}, true},
    {.name = kMiniPreparedRoundFold, .env = "STRIX_INK_PREPARED_ROUND_FOLD", .option = "",
     .value = {"", "", "", ""}, .builtin = "0",
     .kind = profile::Kind::exact, .ready = {true, true, true, true}, .runnable = {true, true, true, true},
     .target = "", .off = "0",
     .source = "tests/core/round_fold_hip.hip (every store: the separate round's bits) and the isolated fold off/on "
               "identity ticket of the HIP prepared attention",
     .omit_builtin_fingerprint = true},
    {kMiniExpertsNvfp4, "STRIX_MOE_EXPERTS_NVFP4", "", {"", "", "", "1"}, "0", profile::Kind::bit_changing,
     {}, {false, true, false, true}, "1", "0",
     "waits for gate v2 (decode and prefill, quality against BF16) and a served B=1 / C8 / C16 receipt",
     false, {}, true},
    {kMiniControllerPlanes, "STRIX_INK_CONTROLLER_PLANES", "--controller-planes", {}, "3", profile::Kind::bit_changing,
     {}, {false, true, false, true}, "1", "3", "awaits CUDA device identity and model qualification",
     true, {}, true},
    {kMiniAttnProjectedBias, "STRIX_INK_ATTN_PROJECTED_BIAS", "", {}, "0", profile::Kind::exact,
     {}, {true, true, true, true}, "1", "0",
     "awaits device identity on the landing head (logits, ids and state bitwise the image route) before any default",
     false, {}, true},
    // Appended after main's rows (main's row positions unchanged); the block-candidate row stays last.
    {.name = kMiniServedCliRoute, .env = "STRIX_INK_SERVED_CLI_ROUTE", .option = "",
     .value = {"", "", "", ""}, .builtin = "0",
     .kind = profile::Kind::bit_changing, .ready = {false, false, false, false},
     .runnable = {true, true, true, true}, .target = "", .off = "0",
     .source = "waits for the served migration identity (served lanes against the incumbent served route); device "
               "evidence so far: served cohort lanes == the same served decoder's lone lane at cohort widths 2..17",
     .omit_builtin_fingerprint = true},
    spec::kBlockDraftLever,
}};

// The lever table must stay bound here: an unbound row is never resolved, so its switch is silently ignored.
inline constexpr profile::Engine kMiniProfile{
    "inkling2-mini", "", kMiniLevers, false, "inkling2-mini-turbo-v1",
};

}  // namespace strix::inkling2
