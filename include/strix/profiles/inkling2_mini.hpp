#pragma once

#include "strix/engine_profile.hpp"
#include "strix/spec/block_draft_policy.hpp"
#include "strix/weight/levers.hpp"
#include "strix/weight/registry.hpp"

#include <array>
#include <cstdint>
#include <stdexcept>
#include <string>
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
// The explicit TURBO lever can admit this mix through a matching low-bit decision row.
inline constexpr std::string_view kMiniLowBitTurbo = "inkling2-mini-lowbit-turbo@1";
inline constexpr std::string_view kMiniLowBitPackFile = "inkling2-mini-q4.lowbit-pack";
inline constexpr weight::WeightLoadPolicy kMiniTurboWeightPolicy = weight::kAttentionHeadQ4Policy;
// The row covers decode steps only. The low-bit copies are resident beside the BF16 image used for
// prefill, verify and decode widths outside the receipt.
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

// The receipt measured these active decode widths. Configured context and lanes must also be known
// and within the row's evidence caps.
inline constexpr bool mini_lowbit_receipt_width(std::uint32_t rows) noexcept {
    return rows == 1U || rows == 8U || rows == kMiniLowBitStepBand;
}

inline constexpr std::array<weight::WeightLoadDecision, 1> kMiniWeightLoadDecisionRows{{
    {kMiniTurboWeightPolicy.id,
     {weight::Backend::hip, "gfx1151", "hip-clang-24.0", kMiniWeightRegime, kMiniLowBitBuild, kMiniCheckpoint},
     {},
     weight::digest_from_hex("54182047338baf7005882cf2062d334aea4b8bb662d6cb484f1ccfce6926508b"),
     {.receipt = "docs/engine/lowbit-calibration.md", .date = "2026-11-24", .rows = 2457,
      .max_context = kMiniLowBitMeasuredExtent.context, .max_lanes = kMiniLowBitMeasuredExtent.lanes,
      .speedup = 27.03 / 23.36,  // smallest measured gain against BF16 across widths 1, 8 and 16
      .tier = profile::Name::turbo, .max_rows = kMiniLowBitStepBand},
     "vs BF16 decode (bf16 prefill, q4 decode) kl=0.0033 top1=0.980 dnll=+0.0027 flips=0.00x windows_failed=0/39 "
     "pack=b0bd0103dcdd9ffd"},
}};
inline constexpr std::span<const weight::WeightLoadDecision> kMiniLowBitDecisions{kMiniWeightLoadDecisionRows};

// `extent`: what the instance holds, the same rule for the CLI and the server: context = the resident positions
// (serial CLI: prompt + generated tokens of its one sequence; server: its configured context), lanes = concurrent
// decode lanes (serial CLI: 1; server: its slot cap, 0 when the slot count is left to the memory budget). A banded
// row applies only inside its bands. `first`: the lower endpoint of the resident image (0 = the `extent` value).
inline weight::WeightLoadPlan resolve_mini_weight_load(
    profile::Name tier, const weight::WeightLoadScope& scope,
    std::span<const weight::FormatId> consumer_formats, bool enabled, bool qualify,
    std::span<const weight::WeightLoadDecision> decisions = kMiniWeightLoadDecisions, weight::WeightLoadExtent extent = {},
    weight::WeightLoadExtent first = {}) {
    return weight::resolve_weight_load(weight::builtin_registry(), kMiniWeightLoadPolicy,
        {tier, scope, consumer_formats, enabled, qualify, kMiniWeightLoadRegimes, extent, first}, decisions);
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
// Exact lever (opt-in): CUDA greedy picks that return no score row run the shared EXACT screened head
// (cuda/core/head_screened.cuh over the BF16 unembed.weight, 8-bit plane built at load): the token and status of the full
// head by construction (rigorous bound, every possible argmax is rescored by the incumbent row body and picked by the
// incumbent rule). Steps that return or read scores (host logits, sampled lanes, verify rows, cohort programs) keep the
// full head; a plane that cannot be built, or (served) would cost a KV slot, keeps the full head. CUDA only: HIP keeps its
// own binding (screened by default, STRIX_SCREENED_HEAD=0 = full head). Off in every tier (builtin "0": the full head);
// an explicit 1 selects it and is a distinct identity until an RTX identity receipt makes it a default.
// Switch: STRIX_INK_SCREENED_HEAD=0/1.
inline constexpr std::string_view kMiniScreenedHead = "inkling2-mini-screened-head@1";
// Exact lever (opt-in): a HIP verify window of 2..kMiniMaxVerifyRows rows projects its score rows through the decoder's
// one pitched rows head (project_head_rows: the head read once, every row on its one-row body), attends its rows in one
// lanes launch set (each row at its own position with its one-row decode geometry), and records its convolution inputs
// (the shared session tape), so a partial commit rebuilds the histories by copies instead of replaying the kept rows.
// Off in every tier (builtin 0: per-row head launches, per-row attention launches, replayed partial commits); CUDA
// declines it. Select it with STRIX_INK_VERIFY_ROWS=1 (any profile), through the one resolver the CLI, the gate and the
// server share; a distinct identity until device identity (scores, picks and continuation after a partial commit
// bitwise the incumbent) makes it a default. Value 2 adds the lanes pick: greedy picks whose scores nobody reads (verify
// windows, served cohorts whose groups declare no score read) take the lanes form of the EXACT screened head (BF16).
inline constexpr std::string_view kMiniVerifyRows = "inkling2-mini-verify-rows@1";
// Exact lever, builtin auto: the default of every tier, and the legacy identity (omit_builtin_fingerprint: only an explicit
// off is fingerprinted; receipts in its source). Greedy speculation, exact by construction (a drafted row is committed
// only when it equals the target's own pick, so the ids are the ids of one-row decode). "auto" resolves the speculation
// mode of the command line and the server alike (an explicit --speculation wins): the command line's resident route
// loads the model's MTP block drafter beside the checkpoint (strix/spec/drafter_source.hpp; --dspark FILE|off wins) and
// buys windows through the shared block drafter; a HIP server drafts from each lane's context (the shared cohort
// drafter) through one-lane verify windows, while one lane is live (the model's max_draft_by_live). Either way the
// verify body is the verify rows lever at level >= 1. Speculation and a requested low-bit row never share a run
// (mini_speculation_policy). CUDA: the step program declines (no verifier), the server
// declines. Values: auto (builtin), off.
inline constexpr std::string_view kMiniSpeculation = "inkling2-mini-speculation@1";
// Image pre-processing recipe of the one vision adapter (src/inkling2/inkling2_vision.*), resolved here with the other
// rows by the resolver the CLI and the server share. "legacy" (builtin, image identity inkling2-vision-v1) or "axial@1"
// (the reference-checked axial recipe, image identity inkling2-vision-axial@1: checkerboard alpha, symmetric
// normalisation, cell-major patches, leading class row, axial RoPE; it can change the prompt length). Bit-changing and
// not qualified yet: no column selects it, an explicit STRIX_INKLING2_VISION_RECIPE=axial@1 runs under --profile fast
// or turbo stamped "unqualified", and every other profile refuses it until a ready column carries the value (the row's
// own identity evidence, Resolution::explicit_value_qualified). Only the image opener reads it (mini_vision_axial on
// mini_vision_media_tier); a text-only run never applies it (mini_defer_vision_recipe), and at its builtin the row
// stays out of the fingerprint and the receipt, so a text-only run resolves as before the row.
inline constexpr std::string_view kMiniVisionRecipe = "inkling2-vision-recipe";
inline constexpr std::array<profile::Lever, 13> kMiniLevers{{
    {kMiniMoeW4a4, "STRIX_INK_MOE_W4A4", "", {"", "1", "", ""}, "0", profile::Kind::bit_changing,
     {}, {false, true, false, true}, "1", "0",
     "waits for the FAST gate (prefill regime, routes replayed) and an RTX prefill receipt",
     false, {}, true, {}, {{{"0", "v1"}, {"1", "v1"}}}},
    {kMiniCombineImage, "STRIX_INK_COMBINE_IMAGE", "", {"", "", "", ""}, "0", profile::Kind::exact,
     {true, true, true, true}, {true, true, true, true}, "", "0",
     "BF16 down image + combine: output bits of the FP32 down rows (device A/B through the grouped down producer, NaN "
     "operand-position fixture); gfx1151 only",
     false, {}, true, {}, {{{"0", "v1"}, {"1", "v1"}}}},
    {.name = kMiniPreparedRoundFold, .env = "STRIX_INK_PREPARED_ROUND_FOLD", .option = "",
     .value = {"", "", "", ""}, .builtin = "0",
     .kind = profile::Kind::exact, .ready = {true, true, true, true}, .runnable = {true, true, true, true},
     .target = "", .off = "0",
     .source = "tests/core/round_fold_hip.hip (every store: the separate round's bits) and the isolated fold off/on "
               "identity ticket of the HIP prepared attention",
     .omit_builtin_fingerprint = true, .identity = {{{"0", "v1"}, {"1", "v1"}}}},
    {kMiniExpertsNvfp4, "STRIX_MOE_EXPERTS_NVFP4", "", {"", "", "", "1"}, "0", profile::Kind::bit_changing,
     {}, {false, true, false, true}, "1", "0",
     "waits for gate v2 (decode and prefill, quality against BF16) and a served B=1 / C8 / C16 receipt",
     false, {}, true, {}, {{{"0", "v1"}, {"1", "v1"}}}},
    {kMiniControllerPlanes, "STRIX_INK_CONTROLLER_PLANES", "--controller-planes", {}, "3", profile::Kind::bit_changing,
     {}, {false, true, false, true}, "1", "3", "awaits CUDA device identity and model qualification",
     true, {}, true, {}, {{{"3", "v1"}, {"1", "v1"}}}},
    {kMiniAttnProjectedBias, "STRIX_INK_ATTN_PROJECTED_BIAS", "", {}, "0", profile::Kind::exact,
     {}, {true, true, true, true}, "1", "0",
     "awaits device identity on the landing head (logits, ids and state bitwise the image route) before any default",
     false, {}, true, {}, {{{"0", "v1"}, {"1", "v1"}}}},
    {.name = kMiniServedCliRoute, .env = "STRIX_INK_SERVED_CLI_ROUTE", .option = "",
     .value = {"", "", "", ""}, .builtin = "0",
     .kind = profile::Kind::bit_changing, .ready = {false, false, false, false},
     .runnable = {true, true, true, true}, .target = "", .off = "0",
     .source = "waits for the served migration identity (served lanes against the incumbent served route); device "
               "evidence so far: served cohort lanes == the same served decoder's lone lane at cohort widths 2..17",
     .omit_builtin_fingerprint = true, .identity = {{{"0", "v1"}, {"1", "v1"}}}},
    {.name = kMiniScreenedHead, .env = "STRIX_INK_SCREENED_HEAD", .option = "",
     .value = {"", "", "", ""}, .builtin = "0",
     .kind = profile::Kind::exact, .ready = {false, false, false, false},
     .runnable = {true, true, true, true}, .target = "1", .off = "0",
     .source = "awaits the RTX identity receipt on the landing head (greedy ids against the full head, CLI and served) "
               "before any default",
     .omit_builtin_fingerprint = true, .identity = {{{"0", "v1"}, {"1", "v1"}}}},
    {.name = kMiniLowBitTurbo, .env = "STRIX_INK_LOWBIT_TURBO", .option = "",
     .value = {"", "", "", ""}, .builtin = "0", .kind = profile::Kind::bit_changing,
     .ready = {}, .runnable = {false, false, false, true}, .target = "1", .off = "0",
     .source = "docs/engine/lowbit-calibration.md (calibrated HIP decode at widths 1, 8 and 16)",
     .omit_builtin_fingerprint = true, .identity = {{{"0", "v1"}, {"1", "v1"}}}},
    {.name = kMiniVerifyRows, .env = "STRIX_INK_VERIFY_ROWS", .option = "",
     .value = {"", "", "", ""}, .builtin = "0",
     .kind = profile::Kind::exact, .ready = {}, .runnable = {true, true, true, true}, .target = "1", .off = "0",
     .source = "awaits device identity on the landing head (verify score rows, picks and the continuation after a "
               "partial commit bitwise the per-row incumbent, BF16 and Q8_0 heads); HIP only",
     .omit_builtin_fingerprint = true, .identity = {{{"0", "v1"}, {"1", "v1"}}}},
    {.name = kMiniSpeculation, .env = "STRIX_INK_SPECULATION", .option = "",
     .value = {"", "", "", ""}, .builtin = "auto",
     .kind = profile::Kind::exact, .ready = {}, .runnable = {true, true, true, true},
     .target = "", .off = "off",
     .source = "ink_draft t1c/t2 (cluster1 gfx1151, BF16, greedy): ids with drafting == without, CLI 3/3 and served "
               "46/46; CLI 160 tokens code / prose / JSON 28.40 / 28.31 / 29.06 -> 29.46 / 32.77 / 42.64 tok/s, served "
               "B=1 23.43 / 23.28 / 23.69 -> 28.58 / 31.88 / 28.67; windows only while one lane is live "
               "(max_draft_by_live), the previous one-row route at two or more; CUDA declines (no verifier)",
     .omit_builtin_fingerprint = true, .identity = {{{"off", "v1"}, {"auto", "v2"}}}},
    {kMiniVisionRecipe, "STRIX_INKLING2_VISION_RECIPE", "", {}, "legacy", profile::Kind::bit_changing,
     {}, {false, true, false, true}, "axial@1", "legacy",
     "awaits the real-checkpoint reference gate (patches, hidden and projected rows) and CLI image identity on HIP "
     "and CUDA",
     false, {}, true, {}, {{{"legacy", "v1"}, {"axial@1", "v1"}}}},
    spec::kBlockDraftLever,
}};

// The lever table must stay bound here: an unbound row is never resolved, so its switch is silently ignored.
inline constexpr profile::Engine kMiniProfile{
    "inkling2-mini", "", kMiniLevers, false, "inkling2-mini-turbo-v1",
};

// The image recipe applies only to a request with image input. The text resolution (resolve_mini_profile, shared by
// the CLI and the server) holds an explicit setting of the row as `requested`, effective builtin, status off, so a
// text-only run starts, summarises, qualifies and fingerprints as with the row unset (nothing is validated there).
// The image opener re-admits it with mini_vision_media_tier when a request carries images.
inline constexpr std::string_view kMiniVisionRecipeDeferred = "image input only: applied when a request carries images";
inline void mini_defer_vision_recipe(profile::Resolution& tier) {
    for (auto& entry : tier.entries) {
        if (entry.lever == nullptr || entry.lever->name != kMiniVisionRecipe) continue;
        const bool explicit_setting =
            entry.status == profile::Status::explicit_env || entry.status == profile::Status::explicit_cli;
        if (!explicit_setting && entry.value == entry.lever->builtin) continue;
        entry.requested = entry.value;
        entry.value = std::string(entry.lever->builtin);
        entry.status = profile::Status::off;
        entry.reason = std::string(kMiniVisionRecipeDeferred);
    }
}
// The resolution an image request runs: `tier` with a deferred image recipe re-admitted as the explicit setting it was
// (the row's one switch is its environment variable); mini_vision_axial then admits or refuses it.
inline profile::Resolution mini_vision_media_tier(profile::Resolution tier) {
    for (auto& entry : tier.entries) {
        if (entry.lever == nullptr || entry.lever->name != kMiniVisionRecipe) continue;
        if (entry.status != profile::Status::off || entry.reason != kMiniVisionRecipeDeferred) continue;
        entry.value = entry.requested;
        entry.status = profile::Status::explicit_env;
        entry.reason.clear();
    }
    return tier;
}

// The image recipe of a resolved run: true selects axial@1. A value the row does not qualify is admitted only under
// --profile fast or turbo; anything but legacy / axial@1 is refused. The one reader of the row (the image opener, on
// mini_vision_media_tier).
inline bool mini_vision_axial(const profile::Resolution& tier) {
    const auto* entry = tier.find(kMiniVisionRecipe);
    if (entry == nullptr || entry->value == "legacy") return false;
    if (entry->value != "axial@1")
        throw std::invalid_argument("STRIX_INKLING2_VISION_RECIPE must be legacy or axial@1");
    if (!profile::Resolution::explicit_value_qualified(*entry) && tier.profile != profile::Name::fast &&
        tier.profile != profile::Name::turbo)
        throw std::invalid_argument("image recipe axial@1 is an unqualified bit-changing image identity: not admitted "
                                    "under --profile " + std::string(tier.word()) + " (admitted: fast, turbo)");
    return true;
}

}  // namespace strix::inkling2
