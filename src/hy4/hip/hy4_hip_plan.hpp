#pragma once
// Hy4-flash HIP binding: the stage map and the cache plan (host-only; no device code).
//
// Stage map: the per-layer order of the forward pass (the order of the host oracle, src/hy4/ref) where
// EVERY stage names the shared entry it calls; the adapter adds no kernel.
// Cache plan: the attention caches of one layer as shared cache-row layouts (strix/attn/cache_rows.hpp):
//   sliding layers: latent rows [512 | 64 rope] in the power-of-two ring the sparse flash Window reader masks;
//   indexed layers: latent rows by position (List selection), token keys [128] in the indexer's token ring
//   (group_indexer::token_ring_slots), pooled group keys [128] by group (one row per closed group of 4).
#include "hy4/ref/hy4_host_ref.hpp"
#include "strix/attn/cache_rows.hpp"
#include "strix/attn/group_indexer.hpp"
#include "strix/hy4/presets.hpp"
#include "strix/pair/layer_pipeline.hpp"
#include "strix/pair/stage_map.hpp"
#include "strix/session/prompt_windows.hpp"

#include <cstdint>
#include <span>
#include <stdexcept>
#include <string_view>
#include <vector>

namespace strix::hy4::hipbind {

namespace cr = strix::attn::cache_rows;

struct LayerCaches {
    cr::Rows latent{};          // width kv_lora + qk_rope
    std::uint64_t latent_rows = 0;
    cr::Rows token_keys{};      // indexed layers only (width 0 otherwise)
    std::uint64_t token_rows = 0;
    cr::Rows group_keys{};      // indexed layers only
    std::uint64_t group_rows = 0;
    [[nodiscard]] std::uint64_t bytes() const {
        return 2ULL * (latent_rows * latent.width + token_rows * token_keys.width + group_rows * group_keys.width);
    }
};

// context: positions the session may hold; max_rows: rows written by one launch (decode 1, verify / prefill chunk).
[[nodiscard]] inline LayerCaches layer_caches(const Config& c, std::uint32_t layer, std::uint32_t context, std::uint32_t max_rows) {
    LayerCaches k;
    const unsigned width = c.kv_lora + c.qk_rope;
    if (c.indexed.at(layer) == 0U) {
        const unsigned ring = cr::window_ring(c.window, max_rows);
        k.latent = {width, ring};
        k.latent_rows = ring;
        return k;
    }
    namespace gi = strix::attn::group_indexer;
    k.latent = {width, 0U};
    k.latent_rows = context;
    const unsigned tring = gi::token_ring_slots(preset::indexer, max_rows);
    k.token_keys = {c.index_dim, tring};
    k.token_rows = tring;
    k.group_keys = {c.index_dim, 0U};
    k.group_rows = context / c.index_ratio;
    return k;
}

[[nodiscard]] inline std::uint64_t cache_bytes(const Config& c, std::uint32_t context, std::uint32_t max_rows) {
    std::uint64_t total = 0;
    for (std::uint32_t l = 0; l < c.layers; ++l) total += layer_caches(c, l, context, max_rows).bytes();
    return total;
}

// ---- indexer binding (indexed layers): constants the device views use, testable on the host -------------------------
namespace gi = strix::attn::group_indexer;
// Per-row stride of the selected token lists (the longest list: 511 groups x 4 + the own group's 4 slots).
inline constexpr unsigned kIndexListStride = gi::max_tokens(preset::indexer);
// w_h scale heads^-0.5 * head_dim^-0.5 as the oracle's fp32 (1 / declared_sqrt(16)) * (1 / declared_sqrt(128)); the
// device scan view multiplies by this constant (hy4_hip_plan_tests checks it against both host formulas).
inline constexpr std::uint32_t kIndexHeadScaleBits = 0x3CB504F3U;
// Tokens the flash attends for the query at slot t of an indexed layer: the causal prefix while every visible group is
// selected (t < 2048), else 511 picked groups x 4 + the own group's slots up to t (group_select_kernel's list length).
[[nodiscard]] constexpr std::uint32_t index_list_tokens(std::uint32_t t) {
    if (gi::dense(preset::indexer, t)) return t + 1U;
    return gi::pick_count(preset::indexer) * preset::indexer.ratio + (t - gi::tail_begin(preset::indexer, t) + 1U);
}

// ---- batched prompt (windows of rows) -----------------------------------------------------------------------------
// The model's window rule on the shared planner (strix/session/prompt_windows.hpp): a window feeds the grouped expert
// prefill (rows % 8 == 0, rows >= 16) and starts 8-aligned, so every group of 4 it closes has its 4 token keys inside
// the window (the token ring holds 3 + rows slots).
inline constexpr strix::session::prompt_windows::Rule kPromptRule{8U, 16U};
using PromptWindow = strix::session::prompt_windows::Window;
using PromptPlan = strix::session::prompt_windows::Plan;
[[nodiscard]] constexpr bool prompt_window_ok(std::uint32_t rows) { return strix::session::prompt_windows::window_ok(kPromptRule, rows); }
[[nodiscard]] inline PromptPlan prompt_plan(std::uint32_t pos, std::uint32_t count, std::uint32_t rmax) {
    return strix::session::prompt_windows::plan(kPromptRule, pos, count, rmax);
}
// One window of the shared layer pipeline (strix/pair/layer_pipeline.hpp layers::windows(prompt, rmax)): the pair's bits
// equal the one-node prefill at the same rmax (shared stage-plan invariant).
[[nodiscard]] inline PromptPlan stage_window_plan(std::uint32_t start, std::uint32_t rows, std::uint32_t rmax) {
    return strix::session::prompt_windows::stage_plan(kPromptRule, start, rows, rmax);
}
// The default cut of a two-stage layer split: the middle layer. On the pack each half holds 6 indexed + 15 sliding
// layers and 2 Engram layers (indexed 0,1,5,..,41; Engram 1,11,21,31); the dense layer 0 sits with the embedding, the
// head with the last stage.
[[nodiscard]] constexpr std::uint32_t pair_cut(std::uint32_t layers) { return layers / 2U; }
// Window width of the pair from a calibration (strix-hy4-rank --window-costs ROWS:MS,...): the shared picker.
using PairWindowCost = strix::pair::layers::WindowCost;
[[nodiscard]] inline std::uint32_t pair_window_rows(std::size_t prompt, std::span<const PairWindowCost> costs) {
    return strix::pair::layers::two_stage_window_rows(prompt, costs);
}
// Measured cost of the Hy4 pair layer pipeline (two Strix Halo nodes over USB4, FAST A8 rows prefill, real text, 11-20
// receipts): a window of u rows at prompt length n costs 267.9 + (1.3666 + 2.344e-6 n) u ms over both ranks (per-window
// costs 4K/1024 rows 1,677, 8K/2048 3,106, 32K/2048 3,224 ms); one stage-boundary frame 32 + 140 / 2048 ms per row
// (100.7 MB of FP32 seam rows per 2048 rows: 172 ms) on ONE link for both directions (the host route copies both
// directions through one path). Fitted to sixteen receipts (v = 1-5, rows 1024-4096, 2K-32K; rms 3.4 %): the frames
// stall the pipeline once the two ranks' frames outlast a stage (8K v = 5: 2 x 172 > 311 ms, seal waits 0.3-0.6 s;
// 32K rows 4096 v = 5), never at 4K rows 1024 v = 4 (2 x 102 < 210 ms). The former fit (196 + 26 / 2048 ms per
// direction) priced every frame by its latency alone and picked v = 5 at 8K, 3.7 % slower than v = 3. The link state
// varies between sessions: on a faster link (11-21, no stall) v = 5 was 1.1 % faster than v = 4 at 8K and tied at 32K.
// These constants keep the slow state, so the pick (v = 4) does not lose to v = 3 in either (measured +0.6 % at 8K,
// +1.2 % at 32K on the fast state; v = 4 keeps 2 x 172 < 388 ms of link slack on the slow one).
// Data for the shared planner (strix/pair/stage_map.hpp plan_stages). Widths: the measured ones (bitwise equal).
inline constexpr strix::pair::LayerPipeCost kPairPipeCost{267.9, 1.3666, 2.344e-6, 32.0, 140.0 / 2048.0, true};
inline constexpr std::uint32_t kPairWindowRows[] = {1024U, 2048U, 4096U};
inline constexpr std::uint32_t kPairVirtualStages[] = {1U, 2U, 3U, 4U, 5U};
// Rows per window and stage map of the pair for a prompt; `width` != 0 keeps that width (the planner picks v only).
[[nodiscard]] inline strix::pair::StagePlan pair_stage_plan(std::size_t prompt, std::uint32_t layers, std::uint32_t width = 0U) {
    const std::uint32_t one[] = {width};
    const std::span<const std::uint32_t> rows = width != 0U ? std::span<const std::uint32_t>(one) : std::span<const std::uint32_t>(kPairWindowRows);
    return strix::pair::plan_stages(prompt, layers, kPairPipeCost, rows, [](std::uint32_t r) { return prompt_window_ok(r); },
                                    kPairVirtualStages);
}
// Rows of the window [p0, p0 + rows) of an indexed layer that attend the causal prefix (slots < 2048); the rest select.
[[nodiscard]] constexpr std::uint32_t index_dense_rows(std::uint32_t p0, std::uint32_t rows) {
    std::uint32_t n = 0;
    while (n < rows && gi::dense(preset::indexer, p0 + n)) ++n;
    return n;
}
// Flash launches of one layer for a verify window [p0, p0 + rows): the first `batched` rows in ONE launch (a window
// layer: every row; an indexed layer: its causal-prefix rows), then the rest (indexed rows that select) one launch per
// row after their own index_select (the selection list is one shared buffer). Row r of the batched launch attends the
// keys its own launch would: a window row min(p0 + r + 1, window) by its position, an indexed row iota[p0 + 1 + r] =
// p0 + r + 1 = index_list_tokens(p0 + r).
struct VrowsFlashSplit {
    std::uint32_t batched = 0, per_row = 0;
};
[[nodiscard]] constexpr VrowsFlashSplit vrows_flash_split(bool indexed, std::uint32_t p0, std::uint32_t rows) {
    const std::uint32_t b = indexed ? index_dense_rows(p0, rows) : rows;
    return {b, rows - b};
}
// attn_k_b (Q8_0, input 192) re-laid at load as 256-wide rows for the A8 rows GEMM (input % 128): two zero-scale Q8_0
// blocks per row, so the padded columns (the q rope tail of each head row) add exact zero terms.
inline constexpr std::uint32_t kKbPadInput = 256U;

enum class Stage : std::uint8_t {
    hc_mix, hc_collapse, norm, project, rope, cache_row, absorb, flash, latent_out, sigmoid_gate, index_score,
    index_pool, index_select, hc_expand, engram_rows, engram_inject, router, route, experts_gate_up, experts_down,
    merge, swiglu, embed, hc_head, head, argmax
};

struct Step {
    Stage stage;
    std::string_view what;   // model tensor / role (data)
    std::string_view entry;  // the shared entry the adapter calls
};

// The steps of trunk layer `layer`, in execution order.
[[nodiscard]] inline std::vector<Step> layer_steps(const Config& c, std::uint32_t layer) {
    constexpr std::string_view q8 = "hip_q8_rows_sites.hpp LinearF32 / HeadsF32", norm = "ops/norm/norm_presets.hpp DeclaredF32Traits",
                               hc = "core/hc_core.hpp enqueue_mixes / enqueue_collapse / enqueue_expand",
                               rope = "ops/rope enqueue_rope (adjacent pairs)", rows = "attn_core/cache_rows.hpp store_rows",
                               dense = "hip_lm_head_launcher launch_hip_dense_rows";
    const bool idx = c.indexed.at(layer) != 0U, moe = c.dense.at(layer) == 0U;
    bool engram = false;
    for (const auto e : c.engram_layers) engram = engram || e == layer;
    std::vector<Step> s;
    if (engram) {
        s.push_back({Stage::engram_rows, "engram rows (hash, gather, E4M3 decode, head norm)", "engram/hip row_kernel head_rms_bf16"});
        s.push_back({Stage::project, "engram_kv / engram value+keys", q8});
        s.push_back({Stage::engram_inject, "engram inject", "engram/hip/inject_kernel.hpp normalized_bf16"});
    }
    s.push_back({Stage::hc_mix, "attn hc mix", hc});
    s.push_back({Stage::hc_collapse, "attn hc collapse", hc});
    s.push_back({Stage::norm, "attn_norm (form w * bf16(x * r): a norm Traits)", norm});
    s.push_back({Stage::project, "attn_gate, attn_q_a, attn_kv_a_mqa", q8});
    s.push_back({Stage::norm, "attn_q_a_norm, attn_kv_a_norm (latent 512)", norm});
    s.push_back({Stage::project, "attn_q_b", q8});
    s.push_back({Stage::rope, "q rope tail (64 of 256 per head), k rope tail (64)", rope});
    s.push_back({Stage::cache_row, "latent row [512 | 64]", rows});
    if (idx) {
        s.push_back({Stage::project, "indexer.attn_k, indexer.attn_q_b", q8});
        s.push_back({Stage::project, "indexer.proj (F32 head weights)", dense});
        s.push_back({Stage::norm, "indexer.k_norm", norm});
        s.push_back({Stage::cache_row, "token key [128] (token ring)", rows});
        s.push_back({Stage::index_pool, "closed group: gated pool + norm + rope at group_rope_slot", "attn_core/group_pool.hpp pool_groups_kernel<View>"});
        s.push_back({Stage::rope, "index query rope (trailing 64 of 128)", rope});
        s.push_back({Stage::index_score, "group scores", "attn_core/group_score_scan.hpp"});
        s.push_back({Stage::index_select, "top 511 + own group (dense when kv_len <= 2048)", "attn_core/group_select_kernel.hpp"});
    }
    s.push_back({Stage::absorb, "attn_k_b per head (q_nope -> latent)", q8});
    s.push_back({Stage::flash, idx ? "sparse MLA List + sinks" : "sparse MLA Window 512 + sinks", "attn_core/sparse_flash_mla.hpp LatentPolicy"});
    s.push_back({Stage::latent_out, "attn_v_b per head (latent -> v)", q8});
    s.push_back({Stage::sigmoid_gate, "o *= sigmoid(gate)", "ops/glu sigmoid gate"});
    s.push_back({Stage::project, "attn_output", q8});
    s.push_back({Stage::norm, "post_attention_norm", norm});
    s.push_back({Stage::hc_expand, "attn hc expand", hc});
    s.push_back({Stage::hc_mix, "ffn hc mix", hc});
    s.push_back({Stage::hc_collapse, "ffn hc collapse", hc});
    s.push_back({Stage::norm, "ffn_norm", norm});
    if (!moe) {
        s.push_back({Stage::project, "dense ffn_gate, ffn_up", q8});
        s.push_back({Stage::swiglu, "silu(g) * u (clamp, bf16 round)", "ops/glu enqueue_swiglu_f32"});
        s.push_back({Stage::project, "dense ffn_down (K 13824: non-A8)", q8});
    } else {
        s.push_back({Stage::project, "latent down 3072 -> 1536, shared gate/up", q8});
        s.push_back({Stage::router, "router logits 512 (F32)", dense});
        s.push_back({Stage::route, "sigmoid + bias top-16 x routed scale", "moe_core/route_wave_kernel.hpp strix::hy4::preset::route"});
        s.push_back({Stage::experts_gate_up, "routed gate/up Q4_K A8", "hip_moe_q4k_a8_probe.hpp enqueue_probe"});
        s.push_back({Stage::experts_down, "routed down Q4_K/Q6_K A8 + clamp", "hip_moe_a8_down.hpp enqueue_down_*"});
        s.push_back({Stage::swiglu, "shared expert silu(g) * u", "ops/glu enqueue_swiglu_f32"});
        s.push_back({Stage::project, "shared down", q8});
        s.push_back({Stage::merge, "ordered weighted merge of the 16 rawdots + shared add", "moe_core/route_wave_kernel.hpp combine"});
        s.push_back({Stage::project, "latent up 1536 -> 3072", q8});
    }
    s.push_back({Stage::norm, "post_ffw_norm", norm});
    s.push_back({Stage::hc_expand, "ffn hc expand", hc});
    return s;
}

// Before layer 0 and after layer 41.
[[nodiscard]] inline std::vector<Step> edge_steps(bool after) {
    if (!after) return {{Stage::embed, "token_embd Q8_0 + 4-stream expansion", "hip_embedding_launcher.hpp launch_hip_embedding_rows"}};
    return {{Stage::hc_head, "hc head collapse", "core/hc_core.hpp (head site)"},
            {Stage::norm, "output_norm", "ops/norm/norm.hpp rmsnorm_kernel<Traits>"},
            {Stage::head, "output BF16 120832 x 3072", "hip_lm_head_launcher launch_hip_dense_rows"},
            {Stage::argmax, "greedy", "hip_generation_output.hpp enqueue_hip_greedy_rows"}};
}

}  // namespace strix::hy4::hipbind
