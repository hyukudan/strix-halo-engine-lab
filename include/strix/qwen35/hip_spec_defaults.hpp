#pragma once
// Qwen dense (qwen35) HIP speculative decode: ONE resolver for every entry point that drives the HipEngine
// (strix-hip-qwen35-decode, strix-hip-qwen35-spec; no server binds the qwen35 HipEngine).
//   MTP source  "auto" (default) = the model's own next-token-prediction layer when the GGUF carries it (the IU4 pack keeps
//               blk.<layers>.* as Q8_0: 0.45 GB per draft position instead of 0.86 GB for the bf16 side file), else none;
//               a path = that side GGUF; "off" / "none" = plain greedy decode.
//   draft rows  draft-only subset head (strix/spec/draft_rows.hpp, group 16 = one IU4 tile column), on whenever MTP
//               drafting is on; off = the full 248K-row head per draft position.
//   draft packs --draft-pack F1,F2: id files (tools/build_draft_vocab.py, e.g. --script cjk) that join the subset when
//               the prompt holds >= pack_min_hits of their ids outside the core (the CUDA dense app's flag and rule).
//               Without one, a non-Latin request drafts only the core + prompt + emitted ids of its script.
//   costs       --spec-costs calibrated (default: this device's walls, hip_dense_measure_costs) | table (compiled
//               fallback) | learned (previous form). --cost-kv online (default: the shared context cost fit from this
//               device's verify walls, spec::ArbiterConfig::cost_fit) | off | KV (fixed). Flags parse through parse_hip_dense_spec_arg (hip_dense_spec_costs.hpp):
//               both apps accept the same flags with the same meaning and end with the same plan, block config and costs.
// Speculation is exact: verify windows are bitwise equal to row-by-row decode (hip_engine.hpp), and drafts (MTP layer
// precision, head subset) only steer how many tokens a window accepts, never which tokens are emitted.
// Home: shared host resolver of the qwen35 HIP family (both apps call it); the mechanisms are shared (strix/spec).
#include <algorithm>
#include <chrono>
#include <cstddef>
#include <cstdint>
#include <iterator>
#include <string>
#include <vector>

#include "strix/qwen35/hip_dense_spec_costs.hpp"
#include "strix/qwen35/hip_engine.hpp"
#include "strix/qwen35/iu4_draft_head.hpp"
#include "strix/spec/block_drafter.hpp"
#include "strix/spec/draft_rows.hpp"
#include "strix/spec/mtp_chain.hpp"

namespace strix::qwen35 {

struct HipDenseSpecPlan {
    std::string mtp_source;  // GGUF with the MTP layer; empty = no MTP drafting
    std::string mtp_reason;  // "model" | "file" | "off" | "absent"
    bool draft_rows = false;
    std::vector<std::string> draft_packs;  // pack id files (comma list of --draft-pack)
    HipDenseCostMode costs = HipDenseCostMode::calibrated;
    std::string cost_kv = "auto";  // context cost of the calibrated / table model (hip_dense_cost_kv)
};

// `mtp`: "auto" / "" / "off" / "none" / path; `draft_rows`: false = full head (off switch). `has_layer` answers whether
// the model GGUF carries the MTP layer (HipEngine::has_mtp_layer in the apps; a stand-in in host tests).
// `plain` = the run needs every position's logits (--dump-logits): no speculation, whatever the flags say.
template <class HasLayer>
HipDenseSpecPlan resolve_hip_dense_spec(const std::string& model, const HipDenseSpecArgs& args, bool plain, HasLayer&& has_layer) {
    HipDenseSpecPlan p;
    p.costs = args.costs;
    p.cost_kv = args.cost_kv;
    const std::string mtp = plain ? std::string("off") : args.mtp;
    const bool draft_rows = args.draft_rows;
    const std::string& draft_packs = args.draft_packs;
    if (mtp == "off" || mtp == "none") {
        p.mtp_reason = "off";
    } else if (mtp.empty() || mtp == "auto") {
        if (has_layer(model)) {
            p.mtp_source = model;
            p.mtp_reason = "model";
        } else {
            p.mtp_reason = "absent";
        }
    } else {
        p.mtp_source = mtp;
        p.mtp_reason = "file";
    }
    p.draft_rows = draft_rows && !p.mtp_source.empty();
    for (std::size_t b = 0, e = 0; p.draft_rows && b < draft_packs.size(); b = e + 1) {
        e = draft_packs.find(',', b);
        if (e == std::string::npos) e = draft_packs.size();
        if (e > b) p.draft_packs.push_back(draft_packs.substr(b, e - b));
    }
    return p;
}
inline HipDenseSpecPlan resolve_hip_dense_spec(const std::string& model, const HipDenseSpecArgs& args, bool plain = false) {
    return resolve_hip_dense_spec(model, args, plain, [](const std::string& m) { return HipEngine::has_mtp_layer(m); });
}

// Subset rows of the draft head: the shared DraftRows defaults (core + fallback prefix, prompt and emitted ids) at the IU4
// tile-column granularity.
inline spec::DraftRowsConfig hip_dense_draft_rows_config(std::uint32_t vocab, const HipDenseSpecPlan& plan) {
    spec::DraftRowsConfig c;
    c.vocab = vocab;
    c.group = kIu4DraftGroup;
    for (const auto& f : plan.draft_packs) c.packs.push_back(spec::DraftRows::read_ids(f, vocab));
    return c;
}

// Drafting block of the qwen35 HIP family: windows up to 16 rows (the engine's max), lookup + MTP chain + hybrid, priced
// by the node work model (hip_dense_spec_costs.hpp: fixed costs, chain cap 15, the CUDA sub15d arm's mechanism) unless
// `fixed` is false (the previous learned form, cap 7). head_fraction = draft head rows / vocab (1 = full head).
// The calibrated / table model also carries the shared context cost (--cost-kv, default auto = hip_dense_cost_kv).
// auto, measured on one Strix Halo node (Qwen 27B dense IU4, ids equal in every arm): calibrated seeds are
// the walls at the prompt's own context, so the online fit has nothing to price: 0.0 % at 8K code, 32K code / think and
// short code / think -> off. The ctx-256 table fallback
// misprices a long context (r8 82.6 ms priced, ~150 ms at 29.7K): online +10.4 % at 32K code (18.22 -> 20.12 tok/s) -> online.
[[nodiscard]] inline std::string hip_dense_cost_kv(const std::string& requested, HipDenseCostSource source) {
    if (requested != "auto") return requested;
    return source == HipDenseCostSource::calibrated ? "off" : "online";
}
inline spec::BlockConfig hip_dense_block_config(const HipDenseSpecPlan& plan, double head_fraction, const HipDenseCosts& costs) {
    auto bc = hip_dense_spec_block_config(HipEngine::max_rows(), !plan.mtp_source.empty(), plan.costs != HipDenseCostMode::learned,
                                          head_fraction, costs);
    if (bc.fixed_costs) spec::set_context_cost(bc.arbiter, hip_dense_cost_kv(plan.cost_kv, costs.source));
    return bc;
}
// The plan's costs for this engine (F7: nothing lab-specific by default). calibrated: after the prompt (real context),
// one warm-up and one timed verify window per row bucket (1 / 2 / 4 / 8 / 16 rows), each discarded (commit(0): the
// committed state is untouched, as for any rejected window); about 2 x (V1 + V2 + V4 + V8 + V16), ~0.9 s on a 27B pack.
// Invalid walls fall back to the table (source table, reported). table / learned: the compiled table.
// One timed window of `rows` rows of token `token`, discarded (commit(0): the committed state is untouched, as for any
// rejected window): the verify target's request (spec/engine_target.hpp) up to the host wait for its results, ms.
template <class Engine>
double hip_dense_time_window(Engine& e, std::uint32_t token, std::uint32_t rows) {
    std::uint32_t t[Engine::max_rows()];
    for (auto& v : t) v = token;
    const auto t0 = std::chrono::steady_clock::now();
    (void)e.forward_rows(t, rows, Engine::LogitRows::all);
    e.commit(0);
    e.sync();
    return std::chrono::duration<double, std::milli>(std::chrono::steady_clock::now() - t0).count();
}
// The one-row pass of this device after the prompt (a placement input: split::with_measured_single): one warm-up window,
// then the median of three timed windows, all discarded. ~4 x the plain step (~0.25 s on a 27B pack).
template <class Engine>
double hip_dense_measure_pass(Engine& e, std::uint32_t token) {
    (void)hip_dense_time_window(e, token, 1U);
    double w[3];
    for (auto& v : w) v = hip_dense_time_window(e, token, 1U);
    std::sort(std::begin(w), std::end(w));
    return w[1];
}
template <class Engine>
HipDenseCosts hip_dense_measure_costs(Engine& e, HipDenseCostMode mode, bool scalar_x, std::uint32_t token) {
    if (mode != HipDenseCostMode::calibrated) return hip_dense_table_costs(scalar_x);
    const std::uint32_t rows[5] = {1U, 2U, 4U, 8U, 16U};
    double walls[5] = {};
    for (int pass = 0; pass < 2; ++pass)  // pass 0 warms every bucket's kernels (first launches), pass 1 is timed
        for (std::uint32_t i = 0; i < 5U; ++i) walls[i] = hip_dense_time_window(e, token, rows[i]);
    return hip_dense_calibrated_costs(walls, e.weight_bytes(), e.mtp_weight_bytes(), e.head_weight_bytes(), scalar_x);
}

// Device glue of the shared MTP chain (strix/spec/mtp_chain.hpp) over HipEngine.
struct HipChainDev {
    static float* alloc(HipEngine& e, std::size_t n) { return e.alloc_f32(n); }
    static void release(HipEngine& e, float* p) { e.free_f32(p); }
    static void copy(HipEngine& e, float* d, const float* s, std::size_t n) { e.copy_f32_async(d, s, n); }
};
using HipMtpChain = spec::MtpChain<HipEngine, HipChainDev>;

}  // namespace strix::qwen35
