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
// Speculation is exact: verify windows are bitwise equal to row-by-row decode (hip_engine.hpp), and drafts (MTP layer
// precision, head subset) only steer how many tokens a window accepts, never which tokens are emitted.
// Home: shared host resolver of the qwen35 HIP family (both apps call it); the mechanisms are shared (strix/spec).
#include <cstddef>
#include <cstdint>
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
};

// `mtp`: "auto" / "" / "off" / "none" / path; `draft_rows`: false = full head (off switch).
inline HipDenseSpecPlan resolve_hip_dense_spec(const std::string& model, const std::string& mtp, bool draft_rows,
                                               const std::string& draft_packs = {}) {
    HipDenseSpecPlan p;
    if (mtp == "off" || mtp == "none") {
        p.mtp_reason = "off";
    } else if (mtp.empty() || mtp == "auto") {
        if (HipEngine::has_mtp_layer(model)) {
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
inline spec::BlockConfig hip_dense_block_config(const HipDenseSpecPlan& plan, bool fixed, double head_fraction) {
    return hip_dense_spec_block_config(HipEngine::max_rows(), !plan.mtp_source.empty(), fixed, head_fraction);
}

// Device glue of the shared MTP chain (strix/spec/mtp_chain.hpp) over HipEngine.
struct HipChainDev {
    static float* alloc(HipEngine& e, std::size_t n) { return e.alloc_f32(n); }
    static void release(HipEngine& e, float* p) { e.free_f32(p); }
    static void copy(HipEngine& e, float* d, const float* s, std::size_t n) { e.copy_f32_async(d, s, n); }
};
using HipMtpChain = spec::MtpChain<HipEngine, HipChainDev>;

}  // namespace strix::qwen35
