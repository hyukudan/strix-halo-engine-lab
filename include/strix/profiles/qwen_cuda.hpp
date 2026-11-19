#pragma once
// CUDA Qwen profile rows over the SAME resolver as HIP (strix::qwen::profile::resolve / take_option, rows = this table).
// Host-only: no CUDA types. Same lever names where the arithmetic is the same (moe_route_bf16); the HIP-only rows
// (qsa_budget, gdn_fp16_chunk, output_head) are listed as pending in the fast column and never used on CUDA; the FAST
// prefill forms hc_mma / mtp_mma / qsa_mma are CUDA numerical identities (FP16 tensor cores, FP32 accumulation).
#include "strix/profiles/qwen.hpp"

#include <cstdint>
#include <optional>
#include <span>
#include <stdexcept>
#include <string>
#include <string_view>

namespace strix::qwen::cuda_profile {
namespace pr = ::strix::qwen::profile;
using pr::Lever;
using pr::Kind;

// Columns: default, fast, quality, turbo. A form is provisional in fast until its gate receipt flips `ready`. An empty turbo
// cell is the fast cell (engine_profile.hpp value_column): TURBO runs every FAST lever plus the TURBO-only rows below, each
// in the regime where it wins (expert_turbo: prefill blocks of >= expert_turbo_rows rows).
inline constexpr Lever table[] = {
    // Pending in fast: the engine refuses this storage with the MTP sidecar, graphs, cohorts, node hand-off and the prefix
    // cache (qwen_route::require_composition), i.e. in every shipped CUDA composition. Runnable only by an explicit
    // option/env or --allow-pending (then unqualified), never provisional.
    {.name = "moe_route_bf16", .env = "STRIX_QPF2K_MOE_ROUTE_BF16", .option = "--prefill-moe-route-bf16",
     .value = {"", "1", ""}, .builtin = "0",
     .kind = Kind::bit_changing, .ready = {true, false, true}, .runnable = {true, true, true},
     .target = "BF16 MoE route storage (cuda-qwen-a8-prefill+moeroute-bf16-rne-v1)", .off = "0",
     .source = "same arithmetic as the HIP row; unqualified; the engine declines it with MTP sidecar/graphs/cohort/hand-off/prefix cache; changes logits"},
    {.name = "qsa_budget", .env = "STRIX_QPF2K_QSA_BUDGET", .option = "--qsa-budget",
     .value = {"", "1536", ""}, .builtin = "default",
     .kind = Kind::bit_changing, .ready = {true, false, true}, .runnable = {false, false, false},
     .target = "1536", .off = "0", .source = "HIP prefill ranked-block budget; not implemented on CUDA",
     .provisional = {false, true, false}},
    {.name = "gdn_fp16_chunk", .env = "STRIX_QPF2K_GDN_FP16_CHUNK", .option = "--gdn-fp16-chunk",
     .value = {"", "1", ""}, .builtin = "0",
     .kind = Kind::bit_changing, .ready = {true, false, true}, .runnable = {false, false, false},
     .target = "qwen-gdn-fp16-chunk", .off = "0", .source = "HIP chunked GDN prefill on FP16 WMMA; not implemented on CUDA",
     .provisional = {false, true, false}},
    {.name = "output_head", .env = "", .option = "--output-head-profile",
     .value = {"", "fp16-output-head", ""}, .builtin = "default",
     .kind = Kind::bit_changing, .ready = {true, false, true}, .runnable = {false, false, false},
     .target = "fp16 output-head projection", .off = "off", .source = "pending output-head implementation and qualification"},
    {.name = "hc_mma", .env = "STRIX_QRTX_HC_MMA", .option = "--hc-mma",
     .value = {"", "1", ""}, .builtin = "0",
     .kind = Kind::bit_changing, .ready = {true, false, true}, .runnable = {false, true, false},
     .target = "qwen-cuda-hc-f16mma-v2", .off = "0",
     .source = "hyper-connection projection/output on FP16 tensor cores, FP32 accumulation; qualification in progress; changes logits",
     .provisional = {false, true, false}},
    // Not a member of the FAST or TURBO bundle: it only changes the draft head's prompt K/V (prompt MTP rows, after the
    // first token under mtp_defer_prompt), which the prompt-logit gate does not see, and its effect on draft acceptance in
    // decode is unmeasured, so TURBO decode would no longer be FAST decode. Explicit in both (--mtp-mma on).
    {.name = "mtp_mma", .env = "STRIX_QRTX_MTP_MMA", .option = "--mtp-mma",
     .value = {"", "", "", ""}, .builtin = "0",
     .kind = Kind::bit_changing, .ready = {true, false, true, false}, .runnable = {false, true, false, true},
     .target = "qwen-cuda-mtp-f16mma-v1", .off = "0",
     .source = "MTP fused projection on FP16 tensor cores, FP32 accumulation; qualification in progress; changes logits",
     .provisional = {false, true, false, true}},
    {.name = "qsa_mma", .env = "STRIX_QRTX_QSA_MMA", .option = "--qsa-mma",
     .value = {"", "1", ""}, .builtin = "0",
     .kind = Kind::bit_changing, .ready = {true, false, true}, .runnable = {false, true, false},
     .target = "qwen-cuda-qsa-bf16mma-v2", .off = "0",
     .source = "QSA prefill attention on the shared row-group core (core/attention_rowgroup.cuh: union of 4 rows' keys, BF16 "
               "tensor cores, FP32 accumulation); union byte-equal in its fixture; logit gate pending; changes logits",
     .provisional = {false, true, false}},
    // Opt-in under fast (not a FAST bundle member until its gate receipt): the PLE key/value projections (Q8_0 weights) on
    // the shared dense A8 projection (qwen_dense::quantize_project, int8 tensor cores, one FP32 FMA chain per output, the
    // dense-a8-v1 projection contract of the GDN/QSA projections) instead of the FP32 lane chain of ple_project_kernel.
    {.name = "ple_a8", .env = "STRIX_QRTX_PLE_A8", .option = "--ple-a8",
     .value = {"", "", ""}, .builtin = "0",
     .kind = Kind::bit_changing, .ready = {true, false, true}, .runnable = {false, true, false},
     .target = "qwen-cuda-ple-a8-v1", .off = "0",
     .source = "PLE key/value projections on the shared dense A8 projection (int8 tensor cores); FAST gate v2 pending; changes logits",
     .provisional = {false, true, false}},
    // fast cell empty and turbo cell "0" (builtin): FAST experts are the expert_fast row, TURBO experts the integer
    // expert_turbo row; the single-chain moe_mma form refuses a non-incumbent expert arithmetic (qwen_options.hpp resolve),
    // and `--moe-mma on` (with `--expert-fast off` under fast) stays available explicitly, e.g. for a non-GSQ pack.
    {.name = "moe_mma", .env = "STRIX_QRTX_MOE_MMA", .option = "--moe-mma",
     .value = {"", "", "", "0"}, .builtin = "0",
     .kind = Kind::bit_changing, .ready = {true, false, true, true}, .runnable = {false, true, false, true},
     .target = "qwen-cuda-moe-i8fma-v1", .off = "0",
     .source = "routed-MoE prefill gate/up and down on int8 tensor cores with one FP32 FMA chain per output; qualification in progress; changes logits",
     .provisional = {false, true, false}},
    // TURBO only: routed GSQ experts on int8 tensor cores (gsq-iq-i8mma-turbo-cuda-v1, src/cuda/core/turbo_i8.hpp), selected
    // ahead of qualification by `--profile turbo` (provisional), in its regime only (expert_turbo_rows): prefill blocks of
    // at least that many rows; decode/verify windows and smaller blocks keep the run's other experts (under TURBO the
    // FAST expert_fast row, so TURBO decode is FAST decode). GSQ expert storage only: another pack refuses the start at
    // admission; `--expert-turbo off` (or STRIX_QRTX_EXPERT_TURBO=0) keeps the FAST experts under TURBO.
    {.name = "expert_turbo", .env = "STRIX_QRTX_EXPERT_TURBO", .option = "--expert-turbo",
     .value = {"", "", "", "1"}, .builtin = "0",
     .kind = Kind::bit_changing, .ready = {true, true, true, false}, .runnable = {false, false, false, true},
     .target = "gsq-iq-i8mma-turbo-cuda-v1", .off = "0",
     .source = "routed GSQ experts: A8 int8 activations x integer codes on int8 IMMA, one FP32 FMA per group; 32K logit gate "
               "(4 prompts x 128 rows) PASS: regime (default rows) KL 0.00906 top-1 0.9844 dNLL +0.00102, int8 in every call "
               "KL 0.00948 top-1 0.9785 dNLL +0.00543; speed regime per expert_turbo_rows; changes logits",
     .provisional = {false, false, false, true}},
    // Regime of expert_turbo (read only when it is on): prefill blocks of at least N rows run the int8 experts, every other
    // MoE call the run's other expert arithmetic; `off` = int8 in every MoE call (decode windows included). Measured: the
    // no-model expert bench (one routed-expert call, IQ3_S gate/up + IQ4_NL down) has int8 / BF16 time 0.954 / 0.967 /
    // 0.953 / 0.957 / 0.898 / 0.915 at 1 / 16 / 128 / 512 / 2048 / 8192 rows, yet int8 in every decode window cost 9-15 %
    // per step in situ (mtp_mma on); the bench does not reproduce that loss, so windows stay FAST. 512 stays because every
    // scheduled prefill block (1024/2048/4096/8192, qwen_prefill.hpp block_schedule) is above it: the gated 32K
    // arithmetic is the int8 one, and no window or tail below 512 rows changes arithmetic without a measurement.
    {.name = "expert_turbo_rows", .env = "STRIX_QRTX_EXPERT_TURBO_ROWS", .option = "--expert-turbo-rows",
     .value = {"", "", "", "512"}, .builtin = "512",
     .kind = Kind::bit_changing, .ready = {true, false, true, false}, .runnable = {true, true, true, true},
     .target = "", .off = "off",
     .source = "int8 experts on prefill blocks of >= N rows only (decode windows: the FAST experts); expert bench int8/BF16 "
               "0.95-0.97 at 1-512 rows, 0.90-0.92 at 2048-8192, in-situ decode windows +9-15 % per step with int8: N = 512",
     .provisional = {false, false, false, true}},
    // FAST (and TURBO outside expert_turbo's regime): routed GSQ experts on BF16 tensor cores (gsq-iq-bf16mma-cuda-v1), the
    // expert member of the FAST bundle; its empty turbo cell is the fast cell.
    // Same admission as expert_turbo: GSQ expert storage only (another pack refuses the start); `--expert-fast off` (or
    // STRIX_QRTX_EXPERT_FAST=0) keeps the pack's own experts; an explicit --expert-arithmetic keeps its own contract.
    {.name = "expert_fast", .env = "STRIX_QRTX_EXPERT_FAST", .option = "--expert-fast",
     .value = {"", "1", ""}, .builtin = "0",
     .kind = Kind::bit_changing, .ready = {true, false, true, true}, .runnable = {false, true, false, false},
     .target = "gsq-iq-bf16mma-cuda-v1", .off = "0",
     .source = "routed GSQ experts: BF16 activations x exact BF16 codes on tensor cores, FP32 accumulation; alone: logit gate "
               "PASS (pooled KL 0.00679, top-1 0.9912, dNLL -0.00145); bundle gate pending; changes logits",
     .provisional = {false, true, false, false}},
    // Exact (logits and state unchanged): the prompt's MTP rows run after the first token, as the identical call. Member
    // of the FAST bundle (moves the MTP prompt K/V out of TTFT); runs only with an MTP sidecar.
    {.name = "mtp_defer_prompt", .env = "STRIX_QRTX_MTP_DEFER_PROMPT", .option = "--mtp-defer-prompt",
     .value = {"", "1", ""}, .builtin = "0",
     .kind = Kind::exact, .ready = {true, true, true, true}, .runnable = {true, true, true, true},
     .target = "", .off = "0",
     .source = "deferred prompt MTP rows: exact (the identical MTP call after the first token; MTP fixture byte-equal)"},
    // Exact schedule trait of the tensor-core GSQ experts (FAST / expert_fast arithmetic only): steps of <= 16 rows (decode
    // and MTP verify windows) run the decode tile form (one m16 block, three-group weight-field ring), bitwise the tile
    // form (same field values, same MMA and scale-FMA order per output). On in every profile (an absent row = on: the
    // empty cells add no entry); `--expert-decode-tile off` / STRIX_QRTX_EXPERT_DECODE_TILE=0 runs the tile form.
    {.name = "expert_decode_tile", .env = "STRIX_QRTX_EXPERT_DECODE_TILE", .option = "--expert-decode-tile",
     .value = {"", "", "", ""}, .builtin = "1",
     .kind = Kind::exact, .ready = {true, true, true, true}, .runnable = {true, true, true, true},
     .target = "", .off = "0",
     .source = "decode tile form of the tensor-core GSQ experts: exact (bitwise the tile form; greedy ids equal on prose/edit/json)"},
};
inline std::span<const Lever> levers() { return table; }

// The RTX Qwen FAST bundle = the fast column above: its members are the rows the fast profile selects, qualified TOGETHER
// by one logit gate (KL <= 0.02, top-1 >= 0.93, dNLL <= 0.01 against the exact reference). The id names that set; any
// member overridden (or any other bit-changing lever on) leaves the run outside the bundle (fast_bundle_status says why).
// Not a member: qsa fused (exact opt-in STRIX_QRTX_QSA_FUSED): with qsa_mma on, attention() returns from the FAST core
// before the fused branch (qwen_qsa.cu attention), so it would change nothing in the bundle's prefill.
inline constexpr std::string_view kFastBundle = "qwen-cuda-fast-bundle-v1";
inline constexpr std::string_view kFastBundleMembers[] = {"expert_fast", "qsa_mma", "hc_mma", "mtp_defer_prompt"};

inline constexpr std::string_view kHcIdentity = "qwen-cuda-hc-f16mma-v2";
inline constexpr std::string_view kMtpIdentity = "qwen-cuda-mtp-f16mma-v1";
inline constexpr std::string_view kQsaIdentity = "qwen-cuda-qsa-bf16mma-v2";
inline constexpr std::string_view kMoeIdentity = "qwen-cuda-moe-i8fma-v1";
inline constexpr std::string_view kPleIdentity = "qwen-cuda-ple-a8-v1";

// The runnable levers a resolution selects. Pending, off and builtin-off rows select nothing.
struct Choice {
    bool moe_route_bf16 = false, hc_mma = false, mtp_mma = false, qsa_mma = false, moe_mma = false;
    bool ple_a8 = false;  // after the five above, so their positional initialisers keep their meaning
    bool expert_turbo = false;  // maps to Config::expert_arithmetic (its identity is the arithmetic name, not a form)
    bool expert_fast = false;   // likewise (gsq-iq-bf16mma-cuda-v1)
    // Regime of expert_turbo (row expert_turbo_rows): int8 experts on prefill blocks of at least this many rows; nullopt:
    // every MoE call. Read only when expert_turbo is on.
    std::optional<std::uint32_t> expert_turbo_rows{512};
    bool mtp_defer_prompt = false;  // exact: Config::defer_prompt_mtp
    bool expert_decode_tile = true; // exact: Config::expert_decode_tile (absent row = on)
    bool fast() const { return moe_route_bf16 || hc_mma || mtp_mma || qsa_mma || moe_mma || ple_a8; }
    bool operator==(const Choice&) const = default;
};

inline std::string normalise(std::string_view lever, std::string_view value) {
    if (value == "1" || value == "on" || value == "true") return "1";
    if (value == "0" || value == "off" || value == "false") return "0";
    throw std::invalid_argument(std::string(lever) + " is on or off (got '" + std::string(value) + "')");
}
// Explicit values of the boolean levers are validated before use; non-boolean rows (pending) are left as they are.
inline bool lever_on(const pr::Resolution& r, std::string_view name) {
    const auto* entry = r.find(name);
    if (!entry || entry->status == pr::Status::pending || entry->status == pr::Status::off) return false;
    return normalise(name, entry->value) == "1";
}
// expert_turbo_rows: a decimal N >= 1, or off (nullopt). Absent from the resolution (exact/fast with no switch): builtin.
inline std::optional<std::uint32_t> turbo_rows(const pr::Resolution& r) {
    const auto* entry = r.find("expert_turbo_rows");
    if (entry && entry->status == pr::Status::off) return std::nullopt;
    std::string_view text;
    for (const auto& l : levers()) if (l.name == "expert_turbo_rows") text = l.builtin;
    if (entry && entry->status != pr::Status::pending) text = entry->value;
    std::uint64_t n = 0;
    for (const char ch : text) {
        if (ch < '0' || ch > '9' || n > 0xffffffffULL / 10U)
            throw std::invalid_argument("expert_turbo_rows is a row count >= 1 or off (got '" + std::string(text) + "')");
        n = n * 10U + static_cast<std::uint64_t>(ch - '0');
    }
    if (text.empty() || n == 0 || n > 0xffffffffULL)
        throw std::invalid_argument("expert_turbo_rows is a row count >= 1 or off (got '" + std::string(text) + "')");
    return static_cast<std::uint32_t>(n);
}
inline Choice choose(const pr::Resolution& r) {
    Choice c;
    c.moe_route_bf16 = lever_on(r, "moe_route_bf16");
    c.hc_mma = lever_on(r, "hc_mma");
    c.mtp_mma = lever_on(r, "mtp_mma");
    c.qsa_mma = lever_on(r, "qsa_mma");
    c.moe_mma = lever_on(r, "moe_mma");
    c.ple_a8 = lever_on(r, "ple_a8");
    c.expert_turbo = lever_on(r, "expert_turbo");
    c.expert_fast = lever_on(r, "expert_fast");
    c.expert_turbo_rows = turbo_rows(r);
    c.mtp_defer_prompt = lever_on(r, "mtp_defer_prompt");
    c.expert_decode_tile = r.find("expert_decode_tile") == nullptr || lever_on(r, "expert_decode_tile");
    return c;
}
// Flip together with the members' fast-column `ready` cells when the whole-bundle gate receipt passes.
inline constexpr bool kFastBundleQualified = false;
// Whether a resolved run IS the FAST bundle, decided on the effective values (not on how they were set): the expert
// arithmetic (expert_bf16mma = Config::expert_arithmetic is gsq-iq-bf16mma-cuda-v1), forms exactly {hc_mma, qsa_mma}, no
// route storage, and the prompt MTP deferred. id empty = outside the bundle; detail names every differing member.
struct BundleStatus {
    std::string id, detail;
};
namespace bundle_detail {
struct Missing {
    std::string text;
    void need(bool ok, std::string_view what) {
        if (!ok) text += (text.empty() ? "" : ", ") + std::string(what);
    }
};
inline BundleStatus status(std::string_view id, std::span<const std::string_view> members, const Missing& m, bool qualified) {
    if (!m.text.empty()) return {"", "outside " + std::string(id) + ": " + m.text};
    std::string list;
    for (const auto x : members) list += (list.empty() ? "" : ",") + std::string(x);
    return {std::string(id), "members " + list + (qualified ? "; qualified" :
        "; provisional: the whole-bundle logit gate has not passed")};
}
}  // namespace bundle_detail
inline BundleStatus fast_bundle_status(const Choice& c, bool expert_bf16mma) {
    bundle_detail::Missing m;
    m.need(expert_bf16mma, "experts are not gsq-iq-bf16mma-cuda-v1");
    m.need(c.qsa_mma, "qsa_mma off");
    m.need(c.hc_mma, "hc_mma off");
    m.need(c.mtp_defer_prompt, "mtp_defer_prompt off");
    m.need(!c.moe_mma, "moe_mma on (not a member)");
    m.need(!c.mtp_mma, "mtp_mma on (not a member)");
    m.need(!c.moe_route_bf16, "moe_route_bf16 on (not a member)");
    m.need(!c.ple_a8, "ple_a8 on (not a member)");
    return bundle_detail::status(kFastBundle, kFastBundleMembers, m, kFastBundleQualified);
}

// The RTX Qwen TURBO bundle = the turbo column = the FAST bundle (expert_fast, qsa_mma, hc_mma, mtp_defer_prompt) plus the
// integer TURBO experts (gsq-iq-i8mma-turbo-cuda-v1) in their regime only: prefill blocks of at least expert_turbo_rows
// rows; decode/verify windows and smaller blocks run the FAST experts, so TURBO decode is FAST decode (tiers nested; the
// shared test strix-profile-fast-set-tests asserts that every FAST cell is in the turbo column). Pack's own dense
// weights. One id for the set that the TURBO logit gate qualifies (FAST thresholds, against the exact reference). Any
// member overridden (another expert arithmetic, the regime off, --turbo-weights onfly, a form off, mtp_mma, moe_mma or
// route storage on) leaves the run outside it. v1 was int8 experts in every MoE call plus mtp_mma.
inline constexpr std::string_view kTurboBundle = "qwen-cuda-turbo-bundle-v2";
inline constexpr std::string_view kTurboBundleMembers[] = {"expert_fast", "qsa_mma", "hc_mma", "mtp_defer_prompt", "expert_turbo",
                                                           "expert_turbo_rows"};
// Flip when the TURBO gate receipts pass on this arithmetic (32K: 4 prompts x 128 rows; the same gate on this tree).
inline constexpr bool kTurboBundleQualified = false;
// expert_i8turbo_v1: Config::expert_arithmetic is gsq-iq-i8mma-turbo-cuda-v1; narrow_bf16mma: Config::expert_narrow is
// gsq-iq-bf16mma-cuda-v1 (the regime split is on and runs the FAST experts outside it).
inline BundleStatus turbo_bundle_status(const Choice& c, bool expert_i8turbo_v1, bool narrow_bf16mma, bool native_weights) {
    bundle_detail::Missing m;
    m.need(expert_i8turbo_v1, "experts are not gsq-iq-i8mma-turbo-cuda-v1");
    m.need(narrow_bf16mma, "outside the expert_turbo regime the experts are not gsq-iq-bf16mma-cuda-v1");
    m.need(c.qsa_mma, "qsa_mma off");
    m.need(c.hc_mma, "hc_mma off");
    m.need(c.mtp_defer_prompt, "mtp_defer_prompt off");
    m.need(native_weights, "dense weights not the pack's (--turbo-weights onfly)");
    m.need(!c.mtp_mma, "mtp_mma on (not a member)");
    m.need(!c.moe_mma, "moe_mma on (not a member)");
    m.need(!c.moe_route_bf16, "moe_route_bf16 on (not a member)");
    m.need(!c.ple_a8, "ple_a8 on (not a member)");
    return bundle_detail::status(kTurboBundle, kTurboBundleMembers, m, kTurboBundleQualified);
}
inline std::string forms_list(const Choice& c) {
    std::string out;
    for (const auto& [on, name] : {std::pair{c.hc_mma, "hc_mma"}, std::pair{c.mtp_mma, "mtp_mma"}, std::pair{c.qsa_mma, "qsa_mma"}, std::pair{c.moe_mma, "moe_mma"},
                                  std::pair{c.ple_a8, "ple_a8"}})
        if (on) out += (out.empty() ? "" : ",") + std::string(name);
    return out;
}
// Identities of the selected forms, e.g. "qwen-cuda-hc-f16mma-v2,qwen-cuda-qsa-bf16mma-v2".
inline std::string form_identities(const Choice& c) {
    std::string out;
    const auto add = [&](bool on, std::string_view id) { if (on) out += (out.empty() ? "" : ",") + std::string(id); };
    add(c.hc_mma, kHcIdentity);
    add(c.mtp_mma, kMtpIdentity);
    add(c.qsa_mma, kQsaIdentity);
    add(c.moe_mma, kMoeIdentity);
    add(c.ple_a8, kPleIdentity);
    return out;
}
// The CUDA arithmetic of the prefill, as a stable string (receipt, hand-off producer id, prefix identity). A FAST run with
// forms carries the forms' versioned identities (";ids=..."), so two versions of one form never share a contract; the
// exact contract keeps its bytes.
inline std::string arithmetic_contract(const Choice& c) {
    if (!c.fast()) return "qwen-cuda;prefill=exact";
    const auto forms = forms_list(c);
    std::string out = "qwen-cuda;prefill=fast;forms=" + (forms.empty() ? std::string("none") : forms) +
        ";moe_route_bf16=" + (c.moe_route_bf16 ? "1" : "0");
    if (!forms.empty()) out += ";ids=" + form_identities(c);
    return out;
}
inline ::strix::profile::ExecutionReceipt receipt(const Choice& c, std::string_view prefill_lineage = {}) {
    ::strix::profile::ExecutionReceipt out;
    out.engine = "qwen-cuda";
    out.identity.arithmetic_contract = arithmetic_contract(c);
    out.identity.prefill_lineage = prefill_lineage;
    return out;
}
inline pr::Resolution resolve(pr::Name profile, const pr::Inputs& inputs = {}) {
    auto out = pr::resolve(profile, inputs, levers());
    (void)choose(out);  // refuses malformed explicit values here, not at first use
    return out;
}
inline bool take_option(int argc, char** argv, int& i, std::optional<pr::Name>& profile, pr::Inputs& inputs) {
    return pr::take_option(argc, argv, i, profile, inputs, levers());
}
}  // namespace strix::qwen::cuda_profile
