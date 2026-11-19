#pragma once
// CUDA Qwen profile rows over the SAME resolver as HIP (strix::qwen::profile::resolve / take_option, rows = this table).
// Host-only: no CUDA types. Same lever names where the arithmetic is the same (moe_route_bf16); the HIP-only rows
// (qsa_budget, gdn_fp16_chunk, output_head) are listed as pending in the fast column and never used on CUDA; the FAST
// prefill forms hc_mma / mtp_mma / qsa_mma are CUDA numerical identities (FP16 tensor cores, FP32 accumulation).
#include "strix/profiles/qwen.hpp"

#include <cstdint>
#include <span>
#include <stdexcept>
#include <string>
#include <string_view>

namespace strix::qwen::cuda_profile {
namespace pr = ::strix::qwen::profile;
using pr::Lever;
using pr::Kind;

// Columns: default, fast, quality, turbo. A form is provisional in fast until its gate receipt flips `ready`. An empty turbo
// cell is the fast cell (engine_profile.hpp value_column): TURBO runs every FAST form plus the TURBO-only rows below.
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
    // Not a member of the FAST bundle (kFastBundle): it only changes the draft head, which the prompt-logit gate does not
    // see, and under the bundle's mtp_defer_prompt its rows run after the first token (no TTFT effect). Explicit in fast;
    // its turbo cell keeps it in TURBO as before.
    {.name = "mtp_mma", .env = "STRIX_QRTX_MTP_MMA", .option = "--mtp-mma",
     .value = {"", "", "", "1"}, .builtin = "0",
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
    // ahead of qualification by `--profile turbo` (provisional). GSQ expert storage only: another pack refuses the start
    // at admission; `--expert-turbo off` (or STRIX_QRTX_EXPERT_TURBO=0) keeps the pack's own experts under TURBO.
    {.name = "expert_turbo", .env = "STRIX_QRTX_EXPERT_TURBO", .option = "--expert-turbo",
     .value = {"", "", "", "1"}, .builtin = "0",
     .kind = Kind::bit_changing, .ready = {true, true, true, false}, .runnable = {false, false, false, true},
     .target = "gsq-iq-i8mma-turbo-cuda-v1", .off = "0",
     .source = "routed GSQ experts: A8 int8 activations x integer codes on int8 IMMA, one FP32 FMA per group; TURBO bar pending; changes logits",
     .provisional = {false, false, false, true}},
    // FAST only: routed GSQ experts on BF16 tensor cores (gsq-iq-bf16mma-cuda-v1), the expert member of the FAST bundle.
    // Same admission as expert_turbo: GSQ expert storage only (another pack refuses the start); `--expert-fast off` (or
    // STRIX_QRTX_EXPERT_FAST=0) keeps the pack's own experts; an explicit --expert-arithmetic keeps its own contract.
    {.name = "expert_fast", .env = "STRIX_QRTX_EXPERT_FAST", .option = "--expert-fast",
     .value = {"", "1", "", "0"}, .builtin = "0",
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
    // Exact schedule trait of the tensor-core GSQ experts: launches averaging >= N rows per expert (the row's value; `on` =
    // native_mma::kWideMinAvgRows, PROVISIONAL; N from the device's no-model tile bench) run 128-row CSR tiles in
    // 512-thread blocks (one weight row per lane), so a 128-row run of an expert decodes its codes once instead of twice;
    // bitwise the tile form (same MMA and scale-FMA order per output, same m16 position per row). Opt-in until its timing
    // receipt: `--expert-wide-tile on|N` / STRIX_QRTX_EXPERT_WIDE_TILE=1|N.
    {.name = "expert_wide_tile", .env = "STRIX_QRTX_EXPERT_WIDE_TILE", .option = "--expert-wide-tile",
     .value = {"", "", "", ""}, .builtin = "0",
     .kind = Kind::exact, .ready = {true, true, true, true}, .runnable = {true, true, true, true},
     .target = "", .off = "0",
     .source = "wide 128-row tile form of the tensor-core GSQ experts: exact by construction (bitwise the tile form); timing pending"},
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
    bool mtp_defer_prompt = false;  // exact: Config::defer_prompt_mtp
    bool expert_decode_tile = true; // exact: Config::expert_decode_tile (absent row = on)
    bool expert_wide_tile = false;  // exact: Config::expert_wide_tile (opt-in)
    std::uint32_t expert_wide_min_rows = 0;  // its planner parameter (value N of the row; 0 = the builtin threshold)
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
// expert_wide_tile takes on / off or N = the minimum average rows per expert of a launch for the wide form (1..65535;
// on = the builtin threshold). N comes from the device's no-model tile bench; it changes the schedule only (exact row).
inline void wide_tile_value(const pr::Resolution& r, bool& on, std::uint32_t& min_rows) {
    on = false;
    min_rows = 0;
    const auto* entry = r.find("expert_wide_tile");
    if (!entry || entry->status == pr::Status::pending || entry->status == pr::Status::off) return;
    const std::string_view v = entry->value;
    const bool digits = !v.empty() && v.size() <= 5 && v.find_first_not_of("0123456789") == std::string_view::npos;
    if (!digits || v == "0" || v == "1") {
        on = normalise("expert_wide_tile", v) == "1";
        return;
    }
    const unsigned long n = std::stoul(std::string(v));
    if (n > 65535UL) throw std::invalid_argument("expert_wide_tile is on, off or rows per expert 2..65535 (got '" + std::string(v) + "')");
    on = true;
    min_rows = static_cast<std::uint32_t>(n);
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
    c.mtp_defer_prompt = lever_on(r, "mtp_defer_prompt");
    c.expert_decode_tile = r.find("expert_decode_tile") == nullptr || lever_on(r, "expert_decode_tile");
    wide_tile_value(r, c.expert_wide_tile, c.expert_wide_min_rows);
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
inline BundleStatus fast_bundle_status(const Choice& c, bool expert_bf16mma) {
    std::string missing;
    const auto need = [&](bool ok, std::string_view what) {
        if (!ok) missing += (missing.empty() ? "" : ", ") + std::string(what);
    };
    need(expert_bf16mma, "experts are not gsq-iq-bf16mma-cuda-v1");
    need(c.qsa_mma, "qsa_mma off");
    need(c.hc_mma, "hc_mma off");
    need(c.mtp_defer_prompt, "mtp_defer_prompt off");
    need(!c.moe_mma, "moe_mma on (not a member)");
    need(!c.mtp_mma, "mtp_mma on (not a member)");
    need(!c.moe_route_bf16, "moe_route_bf16 on (not a member)");
    need(!c.ple_a8, "ple_a8 on (not a member)");
    if (!missing.empty()) return {"", "outside " + std::string(kFastBundle) + ": " + missing};
    std::string members;
    for (const auto m : kFastBundleMembers) members += (members.empty() ? "" : ",") + std::string(m);
    return {std::string(kFastBundle), "members " + members + (kFastBundleQualified ? "; qualified" :
        "; provisional: the whole-bundle logit gate has not passed")};
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
