#pragma once
// ONE engine-option parser for every CUDA Qwen entry point (serve, prefill service, gate tools). Host-only.
// Each app keeps only its own non-engine options and calls take() for the rest; resolve() then turns the request into a
// Config for its entry, and names every value that differs between entries (serve_differences) with its reason.
#include "strix/cuda/qwen_prefill.hpp"
#include "strix/cuda/qwen_turbo.hpp"
#include "strix/profiles/qwen_cuda.hpp"
#include "strix/env_flags.hpp"
#include "strix/profile/all_on_engaged.hpp"
#include "strix/lever.hpp"

#include <algorithm>
#include <cstdint>
#include <iostream>
#include <limits>
#include <ostream>
#include <optional>
#include <stdexcept>
#include <string>
#include <string_view>
#include <vector>

namespace strix::cuda::qwen_options {
namespace qp = ::strix::cuda::qwen_prefill;
namespace cp = ::strix::qwen::cuda_profile;
namespace pr = ::strix::qwen::profile;

enum class Entry { serve, prefill_service, cli };
inline const char* entry_name(Entry e) {
    return e == Entry::serve ? "serve" : e == Entry::prefill_service ? "prefill-service" : "cli";
}

struct Request {
    qp::Config engine;                     // common engine options as typed; entry roles are applied by resolve()
    std::optional<pr::Name> profile;       // --profile (default | exact | fast | quality)
    pr::Inputs inputs;                     // lever options (cli) and the environment lookup (empty: process env)
    bool print_effective_config = false;   // --print-effective-config
    std::optional<bool> release_by_layer;  // typed explicitly
    bool blocks_typed = false;             // --blocks / --block-rows typed: the schedule is the caller's
    ::strix::session::kv_precision::Request kv;  // --kv (strix/session/kv_precision.hpp)
};

// Flat prefill blocks: an untyped schedule becomes one entry, the largest block of the configured schedule (same
// allocation: every workspace is sized from that maximum). Scheduling only: every schedule produces bit-identical
// state. Each block pays the routed-MoE expert weights and the per-kernel launch/skeleton costs once whatever its
// rows, so fewer, larger blocks are less work per token. ON where measured: the serve entry (state stays on the
// device). The hand-off entries keep the ramp, whose small first block releases state early; unmeasured there.
inline constexpr ::strix::lever::Decl flat_blocks_lever{
    .name = "qwen_cuda_flat_blocks", .env = "STRIX_QRTX_FLAT_BLOCKS", .exactness = ::strix::lever::Exactness::bitwise,
    .default_on = true, .source = "docs/engine/serving/qwen-cuda-prefill.md: flat-block receipt (strict ABAB, bit-equal logit dumps)"};

// Block fit: the block workspace is allocated after everything else and its capacity steps down (1024 rows) until it fits
// the free VRAM, instead of refusing the load (Qwen3.8-Flash-Next at 256K: the 8192-row block leaves no room for the
// session KV; 4096 fits). Every entry, so CLI = serve. Bitwise: block rows do not change bits; with room for the schedule
// maximum nothing changes but the allocation order. A typed --blocks / --block-rows stays strict.
inline constexpr ::strix::lever::Decl block_fit_lever{
    .name = "qwen_cuda_block_fit", .env = "STRIX_QRTX_BLOCK_FIT", .exactness = ::strix::lever::Exactness::bitwise,
    .default_on = true, .source = "wave4/next1112/RTXLONG-RESULT.md: 256K fitted-block receipt (bit-equal logit dumps)"};

struct Difference {
    std::string name, value, reason;
};
struct Resolved {
    Entry entry = Entry::cli;
    qp::Config config;
    pr::Resolution resolution;
    cp::Choice choice;
    std::string contract;                  // arithmetic_contract
    std::vector<Difference> differences;   // named entry-role differences, each with its reason
    std::string moe_real_rows_origin = "default";  // lever origin of config.moe_real_rows (default | explicit)
    ::strix::session::kv_precision::Resolution kv;  // KV precision of the run (always in the effective config)
};

inline std::uint64_t number(std::string_view option, const std::string& text, std::uint64_t max = UINT32_MAX) {
    std::size_t end = 0;
    if (text.empty() || text[0] == '-' || text[0] == '+') throw std::invalid_argument(std::string(option) + " requires a nonnegative integer");
    unsigned long long n = 0;
    try { n = std::stoull(text, &end); } catch (const std::exception&) { end = 0; }
    if (end != text.size() || n > max) throw std::invalid_argument(std::string(option) + " value out of range");
    return n;
}

// Consumes one common engine / profile option at argv[i] (advancing i over its value). false: not a common option.
inline bool take(int argc, char** argv, int& i, Request& r) {
    const std::string a = argv[i];
    const auto next = [&]() -> std::string {
        if (i + 1 >= argc) throw std::invalid_argument(a + " needs a value");
        return argv[++i];
    };
    auto& c = r.engine;
    if (a == "--print-effective-config") r.print_effective_config = true;
    else if (a == "--model") c.model_gguf = next();
    else if (a == "--expert-arithmetic") {
        const auto value=next();
        if(value=="incumbent") c.expert_arithmetic=qwen_packs::ExpertArithmetic::incumbent;
        else if(value==qwen_packs::gsq_a8_contract) c.expert_arithmetic=qwen_packs::ExpertArithmetic::gsq_a8_dot4;
        else if(value==qwen_packs::gsq_bf16_contract) c.expert_arithmetic=qwen_packs::ExpertArithmetic::gsq_bf16_mma;
        else if(value==qwen_packs::gsq_i8_turbo_contract) c.expert_arithmetic=qwen_packs::ExpertArithmetic::gsq_i8_turbo;
        else throw std::invalid_argument("--expert-arithmetic requires incumbent, gsq-iq3s-a8-dot4-cuda-v1, gsq-iq-bf16mma-cuda-v1 or gsq-iq-i8mma-turbo-cuda-v1");
    }
    else if (a == "--mtp-sidecar") c.mtp_sidecar_gguf = next();
    else if (a == "--draft-vocab") c.draft_vocab = next();
    else if (a == "--tables") c.tables_dir = next();
    else if (a == "--rope-table") c.rope_table = next();
    else if (a == "--max-positions") c.max_positions = static_cast<std::uint32_t>(number(a, next()));
    else if (a == "--block-schedule") {
        const auto v = next();
        if (v != "default" && v != "tail-large") throw std::invalid_argument("invalid block schedule");
        c.tail_large = v == "tail-large";
    } else if (a == "--final-block-rows") {
        const auto v = next();
        const auto n = number(a, v, qp::kMaxBlockRows);
        c.final_block_rows = static_cast<std::uint32_t>(n);
    } else if (a == "--blocks") {
        r.blocks_typed = true;
        c.block_schedule.clear();
        const auto v = next();
        for (std::size_t at = 0; at < v.size();) {
            const auto comma = v.find(',', at);
            c.block_schedule.push_back(static_cast<std::uint32_t>(number(a, v.substr(at, comma - at))));
            at = comma == std::string::npos ? v.size() : comma + 1;
        }
    } else if (a == "--block-rows") {
        const auto rows = static_cast<std::uint32_t>(number(a, next()));
        if (rows < 1024 || rows % 16) throw std::invalid_argument("--block-rows must be a multiple of 16 and >= 1024");
        r.blocks_typed = true;
        c.block_schedule = {rows};
    } else if (a == "--release-by-layer") r.release_by_layer = true;
    else if (a == "--no-release-by-layer") r.release_by_layer = false;
    else if (::strix::session::kv_precision::take(argc, argv, i, r.kv)) {}
    else return cp::take_option(argc, argv, i, r.profile, r.inputs);
    return true;
}

// Maps a lever Resolution onto the engine Config: the only place that does.
inline void apply(const cp::Choice& choice, qp::Config& config) {
    config.forms = {choice.hc_mma, choice.mtp_mma, choice.qsa_mma, choice.moe_mma, choice.ple_a8};
    config.moe_route_bf16 = choice.moe_route_bf16;  // explicit: the resolver already read the environment
    // Config::defer_prompt_mtp (exact): the lever row `mtp_defer_prompt` (--mtp-defer-prompt on|off, env, fast profile).
    config.defer_prompt_mtp = config.defer_prompt_mtp || choice.mtp_defer_prompt;
    // Config::expert_decode_tile (exact schedule trait): the lever row `expert_decode_tile` (on unless switched off).
    config.expert_decode_tile = config.expert_decode_tile && choice.expert_decode_tile;
    // Config::expert_wide_tile (exact schedule trait): the lever row `expert_wide_tile` (opt-in).
    config.expert_wide_tile = config.expert_wide_tile || choice.expert_wide_tile;
    if (config.expert_wide_min_rows == 0U) config.expert_wide_min_rows = choice.expert_wide_min_rows;
    // TURBO / FAST expert rows: an explicit --expert-arithmetic other than incumbent keeps its own contract (every MoE call).
    // expert_turbo runs the int8 experts in its regime (expert_turbo_rows: prefill blocks of >= N rows); every other MoE
    // call keeps the arithmetic the run has without it (expert_fast: the BF16 FAST experts, else the pack's own), so
    // TURBO = FAST + int8 experts where they win. expert_turbo_rows off: int8 in every MoE call.
    if (config.expert_arithmetic != qwen_packs::ExpertArithmetic::incumbent) return;
    const auto narrow = choice.expert_fast ? qwen_packs::ExpertArithmetic::gsq_bf16_mma : qwen_packs::ExpertArithmetic::incumbent;
    if (!choice.expert_turbo) {
        config.expert_arithmetic = narrow;
        return;
    }
    config.expert_arithmetic = qwen_packs::ExpertArithmetic::gsq_i8_turbo;
    if (choice.expert_turbo_rows) {
        config.expert_narrow = narrow;
        config.expert_regime_min_rows = *choice.expert_turbo_rows;
    }
}

// --all-on records (strix/profile/all_on_engaged.hpp): the switches --all-on exports as arguments of this engine whose
// resolved value the engine does not run. Pure over the resolved Choice and Config; prints each declined lever it was asked
// to export (nothing for a switch the user typed by hand). ple_a8 runs on prefill blocks only (decode and verify windows
// keep the exact projection); that is its phase, not a decline.
inline void report_all_on(const cp::Choice& choice, const qp::Config& c, std::ostream& out) {
    namespace ao = ::strix::profile::all_on;
    (void)ao::engaged("--ple-a8", choice.ple_a8 && c.forms.ple_a8,
                      "the resolved profile did not select the PLE A8 projection", out);
    // The wide tile form is a schedule of the BF16 tensor-core GSQ experts: only a run whose experts (or, under the
    // integer regime, its other-regime experts) are gsq-iq-bf16mma-cuda-v1 reaches it.
    const bool bf16_experts = c.expert_arithmetic == qwen_packs::ExpertArithmetic::gsq_bf16_mma ||
                              (c.expert_narrow && *c.expert_narrow == qwen_packs::ExpertArithmetic::gsq_bf16_mma);
    (void)ao::engaged("--expert-wide-tile", c.expert_wide_tile && bf16_experts,
                      c.expert_wide_tile ? "the experts are not the BF16 tensor-core form"
                                         : "the resolved profile did not select the wide tile form", out);
    // The regime row is read only with the integer experts on; without them every MoE call keeps the other experts.
    const bool regime = c.expert_arithmetic == qwen_packs::ExpertArithmetic::gsq_i8_turbo && c.expert_narrow.has_value();
    (void)ao::engaged("--expert-turbo-rows", regime,
                      c.expert_arithmetic == qwen_packs::ExpertArithmetic::gsq_i8_turbo
                          ? "the integer experts run in every MoE call (no regime split)"
                          : "the integer experts are not selected, so no block runs them", out);
}

inline qp::Config resolve_route(qp::Config config, const pr::Lookup& lookup = {}) {
    if (!config.route_policy) {
        const auto get = [&](const char* key) { return lookup ? lookup(key) : ::strix::envcfg::text(key); };
        config.route_policy = ::strix::cuda::qwen_route::resolve(config.moe_route_bf16, get("STRIX_QPF2K_MOE_ROUTE_BF16"),
            get("STRIX_BASELINE"), get("STRIX_QPF2K_DEFAULTS"), get("STRIX_PREFILL_EXACT_FP16"));
    }
    config.moe_route_bf16 = config.route_policy->enabled;
    return config;
}

// The reported arithmetic contract: the form/route contract plus the expert arithmetic when it is not the incumbent.
// One home for every receipt of a resolved config (effective config, service /health), so they cannot disagree.
inline std::string with_expert_arithmetic(std::string contract, qwen_packs::ExpertArithmetic expert) {
    if (expert != qwen_packs::ExpertArithmetic::incumbent) contract += "+" + std::string(qwen_packs::expert_arithmetic_name(expert));
    return contract;
}
// The per-regime expert rule (Config::expert_narrow set) as a stable string, e.g.
// "prefill_rows>=512:gsq-iq-i8mma-turbo-cuda-v1,else:gsq-iq-bf16mma-cuda-v1"; empty without a split.
inline std::string expert_regime(const qp::Config& c) {
    if (!c.expert_narrow) return {};
    return "prefill_rows>=" + std::to_string(c.expert_regime_min_rows) + ":" +
        std::string(qwen_packs::expert_arithmetic_name(c.expert_arithmetic)) + ",else:" +
        std::string(qwen_packs::expert_arithmetic_name(*c.expert_narrow));
}
// Contract of a resolved config: without a split, the historical bytes; with one, the rule in brackets.
inline std::string with_expert_arithmetic(std::string contract, const qp::Config& c) {
    if (!c.expert_narrow) return with_expert_arithmetic(std::move(contract), c.expert_arithmetic);
    return contract + "+experts[" + expert_regime(c) + "]";
}

inline Resolved resolve(const Request& r, Entry entry, std::string_view default_model = {}) {
    Resolved out;
    out.entry = entry;
    out.config = r.engine;
    auto& c = out.config;
    const auto profile = r.profile.value_or(pr::Name::standard);
    out.kv = ::strix::session::kv_precision::resolve(profile, r.kv, cp::kKvCaps);  // refuses a precision without a path
    if (profile == pr::Name::turbo) {
        // TURBO on CUDA (qwen-turbo-v1): the weight toggle of strix/cuda/qwen_turbo.hpp plus this engine's lever table read
        // through its turbo cells: every FAST form (empty turbo cell = fast cell) and the TURBO-only integer expert row.
        // Lever options override rows as in every other profile. Throws when the weight request cannot be honoured.
        const auto turbo = ::strix::cuda::qwen_turbo::resolve({r.profile, r.inputs, r.print_effective_config}, cp::levers());
        out.resolution = turbo.resolution;
        ::strix::cuda::qwen_turbo::apply(turbo, c);
        out.choice = cp::choose(out.resolution);
    } else {
        out.resolution = cp::resolve(profile, r.inputs);
        out.choice = cp::choose(out.resolution);
    }
    apply(out.choice, c);
    c.route_policy.reset(); // a new request resolves its supplied environment once
    c = resolve_route(std::move(c), r.inputs.env);
    out.choice.moe_route_bf16 = c.route_policy->enabled;
    if (c.route_policy->requested && !c.route_policy->enabled) {
        for (auto& row : out.resolution.entries) if (row.lever->name == "moe_route_bf16") {
            row.value = "0";
            row.status = pr::Status::off;
            row.reason = c.route_policy->reason;
            if (!row.ladder.empty()) row.ladder += " / declined by route: " + row.reason + " -> default=" + row.value;
        }
    }
    if(c.expert_arithmetic!=qwen_packs::ExpertArithmetic::incumbent) {
        const std::string name(qwen_packs::expert_arithmetic_name(c.expert_arithmetic));
        if(c.forms.moe_mma) throw std::invalid_argument(name+": single-chain moe_mma composition unqualified; use --moe-mma off");
        if(c.route_policy->enabled) throw std::invalid_argument(name+": native IQ RouteBF16 composition unqualified");
    }
    out.contract = with_expert_arithmetic(cp::arithmetic_contract(out.choice), c);
    {  // once per process: the first resolution is the run's
        static const bool all_on_record = (report_all_on(out.choice, c, std::cerr), true);
        (void)all_on_record;
    }
    const auto name = [&](std::string n, std::string v, std::string why) { out.differences.push_back({std::move(n), std::move(v), std::move(why)}); };
    if (entry == Entry::serve) {
        if (r.release_by_layer.value_or(false))
            throw std::invalid_argument("--release-by-layer: serve keeps the state on the device (no pieces to release)");
        c.decode = true;
        c.export_state = false;
        c.release_by_layer = false;
        name("decode", "1", "serve runs decode/verify steps (LM head + final mix uploaded, step() enabled)");
        name("export_state", "0", "a decode session never hands its state off; no D2H staging");
        name("release_by_layer", "0", "nothing is exported, so there are no pieces to release per layer");
        if (!r.blocks_typed && !c.block_schedule.empty()) {
            const std::string key(flat_blocks_lever.env);
            const char* raw = r.inputs.env ? r.inputs.env(key.c_str()) : ::strix::envcfg::text(key.c_str());
            if (::strix::lever::select(flat_blocks_lever, {}, raw).on) {
                const auto cap = *std::max_element(c.block_schedule.begin(), c.block_schedule.end());
                if (c.block_schedule != std::vector<std::uint32_t>{cap}) {
                    c.block_schedule = {cap};
                    name("block_schedule", std::to_string(cap),
                         "flat blocks (qwen_cuda_flat_blocks, bitwise): no hand-off needs an early small block, and each block "
                         "re-reads the expert weights once; off: STRIX_QRTX_FLAT_BLOCKS=0 or a typed --blocks");
                }
            }
        }
    } else if (r.release_by_layer) c.release_by_layer = *r.release_by_layer;
    {  // every entry: CLI = serve
        const std::string key(block_fit_lever.env);
        const char* raw = r.inputs.env ? r.inputs.env(key.c_str()) : ::strix::envcfg::text(key.c_str());
        c.block_fit = !r.blocks_typed && ::strix::lever::select(block_fit_lever, {}, raw).on;
    }
    {  // every entry: CLI = serve
        const std::string key(qp::moe_real_rows_lever.env);
        const char* raw = r.inputs.env ? r.inputs.env(key.c_str()) : ::strix::envcfg::text(key.c_str());
        const auto pick = ::strix::lever::select(qp::moe_real_rows_lever, {}, raw);
        c.moe_real_rows = pick.on;
        out.moe_real_rows_origin = ::strix::lever::origin_name(pick.origin);
    }
    if (entry != Entry::serve && c.model_gguf.empty() && !default_model.empty()) {
        c.model_gguf = std::string(default_model);
        name("model_gguf", c.model_gguf, "prefill service default file name when --model is absent");
    }
    if (c.model_gguf.empty() || c.tables_dir.empty() || c.rope_table.empty())
        throw std::invalid_argument("--model, --tables and --rope-table are required");
    if (c.block_schedule.empty() ||
        c.final_block_rows > *std::max_element(c.block_schedule.begin(), c.block_schedule.end()))
        throw std::invalid_argument("invalid final block rows");
    return out;
}

inline ::strix::serve::Json forms_json(const qwen_prefill::Config::Forms& f, bool moe_route_bf16) {
    using J = ::strix::serve::Json;
    auto forms = J::object();
    forms.set("hc_mma", J(f.hc_mma));
    forms.set("mtp_mma", J(f.mtp_mma));
    forms.set("qsa_mma", J(f.qsa_mma));
    forms.set("moe_mma", J(f.moe_mma));
    forms.set("ple_a8", J(f.ple_a8));
    forms.set("moe_route_bf16", J(moe_route_bf16));
    return forms;
}

// Whether the resolved run IS the TURBO bundle (cp::kTurboBundle), from the effective values of either entry.
inline cp::BundleStatus turbo_bundle(const Resolved& r) {
    return cp::turbo_bundle_status(r.choice, r.config.expert_arithmetic == qwen_packs::ExpertArithmetic::gsq_i8_turbo,
                                   r.config.expert_narrow == qwen_packs::ExpertArithmetic::gsq_bf16_mma,
                                   !r.config.turbo_onfly_nvfp4);
}

inline std::string effective_config_json(const Resolved& r) {
    using J = ::strix::serve::Json;
    const auto& c = r.config;
    auto out = J::object();
    out.set("entry", J(entry_name(r.entry)));
    out.set("profile", J(pr::name(r.resolution.profile)));
    out.set("kv", ::strix::serve::parse_json(r.kv.json()));
    out.set("arithmetic_contract", J(r.contract));
    out.set("expert_arithmetic",J(std::string(qwen_packs::expert_arithmetic_name(c.expert_arithmetic))));
    if (c.expert_narrow) {  // per-regime experts (TURBO): absent otherwise, so exact/fast receipts keep their bytes
        auto regime = J::object();
        regime.set("prefill_blocks_min_rows", J(static_cast<std::uint64_t>(c.expert_regime_min_rows)));
        regime.set("prefill_blocks", J(std::string(qwen_packs::expert_arithmetic_name(c.expert_arithmetic))));
        regime.set("windows_and_smaller_blocks", J(std::string(qwen_packs::expert_arithmetic_name(*c.expert_narrow))));
        regime.set("rule", J(expert_regime(c)));
        out.set("expert_regime", std::move(regime));
    }
    out.set("forms", forms_json(c.forms, c.moe_route_bf16.value_or(false)));
    auto route = J::object();
    route.set("requested", J(c.route_policy->requested));
    route.set("effective", J(c.route_policy->enabled));
    route.set("reason", J(std::string(c.route_policy->reason)));
    out.set("route_storage", std::move(route));
    if (r.resolution.is_turbo()) {  // absent outside turbo: every other effective config keeps its bytes
        auto turbo = J::object();
        turbo.set("onfly_nvfp4", J(c.turbo_onfly_nvfp4));
        turbo.set("cache_namespace", J(c.turbo_namespace));
        turbo.set("decode_arithmetic", J(std::string(c.turbo_onfly_nvfp4 ? "nvfp4-a8-v1 (dense Q8_0 roles converted at load)" : "native pack")));
        out.set("turbo_weights", std::move(turbo));
    }
    // FAST bundle (kFastBundle): under --profile fast, or whenever the effective values are the bundle; absent otherwise,
    // so every other effective config keeps its bytes.
    const auto bundle = cp::fast_bundle_status(r.choice, c.expert_arithmetic == qwen_packs::ExpertArithmetic::gsq_bf16_mma);
    if (r.resolution.profile == pr::Name::fast || !bundle.id.empty()) {
        auto fb = J::object();
        fb.set("id", J(bundle.id.empty() ? std::string("none") : bundle.id));
        fb.set("detail", J(bundle.detail));
        out.set("fast_bundle", std::move(fb));
    }
    // TURBO bundle (kTurboBundle): under --profile turbo, or whenever the effective values are the bundle; absent otherwise.
    const auto tbundle = turbo_bundle(r);
    if (r.resolution.is_turbo() || !tbundle.id.empty()) {
        auto tb = J::object();
        tb.set("id", J(tbundle.id.empty() ? std::string("none") : tbundle.id));
        tb.set("detail", J(tbundle.detail));
        out.set("turbo_bundle", std::move(tb));
    }
    out.set("form_identities", J(cp::form_identities(r.choice)));
    out.set("resolution", ::strix::serve::parse_json(r.resolution.json()));
    auto notices = J::array();
    for (const auto& n : r.resolution.notices()) notices.push(J(n));
    out.set("notices", std::move(notices));
    auto engine = J::object();
    engine.set("max_positions", J(static_cast<std::uint64_t>(c.max_positions)));
    engine.set("tail_large", J(c.tail_large));
    engine.set("final_block_rows", J(static_cast<std::uint64_t>(c.final_block_rows)));
    engine.set("defer_prompt_mtp", J(c.defer_prompt_mtp));
    engine.set("block_fit", J(c.block_fit));
    {  // requested here; the loaded engine's moe_rows receipt adds the tiered fallback (still padded) and window counts
        const auto p = qp::moe_rows_policy(c.moe_real_rows, false);
        auto rows = J::object();
        rows.set("lever", J(std::string(qp::moe_real_rows_lever.name)));
        rows.set("requested", J(p.requested));
        rows.set("source", J(r.moe_real_rows_origin));
        rows.set("qualified_default", J(qp::moe_real_rows_lever.default_on));
        rows.set("reason", J(std::string(p.reason)));
        rows.set("tiered", J(std::string(qp::moe_rows_policy(c.moe_real_rows, true).reason)));
        rows.set("restart", J(std::string("read once at engine build; A/B needs separate processes")));
        engine.set("moe_window_rows", std::move(rows));
    }
    auto blocks = J::array();
    for (const auto b : c.block_schedule) blocks.push(J(static_cast<std::uint64_t>(b)));
    engine.set("block_schedule", std::move(blocks));
    engine.set("decode", J(c.decode));
    engine.set("export_state", J(c.export_state));
    engine.set("release_by_layer", J(c.release_by_layer));
    engine.set("mtp_sidecar", J(!c.mtp_sidecar_gguf.empty()));
    engine.set("draft_vocab", J(!c.draft_vocab.empty()));
    out.set("engine", std::move(engine));
    auto diffs = J::array();
    for (const auto& d : r.differences) {
        auto row = J::object();
        row.set("name", J(d.name));
        row.set("value", J(d.value));
        row.set("reason", J(d.reason));
        diffs.push(std::move(row));
    }
    out.set("serve_differences", std::move(diffs));
    return out.dump();
}

inline constexpr std::uint64_t fnv64(std::string_view s) {
    std::uint64_t h = 14695981039346656037ULL;
    for (const char ch : s) { h ^= static_cast<unsigned char>(ch); h *= 1099511628211ULL; }
    return h;
}
inline cp::Choice choice_of(const qp::Config& c) {
    cp::Choice choice{c.moe_route_bf16.value_or(false), c.forms.hc_mma, c.forms.mtp_mma, c.forms.qsa_mma, c.forms.moe_mma};
    choice.ple_a8 = c.forms.ple_a8;
    return choice;
}
// Resolved::contract recomputed from a resolved Config (a running service holds only the Config).
inline std::string config_arithmetic_contract(const qp::Config& c) {
    return with_expert_arithmetic(cp::arithmetic_contract(choice_of(c)), c);
}
// Routed-expert arithmetic classes the loaded engine MAY dispatch (qwen_moe enqueue_routed), from load-time facts only.
// native_experts (Prefill::native_experts) is true when ANY layer has native expert types: it proves the native IQ FP32
// class is present (independent of --moe-mma) but cannot exclude A8 layers, so the A8 classes stay possible. When it
// is false no layer is native and every layer takes the A8 route. --moe-mma is a requested form, not an execution
// fact: prefill windows request it (run_block passes moe_mma && !window), decode windows never do and enqueue_routed
// may fall back on decode rows, so a request for MMA keeps the four-chain class possible. Names are the wfmt capture
// classes (tests/cuda/qwen_moe_wfmt_capture.inc).
inline std::vector<std::string_view> expert_possible_classes(qwen_packs::ExpertArithmetic expert, bool moe_mma,
                                                             bool native_experts) {
    std::vector<std::string_view> out;
    if (native_experts) out.push_back("cuda-native-iq-fp32@1");
    if (expert == qwen_packs::ExpertArithmetic::gsq_a8_dot4) {
        out.push_back(qwen_packs::gsq_a8_contract);
    } else {
        if (moe_mma) out.push_back("cuda-moe-mma@1");
        out.push_back("cuda-four-chain-imma@1");
    }
    return out;
}
// Status receipt from loaded facts (Prefill::weight_pack_id / native_experts / moe_rows) plus the declared form. The
// scope is load-time presence: the class each window executed is reported per run (Timing::executed_forms) and is
// not observed here. Reporting only: the declared contract, numerical identity and hand-off ids keep their bytes.
inline ::strix::serve::Json expert_dispatch_json(const qp::Config& c, std::string_view weight_pack, bool native_experts,
                                                 const qp::MoeRowsPolicy& rows) {
    using J = ::strix::serve::Json;
    auto out = J::object();
    out.set("engine", J(std::string("cuda-qwen-moe")));
    out.set("scope", J(std::string("loaded-presence")));
    auto classes = J::array();
    for (const auto cls : expert_possible_classes(c.expert_arithmetic, c.forms.moe_mma, native_experts))
        classes.push(J(std::string(cls)));
    out.set("possible_classes", std::move(classes));
    out.set("native_class", J(std::string(native_experts ? "present" : "absent")));
    out.set("layer_mix", J(std::string(native_experts ? "unknown" : "a8-route-only")));
    out.set("executed_class", J(std::string("not-observed")));
    out.set("declared_expert_arithmetic", J(std::string(qwen_packs::expert_arithmetic_name(c.expert_arithmetic))));
    if (c.expert_narrow) out.set("declared_expert_regime", J(expert_regime(c)));
    out.set("declared_moe_mma", J(c.forms.moe_mma));
    out.set("declared_contract", J(config_arithmetic_contract(c)));
    auto source = J::object();
    source.set("weight_pack", J(std::string(weight_pack)));
    source.set("native_experts", J(native_experts));
    source.set("moe_window_rows_effective", J(rows.effective));
    out.set("source", std::move(source));
    out.set("reason", J(std::string(
        (native_experts
            ? "at least one layer has native expert types (native IQ FP32, independent of --moe-mma); the loaded fact "
              "is any-layer, so A8 layers are not excluded"
            : "no layer has native expert types: every layer takes the A8 route") +
        std::string(c.expert_arithmetic == qwen_packs::ExpertArithmetic::gsq_a8_dot4
            ? "; A8 layers run the explicit GSQ A8 dot4 contract"
            : c.forms.moe_mma
                ? "; --moe-mma is requested for prefill windows only (decode windows and the decode-row fallback run "
                  "four-chain IMMA)"
                : "; A8 layers run four-chain IMMA") +
        "; per-window executed class is not observed by this receipt")));
    return out;
}
// Preserve historical route-only bytes; every enabled prefill form changes the snapshot target.
inline ::strix::cuda::qwen_route::Identity numerical_identity(const qp::Config& c) {
    namespace route = ::strix::cuda::qwen_route;
    const bool bf16 = c.moe_route_bf16.value_or(false);
    auto id = route::identity(bf16, bf16);
    const auto forms = cp::form_identities(choice_of(c));
    if (!forms.empty()) {
        id.prefill = "cuda-qwen-a8-prefill+forms-v1";
        id.hash = route::hash_part(id.hash, forms);
    }
    if(c.expert_arithmetic==qwen_packs::ExpertArithmetic::gsq_a8_dot4) {
        id.prefill="cuda-qwen-gsq-a8-prefill-v1";
        id.hash=route::hash_part(id.hash,qwen_packs::gsq_a8_contract);
    }
    if(c.expert_arithmetic==qwen_packs::ExpertArithmetic::gsq_bf16_mma) {
        id.prefill="cuda-qwen-gsq-bf16mma-prefill-v1";
        id.hash=route::hash_part(id.hash,qwen_packs::gsq_bf16_contract);
    }
    if(c.expert_arithmetic==qwen_packs::ExpertArithmetic::gsq_i8_turbo) {
        id.prefill="cuda-qwen-gsq-i8turbo-prefill-v1";
        id.hash=route::hash_part(id.hash,qwen_packs::gsq_i8_turbo_contract);
    }
    if(c.expert_narrow) {  // per-regime experts: the prefill state depends on the rule (block rows decide the arithmetic)
        id.prefill="cuda-qwen-gsq-expert-regime-prefill-v1";
        id.hash=route::hash_part(id.hash,expert_regime(c));
    }
    return id;
}
// Hand-off producer id: the exact CUDA path keeps its historical id bytes; a FAST arithmetic folds its contract in, so an
// exact-identity node refuses a FAST state (different producer id).
inline std::uint64_t handoff_numerical_id(std::string_view exact_name, const cp::Choice& choice) {
    if (!choice.fast()) return fnv64(exact_name);
    return fnv64(std::string(exact_name) + "|" + cp::arithmetic_contract(choice) + "|" + cp::form_identities(choice));
}
inline std::uint64_t handoff_numerical_id_config(std::string_view exact_name,const qp::Config& config) {
    if(config.expert_arithmetic==qwen_packs::ExpertArithmetic::incumbent)
        return handoff_numerical_id(exact_name,choice_of(config));
    return fnv64(std::string(exact_name)+"|"+cp::arithmetic_contract(choice_of(config))+"|"+
                 cp::form_identities(choice_of(config))+"|"+(config.expert_narrow?expert_regime(config):
                 std::string(qwen_packs::expert_arithmetic_name(config.expert_arithmetic))));
}
}  // namespace strix::cuda::qwen_options
