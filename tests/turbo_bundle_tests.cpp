// TURBO bundles (turbo column of each family's profile table): what every family's turbo resolution selects, what it
// holds pending, and that default / fast stay untouched. The Qwen rows go through both entry points (CLI option reader
// and the serve-style Inputs) and must resolve to one fingerprint.
#include "strix/deepseek41_pair_profile_table.hpp"
#include "strix/engine_profile.hpp"
#include "strix/glm_pair_profile_table.hpp"
#include "strix/profiles/qwen.hpp"
#include "minimax/engine/mm_profile_table.hpp"
#include "minimax/engine/mm_defaults.hpp"
#include "glm/prefill/glm_pf_attn_wmma.hpp"

#include <cstdio>
#include <map>
#include <string>

namespace {
namespace p = strix::profile;
int failures = 0;
void require(bool value, const char* message) {
    if (!value) { std::fprintf(stderr, "FAIL: %s\n", message); ++failures; }
}
p::Lookup env_of(const std::map<std::string, std::string>& vars) {
    return [&vars](const char* key) -> const char* {
        const auto it = vars.find(key);
        return it == vars.end() ? nullptr : it->second.c_str();
    };
}
template <class R>
const p::Entry* find(const R& r, const char* name) { return r.find(name); }

void generic(const p::Engine& engine, p::Name profile, const std::map<std::string, std::string>& vars, p::Resolution& out) {
    p::Inputs in;
    in.env = env_of(vars);
    in.profile = profile;
    out = p::resolve(engine, in);
}

void qwen_rows() {
    namespace q = strix::qwen::profile;
    const std::map<std::string, std::string> vars;
    q::Inputs in;
    in.env = env_of(vars);
    const auto turbo = q::resolve(p::Name::turbo, in);
    const auto def = q::resolve(p::Name::standard, in);
    const auto fast = q::resolve(p::Name::fast, in);
    for (const char* lever : {"index_proj_wmma", "index_score_wmma", "ple_rows_f16", "gdn_gates_f16"}) {
        const auto* e = find(turbo, lever);
        require(e != nullptr && e->value == "1" && e->status == p::Status::provisional, lever);
        require(find(def, lever) == nullptr && find(fast, lever) == nullptr, "turbo-only row is absent outside turbo");
    }
    for (const char* lever : {"qsa_budget", "moe_route_bf16", "gdn_fp16_chunk"}) {
        require(find(turbo, lever) != nullptr && find(turbo, lever)->value == find(fast, lever)->value, "turbo inherits the FAST cell");
        require(find(def, lever) == nullptr || find(def, lever)->status == p::Status::builtin, "default keeps the lever off");
    }
    const auto* prune = find(turbo, "moe_route_prune");
    require(prune != nullptr && prune->status == p::Status::pending && prune->value == "off",
            "TURBO reports pending route pruning when the prompt length is unknown");
    q::Inputs measured = in;
    measured.regime = {p::Phase::prefill, 8192, 1, "gfx1151"};
    const auto turbo_8k = q::resolve(p::Name::turbo, measured);
    const auto* cap9 = find(turbo_8k, "moe_route_prune");
    require(cap9 != nullptr && cap9->status == p::Status::pending && cap9->value == "off",
            "TURBO keeps cap9 pending at 8K until its composition has evidence");
    // One resolver: CLI spelling and serve-style inputs give the same fingerprint.
    const char* argv[] = {"app", "--profile", "turbo"};
    std::optional<p::Name> name;
    q::Inputs cli;
    int i = 1;
    require(q::take_option(3, const_cast<char**>(argv), i, name, cli), "CLI takes --profile turbo");
    const auto via_cli = q::resolve(name.value_or(p::Name::standard), cli);
    q::Inputs serve;
    serve.env = env_of(vars);
    require(via_cli.fingerprint() == q::resolve(p::Name::turbo, serve).fingerprint(), "CLI and serve share one qwen turbo fingerprint");
    // Opting a member out: the env value 0 wins and drops it from the effective set.
    const std::map<std::string, std::string> leave{{"STRIX_PREFILL_PLE_ROWS_F16", "0"}};
    q::Inputs one_out;
    one_out.env = env_of(leave);
    const auto without = q::resolve(p::Name::turbo, one_out);
    require(find(without, "ple_rows_f16")->status == p::Status::off && without.fingerprint() != turbo.fingerprint(),
            "a single lever can be left out");
}

void families() {
    using strix::deepseek41::pair_defaults::profile_table;
    const p::Engine ds{"deepseek41-pair", "", profile_table, false, "ds41-turbo-v1", {}, {},
                      strix::deepseek41::pair_defaults::ladder_rows};  // the production engine (its incumbent grants)
    const std::map<std::string, std::string> none;
    p::Resolution r;
    generic(ds, p::Name::turbo, none, r);
    require(find(r, "d1_prefill")->value == "0" && find(r, "d1_prefill")->status == p::Status::pending,
            "DeepSeek TURBO defers D1 until a measured request");
    for (const auto tier : {p::Name::fast, p::Name::turbo}) {
        p::Inputs in;
        in.profile = tier;
        in.env = env_of(none);
        in.regime = {p::Phase::prefill, 8192, 1, "gfx1151", p::kBandProbeExecution};
        const auto measured = p::resolve(ds, in);
        require(find(measured, "d1_prefill")->value == "1", "DeepSeek TURBO inherits the measured FAST D1 row");
        require(find(measured, "decoder_replay")->value == "256", "DeepSeek TURBO inherits the measured FAST R256 row");
    }
    require(find(r, "moe_iu4")->value == "0" && find(r, "moe_iu4")->status == p::Status::builtin,
            "DeepSeek IU4 is not a TURBO row (measured slower than FAST)");
    std::size_t engram_rows = 0;
    for (const auto& row : profile_table)
        if (std::string_view(row.env) == "STRIX_DS41_FAST_ENGRAM") ++engram_rows;
    require(engram_rows == 1 && find(r, "engram_wmma") == nullptr, "one real Engram row per env");
    const auto* engram = find(r, "engram_f16wmma");
    require(engram != nullptr && engram->value == "0" && engram->lever->value[3].empty(),
            "DeepSeek startup keeps explicit Engram off");
    require(find(r, "decoder_replay") != nullptr && find(r, "decoder_replay")->status == p::Status::pending, "ds41 replay pending");
    generic(ds, p::Name::fast, none, r);
    require(find(r, "attention_f16wmma")->value == "0", "ds41 startup cannot select standalone attention");
    generic(ds, p::Name::standard, none, r);
    require(find(r, "engram_f16wmma") != nullptr && find(r, "engram_f16wmma")->status == p::Status::builtin, "ds41 default unchanged");

    const p::Engine mm{"minimax-pair", "", strix::minimax::defaults::profile_table, false, "minimax-turbo-v1", {}, {},
                       strix::minimax::defaults::ladder_rows};  // the production engine (its incumbent grants)
    generic(mm, p::Name::turbo, none, r);
    require(strix::minimax::defaults::profile_table.size() == 11,
            "minimax retains the IU4 rows, the FAST router rows, the TP2 exchange row and the dense tail opt-in");
    require(find(r, "router_wmma")->value == "1" && p::applied(find(r, "router_wmma")->status),
            "minimax turbo runs the fast router member (TURBO nests FAST)");
    for (const char* lever : {"idx_wmma", "idx_wmma_layers", "router_wmma_layers", "pf_dense_tail_fixed"}) {
        const auto* e = find(r, lever);
        // turbo cell empty or the builtin
        require(e != nullptr && e->status == p::Status::builtin && e->value == e->lever->builtin &&
                    (std::string_view(e->lever->value[3]).empty() || e->lever->value[3] == e->lever->builtin) &&
                    !e->lever->runnable[3],
                "MiniMax companion rows keep their builtin values");
    }
    // TURBO nests FAST: the pending TURBO flash proposal (mode 1 on 0-59) leaves the FAST member (mode 1 on its FAST
    // layer set); only the 0-59 layer proposal stays pending.
    require(find(r, "msa_flash")->value == "1" && p::applied(find(r, "msa_flash")->status) &&
                find(r, "msa_flash_layers")->status == p::Status::pending &&
                find(r, "msa_flash_layers")->value == std::string(strix::minimax::defaults::kMsaFlashLayers),
            "minimax turbo runs the FAST flash member on the FAST layer set; 0-59 stays pending");
    require(find(r, "iu4_layers")->status == p::Status::pending && find(r, "iu4_layers")->value == "none",
            "minimax IU4 layer set pending (TURBO candidate without a timing row)");
    require(find(r, "model_format")->value == "exl3" && find(r, "iu4_a8")->value == "0", "minimax IU4 format rows explicit-only");
    generic(mm, p::Name::fast, none, r);
    // fast = the qualified bundle mm-fast-pf-v2: its two members' cells (the index rows stay explicit-only, above).
    require(find(r, "msa_flash")->status == p::Status::set && find(r, "msa_flash")->value == "1" &&
                find(r, "router_wmma")->status == p::Status::set && find(r, "router_wmma")->value == "1",
            "minimax fast = bundle members msa_flash + router_wmma");

    namespace md = strix::minimax::defaults;
    const std::map<std::string, std::string> mm_env{{md::kProductProfileEnv, "turbo"}};
    const auto admitted = md::profile_resolution(env_of(mm_env));
    // Review r75 P1: the actual prefill policy of TURBO is the admitted FAST bundle (both members, the FAST flash layer
    // set), and every control that turns the bundle off in FAST turns it off in TURBO.
    const auto same_policy = [&](std::map<std::string, std::string> extra, const char* what) {
        auto fast_env = extra, turbo_env = extra;
        fast_env[md::kProductProfileEnv] = "fast";
        turbo_env[md::kProductProfileEnv] = "turbo";
        std::string fast_error, turbo_error;
        md::PrefillPolicy f, t;
        try { f = md::prefill_policy(env_of(fast_env)); } catch (const std::exception& e) { fast_error = e.what(); }
        try { t = md::prefill_policy(env_of(turbo_env)); } catch (const std::exception& e) { turbo_error = e.what(); }
        require(fast_error.empty() == turbo_error.empty(), what);
        require(f.msa_flash == t.msa_flash && f.msa_flash_layers == t.msa_flash_layers && f.router_wmma == t.router_wmma &&
                    f.router_wmma_layers == t.router_wmma_layers && f.policies == t.policies && f.canonical == t.canonical &&
                    f.bundle_column == t.bundle_column && f.bundle_from_profile == t.bundle_from_profile,
                what);
        return t;
    };
    const auto execution = same_policy({}, "MiniMax TURBO prefill policy equals FAST (default)");
    require(execution.bundle_from_profile && execution.msa_flash == 1 && execution.router_wmma &&
                execution.msa_flash_layers == md::kMsaFlashLayers && find(admitted, "router_wmma")->value == "1",
            "MiniMax TURBO dispatches the whole FAST bundle (flash on the FAST layer set + router)");
    require(find(admitted, "msa_flash")->value == std::to_string(execution.msa_flash),
            "MiniMax actual adapter receipt matches engine flash mode");
    require(find(admitted, "msa_flash_layers")->value == execution.msa_flash_layers,
            "MiniMax actual adapter receipt matches engine layer set");
    const auto off = same_policy({{"STRIX_MM_FAST_PREFILL", "0"}}, "MiniMax TURBO bundle switch 0 equals FAST");
    require(off.msa_flash == 0 && !off.router_wmma, "MiniMax STRIX_MM_FAST_PREFILL=0 turns both members off in TURBO");
    const auto i4 = same_policy({{"STRIX_MMX_MODEL_FORMAT", "fast_i4"}}, "MiniMax TURBO on FAST_I4 equals FAST");
    require(i4.msa_flash == 0 && !i4.router_wmma, "MiniMax FAST_I4 admits no bundle member in TURBO");
    bool refused = false;
    try {
        (void)md::prefill_policy(env_of({{md::kProductProfileEnv, "turbo"}, {md::kBaselineEnv, "1"}}));
    } catch (const std::invalid_argument&) { refused = true; }
    require(refused, "MiniMax --baseline --profile turbo is refused like --baseline --profile fast");
    (void)same_policy({{md::kBaselineEnv, "1"}}, "MiniMax baseline: TURBO and FAST refuse alike");
    std::map<std::string, std::string> exported, fast_exported;
    p::apply(admitted, [&](const char* key, const char* value) { exported[key] = value; return true; });
    p::apply(md::profile_resolution(env_of({{md::kProductProfileEnv, "fast"}})),
             [&](const char* key, const char* value) { fast_exported[key] = value; return true; });
    for (const char* key : {"STRIX_MM_PF_MSA_FLASH", "STRIX_MM_PF_MSA_FLASH_LAYERS", "STRIX_MM_PF_ROUTER_WMMA",
                            "STRIX_MM_PF_ROUTER_WMMA_LAYERS"})
        require(exported.contains(key) == fast_exported.contains(key) &&
                    (!exported.contains(key) || exported[key] == fast_exported[key]),
                "MiniMax TURBO exports the FAST bundle members exactly as FAST (the 0-59 proposal never escapes)");
    require(!exported.contains("STRIX_MM_PF_MSA_FLASH_LAYERS"),
            "MiniMax never exports the pending wider layer set");
    p::Inputs allow;
    allow.profile = p::Name::turbo;
    allow.allow_pending = true;
    const auto pending = p::resolve(mm, allow);
    require(find(pending, "msa_flash_layers")->status == p::Status::pending &&
                find(pending, "msa_flash_layers")->value != "0-59",
            "allow-pending cannot bypass the missing MiniMax 0-59 adapter");

    generic(strix::glm::rank_defaults::product_profile_engine, p::Name::turbo, none, r);
    require(find(r, "pf_wmma")->value == "0", "GLM with no known workload keeps the FAST prefill route");
    require(find(r, "kda_state16")->status == p::Status::pending && find(r, "router_wmma")->status == p::Status::pending &&
                find(r, "idxproj_wmma")->status == p::Status::pending, "GLM router/indexer and KDA16 pending");
}
}  // namespace

int main(int argc, char** argv) {
    try {
        if (argc >= 2 && std::string_view(argv[1]) == "--minimax-cli")
            return strix::minimax::defaults::print_or_apply("rank", argc - 1, argv + 1) ? 0 : 1;
        if (argc >= 3 && std::string_view(argv[1]) == "--glm-arm") {
            for (const char* key : {"STRIX_GLM_PF_WMMA", "STRIX_GLM_PF_ATTN_WMMA", "STRIX_GLM_PF_KDA_WMMA"})
                require(p::export_environment(key, ""), "clear inherited GLM switch");
            for (int i = 3; i < argc; ++i) {
                const std::string assignment = argv[i];
                const auto eq = assignment.find('=');
                require(eq != std::string::npos && p::export_environment(assignment.substr(0, eq).c_str(), assignment.substr(eq + 1).c_str()), "set arm switch");
            }
            const std::string_view arm = argv[2];
            require(strix::glm::pf::wmma_x::enabled() == (arm != "default" && arm != "no-pf_wmma"), "GLM dense execution switch");
            require(strix::glm::pf::attn_wmma::enabled() == (arm != "default" && arm != "no-attn_wmma"), "GLM attention execution switch");
            require(strix::glm::pf::kda_wmma::enabled() == (arm != "default" && arm != "no-kda_wmma"), "GLM KDA execution switch");
            return failures == 0 ? 0 : 1;
        }
        qwen_rows();
        families();
    } catch (const std::exception& e) {
        std::fprintf(stderr, "FAIL: unexpected exception: %s\n", e.what());
        return 1;
    }
    if (failures != 0) return 1;
    std::puts("turbo bundle tests: ok");
    return 0;
}
