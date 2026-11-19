#include "../src/core/dense_linear.hpp"
// strix-cuda-qwen35-spec: exact block drafting (copy / native MTP chain / hybrid blocks, up to 32 verify rows) on the qwen35
// CUDA engine. Verifies output == serial decode token for token, reports accepted tokens per step (mean/p50/p90), tok/s
// average and best 1 s window, and the engine cost table verify_ms(R) / MTP ms / lm_head share.
// Home: specific (CLI binding of the qwen35 engine); drafting logic in src/spec (shared), glue in qwen35/cuda_spec.hpp.
#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <fstream>
#include <sstream>
#include <stdexcept>
#include <string>
#include <vector>

#include "strix/device/cuda_enumerator.hpp"
#include "strix/cuda_core/attention_row_walk.hpp"
#include "strix/cuda_core/dense.hpp"
#include "strix/qwen35/cuda_engine.hpp"
#include "strix/qwen35/cuda_spec.hpp"
#include "strix/qwen35/prefill_controls.hpp"
#include "strix/profiles/qwen_dense_cuda.hpp"
#include "strix/spec/draft_rows.hpp"

#include <memory>
#include <optional>
#include <span>
#include <thread>

namespace q = strix::qwen35;
namespace sp = strix::spec;
using Clock = std::chrono::steady_clock;

namespace {
double now_ms() { return std::chrono::duration<double, std::milli>(Clock::now().time_since_epoch()).count(); }

std::uint64_t mix(std::uint64_t x) {
    x ^= x >> 33; x *= 0xff51afd7ed558ccdULL; x ^= x >> 33; x *= 0xc4ceb9fe1a85ec53ULL; x ^= x >> 33;
    return x;
}

// Test chain: proposes the serial continuation with controlled accuracy (exercises long / partial accepts on the engine).
struct OracleChain final : sp::ChainDrafter {
    const std::vector<std::uint32_t>& ref;
    double acc;
    std::uint32_t stream_len = 1;
    OracleChain(const std::vector<std::uint32_t>& r, double a) : ref(r), acc(a) {}
    void draft(std::span<const std::uint32_t> prefix, std::uint32_t k, std::vector<std::uint32_t>& out) override {
        out.clear();
        for (std::uint32_t j = 0; j < k; ++j) {
            const std::uint64_t pos = std::uint64_t(stream_len) + prefix.size() + j;
            const bool ok = double(mix(pos * 977 + 5) % 1000) / 1000.0 < acc;
            out.push_back(pos < ref.size() && ok ? ref[pos] : 3);
        }
    }
    void accept(std::span<const std::uint32_t> e) override { stream_len += std::uint32_t(e.size()); }
};

std::vector<std::uint32_t> parse_ids(const std::string& text) {
    std::vector<std::uint32_t> v;
    std::string t;
    for (const char c : text + ",") {
        if (c != ',' && c != ' ' && c != '\n' && c != '\r' && c != '\t') t += c;
        else if (!t.empty()) { v.push_back(strix::core::parse_u32(t)); t.clear(); }
    }
    return v;
}

double quantile(std::vector<double> v, double p) {
    if (v.empty()) return 0.0;
    std::sort(v.begin(), v.end());
    return v[std::min<std::size_t>(v.size() - 1, std::size_t(p * double(v.size())))];
}
// One spec arm: controller form, draft head, chain cap (see --arms).
struct ArmSpec { std::string tag; std::uint32_t frontier = 8; bool continuation = true; double loss = 0.05; bool subset = false; std::uint32_t cap = 7;
                 bool fixed = false; };
ArmSpec parse_arm(const std::string& name, std::uint32_t chain_k) {
    ArmSpec a;
    a.cap = chain_k;
    if (!name.empty() && name.find_first_not_of("0123456789") == std::string::npos) {  // the cc_decode form (receipts cd1-cd5)
        a.frontier = strix::core::parse_u32(name);
        a.continuation = false;
        a.loss = 1.0;
        a.tag = "f" + name;
        return a;
    }
    std::string base = name;
    if (base.size() > 1 && base.back() == 'd') { a.fixed = true; base.pop_back(); }  // fixed cost model (--cost-model)
    if (base.size() > 2 && base.compare(base.size() - 2, 2, "15") == 0) { a.cap = 15; base.resize(base.size() - 2); }
    if (base == "sub") a.subset = true;
    else if (base != "new") throw std::runtime_error("--arms: N, new, sub, new15 or sub15, each optionally + d (got " + name + ")");
    a.frontier = sp::ArbiterConfig{}.frontier_every;
    a.loss = sp::ArbiterConfig{}.frontier_loss;
    a.tag = name;
    return a;
}
}  // namespace

int main(int argc, char** argv) {
    strix::device::bind_cuda_main(argc, argv);
    try {
        std::string model, synth, ids_arg, ids_file, sources = "copy,chain,hybrid", chain_hidden = "post", label = "run";
        std::optional<strix::profile::Name> profile;  // --profile exact|default|fast|quality (strix/profiles/qwen_dense_cuda.hpp)
        strix::profile::Pairs lever_cli;             // per-lever options: override the profile, recorded as explicit
        std::uint32_t n_gen = 256, prompt_len = 64, layers_override = 0, max_pos = 0, max_block = 32, chain_k = 7;
        double oracle = -1.0;
        std::uint32_t trace_n = 0, pairs = 1;
        std::string ids_out;
        std::string spec_trace;  // --spec-trace DIR: round trace per spec run (diagnostic, no effect on ids; scripts/bench/spec)
        // Controller / draft-head arms per spec run, interleaved --pairs times (A B A B ...) on the same loaded engine:
        //   N (number) = the cc_decode form: frontier probe every N passes (0 = before it), flat prior, unpriced probe, full head;
        //   new = the ArbiterConfig defaults (continuation prior, probe charged its cost), full head;
        //   sub = new + the draft-only subset head (--draft-core groups of 16 ids + --draft-pack files chosen by the prompt +
        //   prompt / emitted ids); a "15" suffix (new15, sub15) = chain cap 15. Drafts never change emitted tokens.
        //   A trailing "d" (newd, subd, new15d, sub15d) = deterministic controller: BlockConfig::fixed_costs with the work
        //   model --cost-model V1,VROW,DA,DB: verify V(r) = V1 + VROW (r - 1) ms, draft DA + DB f ms per position with
        //   f = draft head rows / vocab (1 = full head). Defaults = PRO 6000 receipts of this identity (cc_decode3-STATE.md).
        //   --cost-kv online|off|KV = the context cost of those arms (spec::set_context_cost, shared with every binder):
        //   online (default) fits it from this device's verify walls (ArbiterConfig::cost_fit), off = context-blind seeds
        //   (the previous form), KV = fixed ms per verify row per 1,024 context positions (deterministic A/B).
        double cm_v1 = 13.30, cm_vrow = 0.113, cm_da = 0.370, cm_db = 0.454;
        std::string cost_kv = "online";
        std::vector<std::string> arms{"new"};
        std::uint32_t draft_core = sp::DraftRowsConfig{}.core_groups, draft_fallback = sp::DraftRowsConfig{}.fallback_groups;
        std::string draft_pack;
        std::unique_ptr<sp::DraftRows> draft_rows;
        std::uint32_t idle_ms = 0;  // host sleep (GPU idle) before every spec run: thermal gaps inside one ticket
        bool cost_table = false, check = true, graphs = true, seed_costs = true, turbo_lm_head = true, pdl = false;
        bool prelaunch = false;  // verify pre-launch: bits unchanged; default off until its timing receipt (DEFAULT ON WHEN PROVEN)
        q::TurboWeights turbo = q::TurboWeights::off;
        for (int i = 1; i < argc; ++i) {
            const std::string a = argv[i];
            auto next = [&]() -> std::string {
                if (i + 1 >= argc) throw std::runtime_error("missing value for " + a);
                return argv[++i];
            };
            if (a == "--model") model = next();
            else if (a == "--synthetic-full") synth = next();
            else if (a == "--layers") layers_override = strix::core::parse_u32(next());
            else if (a == "--prompt-ids") ids_arg = next();
            else if (a == "--prompt-file") ids_file = next();
            else if (a == "--prompt-len") prompt_len = strix::core::parse_u32(next());
            else if (a == "--n-gen") n_gen = strix::core::parse_u32(next());
            else if (a == "--sources") sources = next();
            else if (a == "--max-block") max_block = strix::core::parse_u32(next());
            else if (a == "--chain-k") chain_k = strix::core::parse_u32(next());
            else if (a == "--oracle") oracle = std::stod(next());
            else if (a == "--cost-table") cost_table = true;
            else if (a == "--trace") trace_n = strix::core::parse_u32(next());
            else if (a == "--check") check = next() != "0";
            else if (a == "--graphs") graphs = next() != "0";
            else if (a == "--pdl") pdl = next() != "0";  // EngineOptions::pdl, as the decode CLI
            else if (a == "--prelaunch") prelaunch = next() != "0";  // verify pre-launch (BlockSpeculator::set_prelaunch)
            else if (a == "--profile") profile = strix::qwen35::profile::parse(next());  // one tier word; exact = default
            else if (a == "--split-k") lever_cli.emplace_back("split_k", next());  // exact|fast: overrides the profile
            else if (a == "--decode-attn8") lever_cli.emplace_back("decode_attn8", next());  // on|off: e4m3 decode/verify attention (FAST, identity -d8)
            else if (a == "--seed-costs") seed_costs = next() != "0";
            else if (a == "--cost-model") {
                const std::string v = next();
                if (std::sscanf(v.c_str(), "%lf,%lf,%lf,%lf", &cm_v1, &cm_vrow, &cm_da, &cm_db) != 4 || !(cm_v1 > 0.0) ||
                    !(cm_vrow >= 0.0) || !(cm_da > 0.0) || !(cm_db >= 0.0))
                    throw std::runtime_error("--cost-model V1,VROW,DA,DB (ms; V1 > 0, DA > 0)");
            }
            else if (a == "--cost-kv") {
                cost_kv = next();
                sp::ArbiterConfig probe;
                sp::set_context_cost(probe, cost_kv);  // validates now: online|off|KV
            }
            else if (a == "--chain-hidden") chain_hidden = next();
            else if (a == "--max-pos") max_pos = strix::core::parse_u32(next());
            else if (a == "--label") label = next();
            else if (a == "--frontier") arms = {std::to_string(strix::core::parse_u32(next()))};
            else if (a == "--arms") {
                arms.clear();
                const std::string v = next();
                for (std::size_t b = 0, e = 0; b <= v.size(); b = e + 1) {
                    e = v.find(',', b);
                    if (e == std::string::npos) e = v.size();
                    if (e > b) arms.push_back(v.substr(b, e - b));
                }
                if (arms.empty()) throw std::runtime_error("--arms needs values");
                for (const auto& n : arms) (void)parse_arm(n, chain_k);
            }
            else if (a == "--draft-core") draft_core = strix::core::parse_u32(next());
            else if (a == "--draft-fallback") draft_fallback = strix::core::parse_u32(next());
            else if (a == "--draft-pack") draft_pack = next();
            else if (a == "--idle-ms") idle_ms = strix::core::parse_u32(next());
            else if (a == "--pairs") { pairs = strix::core::parse_u32(next()); if (pairs == 0) throw std::runtime_error("--pairs >= 1"); }
            else if (a == "--ids-out") ids_out = next();
            else if (a == "--spec-trace") spec_trace = next();
            else if (a == "--turbo-weights") {
                const std::string v = next();
                if (v != "off" && v != "native" && v != "onfly" && v != "converted") throw std::runtime_error("--turbo-weights expects off|native|onfly|converted");
                turbo = v == "converted" ? q::TurboWeights::converted : v == "onfly" ? q::TurboWeights::onfly : v == "native" ? q::TurboWeights::native : q::TurboWeights::off;
            } else if (a == "--turbo-lm-head") {
                const std::string v = next();
                if (v != "nvfp4" && v != "bf16") throw std::runtime_error("--turbo-lm-head expects nvfp4|bf16");
                turbo_lm_head = v == "nvfp4";
            } else if (a == "--dense-arith") {
                const std::string v = next();
                namespace cc = strix::cuda_core;
                cc::set_dense_arith(v == "fp32" ? cc::DenseArith::fp32 : v == "mma2" ? cc::DenseArith::mma2
                                    : v == "mma1" ? cc::DenseArith::mma1 : cc::DenseArith::mma3);
            }
            else throw std::runtime_error("unknown argument " + a);
        }
        if (model.empty() == synth.empty())
            throw std::runtime_error("usage: (--model F.gguf | --synthetic-full bf16|q8_0 [--layers N]) [--prompt-ids a,b | --prompt-file F | --prompt-len N] "
                                     "[--n-gen N] [--sources copy,chain,hybrid|none] [--oracle ACC] [--cost-table] [--check 0|1] [--max-block 1..32] "
                                     "[--frontier N | --arms 0,8,new,sub,new15,sub15 --pairs P] [--draft-core G] [--draft-fallback G] [--draft-pack F1,F2] [--idle-ms MS] [--ids-out DIR] [--spec-trace DIR]  (--prompt-file accepts F1,F2,...: one engine load)");
        // (label, ids) per prompt; --prompt-file takes a comma list (label = file stem), each prompt runs serial + spec arms.
        std::vector<std::pair<std::string, std::vector<std::uint32_t>>> prompts;
        if (!ids_arg.empty()) prompts.emplace_back(label, parse_ids(ids_arg));
        if (!ids_file.empty()) {
            std::stringstream list(ids_file);
            for (std::string file; std::getline(list, file, ',');) {
                if (file.empty()) continue;
                std::ifstream f(file);
                if (!f) throw std::runtime_error("cannot open prompt file " + file);
                std::stringstream ss;
                ss << f.rdbuf();
                auto stem = file.substr(file.find_last_of("/\\") + 1);
                stem = stem.substr(0, stem.find('.'));
                prompts.emplace_back(ids_file.find(',') == std::string::npos && label != "run" ? label : stem, parse_ids(ss.str()));
            }
        }
        std::size_t longest = 0;
        for (const auto& p : prompts) longest = std::max(longest, p.second.size());
        std::vector<std::uint32_t> prompt = prompts.empty() ? std::vector<std::uint32_t>{} : prompts.front().second;
        q::EngineOptions opt;
        opt.use_graphs = graphs;
        opt.turbo_weights = turbo;
        opt.turbo_lm_head = turbo_lm_head;
        opt.pdl = pdl;
        // The decode CLI's resolver on the windows route: SpecEngine prefills through forward_rows windows (cuda_spec.hpp
        // prefill), so only the levers of that path are offered (split_k); prefill_arith stays exact. EXACT touches nothing.
        namespace qp = strix::qwen35::profile;
        std::size_t prompt_tokens = prompt_len;
        for (const auto& entry : prompts) prompt_tokens = std::max(prompt_tokens, entry.second.size());
        const auto resolved = qp::resolve(profile, qp::regime_of(opt, qp::Route::windows, prompt_tokens), lever_cli);
        qp::apply(opt, *resolved);
        std::fputs(qp::describe(*resolved).c_str(), stderr);
        opt.max_positions = std::max<std::uint32_t>(max_pos, std::uint32_t(std::max<std::size_t>(longest, prompt_len) + n_gen + 128 +
                                                                          (cost_table ? 4096 : 0)));
        std::unique_ptr<q::CudaEngine> E;
        if (!model.empty()) E = q::CudaEngine::from_gguf(model, opt);
        else {
            q::Config cfg;
            if (layers_override) cfg.layers = layers_override;
            E = q::CudaEngine::synthetic(cfg, synth == "q8_0" ? q::PackFormat::q8_0 : q::PackFormat::bf16, opt, 0x5eedULL);
        }
        const auto& cfg = E->config();
        std::printf("identity: %s\n", E->identity().c_str());
        // the split policy the device really runs (also for --split-k fast without a tier word)
        if (opt.turbo_weights != q::TurboWeights::off) q::print_split_policy(E->options(), stderr);
        // effective decode attention schedule (same bits in every mode; STRIX_DECODE_ATTN_ROW_SHARE=0 = one row per block)
        const auto rsm = strix::cuda_core::decode_row_share();
        std::printf("decode attention row share: %s\n", rsm == strix::cuda_core::DecodeRowShare::off         ? "off"
                                                         : rsm == strix::cuda_core::DecodeRowShare::automatic ? "automatic"
                                                                                                               : "all");
        if (prompt.empty()) {
            std::uint64_t s = 99;
            for (std::uint32_t i = 0; i < prompt_len; ++i) prompt.push_back(std::uint32_t(mix(s + i) % cfg.vocab));
            prompts.assign(1, {label, prompt});
        }
        q::SpecEngine S(*E, chain_hidden == "pre");
        const auto hid = cfg.hidden;
        (void)hid;

        auto prefill = [&](std::uint32_t& anchor) {
            E->reset();
            S.reset();
            for (std::size_t p = 0; p < prompt.size(); p += 32) {
                const auto rows = std::uint32_t(std::min<std::size_t>(32, prompt.size() - p));
                const bool last = p + rows == prompt.size();
                S.prefill(prompt.data() + p, rows, last ? 0 : prompt[p + rows], last, &anchor);
            }
        };
        // Warm every row graph (commit 0 discards: no state change) so capture cost is not in any wall.
        auto warm = [&]() {
            std::vector<std::uint32_t> t(32, 1);
            for (std::uint32_t r = 1; r <= 32; ++r) {
                E->forward_rows(t.data(), r, q::LogitRows::all);
                E->commit(0);
            }
            E->sync();
        };

        if (cost_table) {
            std::uint32_t anchor = 0;
            prefill(anchor);
            warm();
            std::vector<std::uint32_t> t(32, anchor);
            std::printf("cost table (%s, ctx %zu, graphs %d)\n", label.c_str(), prompt.size(), graphs ? 1 : 0);
            std::printf("rows verify_ms(commit full) verify_ms(commit half) tok_per_s_if_all_accepted\n");
            for (const std::uint32_t r : {1U, 2U, 3U, 4U, 6U, 8U, 12U, 16U, 24U, 32U}) {
                double full = 0, half = 0;
                const int reps = 12;
                for (int rep = 0; rep < 2 * reps; ++rep) {
                    const bool h = rep & 1;
                    const double t0 = now_ms();
                    E->forward_rows(t.data(), r, q::LogitRows::all);
                    E->commit(h ? std::max(1U, r / 2) : r);
                    E->sync();
                    (h ? half : full) += now_ms() - t0;
                }
                full /= reps;
                half /= reps;
                std::printf("%u %.3f %.3f %.1f\n", r, full, half, 1000.0 * r / full);
            }
            double none = 0, last = 0;
            for (int rep = 0; rep < 24; ++rep) {
                const bool l = rep & 1;
                const double t0 = now_ms();
                E->forward_rows(t.data(), 1, l ? q::LogitRows::last : q::LogitRows::none);
                E->commit(0);
                E->sync();
                (l ? last : none) += now_ms() - t0;
            }
            std::printf("lm_head share at R=1: none %.3f ms, last %.3f ms -> lm_head %.3f ms (%.1f %%)\n", none / 12, last / 12,
                        (last - none) / 12, 100.0 * (last - none) / last);
            if (E->has_mtp()) {
                const auto o = E->forward_rows(t.data(), 1, q::LogitRows::last);
                E->commit(1);
                E->sync();
                std::vector<double> ms(33, 0.0);
                for (const std::uint32_t r : {1U, 2U, 4U, 8U, 16U, 32U}) {
                    double sum = 0;
                    for (int rep = 0; rep < 12; ++rep) {
                        std::vector<std::uint32_t> nx(32, anchor);
                        const double t0 = now_ms();
                        E->mtp_forward(nx.data(), o.hidden_post, r == 1 ? 1 : r, 0);  // hidden rows: row 0 reused (cost only)
                        E->sync();
                        sum += now_ms() - t0;
                    }
                    std::printf("mtp_forward rows %u: %.3f ms\n", r, sum / 12);
                }
            }
            std::printf("weights %.2f GB/pass, state %.0f MB/pass\n", double(E->weight_bytes()) / 1e9, double(E->state_bytes_per_pass()) / 1e6);
            return 0;
        }

        const auto write_ids = [&](const std::string& name, const std::vector<std::uint32_t>& ids) {
            std::ofstream f(ids_out + "/" + name);
            for (std::size_t i = 0; i < ids.size(); ++i) f << (i ? "," : "") << ids[i];
            if (!f) throw std::runtime_error("cannot write " + ids_out + "/" + name);
        };
        std::size_t bad_total = 0;
        for (const auto& [prompt_label, prompt_ids] : prompts) {
            prompt = prompt_ids;
            label = prompt_label;
            std::printf("prompt[%s]: %zu tokens, n-gen %u\n", label.c_str(), prompt.size(), n_gen);
            // ---- serial reference (greedy)
            std::uint32_t anchor = 0;
            std::vector<std::uint32_t> ref;
            double serial_ms = 0;
            if (check || oracle >= 0.0) {
                prefill(anchor);
                warm();
                ref.push_back(anchor);
                std::uint32_t cur = anchor;
                const double t0 = now_ms();
                while (ref.size() < n_gen) {
                    const auto o = E->forward_rows(&cur, 1, q::LogitRows::last);
                    E->commit(1);
                    E->read_top(o, 1, &cur);
                    ref.push_back(cur);
                }
                serial_ms = now_ms() - t0;
                std::printf("serial[%s]: %u tokens in %.1f ms = %.2f tok/s (%.3f ms/token)\n", label.c_str(), n_gen, serial_ms, 1000.0 * (n_gen - 1) / serial_ms,
                            serial_ms / (n_gen - 1));
            }

            if (!ids_out.empty() && !ref.empty()) write_ids(label + ".serial.ids", ref);
            for (std::uint32_t pair = 0; pair < pairs; ++pair)
            for (const auto& arm_name : arms) {
                const ArmSpec arm = parse_arm(arm_name, chain_k);
                const std::string tag = arms.size() > 1 || pairs > 1 ? label + "/" + arm.tag + "/p" + std::to_string(pair) : label;
                // ---- speculative run
                if (idle_ms) std::this_thread::sleep_for(std::chrono::milliseconds(idle_ms));
                prefill(anchor);
                warm();
                sp::BlockConfig bc;
                bc.arbiter.frontier_every = arm.frontier;
                bc.arbiter.model_continuation = arm.continuation;
                bc.arbiter.frontier_loss = arm.loss;
                bc.max_block = max_block;
                bc.chain_max_k = arm.cap;
                bc.copy = sources.find("copy") != std::string::npos;
                bc.chain = sources.find("chain") != std::string::npos;
                bc.hybrid = sources.find("hybrid") != std::string::npos;
                sp::ChainDrafter* mtp_chain = S.has_chain() ? &S.make_chain() : nullptr;
                OracleChain oracle_chain(ref, oracle);
                sp::ChainDrafter* chain = nullptr;
                if (oracle >= 0.0) chain = &oracle_chain;
                else chain = mtp_chain;
                if (sources == "none") { bc.copy = bc.chain = bc.hybrid = false; }
                if (arm.subset && !draft_rows) {
                    sp::DraftRowsConfig dc;
                    dc.vocab = E->config().vocab;
                    dc.group = 16;  // one NVFP4 tile of the head
                    dc.core_groups = draft_core;
                    dc.fallback_groups = draft_fallback;
                    for (std::size_t b = 0, e = 0; b < draft_pack.size(); b = e + 1) {
                        e = draft_pack.find(',', b);
                        if (e == std::string::npos) e = draft_pack.size();
                        if (e > b) dc.packs.push_back(sp::DraftRows::read_ids(draft_pack.substr(b, e - b), dc.vocab));
                    }
                    draft_rows = std::make_unique<sp::DraftRows>(std::move(dc));
                }
                if (arm.subset) {
                    draft_rows->begin(prompt);
                    draft_rows->admit(std::span<const std::uint32_t>(&anchor, 1));
                }
                E->set_draft_rows(arm.subset ? draft_rows.get() : nullptr);
                double fixed_draft_ms = 0.0;
                if (arm.fixed) {  // the work model prices every decision; walls are reported, never learned
                    const double f = arm.subset ? double(draft_rows->rows()) / double(E->config().vocab) : 1.0;
                    fixed_draft_ms = cm_da + cm_db * f;
                    bc.fixed_costs = true;
                    bc.chain_seed_flush_ms = bc.chain_seed_step_ms = fixed_draft_ms;
                    sp::set_context_cost(bc.arbiter, cost_kv);  // seeded model only (fixed_costs -> learn_costs = false)
                }
                sp::BlockDrafter drafter(bc, chain);
                if (arm.fixed) {
                    for (std::uint32_t r = 1; r <= bc.max_block; ++r) drafter.seed_verify_ms(r, cm_v1 + cm_vrow * double(r - 1));
                    std::printf("costs[%s]: fixed model V(r) = %.3f + %.4f (r-1) ms, draft %.3f ms per position", tag.c_str(),
                                cm_v1, cm_vrow, fixed_draft_ms);
                    const auto& ac = drafter.arbiter().config();
                    if (ac.row_kv_ms_per_kpos > 0.0) std::printf(", + %.4f r ms per 1K context positions", ac.row_kv_ms_per_kpos);
                    std::printf(", context cost %s\n", sp::context_cost_name(ac).c_str());
                } else if (seed_costs) {  // engine's measured verify wall per window shape (commit full), 3 reps each
                    std::vector<std::uint32_t> t(32, anchor);
                    for (const std::uint32_t r : {1U, 2U, 4U, 8U, 16U, 32U}) {
                        double sum = 0;
                        for (int rep = 0; rep < 3; ++rep) {
                            const double t0 = now_ms();
                            E->forward_rows(t.data(), r, q::LogitRows::all);
                            E->commit(0);
                            E->sync();
                            sum += now_ms() - t0;
                        }
                        drafter.seed_verify_ms(r, sum / 3);
                    }
                }
                drafter.commit(prompt);
                std::vector<std::uint32_t> out{anchor};
                drafter.commit(std::span<const std::uint32_t>(out.data(), 1));
                sp::BlockSpeculator spec(drafter, S);
                std::unique_ptr<sp::RoundTraceWriter> round_trace;
                if (!spec_trace.empty()) {
                    std::string file = tag;
                    std::replace(file.begin(), file.end(), '/', '_');
                    round_trace = std::make_unique<sp::RoundTraceWriter>(spec_trace + "/" + file + ".srt", label);
                    spec.set_round_sink(round_trace.get());
                }
                spec.set_prelaunch(prelaunch);
                spec.start(anchor);
                std::vector<double> tick_ms{0.0};  // wall at each step end (relative), tokens cumulative in cum
                std::vector<std::uint32_t> cum{1};
                std::vector<double> per_step;      // emitted per step
                std::uint64_t by_source[3] = {0, 0, 0};
                // Per step: verify_ms = verify + commit wall, draft_ms = chain wall (StepRecord); host = step wall - both (decide,
                // lookup, bookkeeping, launch gaps between them). Model passes by drafted depth k.
                double sum_verify = 0, sum_draft = 0;
                std::vector<std::uint64_t> k_hist(33, 0);
                const auto calls0 = S.chain_calls(), rows0 = S.chain_rows();
                const double t0 = now_ms();
                while (out.size() < n_gen) {
                    sp::StepParams p;
                    p.remaining = n_gen - std::uint32_t(out.size());
                    p.max_rows = 32;
                    p.context = E->committed();
                    const auto r = spec.step(out, p);
                    if (r.emitted == 0) break;
                    if (arm.subset) draft_rows->admit(std::span<const std::uint32_t>(out.data() + out.size() - r.emitted, r.emitted));
                    if (per_step.size() < trace_n) {
                        const auto& A = drafter.arbiter().last_arms();
                        const auto& Sc = drafter.arbiter().last_arm_scores();
                        std::printf("T%zu src%u rows%u prop%u acc%u em%u ver%.2f dr%.2f | ser %.4f cpy k%u e%.2f ms%.2f %.4f | mdl k%u e%.2f ms%.2f %.4f probe%d | cc1 %.2f cc2 %.2f\n",
                                    per_step.size(), unsigned(r.source), r.rows, r.proposed, r.accepted, r.emitted, r.verify_ms, r.draft_ms,
                                    Sc[0], A[1].k, A[1].expected_tokens, A[1].expected_ms, Sc[1], A[2].k, A[2].expected_tokens, A[2].expected_ms,
                                    Sc[2], int(A[2].probe), drafter.chain_cost_ms(1), drafter.chain_cost_ms(2));
                    }
                    ++by_source[std::size_t(r.source)];
                    sum_verify += r.verify_ms;
                    sum_draft += r.draft_ms;
                    if (r.source == sp::Source::model) ++k_hist[std::min<std::uint32_t>(r.proposed, 32)];
                    per_step.push_back(double(r.emitted));
                    tick_ms.push_back(now_ms() - t0);
                    cum.push_back(std::uint32_t(out.size()));
                }
                const double wall = now_ms() - t0;
                std::size_t bad = 0;
                if (!ref.empty())
                    for (std::size_t i = 0; i < out.size() && i < ref.size(); ++i)
                        if (out[i] != ref[i]) { ++bad; if (bad == 1) std::fprintf(stderr, "MISMATCH at token %zu: spec %u serial %u\n", i, out[i], ref[i]); }
                if (!ref.empty() && out.size() != ref.size()) ++bad;  // a short run is not a pass
                // best 1 s window over step ends
                double best = 0;
                for (std::size_t i = 0; i < tick_ms.size(); ++i) {
                    std::size_t j = i;
                    while (j + 1 < tick_ms.size() && tick_ms[j] - tick_ms[i] < 1000.0) ++j;
                    const double dt = tick_ms[j] - tick_ms[i];
                    if (dt >= 900.0) best = std::max(best, 1000.0 * double(cum[j] - cum[i]) / dt);
                }
                const double mean_e = per_step.empty() ? 0.0 : double(out.size() - 1) / double(per_step.size());
                const auto& st = drafter.stats();
                std::printf("spec[%s]: %zu tokens, %zu steps (serial %llu copy %llu chain %llu), E mean %.2f p50 %.0f p90 %.0f, avg %.2f tok/s (%.3f ms/step), "
                            "best-1s %.2f tok/s, hybrid blocks %llu (ext %llu/%llu), mtp calls %llu rows %llu\n",
                            tag.c_str(), out.size(), per_step.size(), (unsigned long long)by_source[0], (unsigned long long)by_source[1],
                            (unsigned long long)by_source[2], mean_e, quantile(per_step, 0.5), quantile(per_step, 0.9),
                            1000.0 * double(out.size() - 1) / wall, wall / double(std::max<std::size_t>(per_step.size(), 1)), best,
                            (unsigned long long)drafter.hybrid_blocks, (unsigned long long)drafter.hybrid_ext_accepted,
                            (unsigned long long)drafter.hybrid_ext_proposed, (unsigned long long)(S.chain_calls() - calls0),
                            (unsigned long long)(S.chain_rows() - rows0));
                std::printf("prelaunch[%s]: %s, pre-launched steps %llu of %zu\n", tag.c_str(), prelaunch ? "on" : "off",
                            (unsigned long long)spec.prelaunched_steps(), per_step.size());
                const double steps_n = double(std::max<std::size_t>(per_step.size(), 1));
                std::string hist;
                for (std::size_t k = 1; k < k_hist.size(); ++k)
                    if (k_hist[k]) hist += " " + std::to_string(k) + ":" + std::to_string(k_hist[k]);
                std::printf("step[%s]: arm %s, per step verify %.3f ms, draft %.3f ms, host %.3f ms; model k histogram%s\n", tag.c_str(),
                            arm.tag.c_str(), sum_verify / steps_n, sum_draft / steps_n, (wall - sum_verify - sum_draft) / steps_n,
                            hist.empty() ? " none" : hist.c_str());
                std::printf("stats %s\n", st.json().c_str());
                if (arm.subset) std::printf("draft[%s]: %s\n", tag.c_str(), draft_rows->describe().c_str());
                E->set_draft_rows(nullptr);
                if (!ids_out.empty()) write_ids(tag.substr(0, tag.find('/')) + ".spec-" + arm.tag + "-p" + std::to_string(pair) + ".ids", out);
                if (!ref.empty()) {
                    std::printf("EXACT[%s]: %s (%zu tokens compared, %zu mismatches)\n", tag.c_str(), bad == 0 ? "PASS" : "FAIL",
                                std::min(out.size(), ref.size()), bad);
                    if (serial_ms > 0) std::printf("speedup vs serial: %.2fx\n", serial_ms / double(n_gen - 1) / (wall / double(out.size() - 1)));
                }
                if (round_trace)
                    std::printf("spec-trace: content=%s rounds=%zu records=%llu ids_equal=%d\n", tag.c_str(), per_step.size(),
                                static_cast<unsigned long long>(round_trace->records()), ref.empty() ? -1 : int(bad == 0));
                bad_total += bad;
            }  // arms x pairs
        }  // prompts
        return bad_total == 0 ? 0 : 2;
    } catch (const std::exception& e) {
        std::fprintf(stderr, "strix-cuda-qwen35-spec: %s\n", e.what());
        return 1;
    }
}
