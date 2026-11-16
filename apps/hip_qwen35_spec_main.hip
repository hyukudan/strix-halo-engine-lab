// strix-hip-qwen35-spec: exact block drafting (shared strix::spec BlockDrafter + Arbiter) over the qwen35 HIP window engine.
// Modes: default = serial reference (--check 1) + speculative run, token-for-token gate (EXACT: PASS/FAIL), E mean/p50/p90,
// avg + best-1s tok/s; --cost-table = verify window ms per rows 1..16 (commit full / half); --prefill-bench N [--pf-sample] =
// prefill tok/s over N synthetic ids in 16-row windows (full run, or sampled windows at positions spread over N).
// Sources: copy (n-gram / prompt lookup) only; the HIP engine has no MTP layer yet (chain/hybrid need mtp_forward).
// Home: specific = app glue; drafting logic + VerifyTarget binding are shared (strix/spec/*.hpp), engine is HipEngine.
#include <algorithm>
#include <chrono>
#include <cstdio>
#include <cstdlib>
#include <fstream>
#include <sstream>
#include <stdexcept>
#include <string>
#include <vector>

#include "strix/qwen35/hip_engine.hpp"
#include "strix/spec/block_drafter.hpp"
#include "strix/spec/engine_target.hpp"
#include "../src/core/dense_linear.hpp"

namespace sp = strix::spec;
namespace q = strix::qwen35;
using Clock = std::chrono::steady_clock;

namespace {
double now_ms() { return std::chrono::duration<double, std::milli>(Clock::now().time_since_epoch()).count(); }
std::uint64_t mix(std::uint64_t x) {
    x ^= x >> 33; x *= 0xff51afd7ed558ccdULL; x ^= x >> 33; x *= 0xc4ceb9fe1a85ec53ULL; x ^= x >> 33;
    return x;
}
std::vector<std::uint32_t> parse_ids(const std::string& text) {
    std::vector<std::uint32_t> v;
    std::string t;
    for (const char c : text + ",") {
        if (c >= '0' && c <= '9') t += c;
        else if (!t.empty()) { v.push_back(strix::core::parse_u32(t)); t.clear(); }
    }
    return v;
}
double quantile(std::vector<double> v, double p) {
    if (v.empty()) return 0.0;
    std::sort(v.begin(), v.end());
    return v[std::min<std::size_t>(v.size() - 1, std::size_t(p * double(v.size())))];
}
}  // namespace

int main(int argc, char** argv) try {
    std::string model, ids_file, label = "run", sources = "copy";
    std::uint32_t n_gen = 256, max_block = 16, max_pos = 0, pf_bench = 0, trace_n = 0, prefill_rows = 16;
    bool check = true, cost_table = false, pf_sample = false, seed_costs = false, pf_twice = false;
    for (int i = 1; i < argc; ++i) {
        const std::string a = argv[i];
        auto next = [&]() -> std::string {
            if (i + 1 >= argc) throw std::invalid_argument("missing value for " + a);
            return argv[++i];
        };
        if (a == "--model") model = next();
        else if (a == "--prompt-file") ids_file = next();
        else if (a == "--n-gen") n_gen = strix::core::parse_u32(next());
        else if (a == "--sources") sources = next();
        else if (a == "--max-block") max_block = strix::core::parse_u32(next());
        else if (a == "--max-pos") max_pos = strix::core::parse_u32(next());
        else if (a == "--check") check = next() != "0";
        else if (a == "--cost-table") cost_table = true;
        else if (a == "--prefill-bench") pf_bench = strix::core::parse_u32(next());
        else if (a == "--pf-sample") pf_sample = true;
        else if (a == "--pf-twice") pf_twice = true;  // two full prefills: cold first, warm last
        else if (a == "--prefill-rows") prefill_rows = strix::core::parse_u32(next());  // >16 = FAST wide windows (dense-q8a8-f16-v1)
        else if (a == "--seed-costs") seed_costs = next() != "0";
        else if (a == "--trace") trace_n = strix::core::parse_u32(next());
        else if (a == "--label") label = next();
        else throw std::invalid_argument("unknown argument " + a);
    }
    if (model.empty() || (ids_file.empty() && !cost_table && !pf_bench))
        throw std::invalid_argument("usage: --model F.gguf (--prompt-file ids | --cost-table | --prefill-bench N [--pf-sample]) [--n-gen N] "
                                    "[--sources copy|none] [--max-block 1..16] [--check 0|1] [--max-pos N] [--label S]");
    std::vector<std::uint32_t> prompt;
    if (!ids_file.empty()) {
        std::ifstream f(ids_file);
        std::stringstream ss;
        ss << f.rdbuf();
        prompt = parse_ids(ss.str());
    }
    q::HipEngineOptions opt;
    opt.max_positions = std::max<std::uint32_t>(max_pos, std::uint32_t(prompt.size()) + n_gen + 128U + (cost_table ? 512U : 0U));
    if (pf_bench) opt.max_positions = std::max(opt.max_positions, pf_bench + 64U);
    opt.quiet = true;
    opt.prefill_rows = prefill_rows;
    auto E = q::HipEngine::from_gguf(model, opt);
    const auto vocab = E->config().vocab;
    std::printf("loaded: weights %.2f GB/pass, state %.0f MB/pass, max_positions %u\n", double(E->weight_bytes()) / 1e9,
                double(E->state_bytes_per_pass()) / 1e6, opt.max_positions);

    auto prefill_ids = [&](const std::vector<std::uint32_t>& ids, std::uint32_t& anchor) {
        E->reset();
        const std::size_t W = E->prefill_rows();
        for (std::size_t p = 0; p < ids.size(); p += W) {
            const auto rows = std::uint32_t(std::min<std::size_t>(W, ids.size() - p));
            const bool last = p + rows == ids.size();
            const auto o = E->forward_commit(ids.data() + p, rows, last ? q::LogitRows::last : q::LogitRows::none);
            if (last) E->read_top(o, 1, &anchor);
        }
        E->sync();
    };
    auto warm = [&]() {  // every window width once (commit 0 discards): kernels / caches hot before any timed wall
        std::vector<std::uint32_t> t(q::HipEngine::max_rows(), 1);
        for (std::uint32_t r = 1; r <= q::HipEngine::max_rows(); ++r) {
            E->forward_rows(t.data(), r, q::LogitRows::all);
            E->commit(0);
        }
        E->sync();
    };

    if (pf_bench) {
        std::vector<std::uint32_t> ids(pf_bench);
        for (std::uint32_t i = 0; i < pf_bench; ++i) ids[i] = std::uint32_t(mix(77 + i) % vocab);
        const std::uint32_t R = E->prefill_rows();
        if (pf_twice) {
            std::uint32_t anchor = 0;
            for (int pass = 0; pass < 2; ++pass) {
                const double t0 = now_ms();
                prefill_ids(ids, anchor);
                const double ms = now_ms() - t0;
                std::printf("prefill[%s] %s: %u tokens in %.1f ms = %.1f tok/s (window %u, anchor %u)\n", label.c_str(), pass ? "WARM" : "COLD",
                            pf_bench, ms, 1000.0 * pf_bench / ms, E->prefill_rows(), anchor);
            }
        } else if (!pf_sample || pf_bench < R) {
            std::uint32_t anchor = 0;
            std::vector<std::uint32_t> warm_ids(ids.begin(), ids.begin() + std::min<std::size_t>(R, ids.size()));
            prefill_ids(warm_ids, anchor);
            const double t0 = now_ms();
            prefill_ids(ids, anchor);
            const double ms = now_ms() - t0;
            std::printf("prefill[%s]: %u tokens in %.1f ms = %.1f tok/s (%.2f ms/window of %u)\n", label.c_str(), pf_bench, ms,
                        1000.0 * pf_bench / ms, ms / (double(pf_bench) / R), R);
        } else {
            // Windows at positions spread over [0, N): window cost is linear in position (attention), so trapezoid over samples.
            std::uint32_t anchor = 0;
            std::vector<std::uint32_t> warm_ids(ids.begin(), ids.begin() + std::min<std::size_t>(R, ids.size()));
            prefill_ids(warm_ids, anchor);
            const unsigned S = 9;
            std::vector<double> pos, cost;
            for (unsigned s = 0; s < S; ++s) {
                const std::uint32_t at = std::uint32_t((std::uint64_t(pf_bench - R) * s / (S - 1)) / R * R);
                E->bench_seek(at);
                double sum = 0;
                for (int rep = 0; rep < 4; ++rep) {
                    const double t0 = now_ms();
                    E->forward_commit(ids.data(), R, q::LogitRows::none);
                    E->sync();
                    sum += now_ms() - t0;
                    E->bench_seek(at);
                }
                pos.push_back(at);
                cost.push_back(sum / 4);
                std::printf("window pos %u: %.2f ms\n", at, sum / 4);
            }
            double total = 0;
            for (unsigned s = 0; s + 1 < S; ++s) total += 0.5 * (cost[s] + cost[s + 1]) * (pos[s + 1] - pos[s]) / R;
            total += cost[S - 1] * (double(pf_bench) - pos[S - 1]) / R;
            std::printf("prefill[%s] SAMPLED: %u tokens ~ %.0f ms = %.1f tok/s (9 positions x 4 windows, trapezoid)\n", label.c_str(), pf_bench,
                        total, 1000.0 * pf_bench / total);
        }
        return 0;
    }

    if (cost_table) {
        std::uint32_t anchor = 0;
        std::vector<std::uint32_t> ids(256);
        for (std::size_t i = 0; i < ids.size(); ++i) ids[i] = std::uint32_t(mix(5 + i) % vocab);
        prefill_ids(ids, anchor);
        warm();
        std::vector<std::uint32_t> t(q::HipEngine::max_rows(), anchor);
        std::printf("cost table (%s, ctx %zu+)\nrows verify_ms(commit full) verify_ms(commit half) tok_per_s_if_all_accepted\n", label.c_str(), ids.size());
        for (const std::uint32_t r : {1U, 2U, 3U, 4U, 6U, 8U, 12U, 16U}) {
            double full = 0, half = 0;
            const int reps = 6;
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
        double none = 0, lst = 0, ser = 0;
        for (int rep = 0; rep < 12; ++rep) {
            const bool l = rep & 1;
            const double t0 = now_ms();
            E->forward_rows(t.data(), 1, l ? q::LogitRows::last : q::LogitRows::none);
            E->commit(0);
            E->sync();
            (l ? lst : none) += now_ms() - t0;
        }
        for (int rep = 0; rep < 6; ++rep) {
            const double t0 = now_ms();
            E->forward_commit(t.data(), 1, q::LogitRows::last);
            E->sync();
            ser += now_ms() - t0;
        }
        std::printf("R=1 window: no-logits %.3f ms, last %.3f ms -> lm_head %.3f ms; forward_commit(1) (no snapshot) %.3f ms\n", none / 6, lst / 6,
                    (lst - none) / 6, ser / 6);
        return 0;
    }

    // ---- exactness reference: serial greedy
    std::uint32_t anchor = 0;
    std::vector<std::uint32_t> ref;
    double serial_ms = 0;
    if (check) {
        prefill_ids(prompt, anchor);
        warm();
        ref.push_back(anchor);
        std::uint32_t cur = anchor;
        const double t0 = now_ms();
        while (ref.size() < n_gen) {
            const auto o = E->forward_commit(&cur, 1, q::LogitRows::last);
            E->read_top(o, 1, &cur);
            ref.push_back(cur);
        }
        serial_ms = now_ms() - t0;
        std::printf("serial: %u tokens in %.1f ms = %.2f tok/s (%.3f ms/token)\n", n_gen, serial_ms, 1000.0 * (n_gen - 1) / serial_ms,
                    serial_ms / (n_gen - 1));
    }

    // ---- speculative run
    const double tp0 = now_ms();
    prefill_ids(prompt, anchor);
    const double prefill_ms = now_ms() - tp0;
    warm();
    sp::BlockConfig bc;
    bc.max_block = max_block;
    bc.copy = sources.find("copy") != std::string::npos;
    bc.chain = false;
    bc.hybrid = false;
    sp::BlockDrafter drafter(bc, nullptr);
    if (seed_costs) {
        std::vector<std::uint32_t> t(q::HipEngine::max_rows(), anchor);
        for (const std::uint32_t r : {1U, 2U, 4U, 8U, 16U}) {
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
    sp::EngineVerifyTarget<q::HipEngine> target(*E);
    sp::BlockSpeculator spec(drafter, target);
    spec.start(anchor);
    std::vector<double> tick_ms{0.0}, per_step;
    std::vector<std::uint32_t> cum{1};
    std::uint64_t windows[17] = {};
    std::uint64_t by_source[3] = {0, 0, 0};
    const double t0 = now_ms();
    while (out.size() < n_gen) {
        sp::StepParams p;
        p.remaining = n_gen - std::uint32_t(out.size());
        p.max_rows = std::min<std::uint32_t>(16U, max_block);
        p.context = E->committed();
        const auto r = spec.step(out, p);
        if (r.emitted == 0) break;
        if (per_step.size() < trace_n)
            std::printf("T%zu src%u rows%u prop%u acc%u em%u ver%.2f dr%.2f\n", per_step.size(), unsigned(r.source), r.rows, r.proposed,
                        r.accepted, r.emitted, r.verify_ms, r.draft_ms);
        ++by_source[std::size_t(r.source)];
        if (r.rows <= 16U) ++windows[r.rows];
        per_step.push_back(double(r.emitted));
        tick_ms.push_back(now_ms() - t0);
        cum.push_back(std::uint32_t(out.size()));
    }
    const double wall = now_ms() - t0;
    std::size_t bad = (out.size() != n_gen || (!ref.empty() && ref.size() != n_gen)) ? 1U : 0U;
    for (std::size_t i = 0; i < out.size() && i < ref.size(); ++i)
        if (out[i] != ref[i]) { ++bad; if (bad == 1) std::fprintf(stderr, "MISMATCH at token %zu: spec %u serial %u\n", i, out[i], ref[i]); }
    double best = 0;
    for (std::size_t i = 0; i < tick_ms.size(); ++i) {
        std::size_t j = i;
        while (j + 1 < tick_ms.size() && tick_ms[j] - tick_ms[i] < 1000.0) ++j;
        const double dt = tick_ms[j] - tick_ms[i];
        if (dt >= 900.0) best = std::max(best, 1000.0 * double(cum[j] - cum[i]) / dt);
    }
    const double mean_e = per_step.empty() ? 0.0 : double(out.size() - 1) / double(per_step.size());
    std::printf("spec[%s]: %zu tokens, %zu steps (serial %llu copy %llu chain %llu), E mean %.2f p50 %.0f p90 %.0f, avg %.2f tok/s "
                "(%.3f ms/step), best-1s %.2f tok/s, prompt %zu tok prefill %.0f ms\n",
                label.c_str(), out.size(), per_step.size(), (unsigned long long)by_source[0], (unsigned long long)by_source[1],
                (unsigned long long)by_source[2], mean_e, quantile(per_step, 0.5), quantile(per_step, 0.9),
                1000.0 * double(out.size() - 1) / wall, wall / double(std::max<std::size_t>(per_step.size(), 1)), best, prompt.size(), prefill_ms);
    for (unsigned r = 1; r <= 16U; ++r)
        if (windows[r]) std::printf("verify-window %u: %llu calls\n", r, (unsigned long long)windows[r]);
    if (!ref.empty()) {
        std::printf("EXACT: %s (%zu tokens compared, %zu mismatches)\n", bad == 0 ? "PASS" : "FAIL", std::min(out.size(), ref.size()), bad);
        std::printf("speedup vs serial: %.2fx\n", serial_ms / double(n_gen - 1) / (wall / double(out.size() - 1)));
    }
    return bad == 0 ? 0 : 2;
} catch (const std::exception& e) {
    std::fprintf(stderr, "strix-hip-qwen35-spec: %s\n", e.what());
    return 1;
}
