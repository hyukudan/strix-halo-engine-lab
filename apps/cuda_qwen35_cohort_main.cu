// strix-cuda-qwen35-cohort: N concurrent requests on the qwen35 CUDA engine through the shared multi-request decoder
// (strix/serve/cohort_decode.hpp: join / leave at step boundaries, plan_rows-priced verify windows, context drafts).
//
// Measures aggregate and per-request tok/s at each concurrency and checks every request's ids against its solo greedy
// run (EXACT: the slot rows are bitwise the slot alone, so ids must be equal). Token-id workloads, no tokenizer:
//   --workload prose  random ids (drafts rarely proposed)
//   --workload edit   a random block, a separator, then the block's start (an induction copy: context drafts accepted)
//   --workload mixed  alternating
//   --prompts FILE    one request per line, comma-separated ids (overrides --workload)
// Usage: strix-cuda-qwen35-cohort (--model GGUF [--turbo-weights converted|native|onfly|off] | --synthetic q8_0|bf16)
//        [--conc 1,2,4,8] [--draft off,on] [--max-tokens 128] [--max-pos 4096] [--workload edit] [--requests 8]
//        [--prior 13.3,0.113,0.13] [--max-seconds 100] [--no-graphs]
#include "strix/qwen35/cuda_engine.hpp"
#include "../src/qwen35/cuda/cohort_backend.hpp"

#include <algorithm>
#include <chrono>
#include <cstdint>
#include <cstdio>
#include <fstream>
#include <map>
#include <sstream>
#include <stdexcept>
#include <string>
#include <vector>

namespace q = strix::qwen35;
namespace sv = strix::serve;
using Clock = std::chrono::steady_clock;

namespace {
std::vector<std::uint32_t> parse_list(const std::string& s) {
    std::vector<std::uint32_t> out;
    std::stringstream ss(s);
    std::string t;
    while (std::getline(ss, t, ',')) if (!t.empty()) out.push_back(static_cast<std::uint32_t>(std::stoul(t)));
    return out;
}

struct Lcg {
    std::uint64_t s;
    std::uint32_t next(std::uint32_t lo, std::uint32_t hi) {
        s = s * 6364136223846793005ULL + 1442695040888963407ULL;
        return lo + static_cast<std::uint32_t>((s >> 33U) % (hi - lo));
    }
};

std::vector<std::uint32_t> make_prompt(const std::string& kind, std::uint32_t index) {
    Lcg r{0x9e3779b97f4a7c15ULL ^ (static_cast<std::uint64_t>(index) * 0x100000001b3ULL)};
    std::vector<std::uint32_t> p;
    if (kind == "prose") {
        const std::uint32_t n = r.next(64, 256);
        for (std::uint32_t i = 0; i < n; ++i) p.push_back(r.next(1000, 100000));
    } else {
        std::vector<std::uint32_t> block;
        for (std::uint32_t i = 0; i < 160; ++i) block.push_back(r.next(1000, 100000));
        p = block;
        p.push_back(198);  // separator
        p.insert(p.end(), block.begin(), block.begin() + 16);
    }
    return p;
}

struct ArmResult {
    double wall_ms = 0, agg = 0, req_mean = 0, ttft_mean = 0;
    std::uint64_t tokens = 0, steps = 0, rows = 0, drafted = 0, accepted = 0;
    std::map<std::uint64_t, std::vector<std::uint32_t>> ids;
    sv::CycleCost fit{};
};

ArmResult run_arm(sv::CohortBackend& be, const std::vector<sv::CohortRequest>& reqs, std::uint32_t conc, bool draft,
                  const sv::CycleCost& prior) {
    sv::CohortConfig cfg;
    cfg.context_draft = draft;
    cfg.prior = prior;
    sv::CohortDecoder dec(be, cfg);
    ArmResult a;
    const auto t0 = Clock::now();
    for (std::uint32_t i = 0; i < conc; ++i) dec.submit(reqs.at(i));
    dec.run();
    a.wall_ms = std::chrono::duration<double, std::milli>(Clock::now() - t0).count();
    double req_sum = 0, ttft_sum = 0;
    for (auto& r : dec.take_finished()) {
        if (!r.error.empty()) throw std::runtime_error("request " + std::to_string(r.id) + " refused: " + r.error);
        a.tokens += r.ids.size();
        a.drafted += r.drafted;
        a.accepted += r.accepted;
        const double span = r.done_ms - r.submit_ms;
        req_sum += span > 0 ? static_cast<double>(r.ids.size()) * 1000.0 / span : 0.0;
        ttft_sum += r.first_ms - r.submit_ms;
        a.ids[r.id] = std::move(r.ids);
    }
    a.agg = a.wall_ms > 0 ? static_cast<double>(a.tokens) * 1000.0 / a.wall_ms : 0.0;
    a.req_mean = req_sum / conc;
    a.ttft_mean = ttft_sum / conc;
    a.steps = dec.steps();
    a.rows = dec.physical_rows();
    a.fit = dec.cost();
    return a;
}
}  // namespace

int main(int argc, char** argv) {
    try {
        std::string model, synth, workload = "edit", prompts_file, turbo = "converted";
        std::vector<std::uint32_t> conc = {1, 2, 4, 8};
        std::vector<std::string> drafts = {"off", "on"};
        std::uint32_t max_tokens = 128, max_pos = 4096, n_req = 0;
        double max_seconds = 100;
        bool graphs = true;
        sv::CycleCost prior{13.3, 0.113, 0.13, 0.0};
        for (int i = 1; i < argc; ++i) {
            const std::string a = argv[i];
            const auto next = [&]() -> std::string {
                if (i + 1 >= argc) throw std::invalid_argument(a + " needs a value");
                return argv[++i];
            };
            if (a == "--model") model = next();
            else if (a == "--synthetic") synth = next();
            else if (a == "--turbo-weights") turbo = next();
            else if (a == "--conc") conc = parse_list(next());
            else if (a == "--draft") {
                drafts.clear();
                std::stringstream ss(next());
                std::string t;
                while (std::getline(ss, t, ',')) drafts.push_back(t);
            } else if (a == "--max-tokens") max_tokens = static_cast<std::uint32_t>(std::stoul(next()));
            else if (a == "--max-pos") max_pos = static_cast<std::uint32_t>(std::stoul(next()));
            else if (a == "--workload") workload = next();
            else if (a == "--prompts") prompts_file = next();
            else if (a == "--requests") n_req = static_cast<std::uint32_t>(std::stoul(next()));
            else if (a == "--max-seconds") max_seconds = std::stod(next());
            else if (a == "--no-graphs") graphs = false;
            else if (a == "--prior") {
                const auto t0 = next();
                if (std::sscanf(t0.c_str(), "%lf,%lf,%lf", &prior.fixed_ms, &prior.row_ms, &prior.lane_ms) != 3)
                    throw std::invalid_argument("--prior fixed,row,lane");
            } else throw std::invalid_argument("unknown argument " + a);
        }
        if (conc.empty()) throw std::invalid_argument("--conc is empty");
        const std::uint32_t max_conc = *std::max_element(conc.begin(), conc.end());
        if (n_req == 0) n_req = max_conc;
        if (n_req < max_conc) throw std::invalid_argument("--requests below the largest --conc");

        std::vector<sv::CohortRequest> reqs;
        if (!prompts_file.empty()) {
            std::ifstream f(prompts_file);
            if (!f) throw std::runtime_error("cannot open " + prompts_file);
            std::string line;
            while (std::getline(f, line) && reqs.size() < n_req)
                if (!line.empty()) reqs.push_back({reqs.size(), parse_list(line), max_tokens, {}});
            if (reqs.size() < n_req) throw std::runtime_error("--prompts has fewer lines than --requests");
        } else {
            for (std::uint32_t i = 0; i < n_req; ++i) {
                const std::string kind = workload == "mixed" ? (i % 2 ? "prose" : "edit") : workload;
                reqs.push_back({i, make_prompt(kind, i), max_tokens, {}});
            }
        }
        std::uint64_t longest = 0;
        for (const auto& r : reqs) longest = std::max<std::uint64_t>(longest, r.prompt.size());

        const auto t_load = Clock::now();
        q::EngineOptions opt;
        if (turbo == "converted") opt.turbo_weights = q::TurboWeights::converted;
        else if (turbo == "native") opt.turbo_weights = q::TurboWeights::native;
        else if (turbo == "onfly") opt.turbo_weights = q::TurboWeights::onfly;
        else if (turbo == "off") opt.turbo_weights = q::TurboWeights::off;
        else throw std::invalid_argument("--turbo-weights expects converted|native|onfly|off");
        opt.use_graphs = graphs;
        opt.max_positions = static_cast<std::uint32_t>(std::max<std::uint64_t>(max_pos, longest + max_tokens + 64U));
        std::unique_ptr<q::CudaEngine> E;
        if (!model.empty()) E = q::CudaEngine::from_gguf(model, opt);
        else if (!synth.empty()) E = q::CudaEngine::synthetic(q::Config{}, synth == "q8_0" ? q::PackFormat::q8_0 : q::PackFormat::bf16, opt, 0x5eedULL);
        else throw std::invalid_argument("--model or --synthetic");
        q::CudaCohortBackend be(*E, n_req);
        const double load_s = std::chrono::duration<double>(Clock::now() - t_load).count();
        std::printf("cohort load %.1f s identity %s lanes %u max_pos %u workload %s requests %u max_tokens %u\n", load_s,
                    E->identity().c_str(), n_req, opt.max_positions, prompts_file.empty() ? workload.c_str() : "file", n_req, max_tokens);
        const auto t_start = Clock::now();
        const auto elapsed = [&] { return std::chrono::duration<double>(Clock::now() - t_start).count(); };

        // warm: graph shapes of drafted windows and the slot prefill path (not timed)
        {
            std::vector<sv::CohortRequest> warm(reqs.begin(), reqs.begin() + max_conc);
            for (auto& w : warm) w.max_tokens = 16;
            (void)run_arm(be, warm, max_conc, true, prior);
        }
        std::printf("cohort warm done %.1f s\n", elapsed());

        // solo references (draft off): request i alone
        std::map<std::uint64_t, std::vector<std::uint32_t>> solo;
        for (std::uint32_t i = 0; i < max_conc; ++i) {
            if (elapsed() > max_seconds) throw std::runtime_error("inner time limit reached in solo references");
            const std::vector<sv::CohortRequest> one{reqs[i]};
            auto a = run_arm(be, one, 1, false, prior);
            std::printf("solo id=%u tokens=%llu tok_s=%.1f ttft_ms=%.1f\n", i, static_cast<unsigned long long>(a.tokens), a.agg, a.ttft_mean);
            solo[i] = a.ids.at(i);
        }
        bool all_equal = true;
        for (const auto& d : drafts) {
            for (const auto c : conc) {
                if (elapsed() > max_seconds) {
                    std::printf("SKIP arm draft=%s C=%u (inner time limit)\n", d.c_str(), c);
                    all_equal = false;
                    continue;
                }
                const auto a = run_arm(be, reqs, c, d == "on", prior);
                std::uint32_t equal = 0;
                for (const auto& [id, ids] : a.ids) equal += ids == solo.at(id) ? 1U : 0U;
                all_equal = all_equal && equal == c;
                std::printf("arm draft=%s C=%u agg_tok_s=%.1f req_tok_s_mean=%.1f ttft_ms_mean=%.1f wall_ms=%.0f tokens=%llu "
                            "steps=%llu rows_per_step=%.2f drafted=%llu accepted=%llu fit=%.2f/%.3f/%.3f ids_equal=%u/%u\n",
                            d.c_str(), c, a.agg, a.req_mean, a.ttft_mean, a.wall_ms, static_cast<unsigned long long>(a.tokens),
                            static_cast<unsigned long long>(a.steps), a.steps ? static_cast<double>(a.rows) / static_cast<double>(a.steps) : 0.0,
                            static_cast<unsigned long long>(a.drafted), static_cast<unsigned long long>(a.accepted),
                            a.fit.fixed_ms, a.fit.row_ms, a.fit.lane_ms, equal, c);
                std::fflush(stdout);
            }
        }
        std::printf("COHORT_DONE ids_equal_all=%s elapsed_s=%.1f\n", all_equal ? "yes" : "no", elapsed());
        return all_equal ? 0 : 3;
    } catch (const std::exception& e) {
        std::fprintf(stderr, "strix-cuda-qwen35-cohort: %s\n", e.what());
        return 1;
    }
}
