// strix-cuda-qwen35-decode: qwen35 (bf16 / q8_0) on one CUDA device through strix::qwen35::CudaEngine. Exact greedy decode,
// C1 timing, real GGUF or on-device synthetic full-size weights (--synthetic-full bf16|q8_0, no disk).
// Home: specific (CLI of the qwen35 family); engine in src/qwen35/cuda, arithmetic in src/cuda_core.
#include <algorithm>
#include <atomic>
#include <chrono>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <sstream>
#include <stdexcept>
#include <string>
#include <thread>
#include <vector>

#include "strix/device/cuda_enumerator.hpp"
#include "strix/qwen35/cuda_engine.hpp"

namespace q = strix::qwen35;

#define CK(x)                                                                                                  \
    do {                                                                                                       \
        const cudaError_t e__ = (x);                                                                           \
        if (e__ != cudaSuccess) {                                                                              \
            throw std::runtime_error(std::string(#x) + " failed: " + cudaGetErrorString(e__) + " at line " + \
                                     std::to_string(__LINE__));                                                \
        }                                                                                                      \
    } while (0)

namespace {

struct ClockSampler {
    std::atomic<bool> stop{false};
    std::vector<int> mhz;
    std::thread th;
    void start() {
        th = std::thread([this] {
            while (!stop.load()) {
                FILE* p = _popen("nvidia-smi --query-gpu=clocks.sm -i 0 --format=csv,noheader,nounits", "r");
                if (p) {
                    char b[64] = {};
                    if (std::fgets(b, sizeof b, p)) mhz.push_back(std::atoi(b));
                    _pclose(p);
                }
                std::this_thread::sleep_for(std::chrono::milliseconds(100));
            }
        });
    }
    void end() {
        stop = true;
        if (th.joinable()) th.join();
    }
};

}  // namespace

int main(int argc, char** argv) {
    strix::device::bind_cuda_main(argc, argv);
    try {
        std::string model, dump, ids_arg, synth;
        std::uint32_t n_gen = 32, rows_cap = 16, max_pos = 4096, prompt_len = 0, ctx_fill = 0, layers_override = 0, decode_rows = 1;
        bool graphs = true;
        for (int i = 1; i < argc; ++i) {
            const std::string a = argv[i];
            auto next = [&]() -> std::string {
                if (i + 1 >= argc) throw std::runtime_error("missing value for " + a);
                return argv[++i];
            };
            if (a == "--model") model = next();
            else if (a == "--synthetic-full") synth = next();
            else if (a == "--prompt-ids") ids_arg = next();
            else if (a == "--prompt-len") prompt_len = static_cast<std::uint32_t>(std::stoul(next()));
            else if (a == "--ctx-fill") ctx_fill = static_cast<std::uint32_t>(std::stoul(next()));
            else if (a == "--n-gen") n_gen = static_cast<std::uint32_t>(std::stoul(next()));
            else if (a == "--rows") rows_cap = static_cast<std::uint32_t>(std::stoul(next()));
            else if (a == "--decode-rows") decode_rows = static_cast<std::uint32_t>(std::stoul(next()));
            else if (a == "--max-pos") max_pos = static_cast<std::uint32_t>(std::stoul(next()));
            else if (a == "--layers") layers_override = static_cast<std::uint32_t>(std::stoul(next()));
            else if (a == "--graphs") graphs = next() != "0";
            else if (a == "--dump-logits") dump = next();
            else throw std::runtime_error("unknown argument " + a);
        }
        if ((model.empty() == synth.empty()) || rows_cap == 0 || rows_cap > 32 || decode_rows == 0 || decode_rows > 32)
            throw std::runtime_error(
                "usage: (--model F.gguf | --synthetic-full bf16|q8_0 [--layers N]) (--prompt-ids 1,2,3 | --prompt-len N) [--ctx-fill N] "
                "[--n-gen N] [--rows 1..32] [--decode-rows 1..32 (timing of an R-row window, committed fully)] [--max-pos N] "
                "[--graphs 0|1] [--dump-logits F]");
        std::vector<std::uint32_t> prompt;
        if (!ids_arg.empty()) {
            std::stringstream ss(ids_arg);
            std::string t;
            while (std::getline(ss, t, ',')) prompt.push_back(static_cast<std::uint32_t>(std::stoul(t)));
        }
        q::EngineOptions opt;
        opt.use_graphs = graphs;
        std::unique_ptr<q::CudaEngine> E;
        const std::uint64_t total_ctx = ctx_fill + prompt.size() + prompt_len + static_cast<std::uint64_t>(n_gen) * decode_rows + 64;
        opt.max_positions = static_cast<std::uint32_t>(std::max<std::uint64_t>(max_pos, total_ctx));
        if (!model.empty()) {
            E = q::CudaEngine::from_gguf(model, opt);
        } else {
            q::Config cfg;
            if (layers_override) cfg.layers = layers_override;
            E = q::CudaEngine::synthetic(cfg, synth == "q8_0" ? q::PackFormat::q8_0 : q::PackFormat::bf16, opt, 0x5eedULL);
        }
        const auto& cfg = E->config();
        std::fprintf(stderr, "qwen35: layers %u hidden %u inter %u vocab %u heads %u/%u hd %u gdn %u/%u x %u mtp %d\n", cfg.layers,
                     cfg.hidden, cfg.intermediate, cfg.vocab, cfg.heads, cfg.kv_heads, cfg.head_dim, cfg.gdn_key_heads,
                     cfg.gdn_value_heads, cfg.gdn_value_dim, E->has_mtp() ? 1 : 0);
        std::uint64_t lcg = 12345;
        auto rnd_tok = [&]() {
            lcg = lcg * 6364136223846793005ull + 1442695040888963407ull;
            return static_cast<std::uint32_t>((lcg >> 33) % cfg.vocab);
        };
        if (prompt.empty()) {
            for (std::uint32_t i = 0; i < prompt_len + ctx_fill; ++i) prompt.push_back(rnd_tok());
            if (prompt.empty()) throw std::runtime_error("give --prompt-ids or --prompt-len");
        }

        std::FILE* df = dump.empty() ? nullptr : std::fopen(dump.c_str(), "wb");
        if (!dump.empty() && !df) throw std::runtime_error("cannot open " + dump);
        std::vector<float> host_logits(static_cast<std::size_t>(cfg.vocab));
        std::vector<std::uint32_t> top1;
        std::uint32_t next_tok = 0;
        const auto tp0 = std::chrono::steady_clock::now();
        for (std::size_t p = 0; p < prompt.size(); p += rows_cap) {
            const std::uint32_t rows = static_cast<std::uint32_t>(std::min<std::size_t>(rows_cap, prompt.size() - p));
            const bool last = p + rows == prompt.size();
            const auto want = df ? q::LogitRows::all : (last ? q::LogitRows::last : q::LogitRows::none);
            const auto o = E->forward_rows(prompt.data() + p, rows, want);
            E->commit(rows);
            if (df) {
                std::uint32_t tp[32];
                E->read_top(o, rows, tp);
                for (std::uint32_t r = 0; r < rows; ++r) {
                    E->read_logits(o, r, host_logits.data());
                    std::fwrite(host_logits.data(), 4, cfg.vocab, df);
                    top1.push_back(tp[r]);
                }
                next_tok = tp[rows - 1];
            } else if (last) {
                E->read_top(o, 1, &next_tok);
            }
        }
        E->sync();
        const double prefill_s = std::chrono::duration<double>(std::chrono::steady_clock::now() - tp0).count();
        if (df) {
            std::fclose(df);
            std::printf("prompt top1:");
            for (auto t : top1) std::printf(" %u", t);
            std::printf("\n");
        }
        std::fprintf(stderr, "prefill %zu tokens in %.2f s (rows %u)\n", prompt.size(), prefill_s, rows_cap);

        // greedy decode; decode_rows > 1 times an R-row window (tokens = the greedy token repeated, all committed)
        std::vector<std::uint32_t> gen;
        std::vector<double> step_ms;
        ClockSampler clk;
        clk.start();
        cudaEvent_t e0, e1;
        CK(cudaEventCreate(&e0));
        CK(cudaEventCreate(&e1));
        const std::uint32_t ctx0 = E->committed();
        for (std::uint32_t i = 0; i < n_gen; ++i) {
            gen.push_back(next_tok);
            if (i + 1 == n_gen) break;
            std::uint32_t toks[32];
            for (std::uint32_t r = 0; r < decode_rows; ++r) toks[r] = next_tok;
            CK(cudaEventRecord(e0, E->stream()));
            const auto o = E->forward_rows(toks, decode_rows, decode_rows == 1 ? q::LogitRows::last : q::LogitRows::all);
            E->commit(decode_rows);
            CK(cudaEventRecord(e1, E->stream()));
            std::uint32_t tp[32];
            E->read_top(o, decode_rows == 1 ? 1 : decode_rows, tp);
            next_tok = tp[decode_rows == 1 ? 0 : decode_rows - 1];
            float ms = 0;
            CK(cudaEventElapsedTime(&ms, e0, e1));
            step_ms.push_back(ms);
        }
        clk.end();
        std::printf("generated:");
        for (auto t : gen) std::printf(" %u", t);
        std::printf("\n");
        if (!step_ms.empty()) {
            std::vector<double> s2(step_ms.begin() + (step_ms.size() > 2 ? 2 : 0), step_ms.end());
            double sum = 0;
            for (double m : s2) sum += m;
            const double mean = sum / s2.size();
            std::sort(s2.begin(), s2.end());
            const double bytes = static_cast<double>(E->weight_bytes()) + static_cast<double>(E->state_bytes_per_pass());
            std::printf("decode: rows %u ctx %u graphs %d nodes %llu: %zu steps, mean %.3f ms (median %.3f min %.3f), %.2f tok/s, "
                        "%.1f GB/s achieved (%.3f GB/pass excl. KV) vs 1792 = %.1f%%\n",
                        decode_rows, ctx0, graphs ? 1 : 0, static_cast<unsigned long long>(E->launches_last_forward()), s2.size(), mean,
                        s2[s2.size() / 2], s2.front(), 1000.0 * decode_rows / mean, bytes / (mean * 1e-3) / 1e9, bytes / 1e9,
                        100.0 * bytes / (mean * 1e-3) / 1e9 / 1792.0);
        }
        if (!clk.mhz.empty()) {
            int lo = 1 << 30, hi = 0;
            double sm = 0;
            for (int m : clk.mhz) {
                lo = std::min(lo, m);
                hi = std::max(hi, m);
                sm += m;
            }
            std::printf("sm clock MHz: min %d avg %.0f max %d (%zu samples)\n", lo, sm / clk.mhz.size(), hi, clk.mhz.size());
        }
        return 0;
    } catch (const std::exception& e) {
        std::fprintf(stderr, "strix-cuda-qwen35-decode: %s\n", e.what());
        return 1;
    }
}
