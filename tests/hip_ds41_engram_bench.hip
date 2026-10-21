// Engram reader micro-benchmark (not a ctest): cold random reads of the real per-token pattern (every
// Engram layer's rows of one token, from the real hash of a token sequence) by the O_DIRECT reader,
// and their overlap with a bandwidth-bound GPU kernel. Rows are spread over the 200 GB tables, so
// repeated hits are rare and O_DIRECT bypasses the page cache: every read is a device read.
// usage: strix-hip-ds41-engram-bench MODEL.gguf [tokens] [--no-gpu]
// Prints JSON lines: decode (one batch per token) latency percentiles per thread count, prefill
// (2048 tokens per batch) rows/s, and with the GPU: reader latency while a kernel streams 351 MB
// of Q8_0 weights in a loop, and that kernel's time with and without the reader running.

#include "strix/deepseek41_model.hpp"
#include "strix/ds41_engram.hpp"
#include "strix/hip_ds41_layer.hpp"

#include <hip/hip_runtime.h>

#include <algorithm>
#include <atomic>
#include <chrono>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <random>
#include <string>
#include <thread>
#include <vector>

namespace {

using namespace strix::ds41;
using Clock = std::chrono::steady_clock;

void check(hipError_t e, const char* what) {
    if (e != hipSuccess) { std::fprintf(stderr, "%s: %s\n", what, hipGetErrorString(e)); std::exit(2); }
}
double ms_since(Clock::time_point t) { return std::chrono::duration<double, std::milli>(Clock::now() - t).count(); }
double pct(std::vector<double> v, double p) {
    std::sort(v.begin(), v.end());
    return v[std::min(v.size() - 1, static_cast<std::size_t>(p * static_cast<double>(v.size() - 1) + 0.5))];
}

struct Pattern {
    std::uint32_t per_token = 0;  // layers * columns
    std::vector<EngramRowRequest> rows;  // [token][layer][column]
};
Pattern make_pattern(const strix::deepseek41::TargetModel& model, std::uint32_t tokens, std::uint32_t seed) {
    EngramHasher hasher(model.engram);
    std::mt19937 rng(seed);
    std::uniform_int_distribution<std::uint32_t> tok(0, model.config.vocab_size - 1);
    std::vector<std::uint32_t> seq(tokens);
    for (auto& t : seq) t = tok(rng);
    const auto h = hasher.hash_sequence(seq);
    Pattern p;
    p.per_token = hasher.layers() * hasher.columns();
    p.rows.resize(h.size());
    for (std::size_t i = 0; i < h.size(); ++i) p.rows[i] = {static_cast<std::uint32_t>((i % p.per_token) / hasher.columns()), h[i]};
    return p;
}

// One batch per token; returns per-token latencies (ms).
std::vector<double> decode_run(EngramRowReader& reader, const Pattern& p, std::uint32_t first, std::uint32_t tokens) {
    std::vector<std::uint8_t> out(std::size_t{p.per_token} * engram_row_bytes);
    std::vector<double> lat;
    for (std::uint32_t t = first; t < first + tokens; ++t) {
        const auto t0 = Clock::now();
        reader.read(std::span<const EngramRowRequest>(p.rows.data() + std::size_t{t} * p.per_token, p.per_token), out);
        lat.push_back(ms_since(t0));
    }
    return lat;
}

// GPU load: the Q8_0 x A8 head-half projection (64640 x 5120, 351 MB), cold by construction.
struct GpuLoad {
    std::uint8_t* w = nullptr;
    std::int8_t* codes = nullptr;
    float* deltas = nullptr;
    float* out = nullptr;
    hipStream_t s{};
    Q8A8View v;
    GpuLoad() {
        const std::size_t bytes = std::size_t{64640} * 5120 / 32 * 34;
        check(hipMalloc(&w, bytes), "w");
        check(hipMemset(w, 0x11, bytes), "memset");
        check(hipMalloc(&codes, 5120), "codes");
        check(hipMemset(codes, 1, 5120), "memset");
        check(hipMalloc(&deltas, 160 * 4), "deltas");
        check(hipMemset(deltas, 0, 160 * 4), "memset");
        check(hipMalloc(&out, 64640 * 4), "out");
        check(hipStreamCreate(&s), "stream");
        v.weight = w; v.width = 5120; v.count = 64640; v.per_group = 64640; v.codes = codes; v.deltas = deltas;
        v.rows = 1; v.x_stride = 5120; v.out_stride = 64640; v.out = out;
    }
    ~GpuLoad() { (void)hipFree(w); (void)hipFree(codes); (void)hipFree(deltas); (void)hipFree(out); (void)hipStreamDestroy(s); }
    // Times `n` back-to-back launches with events; returns per-launch ms.
    std::vector<double> run(int n) {
        std::vector<hipEvent_t> ev(static_cast<std::size_t>(n) + 1);
        for (auto& e : ev) check(hipEventCreate(&e), "event");
        check(hipEventRecord(ev[0], s), "record");
        for (int i = 0; i < n; ++i) {
            enqueue_q8_0_a8(v, s);
            check(hipEventRecord(ev[static_cast<std::size_t>(i) + 1], s), "record");
        }
        check(hipStreamSynchronize(s), "sync");
        std::vector<double> t;
        for (int i = 0; i < n; ++i) {
            float ms = 0;
            check(hipEventElapsedTime(&ms, ev[static_cast<std::size_t>(i)], ev[static_cast<std::size_t>(i) + 1]), "elapsed");
            t.push_back(ms);
        }
        for (auto& e : ev) (void)hipEventDestroy(e);
        return t;
    }
};

void print_latency(const char* mode, unsigned threads, const std::vector<double>& lat, std::uint32_t per_token) {
    double sum = 0;
    for (double x : lat) sum += x;
    std::printf("{\"mode\":\"%s\",\"threads\":%u,\"tokens\":%zu,\"rows_per_token\":%u,\"p50_ms\":%.3f,\"p90_ms\":%.3f,\"p99_ms\":%.3f,"
                "\"max_ms\":%.3f,\"mean_ms\":%.3f,\"rows_per_s\":%.0f}\n",
                mode, threads, lat.size(), per_token, pct(lat, 0.5), pct(lat, 0.9), pct(lat, 0.99), pct(lat, 1.0),
                sum / static_cast<double>(lat.size()), static_cast<double>(lat.size()) * per_token / (sum * 1e-3));
    std::fflush(stdout);
}

} // namespace

int main(int argc, char** argv) {
    if (argc < 2) { std::fprintf(stderr, "usage: %s MODEL.gguf [tokens] [--no-gpu]\n", argv[0]); return 2; }
    const char* path = argv[1];
    const std::uint32_t tokens = argc > 2 && argv[2][0] != '-' ? static_cast<std::uint32_t>(std::atoi(argv[2])) : 1000;
    bool gpu = true;
    for (int i = 2; i < argc; ++i) if (std::strcmp(argv[i], "--no-gpu") == 0) gpu = false;
    const auto model = strix::deepseek41::validate_target(strix::deepseek41::read_file(path));
    const auto tables = engram_tables(model);
    // Fresh tokens for every measurement so no row repeats across runs.
    std::uint32_t seed = 1;
    std::uint32_t per_token = 0;

    // Decode: one batch of every layer's rows per token, threads sweep.
    for (const unsigned threads : {8U, 16U, 24U, 48U}) {
        EngramRowReader reader(path, tables, threads, true);
        const auto p = make_pattern(model, tokens, seed++);
        per_token = p.per_token;
        print_latency("decode", threads, decode_run(reader, p, 0, tokens), p.per_token);
    }
    // Decode, one layer at a time (rows needed by the first Engram layer only).
    {
        EngramRowReader reader(path, tables, 24, true);
        const auto p = make_pattern(model, tokens, seed++);
        Pattern half;
        half.per_token = p.per_token / 2;
        for (std::uint32_t t = 0; t < tokens; ++t)
            for (std::uint32_t k = 0; k < half.per_token; ++k) half.rows.push_back(p.rows[std::size_t{t} * p.per_token + k]);
        print_latency("decode-one-layer", 24, decode_run(reader, half, 0, tokens), half.per_token);
    }
    // Prefill: 2048 tokens per batch.
    for (const unsigned threads : {16U, 48U, 96U}) {
        EngramRowReader reader(path, tables, threads, true);
        const auto p = make_pattern(model, 2048, seed++);
        std::vector<std::uint8_t> out(p.rows.size() * engram_row_bytes);
        const auto t0 = Clock::now();
        reader.read(p.rows, out);
        const double ms = ms_since(t0);
        std::printf("{\"mode\":\"prefill\",\"threads\":%u,\"tokens\":2048,\"rows\":%zu,\"ms\":%.1f,\"rows_per_s\":%.0f,\"us_per_token\":%.1f}\n",
                    threads, p.rows.size(), ms, static_cast<double>(p.rows.size()) / (ms * 1e-3), ms * 1e3 / 2048.0);
        std::fflush(stdout);
    }
    if (!gpu) return 0;

    // Overlap: GPU kernel alone, reader alone, both at once.
    GpuLoad load;
    (void)load.run(5);
    const auto alone = load.run(200);
    std::printf("{\"mode\":\"gpu-alone\",\"kernel\":\"head-half q8a8 351MB\",\"p50_ms\":%.4f,\"p90_ms\":%.4f}\n", pct(alone, 0.5), pct(alone, 0.9));
    std::fflush(stdout);
    {
        EngramRowReader reader(path, tables, 24, true);
        const auto p = make_pattern(model, tokens, seed++);
        std::atomic<bool> stop{false};
        std::vector<double> gpu_times;
        std::thread g([&] { while (!stop.load()) { const auto t = load.run(50); gpu_times.insert(gpu_times.end(), t.begin(), t.end()); } });
        std::this_thread::sleep_for(std::chrono::milliseconds(50));
        const auto lat = decode_run(reader, p, 0, tokens);
        stop = true;
        g.join();
        print_latency("decode-under-gpu", 24, lat, per_token);
        std::printf("{\"mode\":\"gpu-under-reader\",\"kernel\":\"head-half q8a8 351MB\",\"launches\":%zu,\"p50_ms\":%.4f,\"p90_ms\":%.4f}\n",
                    gpu_times.size(), pct(gpu_times, 0.5), pct(gpu_times, 0.9));
    }
    {
        EngramRowReader reader(path, tables, 96, true);
        const auto p = make_pattern(model, 2048, seed++);
        std::vector<std::uint8_t> out(p.rows.size() * engram_row_bytes);
        std::atomic<bool> stop{false};
        std::vector<double> gpu_times;
        std::thread g([&] { while (!stop.load()) { const auto t = load.run(20); gpu_times.insert(gpu_times.end(), t.begin(), t.end()); } });
        std::this_thread::sleep_for(std::chrono::milliseconds(50));
        const auto t0 = Clock::now();
        reader.read(p.rows, out);
        const double ms = ms_since(t0);
        stop = true;
        g.join();
        std::printf("{\"mode\":\"prefill-under-gpu\",\"threads\":96,\"rows\":%zu,\"ms\":%.1f,\"rows_per_s\":%.0f,\"gpu_p50_ms\":%.4f}\n",
                    p.rows.size(), ms, static_cast<double>(p.rows.size()) / (ms * 1e-3), pct(gpu_times, 0.5));
    }
    return 0;
}
