// Engram O_DIRECT row reads as the decode runners issue them (DeviceForward::step_window): one batch per step of
// R rows x every Engram layer x every column, rows from the hasher over a token stream. Time per batch vs reader threads.
//   strix-ds41-engram-reader-bench MODEL.gguf [threads,threads,...] [rows,rows,...] [batches]
#include "strix/deepseek41_model.hpp"
#include "strix/ds41_engram.hpp"
#include "strix/ds41_formats.hpp"

#include <algorithm>
#include <chrono>
#include <cstdio>
#include <cstdlib>
#include <random>
#include <sstream>
#include <string>
#include <vector>

namespace {
std::vector<unsigned> list(const char* s, std::vector<unsigned> fallback) {
    if (s == nullptr) return fallback;
    std::vector<unsigned> out;
    std::stringstream ss(s);
    for (std::string t; std::getline(ss, t, ',');) out.push_back(static_cast<unsigned>(std::stoul(t)));
    return out;
}
}  // namespace

int main(int argc, char** argv) try {
    if (argc < 2) {
        std::fprintf(stderr, "usage: strix-ds41-engram-reader-bench MODEL.gguf [threads,...] [rows,...] [batches]\n");
        return 2;
    }
    const char* path = argv[1];
    const auto threads = list(argc > 2 ? argv[2] : nullptr, {16, 32, 64, 128});
    const auto window = list(argc > 3 ? argv[3] : nullptr, {1, 2, 4, 6});
    const unsigned batches = argc > 4 ? static_cast<unsigned>(std::stoul(argv[4])) : 100U;
    const auto model = strix::deepseek41::validate_target(strix::deepseek41::read_file(path));
    const auto tables = strix::ds41::engram_tables(model);
    strix::ds41::EngramHasher hasher(model.engram);
    const std::uint32_t layers = hasher.layers(), cols = hasher.columns();
    std::mt19937 rng(11);
    std::vector<std::uint64_t> rows(std::size_t{layers} * cols);
    std::printf("Engram tables %zu, %u columns per layer, %u batches per point\n", tables.size(), cols, batches);
    std::printf("threads  rows  requests  mean_ms  p50_ms  p90_ms\n");
    for (const unsigned n : threads) {
        strix::ds41::EngramRowReader reader(path, tables, n, true);
        for (const unsigned R : window) {
            std::vector<double> ms;
            for (unsigned b = 0; b < batches + 2U; ++b) {
                std::vector<strix::ds41::EngramRowRequest> req;
                for (unsigned r = 0; r < R; ++r) {
                    hasher.push(static_cast<std::uint32_t>(rng() % model.config.vocab_size), rows);
                    for (std::uint32_t li = 0; li < layers; ++li)
                        for (std::uint32_t k = 0; k < cols; ++k) req.push_back({li, rows[std::size_t{li} * cols + k]});
                }
                std::vector<std::uint8_t> out(req.size() * strix::ds41::engram_row_bytes);
                const auto t0 = std::chrono::steady_clock::now();
                reader.read(req, out);
                const double t = std::chrono::duration<double, std::milli>(std::chrono::steady_clock::now() - t0).count();
                if (b >= 2U) ms.push_back(t);  // the first batches warm the threads
            }
            std::sort(ms.begin(), ms.end());
            double mean = 0.0;
            for (const double v : ms) mean += v;
            mean /= static_cast<double>(ms.size());
            std::printf("%7u %5u %9zu %8.3f %7.3f %7.3f\n", n, R, std::size_t{R} * layers * cols, mean, ms[ms.size() / 2], ms[ms.size() * 9 / 10]);
            std::fflush(stdout);
        }
    }
    return 0;
} catch (const std::exception& e) {
    std::fprintf(stderr, "error: %s\n", e.what());
    return 1;
}
