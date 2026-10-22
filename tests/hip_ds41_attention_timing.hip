// Timing build of the deepseek41 attention kernels (compiled with STRIX_DS41_ATTN_TIMING): every block records wall
// clock and shader cycles, so each launch gives its span on the GPU, the per-block durations and the shader clock
// actually used. Every case runs cold (after 200 ms idle, one launch) and warm (after 300 back-to-back launches);
// a sampler thread reads the DPM level (pp_dpm_sclk) every 5 ms. The compress kernel also records shader cycles at its
// phase marks (DS41_CMARK; printed as median cycles since the previous mark), default and fast form. Usage:
// [pp_dpm_sclk path] [compress: the compress cases alone]. Not a correctness test.
#include "strix/hip_ds41_attention.hpp"

#include <algorithm>
#include <atomic>
#include <chrono>
#include <cstdio>
#include <cstring>
#include <fstream>
#include <map>
#include <random>
#include <stdexcept>
#include <string>
#include <thread>
#include <vector>

namespace a = strix::ds41::attn;
namespace g = strix::ds41::attn::hip;
namespace {
void check(hipError_t e) { if (e != hipSuccess) throw std::runtime_error(hipGetErrorString(e)); }
template <class T> T* dev(const std::vector<T>& h) {
    T* p; check(hipMalloc(&p, std::max<std::size_t>(h.size(), 1) * sizeof(T)));
    check(hipMemcpy(p, h.data(), h.size() * sizeof(T), hipMemcpyHostToDevice)); return p;
}
template <class T> T* dev_zero(std::size_t n) { T* p; check(hipMalloc(&p, std::max<std::size_t>(n, 1) * sizeof(T))); check(hipMemset(p, 0, n * sizeof(T))); return p; }
std::mt19937 rng(7);
std::vector<float> rnd(std::size_t n, float lo, float hi) { std::vector<float> v(n); for (auto& x : v) x = std::uniform_real_distribution<float>(lo, hi)(rng); return v; }

// DPM sampler: the current sclk level (MHz) every 5 ms
std::atomic<bool> sampling{false}, quit{false};
std::map<int, int> levels;
std::string sclk_path;
int read_sclk() {
    std::ifstream f(sclk_path);
    std::string line;
    while (std::getline(f, line)) if (line.find('*') != std::string::npos) return std::stoi(line.substr(line.find(' ') + 1));
    return -1;
}
void sampler() {
    while (!quit) {
        if (sampling) { const int mhz = read_sclk(); if (mhz > 0) ++levels[mhz]; }
        std::this_thread::sleep_for(std::chrono::milliseconds(5));
    }
}
std::string take_levels() {
    std::string s;
    for (const auto& [mhz, n] : levels) s += std::to_string(mhz) + "MHz x" + std::to_string(n) + " ";
    levels.clear();
    return s.empty() ? "-" : s;
}

unsigned long long* timing;
double wall_mhz;
const char* slot_name[] = {"window", "compress", "qprep", "scores", "select", "chunk", "combine"};
// span per kernel slot (max end - min start), median block duration, shader clock from cycles / wall time
void report(const char* label, float event_us) {
    std::vector<unsigned long long> t(15ULL * 65536 * 4);
    check(hipMemcpy(t.data(), timing, t.size() * 8, hipMemcpyDeviceToHost));
    std::printf("  %-6s event %8.1f us |", label, event_us);
    for (unsigned s = 0; s < 7; ++s) {
        unsigned long long lo = ~0ULL, hi = 0;
        std::vector<double> dur, mhz;
        for (unsigned b = 0; b < 65536; ++b) {
            const unsigned long long* e = &t[(s * 65536ULL + b) * 4];
            if (e[1] == 0) continue;
            lo = std::min(lo, e[0]); hi = std::max(hi, e[1]);
            const double us = double(e[1] - e[0]) / wall_mhz;
            dur.push_back(us);
            const unsigned long long cyc = (e[3] - e[2]) & 0xFFFFFULL;  // 20-bit shader cycle counter
            if (us > 0.5) mhz.push_back(double(cyc) / us);
        }
        if (dur.empty()) continue;
        std::sort(dur.begin(), dur.end()); std::sort(mhz.begin(), mhz.end());
        std::printf(" %s span %.1f us, blocks %zu, block med %.1f max %.1f us, clk %s MHz |", slot_name[s], double(hi - lo) / wall_mhz,
            dur.size(), dur[dur.size() / 2], dur.back(), mhz.empty() ? "-" : std::to_string(int(mhz[mhz.size() / 2])).c_str());
    }
    // chunk-kernel phase marks (slots 7..10): median gap between consecutive marks over the blocks that have both
    std::vector<double> gaps[12];
    for (unsigned b = 0; b < 65536; ++b) {
        unsigned long long prev = t[(5ULL * 65536 + b) * 4];
        if (prev == 0) continue;
        for (unsigned k = 0; k < 12; ++k) {
            const unsigned long long mk = t[((7ULL + k / 4) * 65536 + b) * 4 + k % 4];
            if (mk != 0) { gaps[k].push_back(double(mk - prev) / wall_mhz); prev = mk; }
        }
    }
    if (!gaps[0].empty()) {
        std::printf(" marks (median us since previous):");
        for (unsigned k = 0; k < 12; ++k) if (!gaps[k].empty()) { std::sort(gaps[k].begin(), gaps[k].end()); std::printf(" m%u %.1f", k, gaps[k][gaps[k].size() / 2]); }
    }
    // shader-cycle marks (slots 11..14): median cycles since the previous mark (the first since the block's entry)
    std::vector<double> cgaps[16];
    for (unsigned b = 0; b < 65536; ++b) {
        unsigned long long prev = 0;
        for (unsigned s = 0; s < 7 && prev == 0; ++s) if (t[(s * 65536ULL + b) * 4 + 1] != 0) prev = t[(s * 65536ULL + b) * 4 + 2];
        if (prev == 0) continue;
        for (unsigned k = 0; k < 16; ++k) {
            const unsigned long long mk = t[((11ULL + k / 4) * 65536 + b) * 4 + k % 4];
            if (mk != 0) { cgaps[k].push_back(double((mk - prev) & 0xFFFFFULL)); prev = mk; }
        }
    }
    bool any = false;
    for (unsigned k = 0; k < 16; ++k) if (!cgaps[k].empty()) {
        if (!any) std::printf("\n    cycle marks (median cycles since previous, blocks):");
        any = true;
        std::sort(cgaps[k].begin(), cgaps[k].end());
        std::printf(" c%u %.0f (%zu)", k, cgaps[k][cgaps[k].size() / 2], cgaps[k].size());
    }
    std::printf(" dpm: %s\n", take_levels().c_str());
    check(hipMemset(timing, 0, t.size() * 8));
}
template <class F> void measure(const char* name, F launch) {
    hipEvent_t e0, e1;
    check(hipEventCreate(&e0)); check(hipEventCreate(&e1));
    std::printf("%s\n", name);
    std::this_thread::sleep_for(std::chrono::milliseconds(200));
    check(hipDeviceSynchronize()); check(hipMemset(timing, 0, 15ULL * 65536 * 32)); check(hipDeviceSynchronize());
    take_levels();
    sampling = true;
    check(hipEventRecord(e0)); launch(); check(hipEventRecord(e1)); check(hipEventSynchronize(e1));
    float ms; check(hipEventElapsedTime(&ms, e0, e1));
    report("cold", ms * 1e3F);
    for (int i = 0; i < 300; ++i) launch();
    check(hipDeviceSynchronize()); check(hipMemset(timing, 0, 15ULL * 65536 * 32)); check(hipDeviceSynchronize()); take_levels();
    for (int i = 0; i < 20; ++i) launch();   // keep the clock up right before the measured launch
    check(hipEventRecord(e0)); launch(); check(hipEventRecord(e1)); check(hipEventSynchronize(e1));
    check(hipEventElapsedTime(&ms, e0, e1));
    report("warm", ms * 1e3F);
    sampling = false;
    check(hipEventDestroy(e0)); check(hipEventDestroy(e1));
}
}  // namespace

int main(int argc, char** argv) {
    try {
        sclk_path = argc > 1 ? argv[1] : "/sys/class/drm/card1/device/pp_dpm_sclk";
        int rate_khz = 0;
        check(hipDeviceGetAttribute(&rate_khz, hipDeviceAttributeWallClockRate, 0));
        wall_mhz = rate_khz / 1000.0;
        std::printf("wall clock %.1f MHz, sclk now %d MHz\n", wall_mhz, read_sclk());
        std::thread th(sampler);
        timing = dev_zero<unsigned long long>(15ULL * 65536 * 4);
        g::set_timing_buffer(timing);
        // attention: random ring and compressed caches, 640 visible keys per row
        const unsigned cap = a::kWindow + a::kMaxRows, ncache = 4096;
        std::vector<std::uint8_t> ring(cap * a::kWindowEntryBytes), kv(ncache * a::kKvEntryBytes);
        for (auto& b : ring) b = static_cast<std::uint8_t>(rng() & 0x7E);
        for (std::size_t s = 0; s < cap; ++s) for (unsigned b = 0; b < 16; ++b) ring[s * 528 + 512 + b] = 120;
        for (auto& b : kv) b = static_cast<std::uint8_t>(rng());
        for (std::size_t t = 0; t < ncache; ++t) for (unsigned b = 0; b < 32; ++b) kv[t * 288 + 256 + b] = 0x38;
        auto* dring = dev(ring); auto* dkv = dev(kv);
        const std::string only = argc > 2 ? argv[2] : "";  // "compress": the compress cases alone
        for (const unsigned rows : {1U, 8U}) {
            if (only == "compress") break;
            std::vector<std::uint64_t> pos(rows);
            for (unsigned r = 0; r < rows; ++r) pos[r] = 100000 + r;
            std::vector<a::RopeRow> rope; for (const auto p : pos) rope.push_back(a::rope_row(a::RopeKind::compressed, p));
            std::vector<std::uint32_t> idx(rows * a::kTopK), cnt(rows, a::kTopK);
            for (unsigned r = 0; r < rows; ++r) for (unsigned i = 0; i < a::kTopK; ++i) idx[r * a::kTopK + i] = 7 * i;
            auto* dq = dev(rnd(rows * 64 * 512, -0.5F, 0.5F)); auto* dsink = dev(rnd(64, -2, 2));
            auto* dpos = dev(pos); auto* drope = dev(rope); auto* didx = dev(idx); auto* dcnt = dev(cnt);
            auto* dpart = dev_zero<float>(std::size_t{rows} * 64 * a::kMaxChunks * 514); auto* dout = dev_zero<float>(rows * 64 * 512);
            auto* dtick = dev_zero<std::uint32_t>(rows * g::kAttnGroupCount);
            const g::AttentionView av{rows, cap, a::kTopK, dq, dsink, drope, dpos, dring, dkv, didx, dcnt, dpart, dout, dtick};
            measure(("attention, 640 keys, rows " + std::to_string(rows)).c_str(), [&] { g::enqueue_attention(av, nullptr); });
            g::AttentionView tp = av;
            tp.heads = 32;   // one tensor-parallel rank
            measure(("attention, 640 keys, 32 heads, rows " + std::to_string(rows)).c_str(), [&] { g::enqueue_attention(tp, nullptr); });
        }
        // compress at ratio 1 and 2: the first row at an odd position (a group's last token at ratio 2)
        {
            const auto norm = rnd(512, 0.5F, 1.5F), knorm = rnd(128, 0.5F, 1.5F);
            std::vector<std::uint16_t> wk(128 * 512);
            for (auto& h : wk) h = static_cast<std::uint16_t>(0x1800U + (rng() & 0xFFFU) + ((rng() & 1U) << 15U));
            auto* dnorm = dev(norm); auto* dknorm = dev(knorm); auto* dwk = dev(wk);
            auto* dstate_in = dev(rnd(1024, -3, 3)); auto* dvalid_in = dev(std::vector<std::uint32_t>{1U});
            auto* dstate_out = dev_zero<float>(1024); auto* dvalid_out = dev_zero<std::uint32_t>(1);
            auto* dkvc = dev_zero<std::uint8_t>(std::size_t{60000} * a::kKvEntryBytes);
            auto* didxc = dev_zero<std::uint8_t>(std::size_t{60000} * a::kIdxEntryBytes);
            for (const unsigned ratio : {1U, 2U}) for (const unsigned rows : {1U, 2U, 6U, 8U}) {
                std::vector<std::uint64_t> pos(rows);
                std::vector<a::RopeRow> rope(rows);
                for (unsigned r = 0; r < rows; ++r) { pos[r] = 50001 + r; rope[r] = a::rope_row(a::RopeKind::compressed, pos[r] / ratio * ratio); }
                auto* dkvr = dev(rnd(rows * 512, -3, 3)); auto* dsc = dev(rnd(rows * 512, -4, 4));
                auto* dpos = dev(pos); auto* drope = dev(rope);
                const g::CompressView cv{ratio, rows, dkvr, dsc, dnorm, dwk, dknorm, drope, dpos, dstate_in, dvalid_in, dstate_out, dvalid_out, dkvc, didxc};
                for (const int fast : {0, 1}) {
                    g::set_compress_fast(fast);
                    measure(("compress" + std::string(fast != 0 ? " (fast)" : "") + ", ratio " + std::to_string(ratio) + ", rows " + std::to_string(rows)).c_str(),
                            [&] { g::enqueue_compress(cv, nullptr); });
                }
                g::set_compress_fast(-1);
            }
        }
        // index scores + select at 32K and 131K positions (ratio 1)
        for (const std::uint32_t count : {32768U, 131072U}) for (const unsigned rows : {1U, 8U}) {
            if (only == "compress") break;
            std::vector<std::uint8_t> cache(std::size_t{count} * 68);
            for (auto& b : cache) b = static_cast<std::uint8_t>(rng());
            for (std::size_t t = 0; t < count; ++t) for (unsigned b = 0; b < 4; ++b) cache[t * 68 + 64 + b] = 124;
            std::vector<std::uint64_t> pos(rows);
            for (unsigned r = 0; r < rows; ++r) pos[r] = count - rows + r;
            std::vector<a::RopeRow> rope; for (const auto p : pos) rope.push_back(a::rope_row(a::RopeKind::compressed, p));
            auto* dcache = dev(cache); auto* dq = dev(rnd(rows * 32 * 128, -2, 2)); auto* dw = dev(rnd(rows * 32, -1, 1));
            auto* dpos = dev(pos); auto* drope = dev(rope);
            auto* dscores = dev_zero<float>(std::size_t{rows} * count); auto* dprep = dev_zero<std::uint8_t>(rows * g::kIdxPrepBytes);
            auto* dcand = dev_zero<std::uint8_t>(std::size_t{rows} * (count / 8 + 1)); auto* dkeys = dev_zero<std::uint32_t>(std::size_t{rows} * count);
            auto* didx = dev_zero<std::uint32_t>(rows * a::kTopK); auto* dcnt = dev_zero<std::uint32_t>(rows);
            const g::IndexView iv{rows, 1, count, dq, dw, drope, dpos, dcache, dscores, dprep};
            const g::SelectView sv{rows, 1, count, a::kTopK, a::kCandBlocks, a::kCandBlockSize, a::Pool::candidate_source, dpos, dscores, dcand, dkeys, didx, dcnt};
            const std::string tag = std::to_string(count) + " positions, rows " + std::to_string(rows);
            measure(("index scores, " + tag).c_str(), [&] { g::enqueue_index_scores(iv, nullptr); });
            measure(("select (candidate source), " + tag).c_str(), [&] { g::enqueue_select(sv, nullptr); });
            std::printf("  (index keys: %.1f MB, floor at 220 GB/s: %.1f us)\n", count * 68.0 / 1e6, count * 68.0 / 220e3);
        }
        quit = true; th.join();
    } catch (const std::exception& e) {
        std::fprintf(stderr, "error: %s\n", e.what());
        return 1;
    }
    return 0;
}
