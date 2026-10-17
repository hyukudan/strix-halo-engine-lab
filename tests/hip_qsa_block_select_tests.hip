#include "strix/hip_indexed_attention_launcher.hpp"
#include <hip/hip_runtime.h>
#include <algorithm>
#include <array>
#include <bit>
#include <cmath>
#include <cstdint>
#include <cstdlib>
#include <cstring>
#include <iomanip>
#include <iostream>
#include <limits>
#include <numeric>
#include <stdexcept>
#include <string>
#include <vector>

// Byte-exactness of the parallel QSA top-k block selection against the original
// single-thread loop (and a host model of it), for the C1 kernel (eager and
// graph-captured with a device position) and the known-window rows kernel.
namespace {
using namespace strix;
constexpr std::uint32_t ratio = 4, topk = 512, max_visible = 262144 + 8, max_blocks = max_visible / ratio;
constexpr std::uint32_t max_selected = topk * ratio + ratio, streams = 16;
void require(bool ok, const std::string& text) { if (!ok) throw std::runtime_error(text); }
void check(hipError_t e, const char* text) { if (e != hipSuccess) throw std::runtime_error(std::string(text) + ": " + hipGetErrorString(e)); }
template <class T> T* device(std::size_t count) { void* p = nullptr; check(hipMalloc(&p, count * sizeof(T)), "hipMalloc"); return static_cast<T*>(p); }

struct Rng { std::uint64_t s; std::uint32_t next() { s ^= s << 13; s ^= s >> 7; s ^= s << 17; return static_cast<std::uint32_t>(s >> 11); }
    float unit() { return static_cast<float>(next() >> 8) * (1.0F / 16777216.0F); } };
float bits(std::uint32_t value) { return std::bit_cast<float>(value); }
constexpr float ninf = -std::numeric_limits<float>::infinity();

enum class Dist { uniform, coarse_ties, all_equal, half_ninf, few_valid, all_ninf, specials, raw_bits, low_ulps, ascending, descending, zero_block0 };
constexpr std::array dists{Dist::uniform, Dist::coarse_ties, Dist::all_equal, Dist::half_ninf, Dist::few_valid, Dist::all_ninf,
    Dist::specials, Dist::raw_bits, Dist::low_ulps, Dist::ascending, Dist::descending, Dist::zero_block0};
std::vector<float> scores(Dist dist, std::uint32_t blocks, std::uint64_t seed) {
    Rng rng{seed * 0x9e3779b97f4a7c15ULL + 1U};
    std::vector<float> v(blocks);
    for (std::uint32_t i = 0; i < blocks; ++i) {
        float x = rng.unit() * 3.0F;
        switch (dist) {
        case Dist::uniform: break;
        case Dist::coarse_ties: x = static_cast<float>(rng.next() % 4U) * 0.5F; break;
        case Dist::all_equal: x = 1.25F; break;
        case Dist::half_ninf: if (rng.next() & 1U) x = ninf; break;
        case Dist::few_valid: x = rng.next() % (blocks / 100U + 1U) == 0 ? x : ninf; break;
        case Dist::all_ninf: x = ninf; break;
        case Dist::specials: {
            static constexpr std::array<std::uint32_t, 12> special{0x7fc00000U, 0xffc00000U, 0x7f800001U, 0x7f800000U, 0xff800000U,
                0x00000000U, 0x80000000U, 0x00000001U, 0x807fffffU, 0x3f800000U, 0xbf800000U, 0x00800000U};
            const auto pick = rng.next() % 16U;
            x = pick < special.size() ? bits(special[pick]) : x - 1.5F; break;
        }
        case Dist::raw_bits: x = bits(rng.next() << 1U ^ rng.next()); break;
        case Dist::low_ulps: x = bits(0x3fa00000U + rng.next() % 1500U); break;
        case Dist::ascending: x = static_cast<float>(i) * 0.25F; break;
        case Dist::descending: x = static_cast<float>(blocks - i) * 0.25F; break;
        case Dist::zero_block0: x = i == 0 ? ninf : (i % 7U ? ninf : 0.0F); break;
        }
        v[i] = x;
    }
    return v;
}
// Host model of the serial loop's result: candidates > -inf by (value desc, index asc),
// block 0 for every slot left, ascending block order, then the partial-block tail.
void model(std::vector<float>& s, std::uint32_t visible, std::vector<std::uint32_t>& out) {
    const auto blocks = visible / ratio;
    out.assign(max_selected, 0xffffffffU);
    if (blocks <= topk) { for (std::uint32_t t = 0; t < visible; ++t) out[t] = t; return; }
    std::vector<std::uint32_t> c;
    for (std::uint32_t b = 0; b < blocks; ++b) if (s[b] > ninf) c.push_back(b);
    std::stable_sort(c.begin(), c.end(), [&](auto a, auto b) { return s[a] > s[b]; });
    c.resize(std::min<std::size_t>(c.size(), topk));
    const bool fallback = c.size() < topk;
    for (auto b : c) s[b] = ninf;
    if (fallback) s[0] = ninf;
    c.insert(c.begin(), topk - c.size(), 0U);
    std::sort(c.begin(), c.end());
    for (std::uint32_t i = 0; i < topk; ++i) for (std::uint32_t o = 0; o < ratio; ++o) out[i * ratio + o] = c[i] * ratio + o;
    for (std::uint32_t t = blocks * ratio; t < visible; ++t) out[topk * ratio + t - blocks * ratio] = t;
}

struct Slot { float* scores; std::uint32_t* selected; };
struct Fixture {
    std::array<hipStream_t, streams> stream{};
    std::vector<Slot> slots;
    std::uint64_t* position = device<std::uint64_t>(1);
    std::uint32_t* status = device<std::uint32_t>(1);
    Fixture(std::size_t count) {
        for (auto& s : stream) check(hipStreamCreateWithFlags(&s, hipStreamNonBlocking), "stream");
        for (std::size_t i = 0; i < count; ++i) slots.push_back({device<float>(max_blocks), device<std::uint32_t>(max_selected)});
    }
    void sync() { for (auto s : stream) check(hipStreamSynchronize(s), "sync"); }
    void load(const Slot& slot, const std::vector<float>& s) {
        std::vector<float> padded(max_blocks, bits(0x7fa5a5a5U));
        std::copy(s.begin(), s.end(), padded.begin());
        check(hipMemcpy(slot.scores, padded.data(), max_blocks * 4U, hipMemcpyHostToDevice), "scores upload");
        check(hipMemset(slot.selected, 0xff, max_selected * 4U), "selected sentinel");
    }
    std::pair<std::vector<std::uint32_t>, std::vector<std::uint32_t>> read(const Slot& slot) {
        std::vector<std::uint32_t> s(max_blocks), o(max_selected);
        check(hipMemcpy(s.data(), slot.scores, max_blocks * 4U, hipMemcpyDeviceToHost), "scores read");
        check(hipMemcpy(o.data(), slot.selected, max_selected * 4U, hipMemcpyDeviceToHost), "selected read");
        return {s, o};
    }
};

// One (context, rows) family: every distribution's rows at visible = context + r,
// serial reference on many streams concurrently, candidate on the window/C1 kernel.
unsigned family(Fixture& f, std::uint32_t context, std::uint32_t rows) {
    const auto count = static_cast<std::uint32_t>(dists.size()) * rows;
    std::vector<std::vector<float>> host(count);
    for (std::uint32_t d = 0; d < dists.size(); ++d)
        for (std::uint32_t r = 0; r < rows; ++r) {
            const auto i = d * rows + r;
            host[i] = scores(dists[d], (context + r) / ratio, context * 131U + rows * 17U + i);
            f.load(f.slots[2U * i], host[i]); f.load(f.slots[2U * i + 1U], host[i]);
        }
    for (std::uint32_t i = 0; i < count; ++i) {
        const HipQsaBlockSelectRow row{f.slots[2U * i].scores, f.slots[2U * i].selected, context + i % rows};
        launch_hip_qsa_block_select_probe(true, {&row, 1}, ratio, topk, nullptr, 0, nullptr, f.stream[i % streams]);
    }
    for (std::uint32_t d = 0; d < dists.size(); ++d) {
        std::vector<HipQsaBlockSelectRow> window;
        for (std::uint32_t r = 0; r < rows; ++r) {
            const auto& slot = f.slots[2U * (d * rows + r) + 1U];
            window.push_back({slot.scores, slot.selected, context + r});
        }
        launch_hip_qsa_block_select_probe(false, window, ratio, topk, nullptr, 0, nullptr, f.stream[0]);
    }
    f.sync();
    for (std::uint32_t i = 0; i < count; ++i) {
        const auto visible = context + i % rows;
        const auto reference = f.read(f.slots[2U * i]), candidate = f.read(f.slots[2U * i + 1U]);
        const auto where = " (context " + std::to_string(visible) + ", rows " + std::to_string(rows) + ", dist " +
            std::to_string(i / rows) + ")";
        require(candidate.second == reference.second, "selected indices differ from the serial kernel" + where);
        require(candidate.first == reference.first, "consumed scores differ from the serial kernel" + where);
        std::vector<std::uint32_t> expected;
        auto s = host[i];
        model(s, visible, expected);
        require(expected == reference.second, "serial kernel differs from the host model" + where);
        for (std::uint32_t b = 0; b < s.size(); ++b) require(std::bit_cast<std::uint32_t>(s[b]) == reference.first[b], "host model scores" + where);
    }
    return count;
}

// C1 kernel captured once with a device position; replayed per context.
unsigned graph_family(Fixture& f) {
    hipGraph_t graph{}; hipGraphExec_t exec{};
    const HipQsaBlockSelectRow candidate{f.slots[1].scores, f.slots[1].selected, 0};
    check(hipStreamBeginCapture(f.stream[0], hipStreamCaptureModeThreadLocal), "capture");
    launch_hip_qsa_block_select_probe(false, {&candidate, 1}, ratio, topk, f.position, max_visible, f.status, f.stream[0]);
    check(hipStreamEndCapture(f.stream[0], &graph), "end capture");
    check(hipGraphInstantiate(&exec, graph, nullptr, nullptr, 0), "instantiate");
    unsigned cases = 0;
    for (const std::uint32_t visible : {1U, 2048U, 2051U, 2052U, 2053U, 4096U, 32768U, 131072U, 262144U, 262147U})
        for (const auto dist : {Dist::uniform, Dist::coarse_ties, Dist::specials, Dist::few_valid})
            for (const std::uint32_t state : {0U, 1U, 2U}) {
                // state 1: an earlier failure is published; 2: position past the context.
                const std::uint64_t position = state == 2U ? max_visible : visible - 1U;
                const std::uint32_t status = state == 1U ? 8U : 0U;
                const auto host = scores(dist, max_blocks, visible + static_cast<std::uint32_t>(dist));
                f.load(f.slots[0], host); f.load(f.slots[1], host);
                check(hipMemcpy(f.position, &position, 8U, hipMemcpyHostToDevice), "position");
                check(hipMemcpy(f.status, &status, 4U, hipMemcpyHostToDevice), "status");
                const HipQsaBlockSelectRow reference{f.slots[0].scores, f.slots[0].selected, 0};
                launch_hip_qsa_block_select_probe(true, {&reference, 1}, ratio, topk, f.position, max_visible, f.status, f.stream[0]);
                check(hipGraphLaunch(exec, f.stream[0]), "graph launch");
                f.sync();
                const auto a = f.read(f.slots[0]), b = f.read(f.slots[1]);
                const auto where = " (graph visible " + std::to_string(visible) + ", state " + std::to_string(state) + ")";
                require(a == b, "graph-form selection differs from the serial kernel" + where);
                if (state) require(std::all_of(b.second.begin(), b.second.end(), [](auto x) { return x == 0xffffffffU; }),
                    "rejected graph selection wrote indices" + where);
                ++cases;
            }
    check(hipGraphExecDestroy(exec), "destroy exec"); check(hipGraphDestroy(graph), "destroy graph");
    return cases;
}

float median(std::vector<float> v) { std::sort(v.begin(), v.end()); return v[v.size() / 2]; }
void bench(Fixture& f) {
    hipEvent_t start, stop; check(hipEventCreate(&start), "event"); check(hipEventCreate(&stop), "event");
    float* pristine = device<float>(8U * max_blocks);
    std::cout << "context dist rows | serial_1row_ms parallel_1row_ms rows_kernel_ms T*parallel_ms T*serial_ms\n";
    for (const std::uint32_t context : {2049U, 4096U, 32768U, 131072U, 262144U})
        for (const auto dist : {Dist::uniform, Dist::all_equal}) {
            for (std::uint32_t r = 0; r < 8U; ++r) {
                const auto host = scores(dist, (context + r) / ratio, r + 7U);
                check(hipMemcpy(pristine + r * max_blocks, host.data(), host.size() * 4U, hipMemcpyHostToDevice), "pristine");
            }
            const auto time = [&](bool reference, std::uint32_t rows, unsigned runs) {
                std::vector<HipQsaBlockSelectRow> window;
                for (std::uint32_t r = 0; r < rows; ++r) window.push_back({f.slots[r].scores, f.slots[r].selected, context + r});
                std::vector<float> ms;
                for (unsigned run = 0; run < runs + 2U; ++run) {
                    for (std::uint32_t r = 0; r < rows; ++r)
                        check(hipMemcpyAsync(f.slots[r].scores, pristine + r * max_blocks, max_blocks * 4U, hipMemcpyDeviceToDevice, f.stream[0]), "restore");
                    check(hipEventRecord(start, f.stream[0]), "record");
                    launch_hip_qsa_block_select_probe(reference, window, ratio, topk, nullptr, 0, nullptr, f.stream[0]);
                    check(hipEventRecord(stop, f.stream[0]), "record");
                    check(hipEventSynchronize(stop), "event sync");
                    float elapsed = 0; check(hipEventElapsedTime(&elapsed, start, stop), "elapsed");
                    if (run >= 2U) ms.push_back(elapsed);
                }
                return median(ms);
            };
            const auto serial = time(true, 1, context > 32768U ? 3U : 21U), parallel = time(false, 1, 25U);
            for (const std::uint32_t rows : {1U, 2U, 4U, 8U}) {
                const auto window = rows == 1U ? parallel : time(false, rows, 25U);
                std::cout << context << ' ' << (dist == Dist::uniform ? "uniform" : "all_equal") << ' ' << rows << " | " << std::fixed
                          << std::setprecision(4) << serial << ' ' << parallel << ' ' << window << ' ' << static_cast<float>(rows) * parallel << ' '
                          << static_cast<float>(rows) * serial << '\n';
            }
        }
}
} // namespace

int main() try {
    int devices = 0;
    if (hipGetDeviceCount(&devices) != hipSuccess || devices == 0) { std::cout << "no HIP device\n"; return 77; }
    Fixture f(2U * dists.size() * 8U);
    if (std::getenv("STRIX_QSA_SELECT_BENCH")) { bench(f); return 0; }
    unsigned cases = 0;
    std::vector<std::uint32_t> contexts{2048U, 2049U, 2051U, 2052U, 2053U, 2301U, 4096U, 32768U, 131072U, 262144U};
    if (const char* only = std::getenv("STRIX_QSA_SELECT_CONTEXTS")) { contexts.clear(); contexts.push_back(static_cast<std::uint32_t>(std::atoi(only))); }
    for (const auto context : contexts)
        for (const std::uint32_t rows : {1U, 2U, 4U, 8U}) {
            cases += family(f, context, rows);
            std::cout << "context " << context << " rows " << rows << " ok\n" << std::flush;
        }
    std::cout << "QSA parallel block selection bit-exact row cases passed: " << cases << '\n';
    std::cout << "QSA graph-form selection cases passed: " << graph_family(f) << '\n';
    return 0;
} catch (const std::exception& error) {
    std::cerr << "QSA block selection test failed: " << error.what() << '\n';
    return 1;
}
