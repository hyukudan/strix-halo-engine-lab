// No-model HIP check and bench of the expert-union launches (microscaled_r1::launch_gate_up_union / launch_down_union)
// against the served per-row path (launch_rows, scratch 16 rows = ONE rows launch) on Inkling-shaped experts
// (128 experts, top-8, hidden 2304, ffn 1152, NVFP4 and MXFP8), R = 4 / 8 / 16 routed rows, uniform, served-like and correlated
// routes. Prints per case
//   ID <case> R=<r> union vs rows: EQUAL|DIFF      (every activation and output word bitwise)
//   TIME <case> R=<r> rows <us> union <us> GB/s rows <x> union <y> | ...
//        us per visit = gate_up + down over 16 rotating route sets (expert bytes come from DRAM); GB/s = bytes of the
//        assignments (rows) / of the distinct experts (union) over the time of that form; then the distinct count and
//        the gate / down split of both forms.
// Exit 0 only when every ID line is EQUAL.
#include "strix/hip/legacy_intrinsics.hpp"
#include "strix/hip/legacy_launch.hpp"
#include "moe_core/microscaled_r1.hpp"

#include <algorithm>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <random>
#include <set>
#include <vector>

namespace {

namespace r1 = strix::moe_core::microscaled_r1;
namespace packed = strix::weight::microscaled;
using Expert = strix::moe_core::device_grouped::ExpertWeights;
using packed::MatrixFormat;

constexpr unsigned kExperts = 128U, kTopK = 8U, kHidden = 2304U, kFfn = 1152U, kRowsMax = 16U, kSets = 16U;

#define CHECK_HIP(...)                                                                                  \
    do {                                                                                                \
        if (const hipError_t e = (__VA_ARGS__); e != hipSuccess) {                                                \
            std::fprintf(stderr, "%s:%d %s: %s\n", __FILE__, __LINE__, #__VA_ARGS__, hipGetErrorString(e));       \
            std::exit(2);                                                                               \
        }                                                                                               \
    } while (0)

__global__ void fill_bytes(std::uint8_t* out, std::size_t count, std::uint32_t seed, unsigned kind) {
    for (std::size_t i = blockIdx.x * std::size_t{blockDim.x} + threadIdx.x; i < count;
         i += std::size_t{gridDim.x} * blockDim.x) {
        std::uint32_t h = static_cast<std::uint32_t>(i) * 0x9E3779B1U ^ seed ^ static_cast<std::uint32_t>(i >> 32U);
        h ^= h >> 15U;
        h *= 0x85EBCA77U;
        h ^= h >> 13U;
        auto v = static_cast<std::uint8_t>(h);
        if (kind == 1U) v = static_cast<std::uint8_t>(0x28U + (h >> 8U) % 0x18U);  // NVFP4 E4M3 block scales
        if (kind == 2U && (v & 0x7FU) == 0x7FU) v ^= 1U;                             // MXFP8 E4M3 codes, no NaN
        if (kind == 3U) v = static_cast<std::uint8_t>(0x76U + (h >> 8U) % 8U);      // MXFP8 E8M0 block scales
        out[i] = v;
    }
}

struct Bank {
    MatrixFormat format;
    std::vector<void*> blocks;
    Expert* matrices{};
    double expert_bytes{};

    packed::PackedMatrix make(unsigned rows, unsigned columns, std::uint32_t seed) {
        const bool nv = format == MatrixFormat::nvfp4;
        const std::size_t codes = std::size_t{rows} * columns / (nv ? 2U : 1U);
        const std::size_t scales = std::size_t{rows} * columns / (nv ? 16U : 32U);
        std::uint8_t *c{}, *s{};
        CHECK_HIP(hipMalloc(&c, codes));
        CHECK_HIP(hipMalloc(&s, scales));
        blocks.push_back(c);
        blocks.push_back(s);
        fill_bytes<<<1024, 256>>>(c, codes, seed, nv ? 0U : 2U);
        fill_bytes<<<256, 256>>>(s, scales, seed * 7U + 1U, nv ? 1U : 3U);
        CHECK_HIP(hipGetLastError());
        packed::PackedMatrix m{};
        m.codes = c;
        m.block_scales = s;
        m.rows = rows;
        m.columns = columns;
        m.outer_scale = nv ? 0.03125F : 1.0F;
        return m;
    }

    explicit Bank(MatrixFormat f) : format(f) {
        std::vector<Expert> host(kExperts);
        for (unsigned e = 0; e < kExperts; ++e) {
            host[e].gate_up = make(2U * kFfn, kHidden, 1000U + e);
            host[e].down = make(kHidden, kFfn, 5000U + e);
        }
        const bool nv = f == MatrixFormat::nvfp4;
        expert_bytes = (3.0 * kFfn * kHidden) * (nv ? 0.5 + 1.0 / 16.0 : 1.0 + 1.0 / 32.0);
        CHECK_HIP(hipMalloc(&matrices, sizeof(Expert) * kExperts));
        CHECK_HIP(hipMemcpy(matrices, host.data(), sizeof(Expert) * kExperts, hipMemcpyHostToDevice));
        CHECK_HIP(hipDeviceSynchronize());
    }
    Bank(const Bank&) = delete;
    Bank& operator=(const Bank&) = delete;
    ~Bank() {
        for (void* p : blocks) (void)hipFree(p);
        (void)hipFree(matrices);
    }
};

// Route cases: uniform over the 128 experts; served = uniform over a pool of 64 (R = 8: ~42 distinct experts of 64
// assignments, the 43.9 measured on real C8 cohort passes, coh3 p1); correlated = a pool of 12.
enum class Routes : unsigned { uniform, served, correlated };
constexpr const char* route_name(Routes r) {
    return r == Routes::uniform ? "uniform" : (r == Routes::served ? "served" : "correlated");
}

std::vector<unsigned> make_ids(Routes routes, unsigned rows, std::mt19937& rng) {
    std::vector<unsigned> ids(std::size_t{rows} * kTopK);
    std::vector<unsigned> pool(kExperts);
    for (unsigned e = 0; e < kExperts; ++e) pool[e] = e;
    std::shuffle(pool.begin(), pool.end(), rng);
    const std::vector<unsigned> near(pool.begin(), pool.begin() + (routes == Routes::served ? 64 : 12));
    for (unsigned row = 0; row < rows; ++row) {
        std::vector<unsigned> v;
        if (routes != Routes::uniform) {
            v = near;
            std::shuffle(v.begin(), v.end(), rng);
            v.resize(kTopK);
        } else {
            std::shuffle(pool.begin(), pool.end(), rng);
            v.assign(pool.begin(), pool.begin() + kTopK);
        }
        std::copy(v.begin(), v.end(), ids.begin() + std::ptrdiff_t{row} * kTopK);
    }
    return ids;
}

struct Buffers {
    unsigned *ids{}, *failure{};
    float *weights{}, *input{}, *output{};
    std::uint16_t* activation{};
    Buffers() {
        CHECK_HIP(hipMalloc(&ids, sizeof(unsigned) * kSets * kRowsMax * kTopK));
        CHECK_HIP(hipMalloc(&failure, sizeof(unsigned) * kRowsMax));
        CHECK_HIP(hipMemset(failure, 0, sizeof(unsigned) * kRowsMax));
        CHECK_HIP(hipMalloc(&weights, sizeof(float) * kRowsMax * kTopK));
        CHECK_HIP(hipMalloc(&input, sizeof(float) * kRowsMax * kHidden));
        CHECK_HIP(hipMalloc(&output, sizeof(float) * kRowsMax * kHidden));
        CHECK_HIP(hipMalloc(&activation, sizeof(std::uint16_t) * kRowsMax * kTopK * kFfn));
        std::mt19937 rng(77U);
        std::uniform_real_distribution<float> x(-2.0F, 2.0F), w(0.01F, 0.3F);
        std::vector<float> in(std::size_t{kRowsMax} * kHidden), wt(std::size_t{kRowsMax} * kTopK);
        for (auto& v : in) v = x(rng);
        for (auto& v : wt) v = w(rng);
        CHECK_HIP(hipMemcpy(input, in.data(), sizeof(float) * in.size(), hipMemcpyHostToDevice));
        CHECK_HIP(hipMemcpy(weights, wt.data(), sizeof(float) * wt.size(), hipMemcpyHostToDevice));
    }
    Buffers(const Buffers&) = delete;
    Buffers& operator=(const Buffers&) = delete;
    ~Buffers() {
        (void)hipFree(ids);
        (void)hipFree(failure);
        (void)hipFree(weights);
        (void)hipFree(input);
        (void)hipFree(output);
        (void)hipFree(activation);
    }
    r1::Args args(const Expert* m, unsigned set) const {
        return {m,      ids + std::size_t{set} * kRowsMax * kTopK, weights, failure, input, activation,
                output, kExperts, kHidden, kFfn, 4.0F, 25.0F};
    }
};

// plan = the production decode plan (launch_rows with the expert union admitted: microscaled_r1::launch).
enum class Form { rows, rows_gate, rows_down, uni, uni_gate, uni_down, plan };

template <MatrixFormat F, unsigned R>
hipError_t run(const Bank& bank, const Buffers& b, unsigned set, Form form) {
    const auto args = b.args(bank.matrices, set);
    if (form == Form::rows) return r1::launch_rows<F, kTopK>(args, nullptr, R, kRowsMax, false);
    if (form == Form::plan) return r1::launch_rows<F, kTopK>(args, nullptr, R, kRowsMax, true);
    namespace d = r1::detail;
    if (form == Form::rows_gate) return d::enqueue_gate_up<F, kTopK>(args, nullptr, R);
    if (form == Form::rows_down) {
        d::down_combine_rows<F, kTopK><<<dim3(r1::down_blocks(kHidden, d::kDownColumns), R), d::kThreads, 0, nullptr>>>(args);
        return hipGetLastError();
    }
    if (form == Form::uni_gate) return r1::launch_gate_up_union<F, kTopK, R>(args, R, nullptr);
    if (form == Form::uni_down) return r1::launch_down_union<F, kTopK, R>(args, R, nullptr);
    if (const auto s = r1::launch_gate_up_union<F, kTopK, R>(args, R, nullptr); s != hipSuccess) return s;
    return r1::launch_down_union<F, kTopK, R>(args, R, nullptr);
}

std::uint64_t fnv(const void* data, std::size_t bytes) {
    std::uint64_t h = 0xcbf29ce484222325ULL;
    const auto* p = static_cast<const std::uint8_t*>(data);
    for (std::size_t i = 0; i < bytes; ++i) h = (h ^ p[i]) * 0x100000001B3ULL;
    return h;
}

template <MatrixFormat F, unsigned R>
bool one(const Bank& bank, Buffers& b, std::mt19937& rng, const char* fmt) {
    constexpr unsigned kRounds = 16U;
    hipEvent_t start, stop;
    CHECK_HIP(hipEventCreate(&start));
    CHECK_HIP(hipEventCreate(&stop));
    bool ok = true;
    for (Routes routes : {Routes::uniform, Routes::served, Routes::correlated}) {
        std::vector<unsigned> all(std::size_t{kSets} * kRowsMax * kTopK, 0U);
        double distinct = 0.0;
        for (unsigned s = 0; s < kSets; ++s) {
            const auto ids = make_ids(routes, R, rng);
            std::copy(ids.begin(), ids.end(), all.begin() + std::ptrdiff_t{s} * kRowsMax * kTopK);
            distinct += static_cast<double>(std::set<unsigned>(ids.begin(), ids.end()).size());
        }
        distinct /= kSets;
        CHECK_HIP(hipMemcpy(b.ids, all.data(), sizeof(unsigned) * all.size(), hipMemcpyHostToDevice));
        char name[48];
        std::snprintf(name, sizeof name, "%s-%s", fmt, route_name(routes));
        bool equal = true;
        for (unsigned s = 0; s < kSets; ++s) {
            std::vector<std::uint16_t> act[3];
            std::vector<float> out[3];
            for (unsigned which = 0; which < 3U; ++which) {
                CHECK_HIP(hipMemset(b.output, 0x7B, sizeof(float) * kRowsMax * kHidden));
                CHECK_HIP(hipMemset(b.activation, 0x5A, sizeof(std::uint16_t) * kRowsMax * kTopK * kFfn));
                CHECK_HIP(run<F, R>(bank, b, s, which == 0U ? Form::rows : (which == 1U ? Form::uni : Form::plan)));
                CHECK_HIP(hipDeviceSynchronize());
                act[which].resize(std::size_t{kRowsMax} * kTopK * kFfn);
                out[which].resize(std::size_t{kRowsMax} * kHidden);
                CHECK_HIP(hipMemcpy(act[which].data(), b.activation, sizeof(std::uint16_t) * act[which].size(),
                                    hipMemcpyDeviceToHost));
                CHECK_HIP(hipMemcpy(out[which].data(), b.output, sizeof(float) * out[which].size(),
                                    hipMemcpyDeviceToHost));
            }
            equal = equal && act[0] == act[1] && act[0] == act[2] &&
                    std::memcmp(out[0].data(), out[1].data(), sizeof(float) * out[0].size()) == 0 &&
                    std::memcmp(out[0].data(), out[2].data(), sizeof(float) * out[0].size()) == 0;
            if (s == 0U) {
                std::printf("DIGEST %s R=%u rows %016llx union %016llx\n", name, R,
                            static_cast<unsigned long long>(fnv(out[0].data(), sizeof(float) * out[0].size())),
                            static_cast<unsigned long long>(fnv(out[1].data(), sizeof(float) * out[1].size())));
            }
        }
        std::printf("ID %s R=%u union and plan vs rows: %s\n", name, R, equal ? "EQUAL" : "DIFF");
        ok = ok && equal;
        double us[7]{};
        // Two passes over the seven forms, so a slow drift (thermal) hits the forms alike.
        for (unsigned pass = 0; pass < 2U; ++pass) {
            for (unsigned f = 0; f < 7U; ++f) {
                const Form form = static_cast<Form>(f);
                for (unsigned s = 0; s < kSets; ++s) CHECK_HIP(run<F, R>(bank, b, s, form));
                CHECK_HIP(hipDeviceSynchronize());
                CHECK_HIP(hipEventRecord(start, nullptr));
                for (unsigned i = 0; i < kRounds * kSets; ++i) CHECK_HIP(run<F, R>(bank, b, i % kSets, form));
                CHECK_HIP(hipEventRecord(stop, nullptr));
                CHECK_HIP(hipEventSynchronize(stop));
                float ms = 0.0F;
                CHECK_HIP(hipEventElapsedTime(&ms, start, stop));
                us[f] += 1000.0 * static_cast<double>(ms) / (kRounds * kSets) / 2.0;
            }
        }
        const double bytes = bank.expert_bytes;  // bytes / (us * 1e3) = GB/s
        std::printf("TIME %s R=%u rows %.1f union %.1f us GB/s rows %.1f union %.1f | distinct %.1f/%u | gate rows %.1f "
                    "union %.1f down rows %.1f union %.1f us | plan %.1f us\n",
                    name, R, us[0], us[3], R * kTopK * bytes / (us[0] * 1e3), distinct * bytes / (us[3] * 1e3), distinct,
                    R * kTopK, us[1], us[4], us[2], us[5], us[6]);
        std::fflush(stdout);
    }
    CHECK_HIP(hipEventDestroy(start));
    CHECK_HIP(hipEventDestroy(stop));
    return ok;
}

template <MatrixFormat F>
bool all_rows(const char* fmt, std::mt19937& rng, Buffers& b) {
    const Bank bank(F);
    bool ok = one<F, 4U>(bank, b, rng, fmt);
    ok = one<F, 8U>(bank, b, rng, fmt) && ok;
    ok = one<F, 16U>(bank, b, rng, fmt) && ok;
    return ok;
}

}  // namespace

int main() {
    std::mt19937 rng(20261127U);
    Buffers b;
    bool ok = all_rows<MatrixFormat::nvfp4>("nvfp4", rng, b);
    ok = all_rows<MatrixFormat::mxfp8>("mxfp8", rng, b) && ok;
    std::printf("microscaled_r1_union_hip: %s\n", ok ? "PASS" : "FAIL");
    return ok ? 0 : 1;
}
