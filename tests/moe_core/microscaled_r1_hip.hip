// No-model device check and bench of the R1 decode launch (microscaled_r1::launch) on Inkling-shaped experts
// (128 experts, top-8, hidden 2304, ffn 1152, NVFP4 and MXFP8):
//   check: for 1..8 rows under uniform, skewed and 100 %-collision routes (plus a failed row, an invalid id and
//          non-finite / subnormal inputs), every row of a rows launch equals a one-row launch of that row bit for bit;
//   digest: one FNV-1a line per case over the output and activation words, so two builds of this file (two trees)
//           compare by diffing their digest lines;
//   bench (--bench): ms per launch (whole launch, gate_up only, down_combine only) at 1 / 2 / 4 / 8 rows. Each timed
//          launch takes the next of 16 route sets, so consecutive launches read different experts.
// Exit 0 only when every check passes.
#include "strix/hip/legacy_intrinsics.hpp"
#include "strix/hip/legacy_launch.hpp"
#include "moe_core/microscaled_r1.hpp"

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <limits>
#include <random>
#include <set>
#include <vector>

namespace {

namespace r1 = strix::moe_core::microscaled_r1;
namespace packed = strix::weight::microscaled;
using Expert = strix::moe_core::device_grouped::ExpertWeights;
using packed::MatrixFormat;

constexpr unsigned kExperts = 128U, kTopK = 8U, kHidden = 2304U, kFfn = 1152U, kRowsMax = r1::kMaxRows;
constexpr unsigned kSets = 16U;

#define CHECK_HIP(x)                                                                                    \
    do {                                                                                                \
        if (const hipError_t e = (x); e != hipSuccess) {                                                \
            std::fprintf(stderr, "%s:%d %s: %s\n", __FILE__, __LINE__, #x, hipGetErrorString(e));       \
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

enum class Mode { uniform, skewed, collide };
const char* mode_name(Mode m) { return m == Mode::uniform ? "uniform" : m == Mode::skewed ? "skewed" : "collide"; }

std::vector<unsigned> make_ids(Mode mode, unsigned rows, std::mt19937& rng) {
    std::vector<unsigned> ids(std::size_t{rows} * kTopK);
    std::vector<double> popularity(kExperts);
    for (unsigned e = 0; e < kExperts; ++e) popularity[e] = mode == Mode::skewed ? 1.0 / (1.0 + e) : 1.0;
    for (unsigned row = 0; row < rows; ++row) {
        std::set<unsigned> chosen;
        if (mode == Mode::collide) {
            for (unsigned s = 0; s < kTopK; ++s) chosen.insert(17U * s + 3U);
        } else {
            std::discrete_distribution<unsigned> pick(popularity.begin(), popularity.end());
            while (chosen.size() < kTopK) chosen.insert((pick(rng) * 37U + 11U) % kExperts);
        }
        std::vector<unsigned> v(chosen.begin(), chosen.end());
        std::shuffle(v.begin(), v.end(), rng);
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
        CHECK_HIP(hipMalloc(&weights, sizeof(float) * kRowsMax * kTopK));
        CHECK_HIP(hipMalloc(&input, sizeof(float) * kRowsMax * kHidden));
        CHECK_HIP(hipMalloc(&output, sizeof(float) * kRowsMax * kHidden));
        CHECK_HIP(hipMalloc(&activation, sizeof(std::uint16_t) * kRowsMax * kTopK * kFfn));
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
    // Rows [first, first + count) of the uploaded case, as a launch of `count` rows (set: route set for the bench).
    r1::Args args(const Expert* m, unsigned first = 0U, unsigned set = 0U) const {
        const r1::Args all{m,     ids + std::size_t{set} * kRowsMax * kTopK, weights, failure, input, activation,
                           output, kExperts, kHidden, kFfn, 4.0F, 25.0F};
        return r1::rows_chunk<kTopK>(all, first);
    }
};

void upload_case(Buffers& b, const std::vector<unsigned>& ids, unsigned rows, std::mt19937& rng, int failed_row,
                 bool special) {
    std::uniform_real_distribution<float> x(-2.0F, 2.0F), w(0.01F, 0.3F);
    std::vector<float> input(std::size_t{rows} * kHidden), weights(std::size_t{rows} * kTopK);
    for (auto& v : input) v = x(rng);
    for (auto& v : weights) v = w(rng);
    if (special) {
        // Non-finite and subnormal inputs (the BF16 rounding's NaN, infinity and subnormal cases) in the last row.
        float* row = input.data() + std::size_t{rows - 1U} * kHidden;
        row[5] = std::numeric_limits<float>::quiet_NaN();
        row[700] = std::numeric_limits<float>::infinity();
        row[1500] = 1.0e-40F;
        row[2303] = -3.0e-39F;
        row[31] = 3.3895314e38F;  // rounds up to BF16 infinity
    }
    std::vector<unsigned> failure(rows, 0U);
    if (failed_row >= 0) failure[static_cast<unsigned>(failed_row)] = 1U;
    CHECK_HIP(hipMemcpy(b.ids, ids.data(), sizeof(unsigned) * ids.size(), hipMemcpyHostToDevice));
    CHECK_HIP(hipMemcpy(b.failure, failure.data(), sizeof(unsigned) * rows, hipMemcpyHostToDevice));
    CHECK_HIP(hipMemcpy(b.weights, weights.data(), sizeof(float) * weights.size(), hipMemcpyHostToDevice));
    CHECK_HIP(hipMemcpy(b.input, input.data(), sizeof(float) * input.size(), hipMemcpyHostToDevice));
}

template <class T>
std::vector<T> fetch(const T* device, std::size_t count) {
    std::vector<T> host(count);
    CHECK_HIP(hipDeviceSynchronize());
    CHECK_HIP(hipMemcpy(host.data(), device, sizeof(T) * count, hipMemcpyDeviceToHost));
    return host;
}

void poison(Buffers& b) {
    CHECK_HIP(hipMemset(b.output, 0x7B, sizeof(float) * kRowsMax * kHidden));
    CHECK_HIP(hipMemset(b.activation, 0x5A, sizeof(std::uint16_t) * kRowsMax * kTopK * kFfn));
}

std::uint64_t fnv(std::uint64_t h, const void* data, std::size_t bytes) {
    const auto* p = static_cast<const std::uint8_t*>(data);
    for (std::size_t i = 0; i < bytes; ++i) h = (h ^ p[i]) * 0x100000001B3ULL;
    return h;
}

// One case: the rows launch, then each row alone (its activation lands at the scratch base). Prints the digest of the
// rows launch; returns false when a row's output or activation differs from its one-row launch.
template <MatrixFormat F>
bool run_case(const Bank& bank, Buffers& b, unsigned rows, Mode mode, unsigned variant, std::mt19937& rng,
              const char* name, std::uint64_t& all) {
    auto ids = make_ids(mode, rows, rng);
    const int failed = variant == 1U ? static_cast<int>(rows / 2U) : -1;
    if (variant == 2U) ids[std::size_t{rows - 1U} * kTopK + 3U] = kExperts + 1U;  // invalid id: row not stored
    upload_case(b, ids, rows, rng, failed, variant == 3U);
    poison(b);
    if (r1::launch<F, kTopK>(b.args(bank.matrices), nullptr, rows) != hipSuccess) return false;
    const auto out = fetch(b.output, std::size_t{rows} * kHidden);
    const auto act = fetch(b.activation, std::size_t{rows} * kTopK * kFfn);
    std::uint64_t h = fnv(0xCBF29CE484222325ULL, out.data(), sizeof(float) * out.size());
    h = fnv(h, act.data(), sizeof(std::uint16_t) * act.size());
    all = fnv(all, &h, sizeof(h));
    std::printf("digest %s rows %u mode %-7s variant %u %016llx\n", name, rows, mode_name(mode), variant,
                static_cast<unsigned long long>(h));
    for (unsigned row = 0; row < rows; ++row) {
        poison(b);
        if (r1::launch<F, kTopK>(b.args(bank.matrices, row), nullptr, 1U) != hipSuccess) return false;
        const auto one = fetch(b.output + std::size_t{row} * kHidden, kHidden);
        const auto one_act = fetch(b.activation, std::size_t{kTopK} * kFfn);
        const bool stored = failed != static_cast<int>(row) && !(variant == 2U && row == rows - 1U);
        const bool same_out = !stored || std::memcmp(one.data(), out.data() + std::size_t{row} * kHidden,
                                                     sizeof(float) * kHidden) == 0;
        const bool same_act = failed == static_cast<int>(row) ||
                              std::memcmp(one_act.data(), act.data() + std::size_t{row} * kTopK * kFfn,
                                          sizeof(std::uint16_t) * kTopK * kFfn) == 0;
        if (!same_out || !same_act) {
            std::fprintf(stderr, "%s FAIL rows %u mode %s variant %u row %u: %s differs from the one-row launch\n",
                         name, rows, mode_name(mode), variant, row, same_out ? "activation" : "output");
            return false;
        }
    }
    return true;
}

template <MatrixFormat F>
bool check(const Bank& bank, Buffers& b, std::mt19937& rng, const char* name) {
    unsigned cases = 0U;
    std::uint64_t all = 0xCBF29CE484222325ULL;
    for (unsigned rows = 1U; rows <= kRowsMax; ++rows) {
        for (Mode mode : {Mode::uniform, Mode::skewed, Mode::collide}) {
            for (unsigned variant = 0; variant < 4U; ++variant) {
                if (!run_case<F>(bank, b, rows, mode, variant, rng, name, all)) return false;
                ++cases;
            }
        }
    }
    std::printf("check %s: %u cases, every rows-launch row equals its one-row launch; all-case digest %016llx\n", name,
                cases, static_cast<unsigned long long>(all));
    return true;
}

// part 0: the whole launch; 1: gate_up only; 2: down_combine only (reads the activation already in place).
template <MatrixFormat F>
void launch_part(const Bank& bank, const Buffers& b, unsigned rows, unsigned set, unsigned part) {
    const auto args = b.args(bank.matrices, 0U, set);
    if (part == 0U) {
        (void)r1::launch<F, kTopK>(args, nullptr, rows);
        return;
    }
    // The kernels launch() picks: the plain ones for one row, the rows ones (tail ahead) for several.
    namespace d = r1::detail;
    if (part == 1U) {
        const dim3 grid(r1::gate_blocks(kFfn, d::kGateColumns), rows * kTopK);
        const unsigned lds = r1::staged_input_bytes<F>(kHidden);
        if (rows == 1U) d::gate_up<F, kTopK><<<grid, d::kThreads, lds>>>(args);
        else d::gate_up_rows<F, kTopK><<<grid, d::kThreads, lds>>>(args);
    } else {
        const dim3 grid(r1::down_blocks(kHidden, d::kDownColumns), rows);
        if (rows == 1U) d::down_combine<F, kTopK><<<grid, d::kThreads>>>(args);
        else d::down_combine_rows<F, kTopK><<<grid, d::kThreads>>>(args);
    }
}

template <MatrixFormat F>
void bench(const Bank& bank, Buffers& b, std::mt19937& rng, const char* name) {
    constexpr unsigned kRounds = 8U;
    hipEvent_t start, stop;
    CHECK_HIP(hipEventCreate(&start));
    CHECK_HIP(hipEventCreate(&stop));
    for (unsigned rows : {1U, 2U, 4U, 8U}) {
        for (Mode mode : {Mode::uniform, Mode::skewed, Mode::collide}) {
            if (rows == 1U && mode == Mode::skewed) continue;
            std::vector<unsigned> all(std::size_t{kSets} * kRowsMax * kTopK, 0U);
            double distinct = 0.0;
            for (unsigned s = 0; s < kSets; ++s) {
                const auto ids = make_ids(mode, rows, rng);
                std::copy(ids.begin(), ids.end(), all.begin() + std::ptrdiff_t{s} * kRowsMax * kTopK);
                distinct += static_cast<double>(std::set<unsigned>(ids.begin(), ids.end()).size());
            }
            distinct /= kSets;
            upload_case(b, make_ids(mode, rows, rng), rows, rng, -1, false);
            CHECK_HIP(hipMemcpy(b.ids, all.data(), sizeof(unsigned) * all.size(), hipMemcpyHostToDevice));
            for (unsigned part = 0; part < 3U; ++part) {
                for (unsigned s = 0; s < kSets; ++s) launch_part<F>(bank, b, rows, s, 0U);  // warm + activation
                CHECK_HIP(hipEventRecord(start, nullptr));
                for (unsigned i = 0; i < kRounds * kSets; ++i) launch_part<F>(bank, b, rows, i % kSets, part);
                CHECK_HIP(hipEventRecord(stop, nullptr));
                CHECK_HIP(hipEventSynchronize(stop));
                CHECK_HIP(hipGetLastError());
                float ms = 0.0F;
                CHECK_HIP(hipEventElapsedTime(&ms, start, stop));
                const double per = static_cast<double>(ms) / (kRounds * kSets);
                const double share = part == 0U ? 1.0 : part == 1U ? 2.0 / 3.0 : 1.0 / 3.0;
                std::printf("bench %s rows %u mode %-7s %-4s %.4f ms/launch  assignments %u  distinct %.1f  "
                            "%.1f GB/s distinct  %.1f GB/s assignment\n",
                            name, rows, mode_name(mode), part == 0U ? "both" : part == 1U ? "gate" : "down", per,
                            rows * kTopK, distinct, distinct * bank.expert_bytes * share / (per * 1e6),
                            rows * kTopK * bank.expert_bytes * share / (per * 1e6));
            }
        }
    }
    CHECK_HIP(hipEventDestroy(start));
    CHECK_HIP(hipEventDestroy(stop));
}

}  // namespace

int main(int argc, char** argv) {
    const bool run_bench = argc > 1 && std::strcmp(argv[1], "--bench") == 0;
    std::printf("staged input bytes nvfp4 %u mxfp8 %u (hidden %u)\n", r1::staged_input_bytes<MatrixFormat::nvfp4>(kHidden),
                r1::staged_input_bytes<MatrixFormat::mxfp8>(kHidden), kHidden);
    std::mt19937 rng(20261124U);
    Buffers b;
    bool ok = true;
    {
        const Bank nv(MatrixFormat::nvfp4);
        ok = check<MatrixFormat::nvfp4>(nv, b, rng, "nvfp4") && ok;
        if (run_bench) bench<MatrixFormat::nvfp4>(nv, b, rng, "nvfp4");
    }
    {
        const Bank mx(MatrixFormat::mxfp8);
        ok = check<MatrixFormat::mxfp8>(mx, b, rng, "mxfp8") && ok;
        if (run_bench) bench<MatrixFormat::mxfp8>(mx, b, rng, "mxfp8");
    }
    std::printf("microscaled_r1_hip: %s\n", ok ? "PASS" : "FAIL");
    return ok ? 0 : 1;
}
