// Device fixture and no-model bench of the tall slab schedule of the grouped BF16 expert projection
// (plan: moe_core/grouped_tall_plan.hpp; body: mma::microscaled::detail::projection_tall).
//  fixture: every tall form against the incumbent staged schedule, bitwise (memcmp of the whole output, sentinels included):
//           NVFP4 (reference and shift decode) and MXFP8, contiguous and interleaved rows, K in {32, 96, 1152, 2304},
//           N in {128, 256, 2304}, ragged expert runs of 1 .. 300 rows cut into 128-row tiles, special operand values
//           (signed zeros, subnormals, infinities, NaN, NaN block scales), a device tile count below the launch count, and a
//           raised failure word (no store). The automatic schedule is checked to equal both.
//  --bench [T...]: one Inkling-class grouped projection pair per format per chunk of T tokens (default 2048 and 8192; 8 of
//           128 experts per token, random routes),
//           the incumbent staged schedule against each tall form, interleaved, medians of 15; bitwise check of every arm.
#include "cuda/core/mma_tile.cuh"
#include "moe_core/grouped_tall_plan.hpp"

#include <cuda_runtime.h>

#include <algorithm>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <limits>
#include <numeric>
#include <random>
#include <vector>

namespace {

namespace tile = strix::cuda::core::mma::microscaled;
namespace tall = strix::moe_core::grouped_tall;
namespace view = strix::moe_core::grouped_microscaled;
namespace packed = strix::weight::microscaled;
using Format = packed::MatrixFormat;

#define CK(x)                                                                                              \
    do {                                                                                                   \
        const cudaError_t e_ = (x);                                                                        \
        if (e_ != cudaSuccess) {                                                                           \
            std::printf("CUDA error %s at %s:%d\n", cudaGetErrorString(e_), __FILE__, __LINE__);         \
            std::exit(2);                                                                                  \
        }                                                                                                  \
    } while (0)

int failures = 0;

template <class T>
T* managed(std::size_t n) {
    T* p = nullptr;
    CK(cudaMallocManaged(&p, std::max<std::size_t>(n, 1U) * sizeof(T)));
    return p;
}

std::uint16_t bf16_bits(float v) { return strix::weight::scalar_float::bf16_encode(v); }

// Schedules under test: the incumbent, the default resolution, explicit tall, and every instantiated form.
enum class Arm { staged, automatic, tall, r128_s4, r128_s3, r128_w16, r64, r128_n4 };
constexpr Arm kArms[] = {Arm::staged, Arm::automatic, Arm::tall, Arm::r128_s4, Arm::r128_s3, Arm::r128_w16, Arm::r64,
                         Arm::r128_n4};
const char* arm_name(Arm a) {
    switch (a) {
    case Arm::staged: return "staged";
    case Arm::automatic: return "automatic";
    case Arm::tall: return "tall";
    case Arm::r128_s4: return tall::name(tall::FormId::r128_k32_s4);
    case Arm::r128_s3: return tall::name(tall::FormId::r128_k32_s3);
    case Arm::r128_w16: return tall::name(tall::FormId::r128_w16_k32_s4);
    case Arm::r64: return tall::name(tall::FormId::r64_k32_s4);
    case Arm::r128_n4: return tall::name(tall::FormId::r128_n4_k32_s4);
    }
    return "?";
}

template <Format F>
cudaError_t run(Arm arm, view::Args args, cudaStream_t stream) {
    switch (arm) {
    case Arm::staged: args.schedule = view::Schedule::staged; return tile::launch<F>(args, stream);
    case Arm::automatic: args.schedule = view::Schedule::automatic; return tile::launch<F>(args, stream);
    case Arm::tall: args.schedule = view::Schedule::tall; return tile::launch<F>(args, stream);
    case Arm::r128_s4: return tile::detail::launch_tall_form<F, tall::FormId::r128_k32_s4>(args, stream);
    case Arm::r128_s3: return tile::detail::launch_tall_form<F, tall::FormId::r128_k32_s3>(args, stream);
    case Arm::r128_w16: return tile::detail::launch_tall_form<F, tall::FormId::r128_w16_k32_s4>(args, stream);
    case Arm::r64: return tile::detail::launch_tall_form<F, tall::FormId::r64_k32_s4>(args, stream);
    case Arm::r128_n4: return tile::detail::launch_tall_form<F, tall::FormId::r128_n4_k32_s4>(args, stream);
    }
    return cudaErrorInvalidValue;
}

// A 64-row form takes a 128-row plan as two row slices; every arm sees the same tiles.
struct Case {
    unsigned inputs, outputs;
    packed::RowOrder order;
};

template <Format F>
void fixture_case(const Case& c, view::Decode decode, std::mt19937& rng) {
    using Operand = packed::PackedOperand<F>;
    const std::vector<unsigned> runs = {1U, 15U, 16U, 17U, 63U, 64U, 65U, 100U, 127U, 128U, 129U, 200U, 300U};
    const unsigned experts = static_cast<unsigned>(runs.size());
    const unsigned rows = std::accumulate(runs.begin(), runs.end(), 0U);
    const std::size_t w_codes = std::size_t{c.outputs} * c.inputs / (Operand::kNvfp4 ? 2U : 1U);
    const std::size_t w_scales = std::size_t{c.outputs} * c.inputs / Operand::kBlockColumns;
    auto* codes = managed<std::uint8_t>(w_codes * experts);
    auto* scales = managed<std::uint8_t>(w_scales * experts);
    auto* input = managed<std::uint16_t>(std::size_t{rows} * c.inputs);
    for (std::size_t i = 0; i < w_codes * experts; ++i) {
        const auto bits = static_cast<std::uint8_t>(rng() & 0xffU);
        codes[i] = !Operand::kNvfp4 && (bits & 0x7fU) == 0x7fU ? static_cast<std::uint8_t>(bits - 1U) : bits;
    }
    for (std::size_t i = 0; i < w_scales * experts; ++i) {
        const unsigned r = rng() % 1000U;
        if constexpr (Operand::kNvfp4)  // mostly normal E4M3, some zero, maximum, NaN (0x7f) and subnormal scales
            scales[i] = static_cast<std::uint8_t>(r < 3U ? 0x00U : r < 6U ? 0x7eU : r < 8U ? 0x7fU : r < 12U ? 0x03U
                                                                                   : 0x28U + rng() % 24U);
        else
            scales[i] = static_cast<std::uint8_t>(r < 4U ? 0x7fU : 120U + rng() % 14U);
    }
    for (std::size_t i = 0; i < std::size_t{rows} * c.inputs; ++i) {
        const unsigned r = rng() % 4000U;
        float v = static_cast<float>(static_cast<int>(rng() % 4097U) - 2048) / 1024.0F;
        if (r == 0U) v = -0.0F;
        else if (r == 1U) v = 1.0e-39F;  // BF16 subnormal
        else if (r == 2U) v = (rng() & 1U) != 0U ? std::numeric_limits<float>::infinity()
                                                   : -std::numeric_limits<float>::infinity();
        else if (r == 3U) v = std::numeric_limits<float>::quiet_NaN();
        input[i] = bf16_bits(v);
    }
    // Expert-major runs cut into tiles of at most 128 rows (the plan's CSR cut).
    std::vector<view::TileRows> host_tiles;
    unsigned first = 0, max_rows = 0;
    for (unsigned e = 0; e < experts; ++e) {
        const packed::PackedMatrix w{codes + w_codes * e, scales + w_scales * e, c.outputs, c.inputs, 0.375F, c.order};
        for (unsigned r = 0; r < runs[e]; r += tile::kRowsPerTile) {
            const unsigned live = std::min(tile::kRowsPerTile, runs[e] - r);
            max_rows = std::max(max_rows, live);
            host_tiles.push_back({w, input + std::size_t{first + r} * c.inputs, nullptr, c.inputs, c.outputs, live});
        }
        first += runs[e];
    }
    const unsigned count = static_cast<unsigned>(host_tiles.size());
    const std::size_t out_elems = std::size_t{rows} * c.outputs;
    auto* tiles = managed<view::TileRows>(count);
    auto* reference = managed<float>(out_elems);
    auto* actual = managed<float>(out_elems);
    auto* tile_count = managed<unsigned>(1);
    auto* failure = managed<unsigned>(1);
    const auto bind = [&](float* out) {
        for (unsigned t = 0; t < count; ++t) {
            tiles[t] = host_tiles[t];
            tiles[t].output = out + static_cast<std::size_t>(host_tiles[t].input - input) / c.inputs * c.outputs;
        }
    };
    const auto fill = [&](float* out) { std::fill(out, out + out_elems, -1.0e30F); };
    view::Args args{tiles, count, c.inputs, c.outputs, nullptr, nullptr, max_rows};
    args.decode = decode;
    char what[256];
    std::snprintf(what, sizeof what, "%s K=%u N=%u %s %s", Operand::kNvfp4 ? "nvfp4" : "mxfp8", c.inputs, c.outputs,
                  c.order == packed::RowOrder::contiguous ? "contiguous" : "interleaved",
                  decode == view::Decode::shift ? "shift" : "reference");

    fill(reference);
    bind(reference);
    CK(run<F>(Arm::staged, args, nullptr));
    CK(cudaDeviceSynchronize());
    {  // The incumbent plan (runs cut at 64 rows, one 64-row CTA per tile) gives the same bits as the 128-row plan.
        std::vector<view::TileRows> cut64;
        for (const auto& t : host_tiles)
            for (unsigned r = 0; r < t.rows; r += 64U)
                cut64.push_back({t.weights, t.input + std::size_t{r} * c.inputs, nullptr, c.inputs, c.outputs,
                                 std::min(64U, t.rows - r)});
        auto* tiles64 = managed<view::TileRows>(cut64.size());
        fill(actual);
        for (std::size_t t = 0; t < cut64.size(); ++t) {
            tiles64[t] = cut64[t];
            tiles64[t].output = actual + static_cast<std::size_t>(cut64[t].input - input) / c.inputs * c.outputs;
        }
        view::Args args64{tiles64, static_cast<unsigned>(cut64.size()), c.inputs, c.outputs, nullptr, nullptr,
                          std::min(64U, max_rows)};
        args64.decode = decode;
        CK(run<F>(Arm::staged, args64, nullptr));
        CK(cudaDeviceSynchronize());
        if (std::memcmp(reference, actual, out_elems * sizeof(float)) != 0) {
            std::printf("FAIL %s: 128-row plan differs from the 64-row plan (staged)\n", what);
            ++failures;
        }
        CK(cudaFree(tiles64));
    }
    for (const Arm arm : kArms) {
        if (arm == Arm::staged) continue;
        fill(actual);
        bind(actual);
        CK(run<F>(arm, args, nullptr));
        CK(cudaDeviceSynchronize());
        if (std::memcmp(reference, actual, out_elems * sizeof(float)) != 0) {
            std::size_t i = 0;
            while (std::memcmp(reference + i, actual + i, sizeof(float)) == 0) ++i;
            std::printf("FAIL %s arm %s: first difference at row %zu output %zu (%a vs %a)\n", what, arm_name(arm),
                        i / c.outputs, i % c.outputs, static_cast<double>(reference[i]), static_cast<double>(actual[i]));
            ++failures;
        }
    }
    // A device tile count below the launch count: tiles past it are not touched (same in both schedules).
    *tile_count = count / 2U;
    view::Args bounded = args;
    bounded.tile_count = tile_count;
    fill(reference);
    bind(reference);
    CK(run<F>(Arm::staged, bounded, nullptr));
    fill(actual);
    bind(actual);
    CK(run<F>(Arm::tall, bounded, nullptr));
    CK(cudaDeviceSynchronize());
    if (std::memcmp(reference, actual, out_elems * sizeof(float)) != 0) {
        std::printf("FAIL %s: device tile count bound differs\n", what);
        ++failures;
    }
    // A raised failure word: nothing stored.
    *failure = 1U;
    view::Args failed = args;
    failed.failure = failure;
    fill(actual);
    bind(actual);
    CK(run<F>(Arm::tall, failed, nullptr));
    CK(cudaDeviceSynchronize());
    if (std::any_of(actual, actual + out_elems, [](float v) { return v != -1.0e30F; })) {
        std::printf("FAIL %s: tall stored under a raised failure word\n", what);
        ++failures;
    }
    CK(cudaFree(codes));
    CK(cudaFree(scales));
    CK(cudaFree(input));
    CK(cudaFree(tiles));
    CK(cudaFree(reference));
    CK(cudaFree(actual));
    CK(cudaFree(tile_count));
    CK(cudaFree(failure));
}

void fixture() {
    std::mt19937 rng(0x7a11U);
    const Case cases[] = {{2304U, 2304U, packed::RowOrder::interleaved_halves},
                          {1152U, 2304U, packed::RowOrder::contiguous},
                          {96U, 256U, packed::RowOrder::interleaved_halves},
                          {32U, 128U, packed::RowOrder::contiguous}};
    for (const auto& c : cases) {
        fixture_case<Format::nvfp4>(c, view::Decode::reference, rng);
        fixture_case<Format::nvfp4>(c, view::Decode::shift, rng);
        fixture_case<Format::mxfp8>(c, view::Decode::reference, rng);
    }
    // Admission: explicit tall refuses a shape no form takes (outputs % 128 != 0), staged still runs it.
    view::TileRows dummy{};
    view::Args ragged{&dummy, 1U, 64U, 160U, nullptr, nullptr, 64U};
    ragged.schedule = view::Schedule::tall;
    if (tile::launch<Format::nvfp4>(ragged, nullptr) != cudaErrorInvalidValue) {
        std::printf("FAIL explicit tall accepted outputs %% 128 != 0\n");
        ++failures;
    }
    (void)cudaGetLastError();
}

// ---- bench ----
__global__ void fill_bytes(std::uint8_t* p, std::size_t n, std::uint32_t seed, unsigned base, unsigned span) {
    for (std::size_t i = blockIdx.x * std::size_t{blockDim.x} + threadIdx.x; i < n; i += std::size_t{gridDim.x} * blockDim.x) {
        std::uint32_t h = static_cast<std::uint32_t>(i) * 2654435761U ^ seed;
        h ^= h >> 15;
        h *= 2246822519U;
        h ^= h >> 13;
        const unsigned v = span ? base + h % span : h & 0xffU;
        p[i] = static_cast<std::uint8_t>(span == 0U && (v & 0x7fU) == 0x7fU ? v - 1U : v);
    }
}
__global__ void fill_bf16(std::uint16_t* p, std::size_t n, std::uint32_t seed) {
    for (std::size_t i = blockIdx.x * std::size_t{blockDim.x} + threadIdx.x; i < n; i += std::size_t{gridDim.x} * blockDim.x) {
        std::uint32_t h = static_cast<std::uint32_t>(i) * 2246822519U ^ seed;
        h ^= h >> 16;
        h *= 2654435761U;
        const float v = (static_cast<float>(h & 0xffffU) - 32768.F) / 16384.F;
        p[i] = static_cast<std::uint16_t>(__float_as_uint(v) >> 16);
    }
}
__global__ void count_diff(const float* a, const float* b, std::size_t n, unsigned long long* diff) {
    unsigned long long local = 0;
    for (std::size_t i = blockIdx.x * std::size_t{blockDim.x} + threadIdx.x; i < n; i += std::size_t{gridDim.x} * blockDim.x)
        local += __float_as_uint(a[i]) != __float_as_uint(b[i]);
    if (local) atomicAdd(diff, local);
}

template <Format F>
void bench_format(unsigned tokens, view::Decode decode, const std::vector<unsigned>& counts) {
    using Operand = packed::PackedOperand<F>;
    constexpr unsigned kExperts = 128U, kHidden = 2304U, kInter = 1152U;
    const unsigned assignments = std::accumulate(counts.begin(), counts.end(), 0U);
    const struct {
        const char* name;
        unsigned inputs, outputs;
        packed::RowOrder order;
    } projections[] = {{"gate_up", kHidden, 2U * kInter, packed::RowOrder::interleaved_halves},
                       {"down", kInter, kHidden, packed::RowOrder::contiguous}};
    for (const auto& p : projections) {
        const std::size_t w_codes = std::size_t{p.outputs} * p.inputs / (Operand::kNvfp4 ? 2U : 1U);
        const std::size_t w_scales = std::size_t{p.outputs} * p.inputs / Operand::kBlockColumns;
        std::uint8_t *codes = nullptr, *scales = nullptr;
        std::uint16_t* input = nullptr;
        float *out_ref = nullptr, *out = nullptr;
        unsigned long long* diff = nullptr;
        CK(cudaMalloc(&codes, w_codes * kExperts));
        CK(cudaMalloc(&scales, w_scales * kExperts));
        CK(cudaMalloc(&input, std::size_t{assignments} * p.inputs * 2U));
        CK(cudaMalloc(&out_ref, std::size_t{assignments} * p.outputs * 4U));
        CK(cudaMalloc(&out, std::size_t{assignments} * p.outputs * 4U));
        CK(cudaMalloc(&diff, sizeof(unsigned long long)));
        fill_bytes<<<1024, 256>>>(codes, w_codes * kExperts, 7U, 0U, 0U);
        if constexpr (Operand::kNvfp4) fill_bytes<<<1024, 256>>>(scales, w_scales * kExperts, 9U, 0x28U, 0x18U);
        else fill_bytes<<<1024, 256>>>(scales, w_scales * kExperts, 9U, 120U, 14U);
        fill_bf16<<<1024, 256>>>(input, std::size_t{assignments} * p.inputs, 3U);
        // Two descriptor sets over the same rows (one per output buffer), expert-major, cut at 128 rows.
        std::vector<view::TileRows> ref_tiles, arm_tiles;
        unsigned first = 0;
        for (unsigned e = 0; e < kExperts; ++e) {
            const packed::PackedMatrix w{codes + w_codes * e, scales + w_scales * e, p.outputs, p.inputs, 0.01F, p.order};
            for (unsigned r = 0; r < counts[e]; r += tile::kRowsPerTile) {
                const unsigned live = std::min(tile::kRowsPerTile, counts[e] - r);
                const std::size_t row = first + r;
                ref_tiles.push_back({w, input + row * p.inputs, out_ref + row * p.outputs, p.inputs, p.outputs, live});
                arm_tiles.push_back({w, input + row * p.inputs, out + row * p.outputs, p.inputs, p.outputs, live});
            }
            first += counts[e];
        }
        const auto count = static_cast<unsigned>(ref_tiles.size());
        view::TileRows *d_ref = nullptr, *d_arm = nullptr;
        CK(cudaMalloc(&d_ref, count * sizeof(view::TileRows)));
        CK(cudaMalloc(&d_arm, count * sizeof(view::TileRows)));
        CK(cudaMemcpy(d_ref, ref_tiles.data(), count * sizeof(view::TileRows), cudaMemcpyHostToDevice));
        CK(cudaMemcpy(d_arm, arm_tiles.data(), count * sizeof(view::TileRows), cudaMemcpyHostToDevice));
        view::Args ref_args{d_ref, count, p.inputs, p.outputs, nullptr, nullptr, tile::kRowsPerTile};
        view::Args arm_args{d_arm, count, p.inputs, p.outputs, nullptr, nullptr, tile::kRowsPerTile};
        ref_args.decode = arm_args.decode = decode;
        CK(run<F>(Arm::staged, ref_args, nullptr));
        CK(cudaDeviceSynchronize());

        constexpr Arm timed[] = {Arm::staged, Arm::r128_s4, Arm::r128_s3, Arm::r128_w16, Arm::r64, Arm::r128_n4};
        constexpr unsigned kArmsTimed = sizeof(timed) / sizeof(timed[0]);
        std::vector<float> ms[kArmsTimed];
        unsigned long long mismatch[kArmsTimed]{};
        cudaEvent_t a, b;
        CK(cudaEventCreate(&a));
        CK(cudaEventCreate(&b));
        for (unsigned i = 0; i < kArmsTimed; ++i) CK(run<F>(timed[i], arm_args, nullptr));  // warm-up
        CK(cudaDeviceSynchronize());
        for (unsigned rep = 0; rep < 15U; ++rep) {
            for (unsigned i = 0; i < kArmsTimed; ++i) {
                const unsigned arm = (i + rep) % kArmsTimed;  // rotate the order between repetitions
                CK(cudaEventRecord(a));
                CK(run<F>(timed[arm], arm_args, nullptr));
                CK(cudaEventRecord(b));
                CK(cudaEventSynchronize(b));
                float t = 0.F;
                CK(cudaEventElapsedTime(&t, a, b));
                ms[arm].push_back(t);
            }
        }
        for (unsigned i = 0; i < kArmsTimed; ++i) {
            CK(cudaMemset(out, 0xff, std::size_t{assignments} * p.outputs * 4U));
            CK(run<F>(timed[i], arm_args, nullptr));
            CK(cudaMemset(diff, 0, sizeof(unsigned long long)));
            count_diff<<<1024, 256>>>(out_ref, out, std::size_t{assignments} * p.outputs, diff);
            CK(cudaMemcpy(&mismatch[i], diff, sizeof(unsigned long long), cudaMemcpyDeviceToHost));
        }
        const double flops = 2.0 * assignments * p.outputs * p.inputs;
        float base = 0.F;
        for (unsigned i = 0; i < kArmsTimed; ++i) {
            std::sort(ms[i].begin(), ms[i].end());
            const float med = ms[i][ms[i].size() / 2U];
            if (i == 0) base = med;
            std::printf("BENCH T=%u %s %s.%s rows %u tiles %u %-16s %.3f ms %6.1f TFLOPS x%.2f %s\n", tokens, p.name,
                        Operand::kNvfp4 ? "nvfp4" : "mxfp8", decode == view::Decode::shift ? "shift" : "ref", assignments,
                        count, arm_name(timed[i]), med, flops / med / 1e9, base / med,
                        mismatch[i] == 0 ? "bitwise EQUAL" : "BITS DIFFER");
            if (mismatch[i] != 0) ++failures;
        }
        CK(cudaEventDestroy(a));
        CK(cudaEventDestroy(b));
        CK(cudaFree(codes));
        CK(cudaFree(scales));
        CK(cudaFree(input));
        CK(cudaFree(out_ref));
        CK(cudaFree(out));
        CK(cudaFree(diff));
        CK(cudaFree(d_ref));
        CK(cudaFree(d_arm));
    }
}

void bench(unsigned tokens) {
    // Random routes: each token picks 8 distinct experts of 128.
    std::mt19937 rng(0x8b0aU + tokens);
    std::vector<unsigned> counts(128U, 0U), pool(128U);
    std::iota(pool.begin(), pool.end(), 0U);
    for (unsigned t = 0; t < tokens; ++t) {
        for (unsigned k = 0; k < 8U; ++k) std::swap(pool[k], pool[k + rng() % (128U - k)]);
        for (unsigned k = 0; k < 8U; ++k) ++counts[pool[k]];
    }
    bench_format<Format::nvfp4>(tokens, view::Decode::shift, counts);
    bench_format<Format::nvfp4>(tokens, view::Decode::reference, counts);
    bench_format<Format::mxfp8>(tokens, view::Decode::reference, counts);
}

}  // namespace

int main(int argc, char** argv) {
    int device = 0, optin = 0;
    CK(cudaGetDevice(&device));
    CK(cudaDeviceGetAttribute(&optin, cudaDevAttrMaxSharedMemoryPerBlockOptin, device));
    fixture();
    if (failures != 0) {
        std::printf("grouped_tall_cuda: FAIL (%d)\n", failures);
        return 1;
    }
    std::printf("grouped_tall_cuda: PASS (5 tall forms + automatic = staged bitwise; nvfp4 ref/shift, mxfp8; K 32..2304; "
                "runs 1..300; tile count, failure word; opt-in shared %d B)\n", optin);
    if (argc > 1 && std::strcmp(argv[1], "--bench") == 0) {
        std::vector<unsigned> tokens;
        for (int i = 2; i < argc; ++i) tokens.push_back(static_cast<unsigned>(std::strtoul(argv[i], nullptr, 10)));
        if (tokens.empty()) tokens = {2048U, 8192U};
        for (const unsigned t : tokens) bench(t);
        if (failures != 0) {
            std::printf("grouped_tall_cuda: BENCH BITS DIFFER (%d)\n", failures);
            return 1;
        }
    }
    return 0;
}
