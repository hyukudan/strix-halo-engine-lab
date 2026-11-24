// Device identity of grouped_rows::combine_image (no model): FP32 down sums stored through
// core::f16_tile_gemm::Bf16NanExactStore and combined from that BF16 image give the output bits of grouped_rows::combine
// over the FP32 sums, for quiet and signalling NaNs of both signs and payloads in every slot, +-Inf, Inf x 0 and
// Inf - Inf between slots, two NaN terms in one sum, NaN weights on NaN sums (the product of two NaNs), next to finite
// columns in the same wave. A launch without a NaN
// sum leaves nan_seen at 0 and never writes the FP32 rows; a raised nan_seen with stale FP32 rows changes nothing.
//
// The producer cases run the actual grouped down projection (grouped_microscaled::launch / launch_image, whose tile
// binding addresses the image) over several experts of both weight formats -- nonzero row offsets, partial tiles,
// tiles taller than one row slice -- and compare the combine output bytes (and the image against the FP32 sums) of
// the image route with the FP32 route.
#include "strix/hip/legacy_intrinsics.hpp"
#include "strix/hip/legacy_launch.hpp"
#include "core/f16_tile_gemm.hpp"
#include "moe_core/grouped_microscaled.hpp"
#include "moe_core/grouped_rows.hpp"

#include <hip/hip_runtime.h>

#include <algorithm>
#include <array>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <random>
#include <utility>
#include <vector>

namespace {
namespace grouped = strix::moe_core::grouped_rows;
namespace gemm = strix::core::f16_tile_gemm;
namespace sf = strix::weight::scalar_float;
namespace gm = strix::moe_core::grouped_microscaled;
namespace packed = strix::weight::microscaled;

constexpr unsigned kRows = 37, kTopK = 8, kWidth = 300, kStride = 304, kOutStride = 301;
constexpr unsigned kAssignments = kRows * kTopK;
constexpr float kGuard = -2468.0F;

void ok(hipError_t e, const char* what) {
    if (e != hipSuccess) {
        std::fprintf(stderr, "FAIL combine-image-hip %s: %s\n", what, hipGetErrorString(e));
        std::exit(1);
    }
}

template <class T> T* upload(const std::vector<T>& host) {
    T* d = nullptr;
    ok(hipMalloc(&d, host.size() * sizeof(T)), "malloc");
    ok(hipMemcpy(d, host.data(), host.size() * sizeof(T), hipMemcpyHostToDevice), "upload");
    return d;
}

template <class T> std::vector<T> download(const T* d, std::size_t n) {
    std::vector<T> host(n);
    ok(hipMemcpy(host.data(), d, n * sizeof(T), hipMemcpyDeviceToHost), "download");
    return host;
}

// The down store of one tile, element by element: what the tile epilogue does with each completed sum.
__global__ void store_kernel(const float* sums, gemm::Bf16NanExactStore store, unsigned rows, unsigned width) {
    const unsigned column = blockIdx.x * blockDim.x + threadIdx.x;
    if (column >= width) return;
    for (unsigned row = blockIdx.y; row < rows; row += gridDim.y)
        store.put(0U, row, column, sums[std::size_t{row} * store.y_row + column]);
}

struct Case {
    const char* name;
    bool nans;          // the sums hold NaN / Inf patterns
    bool stale_flag;    // nan_seen already raised and the FP32 rows hold stale values before the launch
};

bool run(const Case& c, std::mt19937& rng) {
    std::normal_distribution<float> dist(0.0F, 3.0F);
    std::vector<float> sums(std::size_t{kAssignments} * kStride, 0.0F);
    for (auto& v : sums) v = dist(rng);
    std::vector<float> weights(kAssignments);
    for (auto& w : weights) w = dist(rng) * 0.25F;
    std::vector<std::uint32_t> combine_rows(kAssignments);
    for (unsigned i = 0; i < kAssignments; ++i) combine_rows[i] = (i * 53U + 11U) % kAssignments;  // a permutation
    if (c.nans) {
        constexpr std::uint32_t patterns[] = {0x7FC00000U, 0xFFC00000U, 0x7FC12345U, 0x7FC10000U, 0x7FC00001U,
                                              0x7F800001U, 0xFFBFFFFFU, 0x7F800000U, 0xFF800000U};
        // Row r slot s: pattern (r + s) % 9 in column (r * 7 + s) % width; finite elsewhere.
        for (unsigned row = 0; row < kRows; ++row)
            for (unsigned slot = 0; slot < kTopK; ++slot) {
                const auto expert_row = combine_rows[row * kTopK + slot];
                sums[std::size_t{expert_row} * kStride + (row * 7U + slot) % kWidth] = sf::u2f(patterns[(row + slot) % 9U]);
            }
        // Two NaN terms in one sum (different payloads), Inf - Inf, Inf x 0 and NaN weights.
        sums[std::size_t{combine_rows[0]} * kStride + 100U] = sf::u2f(0x7FC0ABCDU);
        sums[std::size_t{combine_rows[1]} * kStride + 100U] = sf::u2f(0xFFC01234U);
        sums[std::size_t{combine_rows[8]} * kStride + 101U] = sf::u2f(0x7F800000U);
        sums[std::size_t{combine_rows[9]} * kStride + 101U] = sf::u2f(0xFF800000U);
        weights[combine_rows[16]] = 0.0F;
        sums[std::size_t{combine_rows[16]} * kStride + 102U] = sf::u2f(0x7F800000U);
        // A NaN weight on a NaN sum: the product keeps one NaN by operand position (rows 3, 5, 8, 12, 17).
        weights[combine_rows[24]] = sf::u2f(0x7FC00042U);
        weights[combine_rows[41]] = sf::u2f(0xFFC00007U);
        weights[combine_rows[66]] = sf::u2f(0x7FD00000U);
        weights[combine_rows[99]] = sf::u2f(0x7F800003U);
        weights[combine_rows[140]] = sf::u2f(0xFFFFFFFFU);
        weights[combine_rows[25]] = sf::u2f(0x7F800000U);
    }
    float* d_sums = upload(sums);
    float* d_weights = upload(weights);
    std::uint32_t* d_rows = upload(combine_rows);
    float* d_out_a = upload(std::vector<float>(std::size_t{kRows} * kOutStride, kGuard));
    float* d_out_b = upload(std::vector<float>(std::size_t{kRows} * kOutStride, kGuard));
    const std::vector<float> stale(sums.size(), c.stale_flag ? 12345.0F : kGuard);
    float* d_exact = upload(stale);
    std::uint16_t* d_image = upload(std::vector<std::uint16_t>(sums.size(), 0xFF81U));
    unsigned* d_seen = upload(std::vector<unsigned>{c.stale_flag ? 1U : 0U});

    // Arm A: the incumbent combine over the FP32 sums.
    ok(grouped::combine({d_sums, kStride, d_rows, d_weights, kRows, kTopK, kWidth, d_out_a, kOutStride}, nullptr), "A");
    // Arm B: the down store's image + combine_image.
    const gemm::Bf16NanExactStore store{d_image, d_exact, d_seen, kStride, 0U};
    store_kernel<<<dim3((kWidth + 63U) / 64U, kAssignments), 64>>>(d_sums, store, kAssignments, kWidth);
    ok(hipGetLastError(), "store");
    ok(grouped::combine_image({d_image, d_exact, d_seen, kStride, d_rows, d_weights, kRows, kTopK, kWidth, d_out_b,
                               kOutStride}, nullptr), "B");
    // Refusals before any launch.
    if (grouped::combine_image({d_image, nullptr, d_seen, kStride, d_rows, d_weights, kRows, kTopK, kWidth, d_out_b,
                                kOutStride}, nullptr) != hipErrorInvalidValue ||
        grouped::combine_image({d_image, d_exact, d_seen, kWidth - 1U, d_rows, d_weights, kRows, kTopK, kWidth, d_out_b,
                                kOutStride}, nullptr) != hipErrorInvalidValue)
        ok(hipErrorInvalidValue, "refusal");
    ok(hipDeviceSynchronize(), "sync");

    const auto a = download(d_out_a, std::size_t{kRows} * kOutStride);
    const auto b = download(d_out_b, std::size_t{kRows} * kOutStride);
    const auto exact = download(d_exact, sums.size());
    const auto image = download(d_image, sums.size());
    const unsigned seen = download(d_seen, 1U)[0];
    const bool same = std::memcmp(a.data(), b.data(), a.size() * sizeof(float)) == 0;
    for (std::size_t i = 0, shown = 0; i < a.size() && shown < 6U; ++i) {
        if (sf::f2u(a[i]) == sf::f2u(b[i])) continue;
        ++shown;
        const auto row = i / kOutStride, column = i % kOutStride;
        std::printf("  differ row %zu column %zu combine %08x image %08x terms", row, column, sf::f2u(a[i]), sf::f2u(b[i]));
        for (unsigned slot = 0; slot < kTopK; ++slot) {
            const auto e = combine_rows[row * kTopK + slot];
            std::printf(" [w %08x d %08x]", sf::f2u(weights[e]), sf::f2u(sums[std::size_t{e} * kStride + column]));
        }
        std::printf("\n");
    }
    std::size_t nan_out = 0, image_ok = 0, exact_written = 0;
    for (const float v : a) nan_out += (sf::f2u(v) & 0x7fffffffU) > 0x7f800000U ? 1U : 0U;
    for (std::size_t i = 0; i < sums.size(); ++i) {
        if (i % kStride >= kWidth) {
            image_ok += image[i] == 0xFF81U ? 1U : 0U;  // stride padding untouched
            continue;
        }
        image_ok += image[i] == sf::bf16_encode(sums[i]) ? 1U : 0U;
        const bool nan = (sf::f2u(sums[i]) & 0x7fffffffU) > 0x7f800000U;
        if (nan && std::memcmp(&exact[i], &sums[i], 4U) == 0) ++exact_written;
        if (!nan && std::memcmp(&exact[i], &stale[i], 4U) != 0) return std::printf("FAIL combine-image-hip %s: FP32 row written at a number\n", c.name), false;
    }
    std::size_t nan_sums = 0;
    for (std::size_t i = 0; i < sums.size(); ++i)
        if (i % kStride < kWidth && (sf::f2u(sums[i]) & 0x7fffffffU) > 0x7f800000U) ++nan_sums;
    const bool seen_ok = c.stale_flag ? seen == 1U : seen == (nan_sums != 0U ? 1U : 0U);
    const bool pass = same && image_ok == sums.size() && exact_written == nan_sums && seen_ok &&
                      (!c.nans || nan_out != 0U);
    std::printf("%s combine-image-hip %s output_bitwise=%d nan_outputs=%zu nan_sums=%zu exact_written=%zu image_ok=%zu/%zu "
                "nan_seen=%u\n", pass ? "PASS" : "FAIL", c.name, same ? 1 : 0, nan_out, nan_sums, exact_written, image_ok,
                sums.size(), seen);
    for (void* p : {static_cast<void*>(d_sums), static_cast<void*>(d_weights), static_cast<void*>(d_rows),
                    static_cast<void*>(d_out_a), static_cast<void*>(d_out_b), static_cast<void*>(d_exact),
                    static_cast<void*>(d_image), static_cast<void*>(d_seen)})
        (void)hipFree(p);
    return pass;
}

// The producer cases: experts with these live rows, stored back to back (nonzero offsets after the first), partial tiles
// and one tile of the kRowsPerTile height (two 128-row slices staged, eight 32-row slices narrow).
constexpr std::array<unsigned, 5> kExpertRows{1U, 19U, 130U, 256U, 77U};
constexpr unsigned kProdInputs = 64, kProdWidth = 256, kProdInStride = 72, kProdStride = 264, kProdTopK = 3;

template <packed::MatrixFormat Format>
bool run_producer(gm::Schedule schedule, const char* schedule_name, bool nan_input, std::mt19937& rng) {
    constexpr bool nvfp4 = Format == packed::MatrixFormat::nvfp4;
    constexpr unsigned code_bytes = nvfp4 ? kProdInputs / 2U : kProdInputs, scale_bytes = kProdInputs / (nvfp4 ? 16U : 32U);
    constexpr std::array<std::uint8_t, 8> fp8{0, 0xb0, 0x38, 0xb8, 0x40, 0xc0, 0x30, 0x80};
    constexpr std::array<std::uint8_t, 3> fp4_scales{0x30, 0x38, 0x40};
    constexpr std::array<float, 3> outer{0.5F, 2.0F, 0.25F};
    const unsigned experts = kExpertRows.size();
    unsigned total = 0, max_rows = 0;
    std::array<unsigned, kExpertRows.size()> offset{};
    for (unsigned e = 0; e < experts; ++e) {
        offset[e] = total;
        total += kExpertRows[e];
        max_rows = std::max(max_rows, kExpertRows[e]);
    }
    const std::size_t fp32_count = std::size_t{total + 2U} * kProdStride;  // two spare rows: a shifted image stays in bounds
    std::vector<std::uint16_t> input(std::size_t{total} * kProdInStride, 0U);
    std::uniform_int_distribution<int> small(-6, 6);
    for (auto& v : input) v = sf::bf16_encode(static_cast<float>(small(rng)) * 0.125F);
    if (nan_input) {  // a NaN activation: every sum of that row is NaN (experts 2 and 4, partial tiles)
        input[std::size_t{offset[2] + 5U} * kProdInStride + 3U] = 0x7FC1U;
        input[std::size_t{offset[4]} * kProdInStride + 9U] = 0xFFC0U;
    }
    std::vector<std::uint8_t*> d_codes(experts), d_scales(experts);
    for (unsigned e = 0; e < experts; ++e) {
        std::vector<std::uint8_t> codes(kProdWidth * code_bytes), scales(kProdWidth * scale_bytes);
        for (std::size_t i = 0; i < codes.size(); ++i) {
            const auto seed = i * 17U + i / 3U + i / code_bytes + e;
            codes[i] = nvfp4 ? static_cast<std::uint8_t>(seed % 256U) : fp8[seed % fp8.size()];
        }
        for (std::size_t i = 0; i < scales.size(); ++i) {
            const auto scale = (i + i / scale_bytes + e) % 3U;
            scales[i] = nvfp4 ? fp4_scales[scale] : static_cast<std::uint8_t>(126U + scale);
        }
        d_codes[e] = upload(codes);
        d_scales[e] = upload(scales);
    }
    std::uint16_t* d_input = upload(input);
    float* d_fp32_a = upload(std::vector<float>(fp32_count, kGuard));
    float* d_exact = upload(std::vector<float>(fp32_count, kGuard));
    std::uint16_t* d_image = upload(std::vector<std::uint16_t>(fp32_count, 0xFF81U));
    unsigned* d_seen = upload(std::vector<unsigned>{0U});
    void* tile_storage = nullptr;
    ok(hipMallocManaged(&tile_storage, 2U * experts * sizeof(gm::TileRows)), "tiles");
    auto* tiles_a = static_cast<gm::TileRows*>(tile_storage);
    auto* tiles_b = tiles_a + experts;
    for (unsigned e = 0; e < experts; ++e) {
        const packed::PackedMatrix w{d_codes[e], d_scales[e], kProdWidth, kProdInputs, nvfp4 ? outer[e % 3U] : 1.0F,
                                     e % 2U == 0U ? packed::RowOrder::interleaved_halves : packed::RowOrder::contiguous};
        const auto* in = d_input + std::size_t{offset[e]} * kProdInStride;
        tiles_a[e] = {w, in, d_fp32_a + std::size_t{offset[e]} * kProdStride, kProdInStride, kProdStride, kExpertRows[e]};
        tiles_b[e] = {w, in, d_exact + std::size_t{offset[e]} * kProdStride, kProdInStride, kProdStride, kExpertRows[e]};
        if (!gm::fits_tile<Format>(tiles_a[e], kProdInputs, kProdWidth)) ok(hipErrorInvalidValue, "tile geometry");
    }
    // Combine plan: token t takes kProdTopK distinct grouped rows (a permutation of the rows), one weight per row.
    const unsigned tokens = total / kProdTopK;
    std::vector<std::uint32_t> combine_rows(total);
    for (unsigned i = 0; i < total; ++i) combine_rows[i] = (i * 41U + 7U) % total;
    std::normal_distribution<float> dist(0.0F, 0.5F);
    std::vector<float> weights(total);
    for (auto& w : weights) w = dist(rng);
    std::uint32_t* d_rows = upload(combine_rows);
    float* d_weights = upload(weights);
    constexpr unsigned kProdOutStride = kProdWidth + 3U;
    float* d_out_a = upload(std::vector<float>(std::size_t{tokens} * kProdOutStride, kGuard));
    float* d_out_b = upload(std::vector<float>(std::size_t{tokens} * kProdOutStride, kGuard));

    gm::Args down{tiles_a, experts, kProdInputs, kProdWidth};
    down.max_tile_rows = max_rows;
    down.mean_tile_rows = (total + experts - 1U) / experts;
    down.schedule = schedule;
    gm::DownImageArgs imaged{};
    static_cast<gm::Args&>(imaged) = down;
    imaged.tiles = tiles_b;
    imaged.image_from = d_exact;
    imaged.image = d_image;
    imaged.nan_seen = d_seen;
    // Arm A: the FP32 down + combine. Arm B: the image down + combine_image.
    ok(gm::launch<Format>(down, nullptr), "down A");
    ok(grouped::combine({d_fp32_a, kProdStride, d_rows, d_weights, tokens, kProdTopK, kProdWidth, d_out_a, kProdOutStride},
                        nullptr), "combine A");
    ok(gm::launch_image<Format>(imaged, nullptr), "down B");
    ok(grouped::combine_image({d_image, d_exact, d_seen, kProdStride, d_rows, d_weights, tokens, kProdTopK, kProdWidth,
                               d_out_b, kProdOutStride}, nullptr), "combine B");
    ok(hipDeviceSynchronize(), "sync");

    const auto a = download(d_out_a, std::size_t{tokens} * kProdOutStride);
    const auto b = download(d_out_b, std::size_t{tokens} * kProdOutStride);
    const auto sums = download(d_fp32_a, fp32_count);
    const auto image = download(d_image, fp32_count);
    const unsigned seen = download(d_seen, 1U)[0];
    std::size_t image_bad = 0, nan_sums = 0;
    for (unsigned e = 0; e < experts; ++e)
        for (unsigned row = 0; row < kExpertRows[e]; ++row)
            for (unsigned column = 0; column < kProdWidth; ++column) {
                const auto i = std::size_t{offset[e] + row} * kProdStride + column;
                image_bad += image[i] == sf::bf16_encode(sums[i]) ? 0U : 1U;
                nan_sums += (sf::f2u(sums[i]) & 0x7fffffffU) > 0x7f800000U ? 1U : 0U;
            }
    std::size_t nan_out = 0;
    for (const float v : a) nan_out += (sf::f2u(v) & 0x7fffffffU) > 0x7f800000U ? 1U : 0U;
    const bool same = std::memcmp(a.data(), b.data(), a.size() * sizeof(float)) == 0;
    const bool pass = same && image_bad == 0U && (nan_input ? (nan_sums != 0U && seen == 1U && nan_out != 0U)
                                                             : (nan_sums == 0U && seen == 0U));
    std::printf("%s combine-image-hip producer %s %s%s output_bitwise=%d image_mismatch=%zu nan_sums=%zu nan_seen=%u\n",
                pass ? "PASS" : "FAIL", nvfp4 ? "nvfp4" : "mxfp8", schedule_name, nan_input ? " nan" : "", same ? 1 : 0,
                image_bad, nan_sums, seen);
    for (unsigned e = 0; e < experts; ++e) {
        (void)hipFree(d_codes[e]);
        (void)hipFree(d_scales[e]);
    }
    for (void* p : {static_cast<void*>(d_input), static_cast<void*>(d_fp32_a), static_cast<void*>(d_exact),
                    static_cast<void*>(d_image), static_cast<void*>(d_seen), static_cast<void*>(d_rows),
                    static_cast<void*>(d_weights), static_cast<void*>(d_out_a), static_cast<void*>(d_out_b), tile_storage})
        (void)hipFree(p);
    return pass;
}
}  // namespace

int main() {
    std::mt19937 rng(20261124U);
    bool all = true;
    for (const Case& c : {Case{"finite", false, false}, Case{"finite-stale-flag", false, true},
                          Case{"nan", true, false}, Case{"nan-stale-flag", true, true}})
        all = run(c, rng) && all;
    for (const bool nan_input : {false, true})
        for (const auto& [schedule, name] :
             {std::pair{gm::Schedule::automatic, "auto"}, std::pair{gm::Schedule::staged, "staged"},
              std::pair{gm::Schedule::narrow, "narrow"}, std::pair{gm::Schedule::tall, "tall"}}) {
            all = run_producer<packed::MatrixFormat::nvfp4>(schedule, name, nan_input, rng) && all;
            all = run_producer<packed::MatrixFormat::mxfp8>(schedule, name, nan_input, rng) && all;
        }
    std::puts(all ? "combine_image_hip: PASS" : "combine_image_hip: FAIL");
    return all ? 0 : 1;
}
