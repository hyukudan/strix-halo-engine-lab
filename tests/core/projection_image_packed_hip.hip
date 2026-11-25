// Image projections on the packed-operand core: enqueue_bf16_operand_image + enqueue_bf16_image_rows must store the
// bits enqueue_bf16_rows stores on the FP32 rows (the K64 packed schedule from kPackedImageMinRows rows on whole
// 128-column projections, the K32 image tile otherwise). Cases: a q/k/v-like group of unequal widths, a rounded single
// projection at K 3072, a 128-wide projection, a 64-wide projection (not admitted by the packed route), rows below the
// threshold, at it, and not a multiple of the CTA. Outputs are seeded with a NaN sentinel; every output must be finite.
// Routed cases (enqueue_bf16_rows_routed, one call of any count against enqueue_bf16_rows in runs of kGroupMax): a q-like
// call of six projections (packed, a 24-wide image projection), pitched weights whose pad holds NaN (packed with a row
// pitch), a narrow call the plan keeps on the FP32 rows; also at 8,209 rows (8,192 + 17).
// Route cases: the plan's routes run with the operand image of other rows than the FP32 input, which shows on the device
// which launch served each projection (pitched o, the packed height boundary, K off the K64 stage, mixed weight
// alignment inside one incumbent group); the routed entry must then store the incumbent's bits.
// --bench: ms of the FP32 route and of image + image route (or the routed call) at 8,192 rows. Exit 1 on a failed check.
#include <hip/hip_runtime.h>
#include "strix/hip/legacy_intrinsics.hpp"
#include "strix/hip/legacy_launch.hpp"

#include "attn_core/prepared_gqa.hpp"
#include "core/fma_bf16_rows.hpp"
#include "core/row_transforms.hpp"

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <cstdlib>
#include <cstring>
#include <iostream>
#include <optional>
#include <random>
#include <string>
#include <type_traits>
#include <utility>
#include <vector>

namespace {

namespace fma_rows = strix::core::fma_rows;

// An admitted plan is opaque: callers cannot build one from fields, aggregate-initialise it or write the route, the
// descriptors or the span it carries; the one read is a const admission query.
using Plan = fma_rows::Bf16ImageCallPlan;
template <class T>
concept BuildsFromFields = requires(T) {
    T{{}, {}, static_cast<const std::uint16_t*>(nullptr), std::size_t{1}, 1U, 1U, 1U, std::size_t{1}, true};
};
template <class T>
concept WritesAdmittedFields = requires(T& plan) { plan.admitted = true; } || requires(T& plan) { plan.image_size = 1U; } ||
                               requires(T& plan) { plan.projections[0].output = nullptr; } ||
                               requires(T& plan) { plan.routes[0] = strix::core::ImageRoute::fp32_rows; } ||
                               requires(T& plan) { plan.image = nullptr; };
static_assert(!std::is_aggregate_v<Plan>, "the plan must not be an aggregate");
static_assert(!BuildsFromFields<Plan>, "the plan must not be constructible from its fields");
static_assert(!WritesAdmittedFields<Plan>, "no admitted field may be writable by a caller");
static_assert(std::is_same_v<decltype(std::declval<const Plan&>().admitted()), bool>, "the admission query is const");
// Only the checked admission function builds an admitted plan: no caller-visible default or field constructor, no
// factory function on the type, and copies are the only other way to hold one.
template <class T>
concept HasPlanFactory = requires { T::make(); } || requires { T::admit(); } || requires { T::create(); };
static_assert(!std::is_default_constructible_v<Plan>, "no public default construction");
static_assert(!std::is_constructible_v<Plan, bool> && !std::is_constructible_v<Plan, const Plan*>);
static_assert(!HasPlanFactory<Plan>, "no factory on the plan");
static_assert(std::is_copy_constructible_v<Plan> && !std::is_aggregate_v<Plan>);

// The prepared GQA hook finds the image path of a backend that returns the sealed plan (and not of one that returns the
// old data-member form).
struct SealedPlanBackend {
    using Status = int;
    fma_rows::Bf16ImageCallPlan output_image_plan(unsigned, unsigned, std::span<const int>, const float*,
                                                  const float*) const;
    Status gate_image(const float*, const float*, std::size_t) const;
    Status linear_image(const fma_rows::Bf16ImageCallPlan&, unsigned, unsigned, std::span<const int>) const;
};
struct DataMemberPlan {
    bool admitted;
};
struct DataMemberPlanBackend {
    DataMemberPlan output_image_plan(unsigned, unsigned, std::span<const int>, const float*, const float*) const;
    int gate_image(const float*, const float*, std::size_t) const;
    int linear_image(const DataMemberPlan&, unsigned, unsigned, std::span<const int>) const;
};
static_assert(strix::attn_core::detail::PreparedGqaGateImage<SealedPlanBackend, int>);
static_assert(!strix::attn_core::detail::PreparedGqaGateImage<DataMemberPlanBackend, int>);
using strix::core::Bf16RowProjection;

constexpr unsigned kMaxOutputs = 1U << 20U;

void require(hipError_t status, const char* what) {
    if (status != hipSuccess) {
        std::cerr << "HIP failure: " << what << ": " << hipGetErrorString(status) << '\n';
        std::exit(2);
    }
}

std::uint16_t bf16(float x) {
    std::uint32_t u;
    std::memcpy(&u, &x, 4);
    u += 0x7FFFU + ((u >> 16U) & 1U);
    return static_cast<std::uint16_t>(u >> 16U);
}

template <class T>
T* device(std::size_t n) {
    T* p = nullptr;
    require(hipMalloc(&p, n * sizeof(T)), "malloc");
    return p;
}

struct Case {
    const char* name;
    unsigned K;
    std::vector<unsigned> widths;
    bool round;
    bool routed = false;
    unsigned pitch = 0U;        // weight row elements (0 = K)
    bool expect_image = false;  // routed: the plan takes the image at heights 4,100 and 8,192 (literal, see below)
};

struct Run {
    std::vector<float*> outputs;
    std::vector<Bf16RowProjection> projections;
};

Run make_run(const Case& c, const std::vector<std::uint16_t*>& weights, unsigned rows) {
    Run run;
    for (std::size_t p = 0; p < c.widths.size(); ++p) {
        run.outputs.push_back(device<float>(std::size_t{rows} * c.widths[p]));
        run.projections.push_back({weights[p], c.widths[p], run.outputs.back(), c.round,
                                   c.pitch != 0U ? std::size_t{c.pitch} * 2U : 0U});
    }
    return run;
}

std::vector<float> fetch(const Run& run, unsigned rows) {
    std::vector<float> all;
    for (std::size_t p = 0; p < run.outputs.size(); ++p) {
        std::vector<float> h(std::size_t{rows} * run.projections[p].outputs);
        require(hipMemcpy(h.data(), run.outputs[p], h.size() * 4U, hipMemcpyDeviceToHost), "d2h");
        all.insert(all.end(), h.begin(), h.end());
    }
    return all;
}

void seed(const Run& run, unsigned rows) {
    for (std::size_t p = 0; p < run.outputs.size(); ++p)
        require(hipMemset(run.outputs[p], 0xFF, std::size_t{rows} * run.projections[p].outputs * 4U), "seed");
}

}  // namespace

int main(int argc, char** argv) {
    const bool bench = argc > 1 && std::string(argv[1]) == "--bench";
    const std::vector<Case> cases = {{"qkv_group", 2304U, {3072U, 1024U, 1024U}, false},
                                     {"o_round", 3072U, {2304U}, true},
                                     {"router", 2304U, {128U}, false},
                                     {"narrow64", 2304U, {64U}, false},
                                     {"routed_q_call", 2304U, {3072U, 384U, 24U, 3072U, 1024U, 1024U}, false, true, 0U, true},
                                     {"routed_o_pitched", 3072U, {2304U}, true, true, 3200U, true},
                                     {"routed_o_pitch3328", 3072U, {2304U}, true, true, 3328U, false},  // pitch not measured
                                     {"routed_kv_narrow", 2304U, {1024U, 32U}, false, true, 0U, false},
                                     {"routed_wide_n128_pair", 2304U, {5120U, 128U, 128U}, false, true, 0U, true}};  // unmeasured group: the packed runs are cut into measured launches
    const std::vector<unsigned> heights = {1000U, 1024U, 4096U, 4100U, 8192U, 8209U, 8236U, 9216U, 9232U};  // 9,232: above the measured band
    std::mt19937 rng(20261124U);
    std::normal_distribution<float> nd(0.0F, 1.0F);
    const unsigned max_rows = 9232U, max_k = 3072U, bench_rows = 8192U;
    std::vector<float> host_a(std::size_t{max_rows} * max_k);
    for (auto& v : host_a) v = nd(rng);
    host_a[17] = 0.0F, host_a[18] = -0.0F, host_a[19] = 1.0e-40F;  // signed zero and a subnormal input
    float* a = device<float>(host_a.size());
    require(hipMemcpy(a, host_a.data(), host_a.size() * 4U, hipMemcpyHostToDevice), "a");
    auto* image = device<std::uint16_t>(host_a.size());
    bool ok = true;
    {  // refused before a launch: an image over its own input (whole or partial) and a grid past the launch limit
        auto* over = reinterpret_cast<std::uint16_t*>(a);
        const bool refused =
            fma_rows::enqueue_bf16_operand_image(a, 2048U, 2U, over, nullptr) == hipErrorInvalidValue &&
            fma_rows::enqueue_bf16_operand_image(a, 2048U, 2U, over + 2048U, nullptr) == hipErrorInvalidValue &&
            fma_rows::enqueue_bf16_operand_image(a + 8U, 2048U, 2U, over, nullptr) == hipErrorInvalidValue &&
            fma_rows::enqueue_bf16_operand_image(a, 8192U, (1U << 30U) + 1U, image, nullptr) == hipErrorInvalidValue &&
            hipGetLastError() == hipSuccess;
        std::cout << "rejected_arguments " << (refused ? "PASS" : "FAIL") << '\n';
        ok = ok && refused;
    }
    {  // Which launch served each projection: the plan's routes run with the operand image of OTHER rows (b) while the
       // FP32 rows are a, so an image or packed projection stores the bits enqueue_bf16_rows stores on b and an FP32 one
       // those on a (the two differ). Then the routed entry on a must store the incumbent's bits for every projection.
        using strix::core::ImageRoute;
        constexpr auto P = ImageRoute::packed, I = ImageRoute::image, F = ImageRoute::fp32_rows;
        struct RouteCase {
            const char* name;
            unsigned K, rows;
            std::vector<unsigned> widths;
            bool round;
            unsigned pitch;                  // weight row elements (0 = K)
            std::size_t misaligned;          // projections from this index start 2 bytes past a 16-byte boundary
            std::vector<ImageRoute> expect;  // empty: not planned (every projection keeps the FP32 rows)
        };
        // Pitched o (packed with a row pitch); the q call at the routed height (4,096 routed, 4,095 not); K off the K64
        // stage; one incumbent group of an aligned 2048 and a misaligned 32 (the incumbent runs both on the FMA rows);
        // a planned call whose second group holds the misaligned weight (that whole group keeps the FP32 rows).
        const std::vector<RouteCase> route_cases = {
            {"o_pitched", 3072U, 8192U, {2304U}, true, 3200U, 99U, {P}},
            {"o_pitched_4096", 3072U, 4096U, {2304U}, true, 3200U, 99U, {P}},
            {"o_pitch3328_4096", 3072U, 4096U, {2304U}, true, 3328U, 99U, {}},
            {"q_call_4096", 2304U, 4096U, {3072U, 384U, 24U, 3072U, 1024U, 1024U}, false, 0U, 99U, {P, I, I, P, P, P}},  // 384: no measured geometry
            {"q_call_4095", 2304U, 4095U, {3072U, 384U, 24U, 3072U, 1024U, 1024U}, false, 0U, 99U, {}},
            {"k_off_stage", 2336U, 8192U, {3072U, 1024U}, false, 0U, 99U, {}},
            {"mixed_align", 2304U, 4096U, {2048U, 32U}, false, 0U, 1U, {}},
            {"mixed_groups", 2304U, 4096U, {2048U, 1024U, 24U, 2048U, 32U}, false, 0U, 4U, {P, P, I, F, F}},
        };
        std::vector<float> host_b(host_a.size());
        for (auto& v : host_b) v = nd(rng);
        std::vector<std::uint16_t> host_ib(host_b.size());
        for (std::size_t i = 0; i < host_b.size(); ++i) host_ib[i] = bf16(host_b[i]);
        float* b = device<float>(host_b.size());
        auto* image_b = device<std::uint16_t>(host_ib.size());
        require(hipMemcpy(b, host_b.data(), host_b.size() * 4U, hipMemcpyHostToDevice), "b");
        require(hipMemcpy(image_b, host_ib.data(), host_ib.size() * 2U, hipMemcpyHostToDevice), "image b");
        for (const auto& rc : route_cases) {
            const unsigned ld = rc.pitch != 0U ? rc.pitch : rc.K;
            std::vector<std::uint16_t*> bases;
            Run ref_a, ref_b, got, entry;
            for (std::size_t p = 0; p < rc.widths.size(); ++p) {
                std::vector<std::uint16_t> w(std::size_t{rc.widths[p]} * ld + 8U, std::uint16_t{0x7FC0U});  // pad = NaN
                const std::size_t shift = p >= rc.misaligned ? 1U : 0U;
                for (unsigned o = 0; o < rc.widths[p]; ++o)
                    for (unsigned k = 0; k < rc.K; ++k) w[shift + std::size_t{o} * ld + k] = bf16(0.05F * nd(rng));
                bases.push_back(device<std::uint16_t>(w.size()));
                require(hipMemcpy(bases.back(), w.data(), w.size() * 2U, hipMemcpyHostToDevice), "w");
                for (Run* run : {&ref_a, &ref_b, &got, &entry}) {
                    run->outputs.push_back(device<float>(std::size_t{rc.rows} * rc.widths[p]));
                    run->projections.push_back({bases.back() + shift, rc.widths[p], run->outputs.back(), rc.round,
                                                rc.pitch != 0U ? std::size_t{rc.pitch} * 2U : 0U});
                }
            }
            for (const Run* run : {&ref_a, &ref_b, &got, &entry}) seed(*run, rc.rows);
            const auto incumbent = [&](const float* x, const Run& run) {
                return strix::core::for_each_bf16_projection_group<fma_rows::kGroupMax>(run.projections, hipSuccess,
                    [&](std::span<const Bf16RowProjection> group) {
                        return fma_rows::enqueue_bf16_rows(x, rc.K, rc.rows, group, kMaxOutputs, nullptr);
                    });
            };
            require(incumbent(a, ref_a), "fp32 rows a");
            require(incumbent(b, ref_b), "fp32 rows b");
            ImageRoute routes[fma_rows::kImageRouteMaxProjections]{};
            const bool planned = fma_rows::detail::plan_bf16_image_call(rc.K, rc.rows, got.projections, kMaxOutputs,
                std::span<const std::uint16_t>(image_b, host_ib.size()), routes);
            bool as_planned = planned == !rc.expect.empty();
            for (std::size_t p = 0; p < rc.widths.size(); ++p)
                as_planned = as_planned && routes[p] == (rc.expect.empty() ? F : rc.expect[p]);
            require(planned ? fma_rows::detail::run_bf16_image_routes(a, image_b, rc.K, rc.rows, got.projections,
                                  kMaxOutputs, std::span<const ImageRoute>(routes, rc.widths.size()), nullptr)
                            : incumbent(a, got), "routes");
            require(fma_rows::enqueue_bf16_rows_routed(a, rc.K, rc.rows, entry.projections, kMaxOutputs,
                                                       std::span<std::uint16_t>(image, host_a.size()), nullptr),
                    "routed entry");
            require(hipDeviceSynchronize(), "sync");
            const auto xa = fetch(ref_a, rc.rows), xb = fetch(ref_b, rc.rows), xg = fetch(got, rc.rows),
                       xe = fetch(entry, rc.rows);
            bool served = true;
            for (std::size_t p = 0, at = 0; p < rc.widths.size(); at += std::size_t{rc.rows} * rc.widths[p], ++p) {
                const auto n = std::size_t{rc.rows} * rc.widths[p] * 4U;
                const auto& want = routes[p] != F ? xb : xa;
                const auto& other = routes[p] != F ? xa : xb;
                served = served && std::memcmp(xg.data() + at, want.data() + at, n) == 0 &&
                         std::memcmp(xg.data() + at, other.data() + at, n) != 0;
            }
            const bool same = xe.size() == xa.size() && std::memcmp(xe.data(), xa.data(), xa.size() * 4U) == 0;
            const bool pass = as_planned && served && same;
            ok = ok && pass;
            std::cout << "route " << rc.name << " rows " << rc.rows << (planned ? " planned" : " fp32-rows")
                      << (as_planned ? " routes-as-expected" : " ROUTES-DIFFER")
                      << (served ? " served-as-planned" : " SERVED-ELSEWHERE")
                      << (same ? " entry-same-bits" : " ENTRY-DIFFER") << (pass ? " PASS" : " FAIL") << '\n';
            for (const Run* run : {&ref_a, &ref_b, &got, &entry})
                for (auto* p : run->outputs) require(hipFree(p), "free");
            for (auto* w : bases) require(hipFree(w), "free");
        }
        {  // review r17: an FMA group 0 (projection 0 two bytes past a 16-byte boundary) and a packed group 1 whose
           // projection 3 writes the output of projection 0. The incumbent writes group 0, then group 1 (projection 3
           // wins); the routed entry must keep that sequence (image_routes_order_free refuses the reordering).
            constexpr unsigned K = 2304U, rows = 4096U;
            const unsigned widths[4] = {2048U, 128U, 128U, 2048U};
            std::vector<std::uint16_t*> bases;
            std::vector<float*> outs;
            std::vector<Bf16RowProjection> ref, got;
            float* shared_ref = device<float>(std::size_t{rows} * 2048U);
            float* shared_got = device<float>(std::size_t{rows} * 2048U);
            for (unsigned p = 0; p < 4U; ++p) {
                std::vector<std::uint16_t> w(std::size_t{widths[p]} * K + 8U, std::uint16_t{0x7FC0U});
                const std::size_t shift = p == 0U ? 1U : 0U;
                for (std::size_t i = 0; i < std::size_t{widths[p]} * K; ++i) w[shift + i] = bf16(0.05F * nd(rng));
                bases.push_back(device<std::uint16_t>(w.size()));
                require(hipMemcpy(bases.back(), w.data(), w.size() * 2U, hipMemcpyHostToDevice), "w");
                const bool shared = p == 0U || p == 3U;
                float* out_ref = shared ? shared_ref : device<float>(std::size_t{rows} * widths[p]);
                float* out_got = shared ? shared_got : device<float>(std::size_t{rows} * widths[p]);
                if (!shared) outs.insert(outs.end(), {out_ref, out_got});
                ref.push_back({bases.back() + shift, widths[p], out_ref, false, 0U});
                got.push_back({bases.back() + shift, widths[p], out_got, false, 0U});
            }
            for (const auto* run : {&ref, &got})
                for (const auto& projection : *run)
                    require(hipMemset(projection.output, 0xFF, std::size_t{rows} * projection.outputs * 4U), "seed");
            require(strix::core::for_each_bf16_projection_group<fma_rows::kGroupMax>(
                        std::span<const Bf16RowProjection>(ref), hipSuccess,
                        [&](std::span<const Bf16RowProjection> group) {
                            return fma_rows::enqueue_bf16_rows(a, K, rows, group, kMaxOutputs, nullptr);
                        }),
                    "alias incumbent");
            require(fma_rows::enqueue_bf16_rows_routed(a, K, rows, got, kMaxOutputs,
                                                       std::span<std::uint16_t>(image, host_a.size()), nullptr),
                    "alias routed entry");
            require(hipDeviceSynchronize(), "sync");
            bool same = true;
            for (std::size_t p = 0; p < 4U; ++p) {
                std::vector<float> x(std::size_t{rows} * widths[p]), y(x.size());
                require(hipMemcpy(x.data(), ref[p].output, x.size() * 4U, hipMemcpyDeviceToHost), "d2h");
                require(hipMemcpy(y.data(), got[p].output, y.size() * 4U, hipMemcpyDeviceToHost), "d2h");
                same = same && std::memcmp(x.data(), y.data(), x.size() * 4U) == 0;
            }
            ok = ok && same;
            std::cout << "route alias_fma_first rows " << rows << (same ? " entry-same-bits PASS" : " ENTRY-DIFFER FAIL")
                      << '\n';
            for (auto* p : outs) require(hipFree(p), "free");
            for (auto* p : {shared_ref, shared_got}) require(hipFree(p), "free");
            for (auto* w : bases) require(hipFree(w), "free");
        }
        require(hipFree(b), "free");
        require(hipFree(image_b), "free");
    }
    {  // Which launch the public image entry makes. Expectations are literals (measured geometries: K 2,304 with 128
       // outputs to 8,192 rows and with 1,024 / 2,048 / 2,304 / 3,072 / 5,120 to 9,216, K 3,072 with 2,304 outputs to
       // 9,216, from 4,096 rows), not computed from the policy. Injected device classes go through the public entries.
        using strix::core::ImageLaunch;
        const fma_rows::ImageDeviceClass injected_classes[] = {{"gfx1150", 32U, 40U}, {"gfx1151", 64U, 40U},
                                                               {"gfx1151", 32U, 32U}};
        struct Pick {
            const char* name;
            unsigned K;
            std::vector<unsigned> widths;
            unsigned rows;
            int inject;  // 0: the real device; 1..3: injected unqualified class (gfx1150, wave64, 32 CUs)
            ImageLaunch expect;
        };
        const auto k32 = ImageLaunch::k32_image, k64 = ImageLaunch::k64_packed;
        const std::vector<Pick> picks = {
            {"wide_in_band_8192", 2304U, {3072U, 1024U, 1024U}, 8192U, 0, k64},
            {"wide_in_band_4096", 2304U, {3072U, 1024U, 1024U}, 4096U, 0, k64},
            {"wide_o_in_band", 3072U, {2304U}, 8192U, 0, k64},
            {"qkv_group_rows_8236", 2304U, {3072U, 1024U, 1024U}, 8236U, 0, k64},
            {"qkv_group_rows_9216", 2304U, {3072U, 1024U, 1024U}, 9216U, 0, k64},
            {"n1024_rows_8236", 2304U, {1024U}, 8236U, 0, k64},
            {"n1024_rows_9216", 2304U, {1024U}, 9216U, 0, k64},
            {"n2048_rows_9216", 2304U, {2048U}, 9216U, 0, k64},
            {"k3072_o_rows_9216", 3072U, {2304U}, 9216U, 0, k64},
            {"n128_rows_4096", 2304U, {128U}, 4096U, 0, k64},
            {"n128_rows_8192", 2304U, {128U}, 8192U, 0, k64},
            {"n128_rows_8236", 2304U, {128U}, 8236U, 0, k32},
            {"wide_plus_n128_rows_8236", 2304U, {3072U, 128U}, 8236U, 0, k32},
            // Individually measured widths in a launch group nobody measured: K32 (the narrow members would exit
            // early in most blocks of the wide member's grid).
            {"group_5120_128_128_8192", 2304U, {5120U, 128U, 128U}, 8192U, 0, k32},
            {"group_5120_128_128_4096", 2304U, {5120U, 128U, 128U}, 4096U, 0, k32},
            {"group_3072_1024_8192", 2304U, {3072U, 1024U}, 8192U, 0, k32},
            {"group_3072_3072_8192", 2304U, {3072U, 3072U}, 8192U, 0, k64},  // measured production groups
            {"group_1024_1024_8192", 2304U, {1024U, 1024U}, 8192U, 0, k64},
            {"group_3072_3072_1024_9216", 2304U, {3072U, 3072U, 1024U}, 9216U, 0, k64},
            {"group_3072_3072_1024_4096", 2304U, {3072U, 3072U, 1024U}, 4096U, 0, k64},
            {"group_3072_3072_rows_4095", 2304U, {3072U, 3072U}, 4095U, 0, k32},
            {"group_1024_1024_3072_8192", 2304U, {1024U, 1024U, 3072U}, 8192U, 0, k32},
            {"group_qkv_order_swapped", 2304U, {1024U, 3072U, 1024U}, 8192U, 0, k32},
            {"n256_rows_4096", 2304U, {256U}, 4096U, 0, k32},
            {"n512_rows_8192", 2304U, {512U}, 8192U, 0, k32},
            {"wide_gfx1150", 2304U, {3072U, 1024U, 1024U}, 8192U, 1, k32},
            {"wide_wave64", 2304U, {3072U, 1024U, 1024U}, 8192U, 2, k32},
            {"wide_32cu", 2304U, {3072U, 1024U, 1024U}, 8192U, 3, k32},
            {"wide_rows_9217", 2304U, {3072U, 1024U, 1024U}, 9217U, 0, k32},
            {"wide_rows_9232", 2304U, {3072U, 1024U, 1024U}, 9232U, 0, k32},
            {"wide_rows_4095", 2304U, {3072U, 1024U, 1024U}, 4095U, 0, k32},
            {"k64_n2048_rows_9216", 64U, {2048U}, 9216U, 0, k32},
            {"k64_n2048_rows_8192", 64U, {2048U}, 8192U, 0, k32},
            {"k3072_n3072_unmeasured", 3072U, {3072U}, 8192U, 0, k32},
            {"controller_down_n64_k2304", 2304U, {64U}, 8192U, 0, k32},
            {"controller_down_n64_k3072", 3072U, {64U}, 8192U, 0, k32}};
        for (const auto& pk : picks) {
            const Case c{pk.name, pk.K, pk.widths, false};
            const unsigned ld = pk.K;
            std::vector<std::uint16_t*> weights;
            for (const unsigned width : pk.widths) {
                std::vector<std::uint16_t> w(std::size_t{width} * ld);
                for (auto& v : w) v = bf16(0.05F * nd(rng));
                weights.push_back(device<std::uint16_t>(w.size()));
                require(hipMemcpy(weights.back(), w.data(), w.size() * 2U, hipMemcpyHostToDevice), "w");
            }
            const auto ref = make_run(c, weights, pk.rows);
            const auto img = make_run(c, weights, pk.rows);
            seed(ref, pk.rows);
            seed(img, pk.rows);
            require(strix::core::for_each_bf16_projection_group<fma_rows::kGroupMax>(ref.projections, hipSuccess,
                        [&](std::span<const Bf16RowProjection> group) {
                            return fma_rows::enqueue_bf16_rows(a, pk.K, pk.rows, group, kMaxOutputs, nullptr);
                        }),
                    "fp32 rows");
            require(fma_rows::enqueue_bf16_operand_image(a, pk.K, pk.rows, image, nullptr), "image");
            ImageLaunch launched = ImageLaunch::k32_image;
            bool called = false;
            if (pk.inject != 0) {
                const fma_rows::ScopedImageDeviceClass scoped(injected_classes[pk.inject - 1]);
                called = fma_rows::enqueue_bf16_image_rows(image, pk.K, pk.rows, img.projections, kMaxOutputs, nullptr,
                                                           &launched) == hipSuccess;
            } else {
                called = fma_rows::enqueue_bf16_image_rows(image, pk.K, pk.rows, img.projections, kMaxOutputs, nullptr,
                                                           &launched) == hipSuccess;
            }
            require(hipDeviceSynchronize(), "sync");
            const auto x = fetch(ref, pk.rows), y = fetch(img, pk.rows);
            const bool same = x.size() == y.size() && std::memcmp(x.data(), y.data(), x.size() * 4U) == 0;
            const bool picked = launched == pk.expect;
            ok = ok && called && same && picked;
            std::cout << "select " << pk.name << " rows " << pk.rows << " launch "
                      << (launched == ImageLaunch::k64_packed ? "K64" : "K32") << " expect "
                      << (pk.expect == ImageLaunch::k64_packed ? "K64" : "K32") << (same ? " same-bits" : " DIFFER")
                      << (called && same && picked ? " PASS" : " FAIL") << '\n';
            for (auto* p : ref.outputs) require(hipFree(p), "free");
            for (auto* p : img.outputs) require(hipFree(p), "free");
            for (auto* w : weights) require(hipFree(w), "free");
        }
        {  // Production packed calls are the launches of the previous code (runs of kGroupMax = 3 packed projections), each
           // one launch: the 4-projection call is [3072,3072,1024] + [1024], the q pair and the kv pair one launch each;
           // an unlisted grouping is cut into single launches (literals).
            using strix::core::ImageRoute;
            struct Cut {
                const char* name;
                std::vector<unsigned> widths;
                std::vector<std::size_t> runs;
            };
            const Cut cuts[] = {{"all_call", {3072U, 3072U, 1024U, 1024U}, {3U, 1U}},
                                {"q_pair", {3072U, 3072U}, {2U}},
                                {"kv_pair", {1024U, 1024U}, {2U}},
                                {"qkv_group", {3072U, 1024U, 1024U}, {3U}},
                                {"wide_n128_pair", {5120U, 128U, 128U}, {1U, 1U, 1U}}};
            bool cuts_ok = true;
            for (const auto& cut : cuts)
                for (const unsigned rows : {4096U, 8192U, 8209U, 9216U}) {
                    std::vector<Bf16RowProjection> projections;
                    for (const unsigned width : cut.widths) projections.push_back({nullptr, width, nullptr, false});
                    const ImageRoute routes[4] = {ImageRoute::packed, ImageRoute::packed, ImageRoute::packed,
                                                  ImageRoute::packed};
                    std::vector<std::size_t> got;
                    // rows 8209 / 9216 keep the 128-wide members off the packed table: only the listed groups matter here
                    require(strix::core::for_each_image_route_run<fma_rows::kGroupMax>(
                        std::span<const Bf16RowProjection>(projections),
                        std::span<const ImageRoute>(routes, projections.size()), 2304U, hipSuccess,
                        [&](ImageRoute, std::span<const Bf16RowProjection> run) {
                            got.push_back(run.size());
                            return hipSuccess;
                        },
                        &fma_rows::kImageRoutePolicy, rows), "cut");
                    const bool same = got == cut.runs;
                    cuts_ok = cuts_ok && same;
                    if (!same) std::cout << "select launches " << cut.name << " rows " << rows << " runs " << got.size() << " FAIL\n";
                }
            ok = ok && cuts_ok;
            std::cout << "select launches production_groups_one_launch " << (cuts_ok ? "PASS" : "FAIL") << '\n';
        }
        {  // the call that stores the image: the plan of an unmeasured K / width keeps the FP32 rows (literal)
            const Case c{"k64_n2048_routed", 64U, {2048U}, false, true};
            std::vector<std::uint16_t*> weights{device<std::uint16_t>(std::size_t{2048U} * 64U)};
            require(hipMemset(weights[0], 0, std::size_t{2048U} * 64U * 2U), "w");
            const auto run = make_run(c, weights, 9216U);
            strix::core::ImageRoute routes[fma_rows::kImageRouteMaxProjections]{};
            bool planned_any = false;
            for (const unsigned rows : {8192U, 9216U})
                planned_any = planned_any || fma_rows::detail::plan_bf16_image_call(
                                                 64U, rows, run.projections, kMaxOutputs,
                                                 std::span<std::uint16_t>(image, host_a.size()), routes);
            ok = ok && !planned_any;
            std::cout << "select k64_n2048_routed " << (planned_any ? "PLANNED FAIL" : "fp32-rows PASS") << '\n';
            for (auto* p : run.outputs) require(hipFree(p), "free");
            require(hipFree(weights[0]), "free");
        }
        {  // The device of an explicit stream decides, not the current device (a node has no second GPU: the stream's owner
           // is injected as an unqualified class while the real current device stays the measured one). Default, legacy
           // and per-thread streams are current-device sentinels. Both public entries, guards checked after a sync.
            const fma_rows::ImageDeviceClass owner_other{"gfx1150", 32U, 40U, 1};
            hipStream_t stream = nullptr;
            require(hipStreamCreate(&stream), "stream");
            const Case c{"stream_owner", 2304U, {3072U, 2304U}, false, true};
            std::vector<std::uint16_t*> weights;
            for (const unsigned width : c.widths) {
                std::vector<std::uint16_t> w(std::size_t{width} * c.K);
                for (auto& v : w) v = bf16(0.05F * nd(rng));
                weights.push_back(device<std::uint16_t>(w.size()));
                require(hipMemcpy(weights.back(), w.data(), w.size() * 2U, hipMemcpyHostToDevice), "w");
            }
            const unsigned rows = 8192U;
            const Case ci{"stream_owner_image", 2304U, {3072U, 1024U, 1024U}, false};
            std::vector<std::uint16_t*> iweights;
            for (const unsigned width : ci.widths) {
                std::vector<std::uint16_t> w(std::size_t{width} * ci.K);
                for (auto& v : w) v = bf16(0.05F * nd(rng));
                iweights.push_back(device<std::uint16_t>(w.size()));
                require(hipMemcpy(iweights.back(), w.data(), w.size() * 2U, hipMemcpyHostToDevice), "w");
            }
            struct StreamCase {
                const char* name;
                hipStream_t stream;
                bool foreign_owner;  // the injected owner of `stream` is the unqualified class
                bool expect_packed;
            };
            const StreamCase stream_cases[] = {{"explicit_owner_qualified", stream, false, true},
                                               {"explicit_owner_unqualified", stream, true, false},
                                               {"default_sentinel", nullptr, false, true},
                                               {"per_thread_sentinel", hipStreamPerThread, false, true}};
            const bool legacy_is_current = fma_rows::image_routes_stream_device(hipStreamLegacy) ==
                                           fma_rows::image_routes_current_device();
            ok = ok && legacy_is_current;
            std::cout << "select stream_owner legacy_is_current_device " << (legacy_is_current ? "PASS" : "FAIL") << '\n';
            for (const auto& sc : stream_cases) {
                const auto ref = make_run(c, weights, rows), got = make_run(c, weights, rows);
                const auto iref = make_run(ci, iweights, rows), igot = make_run(ci, iweights, rows);
                for (const Run* run : {&ref, &got, &iref, &igot}) seed(*run, rows);
                require(strix::core::for_each_bf16_projection_group<fma_rows::kGroupMax>(ref.projections, hipSuccess,
                            [&](std::span<const Bf16RowProjection> group) {
                                return fma_rows::enqueue_bf16_rows(a, c.K, rows, group, kMaxOutputs, nullptr);
                            }),
                        "fp32 rows");
                require(strix::core::for_each_bf16_projection_group<fma_rows::kGroupMax>(iref.projections, hipSuccess,
                            [&](std::span<const Bf16RowProjection> group) {
                                return fma_rows::enqueue_bf16_rows(a, ci.K, rows, group, kMaxOutputs, nullptr);
                            }),
                        "fp32 rows");
                require(hipDeviceSynchronize(), "sync");
                bool used = false;
                using strix::core::ImageLaunch;
                ImageLaunch launched = ImageLaunch::k32_image;
                hipError_t routed_status = hipSuccess, image_status = hipSuccess;
                {
                    std::optional<fma_rows::ScopedImageStreamOwner> scoped;
                    if (sc.foreign_owner) scoped.emplace(sc.stream, owner_other);
                    routed_status = fma_rows::enqueue_bf16_rows_routed(a, c.K, rows, got.projections, kMaxOutputs,
                                                                       std::span<std::uint16_t>(image, host_a.size()),
                                                                       sc.stream, &used);
                    require(hipStreamSynchronize(sc.stream), "stream sync");
                    require(fma_rows::enqueue_bf16_operand_image(a, ci.K, rows, image, sc.stream), "image");
                    image_status = fma_rows::enqueue_bf16_image_rows(image, ci.K, rows, igot.projections, kMaxOutputs,
                                                                     sc.stream, &launched);
                }
                require(hipStreamSynchronize(sc.stream), "stream sync");
                require(hipDeviceSynchronize(), "sync");
                const auto x = fetch(ref, rows), y = fetch(got, rows), xi = fetch(iref, rows), yi = fetch(igot, rows);
                const bool same = x.size() == y.size() && std::memcmp(x.data(), y.data(), x.size() * 4U) == 0 &&
                                  xi.size() == yi.size() && std::memcmp(xi.data(), yi.data(), xi.size() * 4U) == 0;
                const bool pass = routed_status == hipSuccess && image_status == hipSuccess && same &&
                                  used == sc.expect_packed && (launched == ImageLaunch::k64_packed) == sc.expect_packed;
                ok = ok && pass;
                std::cout << "select stream_owner " << sc.name << " image_used " << used << " launch "
                          << (launched == ImageLaunch::k64_packed ? "K64" : "K32") << " expect "
                          << (sc.expect_packed ? "image/K64" : "rows/K32") << (same ? " same-bits" : " DIFFER")
                          << (pass ? " PASS" : " FAIL") << '\n';
                for (const Run* run : {&ref, &got, &iref, &igot})
                    for (auto* p : run->outputs) require(hipFree(p), "free");
            }
            for (auto* w : weights) require(hipFree(w), "free");
            for (auto* w : iweights) require(hipFree(w), "free");
            require(hipStreamDestroy(stream), "stream destroy");
        }
        {  // the public routed entry: the real device takes the image routes, every injected class keeps the FP32 rows
            const Case c{"routed_injected", 2304U, {3072U, 2304U}, false, true};
            std::vector<std::uint16_t*> weights;
            for (const unsigned width : c.widths) {
                std::vector<std::uint16_t> w(std::size_t{width} * c.K);
                for (auto& v : w) v = bf16(0.05F * nd(rng));
                weights.push_back(device<std::uint16_t>(w.size()));
                require(hipMemcpy(weights.back(), w.data(), w.size() * 2U, hipMemcpyHostToDevice), "w");
            }
            for (int inject = 0; inject <= 3; ++inject) {
                const unsigned rows = 8192U;
                const auto ref = make_run(c, weights, rows);
                const auto got = make_run(c, weights, rows);
                seed(ref, rows);
                seed(got, rows);
                require(strix::core::for_each_bf16_projection_group<fma_rows::kGroupMax>(ref.projections, hipSuccess,
                            [&](std::span<const Bf16RowProjection> group) {
                                return fma_rows::enqueue_bf16_rows(a, c.K, rows, group, kMaxOutputs, nullptr);
                            }),
                        "fp32 rows");
                bool used = false;
                hipError_t status = hipSuccess;
                {
                    std::optional<fma_rows::ScopedImageDeviceClass> scoped;
                    if (inject != 0) scoped.emplace(injected_classes[inject - 1]);
                    status = fma_rows::enqueue_bf16_rows_routed(a, c.K, rows, got.projections, kMaxOutputs,
                                                                std::span<std::uint16_t>(image, host_a.size()), nullptr,
                                                                &used);
                }
                require(hipDeviceSynchronize(), "sync");
                const auto x = fetch(ref, rows), y = fetch(got, rows);
                const bool same = x.size() == y.size() && std::memcmp(x.data(), y.data(), x.size() * 4U) == 0;
                const bool pass = status == hipSuccess && same && used == (inject == 0);
                ok = ok && pass;
                std::cout << "select routed_injected class " << inject << " image_used " << used << " expect "
                          << (inject == 0) << (same ? " same-bits" : " DIFFER") << (pass ? " PASS" : " FAIL") << '\n';
                for (auto* p : ref.outputs) require(hipFree(p), "free");
                for (auto* p : got.outputs) require(hipFree(p), "free");
            }
            for (auto* w : weights) require(hipFree(w), "free");
        }
    }
    for (const auto& c : cases) {
        std::vector<std::uint16_t*> weights;
        const unsigned ld = c.pitch != 0U ? c.pitch : c.K;
        for (const unsigned width : c.widths) {
            std::vector<std::uint16_t> w(std::size_t{width} * ld, std::uint16_t{0x7FC0U});  // pad = NaN
            for (unsigned o = 0; o < width; ++o)
                for (unsigned k = 0; k < c.K; ++k) w[std::size_t{o} * ld + k] = bf16(0.05F * nd(rng));
            weights.push_back(device<std::uint16_t>(w.size()));
            require(hipMemcpy(weights.back(), w.data(), w.size() * 2U, hipMemcpyHostToDevice), "w");
        }
        for (const unsigned rows : heights) {
            const auto ref = make_run(c, weights, rows);
            const auto img = make_run(c, weights, rows);
            seed(ref, rows);
            seed(img, rows);
            const std::span<std::uint16_t> scratch(image, host_a.size());
            bool image_used = false;  // what the public routed entry reports it did
            const auto fp32 = [&](const Run& run) {
                return strix::core::for_each_bf16_projection_group<fma_rows::kGroupMax>(run.projections, hipSuccess,
                    [&](std::span<const Bf16RowProjection> group) {
                        return fma_rows::enqueue_bf16_rows(a, c.K, rows, group, kMaxOutputs, nullptr);
                    });
            };
            const auto image_arm = [&](const Run& run) {
                if (c.routed)
                    return fma_rows::enqueue_bf16_rows_routed(a, c.K, rows, run.projections, kMaxOutputs, scratch,
                                                              nullptr, &image_used);
                if (const auto status = fma_rows::enqueue_bf16_operand_image(a, c.K, rows, image, nullptr);
                    status != hipSuccess)
                    return status;
                return fma_rows::enqueue_bf16_image_rows(image, c.K, rows, run.projections, kMaxOutputs, nullptr);
            };
            require(fp32(ref), "fp32 rows");
            bool admitted = fma_rows::admits_bf16_image_projections(c.K, rows, img.projections, kMaxOutputs);
            bool expect_used = false;
            if (c.routed) {
                strix::core::ImageRoute routes[fma_rows::kImageRouteMaxProjections]{};
                const bool planned =
                    fma_rows::detail::plan_bf16_image_call(c.K, rows, img.projections, kMaxOutputs, scratch, routes);
                expect_used = c.expect_image && (rows == 4096U || rows == 4100U || rows == 8192U || rows == 8209U || rows == 8236U || rows == 9216U);  // literal
                admitted = planned == expect_used;
            }
            require(image_arm(img), "image rows");
            require(hipDeviceSynchronize(), "sync");
            // The public call itself must report the image route (a wrapper that always fell back would still match bits).
            if (c.routed) admitted = admitted && image_used == expect_used;
            const auto x = fetch(ref, rows), y = fetch(img, rows);
            const bool finite = std::all_of(y.begin(), y.end(), [](float v) { return std::isfinite(v); });
            const bool same = x.size() == y.size() && std::memcmp(x.data(), y.data(), x.size() * 4U) == 0;
            ok = ok && admitted && finite && same;
            std::cout << "check " << c.name << " rows " << rows << (admitted ? " admitted" : " NOT-ADMITTED")
                      << (finite ? " finite" : " NONFINITE") << (same ? " same-bits" : " DIFFER")
                      << (admitted && finite && same ? " PASS" : " FAIL") << '\n';
            if (bench && rows == bench_rows) {
                hipEvent_t e0, e1;
                require(hipEventCreate(&e0), "e");
                require(hipEventCreate(&e1), "e");
                double t[2];
                for (unsigned arm = 0; arm < 2U; ++arm) {
                    std::vector<float> ms;
                    for (unsigned rep = 0; rep < 8U; ++rep) {
                        require(hipEventRecord(e0, nullptr), "r");
                        require(arm == 0U ? fp32(ref) : image_arm(img), "b");
                        require(hipEventRecord(e1, nullptr), "r");
                        require(hipEventSynchronize(e1), "s");
                        float v = 0.0F;
                        require(hipEventElapsedTime(&v, e0, e1), "el");
                        if (rep > 0U) ms.push_back(v);
                    }
                    std::sort(ms.begin(), ms.end());
                    t[arm] = double(ms[ms.size() / 2U]);
                }
                std::cout << "bench " << c.name << " rows " << rows << " fp32_rows_ms " << t[0] << " image_route_ms "
                          << t[1] << " ratio " << t[1] / t[0] << '\n';
            }
            for (auto* p : ref.outputs) require(hipFree(p), "free");
            for (auto* p : img.outputs) require(hipFree(p), "free");
        }
        for (auto* w : weights) require(hipFree(w), "free");
    }
    {  // gate -> image: the gate's own image + enqueue_bf16_rows_from_image == FP32 gated rows + the routed entry
        namespace rt = strix::core::row_transforms;
        const unsigned K = 3072U, N = 2304U, ld = 3200U;  // the pitched output projection
        std::vector<std::uint16_t> w(std::size_t{N} * ld, std::uint16_t{0x7FC0U});
        for (unsigned o = 0; o < N; ++o)
            for (unsigned k = 0; k < K; ++k) w[std::size_t{o} * ld + k] = bf16(0.05F * nd(rng));
        auto* weights = device<std::uint16_t>(w.size());
        require(hipMemcpy(weights, w.data(), w.size() * 2U, hipMemcpyHostToDevice), "w");
        std::vector<float> logits(std::size_t{max_rows} * K);
        for (auto& v : logits) v = 2.0F * nd(rng);
        logits[5] = INFINITY, logits[6] = -INFINITY, logits[7] = NAN, logits[8] = -0.0F;
        auto* gate = device<float>(logits.size());
        require(hipMemcpy(gate, logits.data(), logits.size() * 4U, hipMemcpyHostToDevice), "gate");
        auto* gated = device<float>(logits.size());
        auto* out_a = device<float>(std::size_t{max_rows} * N);
        auto* out_b = device<float>(std::size_t{max_rows} * N);
        const std::span<std::uint16_t> scratch(image, host_a.size());
        for (const unsigned rows : {1000U, 1024U, 8192U, 8209U}) {
            for (const bool rounded : {false, true}) {
                const Bf16RowProjection pa{weights, N, out_a, rounded, std::size_t{ld} * 2U};
                const Bf16RowProjection pb{weights, N, out_b, rounded, std::size_t{ld} * 2U};
                require(hipMemset(out_a, 0xFF, std::size_t{rows} * N * 4U), "seed");
                require(hipMemset(out_b, 0xFF, std::size_t{rows} * N * 4U), "seed");
                const std::size_t n = std::size_t{rows} * K;
                require(rt::elements(n, rt::Bf16SigmoidProduct{a, gate, gated}, nullptr), "gate");
                require(fma_rows::enqueue_bf16_rows_routed(gated, K, rows, {&pa, 1U}, kMaxOutputs, scratch, nullptr),
                        "routed");
                const float* sources[2] = {a, gate};
                const auto plan = fma_rows::plan_bf16_rows_from_image(K, rows, {&pb, 1U}, kMaxOutputs, scratch, sources);
                const bool reads = plan.admitted();
                bool same = reads == (rows == 8192U || rows == 8209U);  // literal: the measured pitched o shape, 4,096..9,216 rows
                if (reads) {
                    require(rt::elements(n, rt::Bf16SigmoidProductImage{a, gate, image}, nullptr), "gate image");
                    require(fma_rows::enqueue_bf16_rows_from_image(plan, scratch, K, rows, {&pb, 1U}, kMaxOutputs,
                                                                   nullptr),
                            "from image");
                } else {
                    same = same && fma_rows::enqueue_bf16_rows_from_image(plan, scratch, K, rows, {&pb, 1U},
                                                                          kMaxOutputs, nullptr) == hipErrorNotSupported;
                    require(fma_rows::enqueue_bf16_rows_routed(gated, K, rows, {&pb, 1U}, kMaxOutputs, scratch, nullptr),
                            "routed b");
                }
                require(hipDeviceSynchronize(), "sync");
                std::vector<float> ha(std::size_t{rows} * N), hb(ha.size());
                require(hipMemcpy(ha.data(), out_a, ha.size() * 4U, hipMemcpyDeviceToHost), "a");
                require(hipMemcpy(hb.data(), out_b, hb.size() * 4U, hipMemcpyDeviceToHost), "b");
                same = same && std::memcmp(ha.data(), hb.data(), ha.size() * 4U) == 0;
                std::cout << "gate_image rows " << rows << " round " << rounded << " reads_image " << reads << ' '
                          << (same ? "EXACT" : "DIFFER") << '\n';
                ok = ok && same;
            }
        }
        {  // review r17: an image over a gate input (attended rows a, gate logits) or over a weight is refused before
           // any launch: the plan is not admitted, the consumer launches nothing, every byte under the image survives.
            const unsigned rows = 8192U;
            const std::size_t n = std::size_t{rows} * K;
            auto* big = device<std::uint16_t>(n + w.size() + 64U);  // weights at 0, an image from byte 16 on
            require(hipMemcpy(big, w.data(), w.size() * 2U, hipMemcpyHostToDevice), "w big");
            const Bf16RowProjection pw{big, N, out_b, false, std::size_t{ld} * 2U};
            const Bf16RowProjection pc{weights, N, out_b, false, std::size_t{ld} * 2U};
            require(hipMemset(out_b, 0xFF, std::size_t{rows} * N * 4U), "seed");
            const float* sources[2] = {a, gate};
            struct Overlap {
                const char* name;
                const Bf16RowProjection* projection;
                std::uint16_t* image;
            };
            const Overlap overlaps[] = {{"attended", &pc, reinterpret_cast<std::uint16_t*>(a) + 8U},
                                        {"gate_logits", &pc, reinterpret_cast<std::uint16_t*>(gate) + 8U},
                                        {"weight", &pw, big + 8U}};
            for (const auto& o : overlaps) {
                const std::span<const std::uint16_t> view(o.image, n);
                const auto plan = fma_rows::plan_bf16_rows_from_image(K, rows, {o.projection, 1U}, kMaxOutputs, view,
                                                                      sources);
                const bool refused =
                    !plan.admitted() && fma_rows::enqueue_bf16_rows_from_image(plan, view, K, rows, {o.projection, 1U},
                                                                             kMaxOutputs, nullptr) == hipErrorNotSupported;
                require(hipDeviceSynchronize(), "sync");
                std::vector<float> ha(host_a.size()), hg(logits.size()), ho(std::size_t{rows} * N);
                std::vector<std::uint16_t> hw(w.size());
                require(hipMemcpy(ha.data(), a, ha.size() * 4U, hipMemcpyDeviceToHost), "a back");
                require(hipMemcpy(hg.data(), gate, hg.size() * 4U, hipMemcpyDeviceToHost), "gate back");
                require(hipMemcpy(hw.data(), big, hw.size() * 2U, hipMemcpyDeviceToHost), "w back");
                require(hipMemcpy(ho.data(), out_b, ho.size() * 4U, hipMemcpyDeviceToHost), "out back");
                const bool guarded = std::memcmp(ha.data(), host_a.data(), ha.size() * 4U) == 0 &&
                                     std::memcmp(hg.data(), logits.data(), hg.size() * 4U) == 0 &&
                                     std::memcmp(hw.data(), w.data(), hw.size() * 2U) == 0 &&
                                     std::all_of(ho.begin(), ho.end(), [](float v) {
                                         std::uint32_t bits = 0;
                                         std::memcpy(&bits, &v, 4U);
                                         return bits == 0xFFFFFFFFU;
                                     });
                std::cout << "gate_image_overlap " << o.name << (refused ? " refused" : " ADMITTED")
                          << (guarded ? " guard-intact" : " GUARD-CHANGED") << (refused && guarded ? " PASS" : " FAIL")
                          << '\n';
                ok = ok && refused && guarded;
            }
            require(hipFree(big), "free");
        }
        {  // review r20: the plan binds the descriptors and the image span it admitted. A consumer call with a substituted
           // descriptor at the same count (the output moved into the image) or a shorter view of the same image is
           // refused before any launch: the output rows and the image keep every byte.
            const unsigned rows = 8192U;
            const Bf16RowProjection pc{weights, N, out_b, false, std::size_t{ld} * 2U};
            const float* sources[2] = {a, gate};
            const auto plan = fma_rows::plan_bf16_rows_from_image(K, rows, {&pc, 1U}, kMaxOutputs, scratch, sources);
            auto moved = pc;
            moved.output = reinterpret_cast<float*>(scratch.data());
            require(hipMemset(out_b, 0xFF, std::size_t{rows} * N * 4U), "seed");
            require(hipMemset(scratch.data(), 0x5A, scratch.size() * 2U), "seed image");
            const bool substituted = fma_rows::enqueue_bf16_rows_from_image(plan, scratch, K, rows, {&moved, 1U},
                                                                            kMaxOutputs, nullptr) == hipErrorNotSupported;
            const bool short_view = fma_rows::enqueue_bf16_rows_from_image(plan, scratch.first(1U), K, rows, {&pc, 1U},
                                                                           kMaxOutputs, nullptr) == hipErrorNotSupported;
            // a refused admission (a one-element span) is a plan that launches nothing
            const auto inert = fma_rows::plan_bf16_rows_from_image(K, rows, {&pc, 1U}, kMaxOutputs, scratch.first(1U), sources);
            const bool inert_refused =
                !inert.admitted() && fma_rows::enqueue_bf16_rows_from_image(inert, scratch, K, rows, {&pc, 1U}, kMaxOutputs,
                                                                           nullptr) == hipErrorNotSupported;
            // device binding: another current device (injected) refuses the plan admitted here, and a plan admitted on
            // another (injected) device is refused here; nothing is enqueued either way
            int here = -1;
            require(hipGetDevice(&here), "device");
            const fma_rows::ImageDeviceClass elsewhere{"gfx1151", 32U, 40U, here + 1};
            bool device_bound = false;
            {
                const fma_rows::ScopedImageDeviceClass scoped(elsewhere);
                device_bound = fma_rows::enqueue_bf16_rows_from_image(plan, scratch, K, rows, {&pc, 1U}, kMaxOutputs,
                                                                     nullptr) == hipErrorNotSupported;
            }
            Plan remote = inert;
            {
                const fma_rows::ScopedImageDeviceClass scoped(elsewhere);
                remote = fma_rows::plan_bf16_rows_from_image(K, rows, {&pc, 1U}, kMaxOutputs, scratch, sources);
            }
            device_bound = device_bound && remote.admitted() &&
                           fma_rows::enqueue_bf16_rows_from_image(remote, scratch, K, rows, {&pc, 1U}, kMaxOutputs,
                                                                  nullptr) == hipErrorNotSupported;
            // the store policy is part of the admitted descriptor: a call that changes only it is refused
            auto policy_changed = pc;
            policy_changed.round_separate = !pc.round_separate;
            const bool policy_bound = fma_rows::enqueue_bf16_rows_from_image(plan, scratch, K, rows, {&policy_changed, 1U},
                                                                           kMaxOutputs, nullptr) == hipErrorNotSupported;
            // a launch stream owned by another device than the admitted one is refused, the current device unchanged
            // (the stream's owner is injected: a node has no second GPU); the default, legacy and per-thread streams
            // belong to the current device
            hipStream_t stream = nullptr;
            require(hipStreamCreate(&stream), "stream");
            const fma_rows::ImageDeviceClass stream_elsewhere{"gfx1151", 32U, 40U, here + 1};
            bool stream_bound = false;
            {
                const fma_rows::ScopedImageStreamOwner owner(stream, stream_elsewhere);
                stream_bound = fma_rows::enqueue_bf16_rows_from_image(plan, scratch, K, rows, {&pc, 1U}, kMaxOutputs,
                                                                      stream) == hipErrorNotSupported;
            }
            // Every refused call above ran before this point: a launch that still returned the refusal code shows in
            // the bytes after the synchronization.
            require(hipStreamSynchronize(stream), "stream sync");
            require(hipDeviceSynchronize(), "sync");
            std::vector<float> ho(std::size_t{rows} * N);
            std::vector<std::uint16_t> hi(scratch.size());
            require(hipMemcpy(ho.data(), out_b, ho.size() * 4U, hipMemcpyDeviceToHost), "out back");
            require(hipMemcpy(hi.data(), scratch.data(), hi.size() * 2U, hipMemcpyDeviceToHost), "image back");
            const bool guarded = std::all_of(ho.begin(), ho.end(), [](float v) {
                                     std::uint32_t bits = 0;
                                     std::memcpy(&bits, &v, 4U);
                                     return bits == 0xFFFFFFFFU;
                                 }) &&
                                 std::all_of(hi.begin(), hi.end(), [](std::uint16_t v) { return v == 0x5A5AU; });
            const bool pass = plan.admitted() && substituted && short_view && inert_refused && device_bound &&
                              policy_bound && stream_bound && guarded;
            // controls after the guards: the same call on the admitted device's own explicit stream, on the per-thread
            // stream and on the default stream is accepted
            bool accepted = true;
            for (hipStream_t s : {stream, hipStreamPerThread, static_cast<hipStream_t>(nullptr)}) {
                accepted = accepted && fma_rows::enqueue_bf16_rows_from_image(plan, scratch, K, rows, {&pc, 1U},
                                                                              kMaxOutputs, s) == hipSuccess;
                require(hipStreamSynchronize(s), "control sync");
            }
            require(hipStreamDestroy(stream), "stream destroy");
            std::cout << "gate_image_plan_binding admitted " << plan.admitted() << " substituted "
                      << (substituted ? "refused" : "ADMITTED") << " short_view " << (short_view ? "refused" : "ADMITTED") << " device_bound " << (device_bound ? "refused" : "LAUNCHED")
                      << " policy_bound " << (policy_bound ? "refused" : "LAUNCHED") << " stream_bound "
                      << (stream_bound ? "refused" : "LAUNCHED") << (guarded ? " guard-intact" : " GUARD-CHANGED")
                      << " own_streams " << (accepted ? "accepted" : "REFUSED") << (pass && accepted ? " PASS" : " FAIL")
                      << '\n';
            ok = ok && pass && accepted;
        }
        for (void* p : {static_cast<void*>(weights), static_cast<void*>(gate), static_cast<void*>(gated),
                        static_cast<void*>(out_a), static_cast<void*>(out_b)})
            require(hipFree(p), "free");
    }
    std::cout << (ok ? "ALL PASS" : "FAILED") << '\n';
    return ok ? 0 : 1;
}
