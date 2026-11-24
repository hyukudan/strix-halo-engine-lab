#include "core/dense_row_plan.hpp"
#include "cuda/core/dense/row_plan.hpp"

#include <algorithm>
#include <array>
#include <bit>
#include <cmath>
#include <cstdint>
#include <initializer_list>
#include <iostream>
#include <limits>
#include <random>
#include <span>
#include <utility>
#include <vector>

namespace {

enum class Status { ok, failed };

bool check_row_chunks() {
    std::vector<std::pair<unsigned, unsigned>> chunks;
    const auto record = [&](unsigned first, unsigned count) {
        chunks.emplace_back(first, count);
        return Status::ok;
    };
    using strix::core::for_each_dense_chunk;
    if (for_each_dense_chunk<32U>(0U, Status::ok, record) != Status::ok || !chunks.empty()) return false;
    if (for_each_dense_chunk<32U>(32U, Status::ok, record) != Status::ok ||
        chunks != decltype(chunks){{0U, 32U}}) return false;
    chunks.clear();
    if (for_each_dense_chunk<32U>(65U, Status::ok, record) != Status::ok ||
        chunks != decltype(chunks){{0U, 32U}, {32U, 32U}, {64U, 1U}}) return false;
    chunks.clear();
    constexpr auto max_rows = std::numeric_limits<unsigned>::max();
    return for_each_dense_chunk<max_rows - 1U>(max_rows, Status::ok, record) == Status::ok &&
           chunks == decltype(chunks){{0U, max_rows - 1U}, {max_rows - 1U, 1U}};
}

bool check_projection_groups() {
    using namespace strix::core;
    std::array<Bf16RowProjection, 9> projections{};
    for (unsigned i = 0; i < projections.size(); ++i) projections[i].outputs = i + 1U;
    std::vector<unsigned> seen;
    std::vector<std::size_t> counts;
    const auto record = [&](std::span<const Bf16RowProjection> group) {
        counts.push_back(group.size());
        for (const auto& projection : group) seen.push_back(projection.outputs);
        return Status::ok;
    };
    if (for_each_bf16_projection_group<4U>(projections, Status::ok, record) != Status::ok ||
        counts != decltype(counts){4U, 4U, 1U} || seen != decltype(seen){1U, 2U, 3U, 4U, 5U, 6U, 7U, 8U, 9U})
        return false;
    unsigned calls = 0;
    const auto fail_second = [&](std::span<const Bf16RowProjection>) {
        return ++calls == 2U ? Status::failed : Status::ok;
    };
    return for_each_bf16_projection_group<4U>(projections, Status::ok, fail_second) == Status::failed && calls == 2U;
}

bool check_cuda_bf16_paths() {
    using strix::cuda_core::Bf16RowPath;
    using strix::cuda_core::kFp32GridMaxRows;
    using strix::cuda_core::plan_bf16_row_projection;
    constexpr unsigned input = 2304U;
    constexpr unsigned short_limit = 32U;
    for (const unsigned rows : {33U, 2048U}) {
        for (const unsigned out : {384U, 1024U, 3072U})
            if (plan_bf16_row_projection(input, out, rows, short_limit, true) != Bf16RowPath::big) return false;
        for (const unsigned out : {16U, 32U})
            if (plan_bf16_row_projection(input, out, rows, short_limit, true) != Bf16RowPath::big_compact) return false;
        if (plan_bf16_row_projection(input, 24U, rows, short_limit, true) != Bf16RowPath::fp32_grid ||
            plan_bf16_row_projection(input + 8U, 32U, rows, short_limit, true) != Bf16RowPath::fp32_grid)
            return false;
    }
    return plan_bf16_row_projection(input, 24U, kFp32GridMaxRows, short_limit, true) == Bf16RowPath::fp32_grid &&
           plan_bf16_row_projection(input, 24U, kFp32GridMaxRows + 1U, short_limit, true) == Bf16RowPath::short_rows &&
           plan_bf16_row_projection(input, 32U, short_limit, short_limit, true) == Bf16RowPath::short_rows &&
           plan_bf16_row_projection(input, 3072U, 2048U, short_limit, false) == Bf16RowPath::short_rows;
}

// FAST mma2 / mma1 take the tall path where mma3 does, except that the
// three-term compact tile becomes plain big. Explicit terms 3 is the default.
bool check_cuda_bf16_fast_terms() {
    using strix::cuda_core::Bf16RowPath;
    using strix::cuda_core::plan_bf16_row_projection;
    constexpr unsigned input = 2304U;
    constexpr unsigned short_limit = 32U;
    for (const unsigned rows : {16U, 32U, 33U, 2048U, 70000U})
        for (const unsigned in : {input, input + 8U})
            for (const unsigned out : {16U, 24U, 32U, 48U, 384U, 3072U})
                for (const bool tall : {false, true}) {
                    const auto exact = plan_bf16_row_projection(in, out, rows, short_limit, tall);
                    if (plan_bf16_row_projection(in, out, rows, short_limit, tall, 3) != exact) return false;
                    for (const int terms : {1, 2}) {
                        const auto fast = plan_bf16_row_projection(in, out, rows, short_limit, tall, terms);
                        const auto expected = exact == Bf16RowPath::big_compact ? Bf16RowPath::big : exact;
                        if (fast != expected) return false;
                    }
                    for (const int terms : {0, 4})
                        if (plan_bf16_row_projection(in, out, rows, short_limit, tall, terms) != Bf16RowPath::short_rows)
                            return false;
                }
    return plan_bf16_row_projection(input, 32U, 2048U, short_limit, true, 1) == Bf16RowPath::big &&
           plan_bf16_row_projection(input, 3072U, 2048U, short_limit, true, 2) == Bf16RowPath::big &&
           plan_bf16_row_projection(input, 3072U, short_limit, short_limit, true, 1) == Bf16RowPath::short_rows;
}

// Host model of the split kernel (RN-even to BF16, FP32 subtraction) and of the
// tall chain order (per K step lo, mid, hi into one accumulator from +0).
float bf16_rn(float value) {
    const auto bits = std::bit_cast<std::uint32_t>(value);
    const std::uint32_t lsb = (bits >> 16U) & 1U;
    return std::bit_cast<float>((bits + 0x7FFFU + lsb) & 0xFFFF0000U);
}

struct Planes {
    float hi, mid, lo;
};

Planes split3(float x) {
    const float hi = bf16_rn(x);
    const float r = x - hi;
    const float mid = bf16_rn(r);
    return {hi, mid, bf16_rn(r - mid)};
}

// A BF16-exact finite operand needs one plane: mid and lo are +0 and the
// three-term chain adds only zero products to an accumulator that is never -0.
bool check_exact_operand_planes() {
    using strix::core::Bf16Operand;
    using strix::core::exact_bf16_planes;
    static_assert(exact_bf16_planes(Bf16Operand::fp32) == 3 && exact_bf16_planes(Bf16Operand::bf16_exact) == 1);
    std::mt19937 rng(20261122U);
    std::uniform_int_distribution<std::uint32_t> high(0U, 0xFFFFU);
    std::normal_distribution<float> normal(0.0F, 1.0F);
    unsigned fp32_with_residual = 0;
    for (unsigned trial = 0; trial < 200U; ++trial) {
        constexpr unsigned k = 64U;
        std::array<float, k> x{}, w{};
        for (unsigned i = 0; i < k; ++i) {
            std::uint32_t bits = high(rng) << 16U;
            if (((bits >> 23U) & 0xFFU) > 0x9FU) bits &= 0xBFFFFFFFU;  // finite, |x| < 2^33 (no product overflow)
            x[i] = std::bit_cast<float>(bits);
            if (trial % 2U == 0U) x[i] = bf16_rn(normal(rng) * (trial % 4U == 0U ? 1.0e-30F : 1.0F));
            w[i] = bf16_rn(normal(rng));
            const auto planes = split3(x[i]);
            if (std::bit_cast<std::uint32_t>(planes.hi) != std::bit_cast<std::uint32_t>(x[i]) ||
                std::bit_cast<std::uint32_t>(planes.mid) != 0U || std::bit_cast<std::uint32_t>(planes.lo) != 0U)
                return false;
            const float y = normal(rng);
            const auto general = split3(y);
            if (std::bit_cast<std::uint32_t>(general.mid) != 0U) ++fp32_with_residual;
        }
        float three = 0.0F, one = 0.0F;
        for (unsigned i = 0; i < k; ++i) {
            const auto planes = split3(x[i]);
            three += planes.lo * w[i];
            three += planes.mid * w[i];
            three += planes.hi * w[i];
            one += planes.hi * w[i];
            if (std::signbit(three) && three == 0.0F) return false;  // the accumulator is never -0
        }
        if (std::bit_cast<std::uint32_t>(three) != std::bit_cast<std::uint32_t>(one)) return false;
    }
    return fp32_with_residual > 0U;  // an undeclared FP32 operand does need the extra planes
}

// A backend whose dense takes the operand class receives the caller's declaration; one without that overload is
// still called with its four arguments (the FP32 operand).
bool check_dense_declared() {
    using strix::core::Bf16Operand;
    using strix::core::Bf16RowProjection;
    struct Declaring {
        int calls = 0;
        Bf16Operand last = Bf16Operand::fp32;
        void dense(const float*, unsigned, unsigned, std::span<const Bf16RowProjection>, Bf16Operand operand) {
            ++calls;
            last = operand;
        }
    };
    struct Plain {
        int calls = 0;
        void dense(const float*, unsigned, unsigned, std::span<const Bf16RowProjection>) { ++calls; }
    };
    const float input[4]{};
    float output[2]{};
    const std::uint16_t weights[8]{};
    const Bf16RowProjection projections[] = {{weights, 2U, output, false, 0U}};
    Declaring declaring;
    strix::core::dense_declared(declaring, input, 4U, 1U, projections, true);
    if (declaring.calls != 1 || declaring.last != Bf16Operand::bf16_exact) return false;
    strix::core::dense_declared(declaring, input, 4U, 1U, projections, false);
    if (declaring.calls != 2 || declaring.last != Bf16Operand::fp32) return false;
    Plain plain;
    strix::core::dense_declared(plain, input, 4U, 1U, projections, true);
    return plain.calls == 1;
}

}  // namespace

/// The image route plan: inside the incumbent groups the WMMA tile serves, whole-tile projections (pitched included)
// go packed, narrow contiguous ones to the image tile, what neither reads keeps the FP32 rows; a group the incumbent
// runs elsewhere (here: a misaligned weight) keeps every projection on the FP32 rows; a call below the height or the
// packed-column bar keeps every FP32 route; runs keep call order per route and split on the group limit, the rounding
// class and the row pitch, and an FP32 run never leaves its incumbent group.
static bool check_image_routes() {
    using strix::core::Bf16RowProjection;
    using strix::core::ImageRoute;
    using R = ImageRoute;
    // Measured shapes of this policy: K 2,304 with 128..3,072 outputs (contiguous) and 2,304 outputs on a 3,200-element
    // pitch (6,400-byte rows).
    static constexpr std::array<strix::core::ImageShapeRow, 2> shapes{{{2304U, 128U, 3072U, 0U},
                                                                      {2304U, 2304U, 2304U, 3200U}}};
    constexpr strix::core::ImageRoutePolicy policy{1024U, 64U, 128U, 2048U, 9216U, shapes};
    constexpr unsigned K = 2304U;
    constexpr unsigned kWide = 1U << 20U;
    const auto* w = std::bit_cast<const std::uint16_t*>(std::uintptr_t{0x10000U});
    const auto* odd = std::bit_cast<const std::uint16_t*>(std::uintptr_t{0x10008U});
    auto* y = std::bit_cast<float*>(std::uintptr_t{0x20000U});
    const auto image_admits = [](const Bf16RowProjection& p) {  // the image tile: contiguous 16-byte aligned rows
        return p.weights != nullptr && p.contiguous(K) && std::bit_cast<std::uintptr_t>(p.weights) % 16U == 0U;
    };
    const auto group_admits = [](std::span<const Bf16RowProjection> group) {  // the incumbent's WMMA condition
        return std::all_of(group.begin(), group.end(), [](const Bf16RowProjection& p) {
            return std::bit_cast<std::uintptr_t>(p.weights) % 16U == 0U;
        });
    };
    std::array<ImageRoute, 8> routes{};
    std::vector<std::pair<R, std::vector<unsigned>>> runs;
    const auto record = [&](R route, std::span<const Bf16RowProjection> run) {
        std::vector<unsigned> widths;
        for (const auto& p : run) widths.push_back(p.outputs);
        runs.emplace_back(route, widths);
        return Status::ok;
    };
    const auto routed = [&](std::initializer_list<R> expect) {
        return std::equal(expect.begin(), expect.end(), routes.begin());
    };
    // Groups of 3: [q 3072, relative 384, sink 24] (WMMA) and [o-like pitched 2304 (6400 B rows, rounded), a misaligned
    // 128, k 1024] (FMA rows for the whole group: the misaligned weight sends the incumbent there).
    const std::array<Bf16RowProjection, 6> call{{{w, 3072U, y, false, 0U}, {w, 384U, y, false, 0U},
        {w, 24U, y, false, 0U}, {w, 2304U, y, true, 6400U}, {odd, 128U, y, false, 0U}, {w, 1024U, y, false, 0U}}};
    if (!strix::core::plan_image_routes<3U>(policy, K, 8192U, call, kWide, routes, group_admits, image_admits) ||
        !routed({R::packed, R::packed, R::image, R::fp32_rows, R::fp32_rows, R::fp32_rows}))
        return false;
    if (strix::core::for_each_image_route_run<3U>(call, routes, K, Status::ok, record) != Status::ok ||
        runs != decltype(runs){{R::packed, {3072U, 384U}}, {R::image, {24U}}, {R::fp32_rows, {2304U}},
                               {R::fp32_rows, {128U, 1024U}}})
        return false;
    // The same call without the misaligned weight: [q, rel, sink] [o, k]: packed [q, rel] then [o] (pitch + round
    // change) then [k]; image [sink].
    const std::array<Bf16RowProjection, 5> aligned{{{w, 3072U, y, false, 0U}, {w, 384U, y, false, 0U},
        {w, 24U, y, false, 0U}, {w, 2304U, y, true, 6400U}, {w, 1024U, y, false, 0U}}};
    runs.clear();
    if (!strix::core::plan_image_routes<3U>(policy, K, 8192U, aligned, kWide, routes, group_admits, image_admits) ||
        !routed({R::packed, R::packed, R::image, R::packed, R::packed}) ||
        strix::core::for_each_image_route_run<3U>(aligned, routes, K, Status::ok, record) != Status::ok ||
        runs != decltype(runs){{R::packed, {3072U, 384U}}, {R::packed, {2304U}}, {R::packed, {1024U}},
                               {R::image, {24U}}})
        return false;
    // Group limit: 4 packed projections in runs of 3 + 1.
    const std::array<Bf16RowProjection, 4> four{{{w, 1024U, y, false, 0U}, {w, 1024U, y, false, 0U},
        {w, 1024U, y, false, 0U}, {w, 1024U, y, false, 0U}}};
    runs.clear();
    if (!strix::core::plan_image_routes<3U>(policy, K, 4096U, four, kWide, routes, group_admits, image_admits) ||
        strix::core::for_each_image_route_run<3U>(four, routes, K, Status::ok, record) != Status::ok ||
        runs != decltype(runs){{R::packed, {1024U, 1024U, 1024U}}, {R::packed, {1024U}}})
        return false;
    // Rounded projections of different store policies (separate / incumbent) never share a run.
    const std::array<Bf16RowProjection, 3> policies{{{w, 1024U, y, true, 0U, true}, {w, 1024U, y, true, 0U, true},
        {w, 1024U, y, true, 0U, false}}};
    runs.clear();
    if (!strix::core::plan_image_routes<3U>(policy, K, 4096U, policies, kWide, routes, group_admits, image_admits) ||
        strix::core::for_each_image_route_run<3U>(policies, routes, K, Status::ok, record) != Status::ok ||
        runs != decltype(runs){{R::packed, {1024U, 1024U}}, {R::packed, {1024U}}})
        return false;
    // Groups of 2: [q, a pitched 96 (FP32 inside a WMMA group)] [misaligned pitched 96, misaligned pitched 64]: the two
    // FP32 runs share pitch and rounding but not the incumbent group, so they stay apart.
    const std::array<Bf16RowProjection, 4> split{{{w, 3072U, y, false, 0U}, {w, 96U, y, false, 6400U},
        {odd, 96U, y, false, 6400U}, {odd, 64U, y, false, 6400U}}};
    runs.clear();
    if (!strix::core::plan_image_routes<2U>(policy, K, 8192U, split, kWide, routes, group_admits, image_admits) ||
        !routed({R::packed, R::fp32_rows, R::fp32_rows, R::fp32_rows}) ||
        strix::core::for_each_image_route_run<2U>(split, routes, K, Status::ok, record) != Status::ok ||
        runs != decltype(runs){{R::packed, {3072U}}, {R::fp32_rows, {96U}}, {R::fp32_rows, {96U, 64U}}})
        return false;
    const auto all_fp32 = [&](std::size_t n) {
        return std::all_of(routes.begin(), routes.begin() + static_cast<std::ptrdiff_t>(n),
                           [](R r) { return r == R::fp32_rows; });
    };
    // One group of an aligned 2048 and a misaligned contiguous 32: the incumbent runs both on the FMA rows, so the
    // 2,048 whole-tile columns do not take the image.
    const std::array<Bf16RowProjection, 2> mixed{{{w, 2048U, y, false, 0U}, {odd, 32U, y, false, 0U}}};
    if (strix::core::plan_image_routes<3U>(policy, K, 8192U, mixed, kWide, routes, group_admits, image_admits) ||
        !all_fp32(2U))
        return false;
    // A row pitch whose outputs x pitch wraps is not packed (2,048 outputs x 2^63 bytes).
    const std::array<Bf16RowProjection, 1> wrapped{{{w, 2048U, y, false, std::size_t{1} << 63U}}};
    if (strix::core::plan_image_routes<3U>(policy, K, 8192U, wrapped, kWide, routes, group_admits, image_admits) ||
        !all_fp32(1U))
        return false;
    // Narrow calls ([k], [k, k_taps]) stay on FP32: 1,024 packed columns < 2,048.
    const std::array<Bf16RowProjection, 2> narrow{{{w, 1024U, y, false, 0U}, {w, 32U, y, false, 0U}}};
    if (strix::core::plan_image_routes<3U>(policy, K, 8192U, narrow, kWide, routes, group_admits, image_admits) ||
        !all_fp32(2U))
        return false;
    // Below the height, K off the stage, outputs above max_outputs (384 packed columns left), too few route slots.
    if (strix::core::plan_image_routes<3U>(policy, K, 1023U, aligned, kWide, routes, group_admits, image_admits) ||
        !all_fp32(5U))
        return false;
    if (strix::core::plan_image_routes<3U>(policy, K + 32U, 8192U, aligned, kWide, routes, group_admits,
                                           image_admits) || !all_fp32(5U))
        return false;
    if (strix::core::plan_image_routes<3U>(policy, K, 8192U, call, 2048U, routes, group_admits, image_admits) ||
        !all_fp32(6U))
        return false;
    // Above the measured band (9,216 rows): the call keeps its FP32 rows; its top edge still routes.
    if (strix::core::plan_image_routes<3U>(policy, K, 16384U, aligned, kWide, routes, group_admits, image_admits) ||
        !all_fp32(5U) ||
        strix::core::plan_image_routes<3U>(policy, K, 9217U, aligned, kWide, routes, group_admits, image_admits) ||
        !strix::core::plan_image_routes<3U>(policy, K, 9216U, aligned, kWide, routes, group_admits, image_admits))
        return false;
    // A call on an existing image takes the packed schedule only on the measured device class, heights and shapes.
    using strix::core::image_packed_schedule;
    const std::array<Bf16RowProjection, 2> wide_image{{{w, 3072U, y, false, 0U}, {w, 1024U, y, false, 0U}}};
    const std::array<Bf16RowProjection, 1> over_wide{{{w, 3200U, y, false, 0U}}};   // above the measured outputs
    const std::array<Bf16RowProjection, 1> off_pitch{{{w, 2304U, y, false, 6656U}}};  // pitch not measured
    const std::array<Bf16RowProjection, 1> partial_tile{{{w, 64U, y, false, 0U}}};    // not a whole tile
    const std::array<Bf16RowProjection, 2> one_bad{{{w, 3072U, y, false, 0U}, {w, 3200U, y, false, 0U}}};
    if (!image_packed_schedule(policy, true, K, 1024U, wide_image) ||
        !image_packed_schedule(policy, true, K, 9216U, wide_image) ||
        image_packed_schedule(policy, false, K, 4096U, wide_image) ||
        image_packed_schedule(policy, false, K, 8192U, wide_image) ||
        image_packed_schedule(policy, true, K, 1023U, wide_image) ||
        image_packed_schedule(policy, true, K, 9217U, wide_image) ||
        image_packed_schedule(policy, true, K, 16384U, wide_image) ||
        image_packed_schedule(policy, true, K, 0U, wide_image) ||
        image_packed_schedule(policy, true, K, 8192U, over_wide) ||
        image_packed_schedule(policy, true, K, 8192U, off_pitch) ||
        image_packed_schedule(policy, true, K, 8192U, partial_tile) ||
        image_packed_schedule(policy, true, K, 8192U, one_bad) ||
        image_packed_schedule(policy, true, K, 8192U, std::span<const Bf16RowProjection>{}))
        return false;
    // A width outside every listed K (K 64, 2,048 outputs, a stage multiple, whole tiles, in band) is not calibrated,
    // for the image entry and for the call that stores the image.
    const std::array<Bf16RowProjection, 1> k64_wide{{{w, 2048U, y, false, 0U}}};
    if (image_packed_schedule(policy, true, 64U, 8192U, k64_wide) ||
        image_packed_schedule(policy, true, 64U, 9216U, k64_wide) ||
        strix::core::plan_image_routes<3U>(policy, 64U, 8192U, k64_wide, kWide, routes, group_admits, image_admits) ||
        !all_fp32(1U) ||
        strix::core::plan_image_routes<3U>(policy, 64U, 9216U, k64_wide, kWide, routes, group_admits, image_admits) ||
        !all_fp32(1U))
        return false;
    // The same width-K shapes through the call that stores the image: a pitch or an output count off the table keeps
    // the FP32 rows.
    const std::array<Bf16RowProjection, 2> off_table{{{w, 3200U, y, false, 0U}, {w, 2304U, y, false, 6656U}}};
    if (strix::core::plan_image_routes<3U>(policy, K, 8192U, off_table, kWide, routes, group_admits, image_admits) ||
        !all_fp32(2U))
        return false;
    constexpr strix::core::ImageRoutePolicy unbounded{1024U, 64U, 128U, 2048U, 0U};  // no measured height: none routes
    if (image_packed_schedule(unbounded, true, K, 4096U, wide_image) ||
        strix::core::plan_image_routes<3U>(unbounded, K, 8192U, aligned, kWide, routes, group_admits, image_admits))
        return false;
    std::array<ImageRoute, 2> few{};
    return !strix::core::plan_image_routes<3U>(policy, K, 8192U, call, kWide, few, group_admits, image_admits);
}

// A routed call reorders launches by route: an output that aliases the input, another output or another weight keeps
// the incumbent sequence (review r17: an FMA group 0 and a packed group 1 writing the same output buffer).
static bool check_image_route_order() {
    using strix::core::Bf16RowProjection;
    constexpr unsigned K = 2304U, rows = 4096U;
    const auto at = [](std::uintptr_t address) { return address; };
    const auto f32 = [](std::uintptr_t address) { return reinterpret_cast<float*>(address); };
    const auto w16 = [](std::uintptr_t address) { return reinterpret_cast<const std::uint16_t*>(address); };
    constexpr std::uintptr_t gib = std::uintptr_t{1} << 30U;
    const float* input = f32(at(1U * gib));
    const unsigned widths[4] = {2048U, 128U, 128U, 2048U};
    std::array<Bf16RowProjection, 4> call{};
    for (unsigned p = 0; p < 4U; ++p)
        call[p] = {w16(at((2U + p) * gib) + (p == 0U ? 2U : 0U)), widths[p], f32(at((8U + p) * gib)), false, 0U};
    if (!strix::core::image_routes_order_free(input, K, rows, call)) return false;  // disjoint: routes may reorder
    auto alias = call;
    alias[3].output = alias[0].output;  // FMA-first group 0 and packed group 1 write the same buffer
    if (strix::core::image_routes_order_free(input, K, rows, alias)) return false;
    auto inside = call;
    inside[3].output = alias[0].output + std::size_t{rows} * widths[0] - 1U;  // last float of output 0
    if (strix::core::image_routes_order_free(input, K, rows, inside)) return false;
    auto onto_weights = call;
    onto_weights[1].output = f32(reinterpret_cast<std::uintptr_t>(call[2].weights) + 64U);
    if (strix::core::image_routes_order_free(input, K, rows, onto_weights)) return false;
    auto onto_input = call;
    onto_input[2].output = const_cast<float*>(input) + 16U;
    if (strix::core::image_routes_order_free(input, K, rows, onto_input)) return false;
    auto adjacent = call;  // output 3 starts on the first byte after output 0: still disjoint
    adjacent[3].output = call[0].output + std::size_t{rows} * widths[0];
    if (!strix::core::image_routes_order_free(input, K, rows, adjacent)) return false;
    auto wrapped = call;  // a range past the top of the address space counts as overlapping
    wrapped[1].output = f32(std::numeric_limits<std::uintptr_t>::max() - 64U);
    return !strix::core::image_routes_order_free(input, K, rows, wrapped);
}

int main() {
    if (!check_row_chunks() || !check_projection_groups() || !check_cuda_bf16_paths() ||
        !check_cuda_bf16_fast_terms() || !check_exact_operand_planes() || !check_image_routes() ||
        !check_image_route_order() || !check_dense_declared()) {
        std::cerr << "Dense row or projection chunk policy failed\n";
        return 1;
    }
    return 0;
}
