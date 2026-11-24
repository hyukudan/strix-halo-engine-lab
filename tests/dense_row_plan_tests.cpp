#include "core/dense_row_plan.hpp"
#include "cuda/core/dense/row_plan.hpp"

#include <array>
#include <bit>
#include <cmath>
#include <cstdint>
#include <iostream>
#include <limits>
#include <random>
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

int main() {
    if (!check_row_chunks() || !check_projection_groups() || !check_cuda_bf16_paths() ||
        !check_cuda_bf16_fast_terms() || !check_exact_operand_planes() ||
        !check_dense_declared()) {
        std::cerr << "Dense row or projection chunk policy failed\n";
        return 1;
    }
    return 0;
}
