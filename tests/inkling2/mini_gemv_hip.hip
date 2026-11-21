#include "inkling2/inkling2_mini_dense_hip.hpp"
#include "strix/weight/formats/scalar_float.hpp"

#include <algorithm>
#include <array>
#include <bit>
#include <cmath>
#include <cstdint>
#include <cstdlib>
#include <iostream>
#include <span>

namespace mini = strix::inkling2;
namespace core = strix::core;
namespace scalar = strix::weight::scalar_float;

namespace {

constexpr unsigned width = 2304U;
constexpr std::array outputs{1024U, 3072U, 2304U};
constexpr unsigned head_live = 257U, head_stored = 264U;
constexpr float sentinel = -999.0F;
constexpr float cap = mini::kLogitSoftcap;
static_assert(width == mini::kHiddenWidth && cap == 30.0F);

void require_hip(hipError_t status) {
    if (status == hipSuccess) return;
    std::cerr << hipGetErrorString(status) << '\n';
    std::exit(1);
}

template <class T>
std::span<T> managed(std::size_t count) {
    void* storage = nullptr;
    require_hip(hipMallocManaged(&storage, count * sizeof(T)));
    return {static_cast<T*>(storage), count};
}

float fixture_value(std::uint32_t& state) {
    state = 1664525U * state + 1013904223U;
    return static_cast<float>(static_cast<int>(state >> 16U) - 32768) * 0x1p-15F;
}

float widen(std::uint16_t bits) { return scalar::u2f(std::uint32_t{bits} << 16U); }

float softcap(float value) { return cap * std::tanh(widen(scalar::bf16_encode(value)) / cap); }

bool same_bits(float lhs, float rhs) {
    return std::bit_cast<std::uint32_t>(lhs) == std::bit_cast<std::uint32_t>(rhs);
}

struct Results {
    std::array<std::span<float>, outputs.size()> dense;
    std::span<float> head;

    explicit Results(unsigned rows) : head(managed<float>(std::size_t{rows} * head_live + 1U)) {
        for (std::size_t p = 0; p < outputs.size(); ++p)
            dense[p] = managed<float>(std::size_t{rows} * outputs[p] + 1U);
    }
    ~Results() {
        for (auto projection : dense) require_hip(hipFree(projection.data()));
        require_hip(hipFree(head.data()));
    }
    Results(const Results&) = delete;
    Results& operator=(const Results&) = delete;
};

using Weights = std::array<std::span<std::uint16_t>, outputs.size()>;

void project(const float* input, unsigned rows, const Weights& weights, Results& result, hipStream_t stream) {
    std::array<mini::MiniHipDenseProjection, outputs.size()> group{};
    for (std::size_t p = 0; p < outputs.size(); ++p) {
        std::ranges::fill(result.dense[p], sentinel);
        group[p] = {weights[p].data(), outputs[p], result.dense[p].data()};
    }
    std::ranges::fill(result.head, sentinel);
    require_hip(mini::enqueue_mini_hip_dense_rows(input, width, rows, group, stream));
    // Use the production head contract with a small padded vocabulary slice.
    require_hip(core::enqueue_bf16_softcap_head(rows, {width, head_stored, head_live, cap},
        weights[0].first(std::size_t{head_stored} * width), result.head, stream,
        [&](std::span<const mini::MiniHipDenseProjection> projections) {
            return mini::enqueue_mini_hip_dense_rows(input, width, rows, projections, stream);
        }));
    require_hip(hipStreamSynchronize(stream));
}

bool check_reference(std::span<const float> input, unsigned rows, const Weights& weights, const Results& result) {
    constexpr float tanh_tolerance = 4.0e-6F;
    bool distinguishes_bf16_rounding = false;
    bool differs_from_serial = false;
    if (result.head.back() != sentinel) return false;
    for (std::size_t p = 0; p < outputs.size(); ++p) {
        if (result.dense[p].back() != sentinel) return false;
        for (unsigned r = 0; r < rows; ++r) {
            for (unsigned o = 0; o < outputs[p]; ++o) {
                float expected = 0.0F, magnitude = 0.0F;
                for (unsigned k = 0; k < width; ++k) {
                    const float x = input[std::size_t{r} * width + k];
                    const float w = widen(weights[p][std::size_t{o} * width + k]);
                    expected = std::fma(w, x, expected);
                    magnitude += std::abs(w * x);
                }
                const float bound = 0x1p-16F * magnitude;
                const float got = result.dense[p][std::size_t{r} * outputs[p] + o];
                differs_from_serial |= !same_bits(got, expected);
                if (!(std::abs(got - expected) <= bound)) {
                    std::cerr << "GEMV reference mismatch at projection " << p << ", row " << r
                              << ", column " << o << " in " << rows << " rows\n";
                    return false;
                }
                if (p == 0 && o < head_live) {
                    const float capped = result.head[std::size_t{r} * head_live + o];
                    // Rounding is monotone, but the reference interval may straddle a BF16 midpoint.
                    if (!(capped >= softcap(expected - bound) - tanh_tolerance &&
                          capped <= softcap(expected + bound) + tanh_tolerance &&
                          std::abs(capped - softcap(got)) <= tanh_tolerance)) {
                        std::cerr << "Head reference mismatch at row " << r << ", column " << o
                                  << " in " << rows << " rows\n";
                        return false;
                    }
                    distinguishes_bf16_rounding |= std::abs(softcap(got) - cap * std::tanh(got / cap)) > tanh_tolerance;
                }
            }
        }
    }
    if (!distinguishes_bf16_rounding || !differs_from_serial)
        std::cerr << "Fixture must distinguish BF16 head rounding and GEMV reduction order\n";
    return distinguishes_bf16_rounding && differs_from_serial;
}

bool matches_single(const Results& batch, unsigned row, const Results& single) {
    for (std::size_t p = 0; p < outputs.size(); ++p) {
        for (unsigned o = 0; o < outputs[p]; ++o) {
            if (!same_bits(batch.dense[p][std::size_t{row} * outputs[p] + o], single.dense[p][o])) return false;
        }
    }
    for (unsigned o = 0; o < head_live; ++o) {
        if (!same_bits(batch.head[std::size_t{row} * head_live + o], single.head[o])) return false;
    }
    return true;
}

}  // namespace

int main() {
    hipStream_t stream{};
    require_hip(hipStreamCreate(&stream));
    std::uint32_t state = 1U;
    Weights weights{};
    for (std::size_t p = 0; p < outputs.size(); ++p) {
        weights[p] = managed<std::uint16_t>(std::size_t{outputs[p]} * width);
        for (auto& bits : weights[p]) bits = scalar::bf16_encode(fixture_value(state));
    }

    bool pass = true;
    for (const unsigned rows : {3U, 16U}) {
        auto input = managed<float>(std::size_t{rows} * width);
        for (float& x : input) x = fixture_value(state);
        const auto batch_pass = [&] {
            Results batch(rows);
            project(input.data(), rows, weights, batch, stream);
            if (!check_reference(input, rows, weights, batch)) return false;
            if (rows == 16U) return true;

            Results single(1U);
            for (unsigned r = 0; r < rows; ++r) {
                const auto row = input.subspan(std::size_t{r} * width, width);
                project(row.data(), 1U, weights, single, stream);
                if (!check_reference(row, 1U, weights, single) || !matches_single(batch, r, single)) {
                    std::cerr << "Single-row GEMV or row independence failed at row " << r << '\n';
                    return false;
                }
            }
            return true;
        }();
        pass &= batch_pass;
        require_hip(hipFree(input.data()));
    }
    for (auto weight : weights) require_hip(hipFree(weight.data()));
    require_hip(hipStreamDestroy(stream));
    return pass ? 0 : 1;
}
