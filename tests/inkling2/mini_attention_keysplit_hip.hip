#include "attn_core/prepared_gqa_hip.hpp"
#include "strix/ds41_scalar_codecs.hpp"

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <cstdlib>
#include <cstring>
#include <iostream>
#include <limits>
#include <memory>
#include <vector>

namespace {
namespace dense = strix::attn_core::dense;
namespace mask = strix::attn_core::mask;
constexpr unsigned dim = 128U, group = 3U, kv_heads = 8U, heads = group * kv_heads;
constexpr unsigned max_rows = 16U, max_slots = 4096U + max_rows - 1U, bias_extent = 1024U;
constexpr float max_relative_error = 0.01F;
constexpr float poison = std::numeric_limits<float>::quiet_NaN();

struct AttentionShape {
    static constexpr unsigned kQueryHeads = heads;
    static constexpr unsigned kHeadWidth = dim;
};

struct Fixture {
    alignas(16) float queries[max_rows * heads * dim];
    alignas(16) float output[max_rows * heads * dim];
    alignas(16) std::uint16_t keys[max_slots * kv_heads * dim];
    alignas(16) std::uint16_t values[max_slots * kv_heads * dim];
    alignas(16) std::uint16_t zeros[dim];
    float bias[max_rows * heads * bias_extent];
    float sinks[max_rows * heads];
    alignas(16) float workspace[strix::attn_core::prepared_gqa_hip_key_split_workspace_floats<AttentionShape>(max_rows)];
    unsigned failure;
};

template <bool Bias, bool Sink>
struct Policy : dense::PreparedBf16Policy<dim, group, Bias, Sink> {
    static constexpr float kScaleLog2 = 0.12751743F;  // log2(e) / sqrt(128)
};

void require_hip(hipError_t result) {
    if (result == hipSuccess) return;
    std::cerr << hipGetErrorString(result) << '\n';
    std::exit(1);
}

float sample(std::uint32_t index) {
    index ^= index >> 16U;
    index *= 0x7feb352dU;
    index ^= index >> 15U;
    index *= 0x846ca68bU;
    index ^= index >> 16U;
    return static_cast<float>(static_cast<int>(index & 255U) - 128) / 128.0F;
}

std::uint16_t bf16(float x) {
    return static_cast<std::uint16_t>(strix::ds41::codec::f2u(x) >> 16U);
}

dense::PreparedBf16Args prepare(Fixture& data, unsigned rows, unsigned keys, unsigned window) {
    const bool sliding = window != 0U;
    const unsigned first = keys - 1U + (sliding ? 37U : 0U);
    const unsigned end = first + rows;
    const unsigned begin = mask::window_lo(first, window);
    const unsigned slots = sliding ? end - begin : end;
    std::fill_n(data.zeros, dim, std::uint16_t{0});
    for (unsigned p = begin; p < end; ++p) {
        // Vary the score distribution across the chunks to exercise max rescaling.
        const float amplitude = 0.5F + 0.25F * static_cast<float>((p / 128U) % 7U);
        const unsigned slot = sliding ? p % slots : p;
        for (unsigned i = 0; i < kv_heads * dim; ++i) {
            const unsigned at = slot * kv_heads * dim + i;
            const unsigned seed = p * kv_heads * dim + i;
            data.keys[at] = bf16(amplitude * sample(seed ^ 0x12345U));
            data.values[at] = bf16(0.25F + 0.75F * sample(seed ^ 0x87230U));
        }
    }
    for (unsigned i = 0; i < rows * heads * dim; ++i) data.queries[i] = sample(i ^ 0x956U);
    for (unsigned i = 0; i < rows * heads * bias_extent; ++i) data.bias[i] = sample(i ^ 0x16bdU);
    for (unsigned i = 0; i < rows * heads; ++i) data.sinks[i] = i % heads == 0U ? 8.0F : sample(i ^ 0x781U);
    return {data.queries, data.output, data.keys, data.values, data.zeros, &data.failure,
            rows, kv_heads, first, window, slots, sliding, {data.bias, heads, bias_extent}, {data.sinks, heads}};
}

template <bool Bias, bool Sink>
bool compare(Fixture& data, const dense::PreparedBf16Args& args, unsigned splits, float& worst) {
    const std::size_t count = std::size_t{args.rows} * heads * dim;
    const auto run = [&](unsigned chunks) {
        data.failure = 0U;
        std::fill_n(data.output, count, poison);
        const auto size = dense::workspace_floats(args.rows, heads, dim, chunks);
        std::fill_n(data.workspace, size, poison);
        require_hip(dense::launch_prepared_bf16<Policy<Bias, Sink>>(args, nullptr, data.workspace, chunks));
        require_hip(hipDeviceSynchronize());
        return data.failure == 0U &&
               std::all_of(data.output, data.output + count, [](float x) { return std::isfinite(x); });
    };
    if (!run(1U)) return false;
    const std::vector<float> reference(data.output, data.output + count);
    if (!run(splits)) return false;
    for (std::size_t base = 0; base < count; base += dim) {
        float magnitude = 0.0F, error = 0.0F;
        for (unsigned d = 0; d < dim; ++d) {
            magnitude = std::max(magnitude, std::abs(reference[base + d]));
            error = std::max(error, std::abs(data.output[base + d] - reference[base + d]));
        }
        if (error > max_relative_error * magnitude || (splits == 1U && error != 0.0F)) {
            std::cerr << "output error=" << error << " reference magnitude=" << magnitude << '\n';
            return false;
        }
        if (magnitude != 0.0F) worst = std::max(worst, error / magnitude);
    }
    const std::vector<float> first(data.output, data.output + count);
    return run(splits) && std::memcmp(first.data(), data.output, count * sizeof(float)) == 0;
}
}  // namespace

int main() {
    if (dense::small_row_key_splits(127U, 1U, 0U) != 1U ||
        dense::small_row_key_splits(128U, 1U, 0U) != 2U ||
        dense::small_row_key_splits(1023U, 1U, 0U) != 8U ||
        dense::small_row_key_splits(1023U, 4U, 1024U) != 9U ||
        dense::small_row_key_splits(4095U, 1U, 1024U) != 8U ||
        dense::small_row_key_splits(4095U, 1U, 0U) != 16U ||
        dense::small_row_key_splits(4095U, 16U, 0U) != 16U ||
        dense::small_row_key_splits(4095U, 17U, 0U) != 1U) {
        std::cerr << "unexpected split plan\n";
        return 1;
    }
    Fixture* data = nullptr;
    require_hip(hipMallocManaged(&data, sizeof(Fixture)));
    std::construct_at(data);
    float worst = 0.0F;
    bool ok = true;
    for (const unsigned rows : {1U, 4U, 16U}) {
        for (const unsigned keys : {1U, 100U, 1024U, 1025U, 4096U}) {
            for (const unsigned window : {0U, keys}) {
                const auto args = prepare(*data, rows, keys, window);
                const auto splits = dense::small_row_key_splits(args.first_position, rows, args.window);
                const auto check = [&]<bool Bias, bool Sink> {
                    if (compare<Bias, Sink>(*data, args, splits, worst)) return;
                    std::cerr << "rows=" << rows << " keys=" << keys << " window=" << args.window
                              << " splits=" << splits << " bias=" << Bias << " sink=" << Sink
                              << " failure=" << data->failure << '\n';
                    ok = false;
                };
                check.template operator()<false, false>();
                check.template operator()<false, true>();
                check.template operator()<true, false>();
                check.template operator()<true, true>();
            }
        }
    }
    require_hip(hipFree(data));
    std::cout << "max relative error across query heads and rows: " << worst << '\n';
    return ok ? 0 : 1;
}
