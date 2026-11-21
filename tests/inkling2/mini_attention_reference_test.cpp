#include "attn_core/causal_gqa_reference.hpp"

#include <algorithm>
#include <array>
#include <cmath>
#include <cstddef>
#include <limits>
#include <span>
#include <stdexcept>

int main() {
    using strix::attn_core::ReferenceAttentionArithmetic;
    using strix::attn_core::ReferenceAttentionCache;
    using strix::attn_core::ReferenceAttentionInput;
    using strix::attn_core::ReferenceAttentionShape;
    using strix::attn_core::causal_gqa_reference;

    constexpr unsigned rows = 5;
    constexpr unsigned query_heads = 4;
    constexpr unsigned kv_heads = 2;
    constexpr unsigned relative_extent = 2;
    constexpr unsigned window = 2;
    constexpr float ln2 = 0.6931471805599453F;
    std::array<float, rows * query_heads> queries{};
    std::array<float, rows * kv_heads> keys{};
    std::array<float, rows * query_heads * relative_extent> relative{};
    std::array<float, rows * query_heads> sinks{};
    for (unsigned row = 0; row < rows; ++row) {
        queries[row * query_heads] = ln2;
        queries[row * query_heads + 2] = ln2;
        keys[row * kv_heads] = 1.0F;
        keys[row * kv_heads + 1] = 2.0F;
        constexpr std::array<float, query_heads * relative_extent> bias{
            ln2, -ln2, 0.0F, ln2, -ln2, ln2, 0.0F, 0.0F,
        };
        std::ranges::copy(bias, relative.begin() + row * query_heads * relative_extent);
    }
    sinks[0] = ln2;
    sinks[17] = ln2;
    sinks[19] = ln2;
    constexpr std::array<float, rows * kv_heads> values{
         2.0F, 13.0F,  4.0F, 17.0F,  8.0F, 19.0F,  16.0F, 23.0F,  32.0F, 29.0F,
    };

    using Output = std::array<float, rows * query_heads>;
    const auto run = [&](const ReferenceAttentionShape& shape, ReferenceAttentionCache cache, Output& out,
                         unsigned first, unsigned count) {
        const ReferenceAttentionInput input{
            first,
            std::span(queries).subspan(first * query_heads, count * query_heads),
            std::span(keys).subspan(first * kv_heads, count * kv_heads),
            std::span(values).subspan(first * kv_heads, count * kv_heads),
            relative_extent,
            std::span(relative).subspan(first * query_heads * relative_extent, count * query_heads * relative_extent),
            std::span(sinks).subspan(first * query_heads, count * query_heads),
        };
        causal_gqa_reference(shape, input, cache, std::span(out).subspan(first * query_heads, count * query_heads));
    };

    const ReferenceAttentionShape full{query_heads, kv_heads, 1};
    Output full_out{};
    std::array<float, rows * kv_heads> full_keys{}, full_values{};
    run(full, {full_keys, full_values}, full_out, 0, rows);
    if (full_keys != keys || full_values != values) return 1;

    const ReferenceAttentionShape sliding{query_heads, kv_heads, 1, window};
    Output sliding_out{};
    std::array<float, window * kv_heads> ring_keys{}, ring_values{};
    run(sliding, {ring_keys, ring_values}, sliding_out, 0, rows);
    if (ring_values != std::array{32.0F, 29.0F, 16.0F, 23.0F}) return 2;

    const auto near = [](float result, float expected) { return std::abs(result - expected) < 1.0e-5F; };
    constexpr std::array first_rows{
        8.0F / 6.0F, 1.0F, 26.0F / 3.0F, 13.0F / 2.0F,
        3.0F, 2.0F, 138.0F / 11.0F, 10.0F,
    };
    for (std::size_t i = 0; i < first_rows.size(); ++i) {
        if (!near(first_rows[i], full_out[i]) || sliding_out[i] != full_out[i]) return 3;
    }

    // Beyond the relative extent the full cache still participates with zero bias.
    // The ring keeps only the preceding and current key. Both include each row's sink.
    constexpr std::array full_last{172.0F / 12.0F, 78.0F / 8.0F, 438.0F / 23.0F, 101.0F / 7.0F};
    constexpr std::array sliding_last{24.0F, 64.0F / 5.0F, 22.0F, 13.0F};
    for (unsigned head = 0; head < query_heads; ++head) {
        if (!near(full_last[head], full_out[16 + head]) ||
            !near(sliding_last[head], sliding_out[16 + head])) return 4;
    }

    // Exercise ring and full-cache prefill/decode parity across chunk boundaries.
    Output chunked{}, full_chunked{};
    std::array<float, window * kv_heads> chunked_keys{}, chunked_values{};
    std::array<float, rows * kv_heads> full_chunked_keys{}, full_chunked_values{};
    run(sliding, {chunked_keys, chunked_values}, chunked, 0, 3);
    const auto checkpoint_keys = chunked_keys;
    const auto checkpoint_values = chunked_values;
    run(sliding, {chunked_keys, chunked_values}, chunked, 3, 1);
    run(sliding, {chunked_keys, chunked_values}, chunked, 4, 1);
    run(full, {full_chunked_keys, full_chunked_values}, full_chunked, 0, 2);
    run(full, {full_chunked_keys, full_chunked_values}, full_chunked, 2, 3);
    if (chunked != sliding_out || chunked_keys != ring_keys || chunked_values != ring_values ||
        full_chunked != full_out || full_chunked_keys != full_keys || full_chunked_values != full_values) return 5;

    chunked_keys = checkpoint_keys;
    chunked_values = checkpoint_values;
    run(sliding, {chunked_keys, chunked_values}, chunked, 3, 2);
    if (chunked != sliding_out || chunked_keys != ring_keys || chunked_values != ring_values) return 6;

    try {
        run(full, {chunked_keys, chunked_values}, chunked, 0, rows);
        return 7;
    } catch (const std::invalid_argument&) {
        if (chunked != sliding_out || chunked_keys != ring_keys || chunked_values != ring_values) return 8;
    }

    // One head with dimension four tests 1/sqrt(head_dim) and sink stability.
    constexpr std::array q{2.0F * ln2, 0.0F, 0.0F, 0.0F};
    constexpr std::array k{1.0F, 0.0F, 0.0F, 0.0F};
    constexpr std::array v{3.0F, 6.0F, 9.0F, 12.0F};
    constexpr std::array sink{0.0F};
    constexpr std::array extreme_sink{1000.0F};
    std::array<float, 4> one_out{}, one_key{}, one_value{};
    causal_gqa_reference({1, 1, 4}, {0, q, k, v, 0, {}, sink}, {one_key, one_value}, one_out);
    if (!near(one_out[0], 2.0F) || !near(one_out[3], 8.0F)) return 9;
    causal_gqa_reference({1, 1, 4}, {0, q, k, v, 0, {}, {}}, {one_key, one_value}, one_out);
    if (one_out != v) return 10;
    causal_gqa_reference({1, 1, 4}, {0, q, k, v, 0, {}, extreme_sink}, {one_key, one_value}, one_out);
    if (one_out != std::array<float, 4>{}) return 11;

    // All operands are widened BF16. The final row has scores [0.1640625,
    // 0.8046875, -0.578125, -0.7421875] and rounded real-key probabilities
    // [0.265625, 0.50390625, 0.1259765625]. Omitting any of the QK, scale,
    // bias, probability, or PV rounds changes at least one expected channel.
    const std::array<float, 9> eager_q{
        0.0F, 0.0F, 0.0F, 0.0F, 0.0F, 0.0F, -0.796875F, -1.0546875F, -0.455078125F,
    };
    const std::array<float, 9> eager_k{
        0.87890625F, -0.5625F, 1.609375F, -0.4296875F, -1.3359375F, 1.2421875F,
        1.7421875F, -0.77734375F, 1.46875F,
    };
    const std::array<float, 9> eager_v{
        1.953125F, 4.46875F, -0.76953125F, 0.58203125F, 3.984375F, -0.61328125F,
        2.0F, -1.359375F, -4.6875F,
    };
    const std::array<float, 9> eager_relative{
        0.0F, 0.0F, 0.0F, 0.0F, 0.0F, 0.0F, 0.1318359375F, 0.115234375F, 0.6484375F,
    };
    const std::array<float, 3> eager_sink{0.0F, 0.0F, -0.7421875F};
    const ReferenceAttentionShape eager_shape{1, 1, 3, 0, ReferenceAttentionArithmetic::bf16_eager};
    const ReferenceAttentionInput eager_input{0, eager_q, eager_k, eager_v, 3, eager_relative, eager_sink};
    std::array<float, 9> eager_out{}, eager_keys{}, eager_values{}, fp32_out{};
    causal_gqa_reference(eager_shape, eager_input, {eager_keys, eager_values}, eager_out);
    constexpr std::array eager_expected{1.0625F, 3.03125F, -1.1015625F};
    if (!std::ranges::equal(std::span(eager_out).last<3>(), eager_expected)) return 12;
    causal_gqa_reference({1, 1, 3}, eager_input, {eager_keys, eager_values}, fp32_out);
    if (std::ranges::equal(std::span(fp32_out).last<3>(), eager_expected)) return 13;

    auto bad_keys = eager_k;
    bad_keys[8] = std::numeric_limits<float>::quiet_NaN();
    auto invalid_input = eager_input;
    invalid_input.keys = bad_keys;
    std::array<float, 9> rejected_out{}, rejected_keys{}, rejected_values{};
    try {
        causal_gqa_reference(eager_shape, invalid_input, {rejected_keys, rejected_values}, rejected_out);
        return 14;
    } catch (const std::invalid_argument&) {
        constexpr std::array<float, 9> zero{};
        return rejected_out == zero && rejected_keys == zero && rejected_values == zero ? 0 : 15;
    }
}
