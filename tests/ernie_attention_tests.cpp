#include "strix/ernie/attention.hpp"

#include "attn_core/compressed_history_reference.hpp"
#include "attn_core/selected_latent_reference.hpp"
#include "attn_core/sparse_index_reference.hpp"
#include "attn_core/value_head_mix_reference.hpp"
#include "core/hadamard_reference.hpp"

#include <bit>

#include <array>
#include <iostream>

namespace {

using namespace strix;
using namespace strix::attn_core;

void check(bool ok, const char* label) {
    if (!ok) throw std::runtime_error(label);
}

void near(float got, float expected, const char* label) {
    if (!(std::fabs(got - expected) < 1e-6F)) {
        std::cerr << "got " << got << ", expected " << expected << '\n';
        throw std::runtime_error(label);
    }
}

void pool_and_merge() {
    constexpr float first[] = {2.0F, 5.0F}, last[] = {6.0F, 1.0F}, zero[] = {0.0F, 0.0F};
    const float log3 = std::log(3.0F), ape[] = {log3, 0.0F, 0.0F, log3};
    std::array<float, 2> pooled;
    projected_ape_pool_reference<2>(first, zero, last, zero, ape, pooled);
    near(pooled[0], 3.0F, "APE softmax with 3:1 preference for first value");
    near(pooled[1], 2.0F, "APE softmax with 1:3 preference for first value");

    const float query[] = {std::sqrt(2.0F), 0.0F}, key[] = {std::log(2.0F), 0.0F}, value[] = {3.0F, -2.0F};
    std::array<float, 2> output;
    hybrid_mqa_reference({2, zero, query, std::span(zero).first(1)}, 1, 1,
        [&](std::size_t) { return std::span(zero); },
        [&](std::size_t) { return std::span(key); },
        [&](std::size_t) { return std::span(value); }, output);
    near(output[0], 1.5F, "both domains and zero-value sink share denominator");
    near(output[1], -1.0F, "compressed value uses the output coordinate frame");
}

void hca_carry() {
    ernie::ReferenceShape s{};
    s.hidden = s.head_dim = s.o_rank = 2;
    s.heads = s.q_rank = s.o_groups = s.vha_rank = 1;
    ernie::AttentionHistory history;
    float kv_gain = 1.0F;
    const auto read = [&](std::string_view name, std::span<float> values, std::size_t) {
        std::ranges::fill(values, 0.0F);
        if (name.ends_with("norm.weight")) std::ranges::fill(values, 1.0F);
        if (name.ends_with("kv_layernorm.weight")) std::ranges::fill(values, kv_gain);
        if (name.ends_with("linear_o_group_proj")) values[0] = values[3] = 1.0F;
    };
    bool fail_output = false;
    const auto project = [&](const std::string& name, std::span<const float> input, std::span<float> output, unsigned) {
        if (name.ends_with(".o_proj.weight")) {
            if (fail_output) throw std::runtime_error("mock output projection failure");
            std::ranges::copy(input, output.begin());
        } else if (name.ends_with("linear_o_group_proj")) {
            std::ranges::copy(input, output.begin());
        } else if (name.ends_with("linear_kv_proj.weight")) {
            std::ranges::fill(output, input.front());
        } else if (name.ends_with("linear_wkv.weight")) {
            std::ranges::fill(output, 1.0F);
        } else {
            std::ranges::fill(output, 0.0F);
        }
    };
    constexpr float first[] = {1.0F, 0.0F}, rest[] = {0.0F, 0.0F};
    const float raw = 1.0F / std::sqrt(1.0F + 1e-6F);
    auto output = ernie::attention_reference(0, s, read, first, history, project);
    near(output[0], raw / 4.0F, "first HCA row attends itself with a sink and a half output gate");
    check(output == ernie::attention_reference_first_hca("model.layers.0.self_attn.", s, read, first, project),
          "position-zero diagnostic matches history before any compressor group is visible");
    for (unsigned p = 1; p < 127; ++p)
        output = ernie::attention_reference(0, s, read, rest, history, project);
    near(output[0], raw / 256.0F, "only the causal window is visible before group completion");
    fail_output = true;
    try {
        static_cast<void>(ernie::attention_reference(0, s, read, rest, history, project));
        check(false, "failed output projection should throw");
    } catch (const std::runtime_error& error) {
        check(std::string_view(error.what()) == "mock output projection failure", "unexpected failure at HCA boundary");
    }
    check(history.position() == 127, "HCA failure preserves committed position");
    fail_output = false;
    const auto saved = history;
    output = ernie::attention_reference(0, s, read, rest, history, project);
    near(output[0], (raw + 1.0F) / 260.0F, "completed group and first raw row both contribute at 127");
    auto replay = saved;
    check(output == ernie::attention_reference(0, s, read, rest, replay, project), "copied carry replays exactly");
    for (unsigned p = 128; p <= 129; ++p) {
        output = ernie::attention_reference(0, s, read, rest, history, project);
        near(output[0], 1.0F / 260.0F, "old raw row evicts while its group remains visible");
    }

    s.contracts.checkpoint_attention = true;
    kv_gain = 2.0F;
    ernie::AttentionHistory weighted;
    output = ernie::attention_reference(0, s, read, first, weighted, project);
    near(output[0], 0.5F, "HCA preserves the learned KV gain through attention and the output gate");
    for (unsigned p = 1; p < 127; ++p)
        output = ernie::attention_reference(0, s, read, rest, weighted, project);
    check(output[0] == 0.0078125F, "checkpoint HCA sees the raw row until the first group closes");
    output = ernie::attention_reference(0, s, read, rest, weighted, project);
    check(output[0] == 0.01153564453125F, "checkpoint HCA includes the completed group alongside the raw row");
    output = ernie::attention_reference(0, s, read, rest, weighted, project);
    check(output[0] == 0.00384521484375F, "checkpoint HCA reuses the committed compressed KV value");

}

void repeated_hca_groups() {
    ernie::ReferenceShape shape{};
    shape.hidden = shape.head_dim = shape.o_rank = 2;
    shape.heads = shape.q_rank = shape.o_groups = shape.vha_rank = 1;
    shape.contracts.checkpoint_attention = true;
    const auto read = [](std::string_view name, std::span<float> values, std::size_t) {
        std::ranges::fill(values, name.ends_with("norm.weight") ? 1.0F : 0.0F);
    };
    const auto project = [](const std::string& name, std::span<const float> input,
                            std::span<float> output, unsigned) {
        std::ranges::fill(output, 0.0F);
        if (name.ends_with(".o_proj.weight") || name.ends_with("linear_o_group_proj"))
            std::ranges::copy(input, output.begin());
        else if (name.ends_with("linear_wkv.weight"))
            output.front() = input.front();
    };
    constexpr unsigned total = 513, window = 128, ratio = 128;
    ernie::AttentionHistory history(total);
    for (unsigned pos = 0; pos < total; ++pos) {
        const std::array input{pos / ratio % 2U == 0 ? 1.0F : -1.0F, 0.0F};
        const auto output = ernie::attention_reference(40, shape, read, input, history, project);
        const unsigned closed = (pos + 1U) / ratio;
        const auto raw = std::min(pos + 1U, window);
        const float sum = closed % 2U == 0 ? 0.0F : 1.4140625F;
        const auto ungated = weight::scalar_float::bf16_round(sum / static_cast<float>(1U + raw + closed));
        check(output.front() == ungated / 2.0F, "each R128 group joins when complete and survives window eviction");
    }
}

void packed_hca_binding() {
    ernie::ReferenceShape shape{};
    shape.hidden = 1;
    shape.head_dim = 128;
    shape.o_rank = 512;
    shape.heads = shape.q_rank = shape.o_groups = shape.vha_rank = 1;
    shape.contracts.checkpoint_attention = true;
    const auto read = [](std::string_view name, std::span<float> values, std::size_t) {
        std::ranges::fill(values, name.ends_with("norm.weight") ? 1.0F : 0.0F);
        if (name.ends_with("kv_layernorm.weight")) values.front() = 1.0625F;
    };
    float grouped_input = 0.0F;
    const auto project = [&](const std::string& name, std::span<const float> input, std::span<float> output, unsigned) {
        if (name.ends_with("linear_o_group_proj")) {
            grouped_input = input.front();
            std::ranges::fill(output, 1.0625F);
        } else if (name.ends_with(".o_proj.weight")) output.front() = input.front();
        else std::ranges::fill(output, name.ends_with("linear_kv_proj.weight") ? 1.0F : 0.0F);
    };
    constexpr std::array input{0.0F};
    ernie::AttentionHistory base(1), candidate(1);
    const auto base_out = ernie::attention_reference(40, shape, read, input, base, project);
    check(grouped_input == 0.53125F && base_out.front() == 0.53125F, "BF16 attention epilogue keeps both midpoint stores");
    shape.contracts.hca_fp8_epilogue = true;
    const auto candidate_out = ernie::attention_reference(40, shape, read, input, candidate, project);
    check(grouped_input == 0.5F && candidate_out.front() == 0.5F, "optional epilogue quantizes both projection inputs");
}

void packed_hca_gate_rows() {
    constexpr unsigned block = 128;
    for (const unsigned width : {8192U, 16384U}) {
        for (const unsigned rows : {1U, 5U, 1000U}) {
            std::vector<float> values(width), gates(width), actual(width), reference(width);
            for (unsigned row = 0; row < rows; ++row) {
                for (unsigned i = 0; i < width; ++i) {
                    const auto value = static_cast<int>((row * 17U + i * 23U) % 61U) - 30;
                    const auto gate = static_cast<int>((row * 13U + i * 19U) % 69U) - 34;
                    values[i] = static_cast<float>(value) / 8.0F;
                    gates[i] = static_cast<float>(gate) / 4.0F;
                    const auto sigmoid = 1.0F / (1.0F + std::exp(-gates[i]));
                    using weight::scalar_float::bf16_round;
                    reference[i] = bf16_round(values[i] * bf16_round(sigmoid));
                }
                core::sigmoid_product_reference(values, gates, actual, core::SigmoidGatePrecision::bf16_sigmoid);
                check(actual == reference, "packed gate retains the unfused BF16 product");
                for (unsigned first = 0; first < width; first += block) {
                    float peak = 1e-10F;
                    for (unsigned i = 0; i < block; ++i) peak = std::max(peak, std::fabs(reference[first + i]));
                    const auto exponent = static_cast<int>(std::ceil(std::log2(static_cast<double>(peak) / 448.0)));
                    const float scale = std::ldexp(1.0F, exponent);
                    for (unsigned i = 0; i < block; ++i) {
                        auto& x = reference[first + i];
                        x = weight::scalar_float::e4m3(weight::scalar_float::e4m3_encode(x / scale)) * scale;
                    }
                }
                core::round_block_fp8_activation_reference(actual, block);
                check(actual == reference, "packed gate agrees with separately quantized BF16 product");
            }
        }
    }
}

void compressed_index() {
    std::array<float, 4> x{2.0F, 0.0F, 0.0F, 0.0F};
    core::hadamard_reference(x);
    check(x == std::array{1.0F, 1.0F, 1.0F, 1.0F}, "orthonormal Hadamard maps impulse to constant row");
    core::hadamard_reference(x);
    check(x == std::array{2.0F, 0.0F, 0.0F, 0.0F}, "Hadamard inverse");

    std::array<float, 32> key0{}, key1{}, key2{};
    key0[0] = 1.0F;
    key0[1] = -1.0F;
    key1[0] = 1.0F;
    key1[1] = 2.0F;
    key2[0] = 2.0F;
    const auto a = pack_index_fp4_reference(key0), b = pack_index_fp4_reference(key1),
               c = pack_index_fp4_reference(key2);
    // Range 1 gives scale 1/4. Codes 6 and 14 store +4 and -4.
    check(a.size() == 17 && a[0] == 0xe6 && a[16] == 125, "index key low-first FP4 and E8M0 scale");
    std::array<float, 32> decoded;
    unpack_index_fp4_reference(a, decoded);
    check(decoded == key0, "packed FP4 key can be used by the scorer");
    const std::array<std::span<const std::uint8_t>, 3> rows{a, b, c};
    std::array<float, 64> queries{};
    queries[0] = 1.0F;
    queries[33] = 1.0F;
    constexpr float weights[] = {1.0F, 10.0F}, gates[] = {1.0F, -1.0F}, fallback[] = {-2.0F, -1.0F};
    const auto at = [&](std::size_t i) { return rows.at(i); };
    constexpr float bias[] = {4.0F, -5.0F};
    const auto selected = index_head_weights_reference(weights, fallback, bias, 1);
    check(selected == std::vector<float>{-2.0F, 0.0F}, "head selection uses biased score but scales by the raw gate");
    check(weighted_index_top_k_reference(queries, selected, 32, 3, 1, at) == std::vector<std::uint32_t>{0},
          "negative indexer weights rank the history by signed weighted ReLU scores");

    check(gated_index_top_k_reference(queries, weights, gates, 32, 3, 2, at) == std::vector<std::uint32_t>{0, 2},
          "index gate suppresses second head and lower position wins tie");
    check(gated_index_top_k_reference(queries, weights, fallback, 32, 3, 1, at) == std::vector<std::uint32_t>{1},
          "index retains strongest head when all gate logits are negative");
    check(gated_index_top_k_reference(queries, weights, gates, 32, 2, 1, at) == std::vector<std::uint32_t>{0},
          "future high-scoring key lies outside the causal prefix");
}

void index_window_rows() {
    struct Case {
        unsigned position, history, window;
    };
    constexpr std::array cases{Case{0, 0, 1}, Case{5, 0, 6}, Case{127, 0, 128}, Case{128, 1, 128},
        Case{200, 73, 128}, Case{2175, 2048, 128}, Case{2176, 2049, 128}, Case{5000, 4873, 128}};
    constexpr unsigned budget = 2048, window = 128;
    for (const auto& c : cases) {
        const SparseWindowReference selection(c.position, window, budget);
        check(selection.dense() == (c.history <= budget), "entire history selected only within budget");
        check(selection.history == c.history, "indexer excludes the entire final window");
        const auto retained = c.history < budget ? c.history : budget;
        std::vector<std::uint32_t> rows(retained);
        for (unsigned i = 0; i < retained; ++i) rows[i] = c.history - retained + i;
        const auto history = rows;
        selection.append_window(rows);
        check(rows.size() == retained + c.window &&
              std::ranges::equal(history, std::span(rows).first(retained)),
              "window append preserves the original index order");
        for (unsigned i = 0; i < c.window; ++i)
            check(rows[retained + i] == c.history + i, "window contains consecutive keys, including self");
    }
}

void index_leading_rotary() {
    constexpr unsigned heads = 64, width = 128, rotary_width = 64, half = rotary_width / 2;
    constexpr double base = 8'000'000.;
    for (const unsigned tokens : {1U, 7U, 4096U}) {
        for (unsigned token = 0; token < tokens; ++token) {
            const auto position = (token * 541U + 9U) % 65536U;
            const RotaryReferenceRow rope(rotary_width, position, base);
            std::vector<float> query(std::size_t{heads} * width), key(width);
            for (std::size_t i = 0; i < query.size(); ++i)
                query[i] = static_cast<float>(static_cast<int>((i * 17U + token * 3U) % 61U) - 30) / 32.0F;
            std::ranges::copy(std::span(query).last(width), key.begin());
            auto reference = query;
            std::array<float, half> c{}, sn{};
            for (unsigned d = 0; d < half; ++d) {
                const double theta = static_cast<double>(position) / std::pow(base, static_cast<double>(d) / half);
                c[d] = static_cast<float>(std::cos(theta));
                sn[d] = static_cast<float>(std::sin(theta));
            }
            for (unsigned head = 0; head < heads; ++head) {
                const auto at = std::size_t{head} * width;
                for (unsigned d = 0; d < half; ++d) {
                    const auto a = query[at + d], b = query[at + half + d];
                    reference[at + d] = a * c[d] - b * sn[d];
                    reference[at + half + d] = a * sn[d] + b * c[d];
                }
            }
            const auto original_query = query.data(), original_key = key.data();
            rope.apply_leading_split_half(query, width);
            rope.apply_leading_split_half(key, width);
            check(query.data() == original_query && key.data() == original_key, "partial rotary stays in place");
            core::round_bf16_reference(reference);
            core::round_bf16_reference(query);
            core::round_bf16_reference(key);
            check(query == reference, "indexer rotates leading split halves and leaves the tail intact");
            check(std::ranges::equal(key, std::span(reference).last(width)), "keys use the same partial rotary layout");
        }
    }
}

void index_hadamard_rows() {
    constexpr unsigned width = 128;
    for (const unsigned rows : {1U, 2U, 5U, 6U, 7U, 64U, 129U, 1026U}) {
        double squared_error = 0.0, squared_reference = 0.0;
        for (unsigned row = 0; row < rows; ++row) {
            std::array<float, width> input{};
            for (unsigned d = 0; d < width; ++d)
                input[d] = static_cast<float>(static_cast<int>((d * 71U + row * 53U) % 127U) - 63) / 32.0F;
            auto output = input;
            core::hadamard_reference(output);
            core::round_bf16_reference(output);
            for (unsigned d = 0; d < width; ++d) {
                double exact = 0.0;
                for (unsigned i = 0; i < width; ++i)
                    exact += (std::popcount(i & d) % 2 == 0 ? 1.0 : -1.0) * input[i];
                exact /= std::sqrt(static_cast<double>(width));
                const double error = static_cast<double>(output[d]) - exact;
                squared_error += error * error;
                squared_reference += exact * exact;
            }
        }
        check(squared_error < 1e-4 * squared_reference, "BF16 Hadamard rows match dense orthonormal transform");
    }
}

void folded_value_mix() {
    constexpr std::array input_factor{1.0F, 2.0F};
    constexpr std::array output_factor{-0.5F, 0.25F};
    const auto folded = fold_value_head_mix_reference(2, 1, output_factor, input_factor, true);
    check(folded == std::vector<float>{0.5F, -1.0F, 0.25F, 1.5F}, "folded VHA is I plus V times transposed U");
    std::array values{2.0F, 4.0F, 6.0F, 8.0F};
    apply_folded_value_head_mix_reference(values, 2, folded);
    check(values == std::array{-5.0F, -6.0F, 9.5F, 13.0F}, "folded VHA mixes across heads only");
}

void head_mix_partition_reference() {
    constexpr unsigned heads = 64, rank = 2, width = 4;
    std::array<float, heads * rank> u{}, v{};
    for (unsigned h = 0; h < heads; ++h) {
        u[rank * h] = h % 2U == 0 ? 0.125F : -0.25F;
        u[rank * h + 1U] = h % 4U == 0 ? -0.25F : 0.125F;
        v[rank * h] = static_cast<float>(h % 8U) / 8.0F;
        v[rank * h + 1U] = h % 3U == 0 ? 0.25F : -0.125F;
    }
    const auto matrix = fold_value_head_mix_reference(heads, rank, v, u, true);
    for (const unsigned world : {2U, 4U}) {
        const unsigned local_heads = heads / world;
        for (const unsigned tokens : {1U, 37U}) {
            for (unsigned row = 0; row < tokens; ++row) {
                std::array<float, heads * width> input{};
                for (unsigned i = 0; i < input.size(); ++i)
                    input[i] = static_cast<float>(static_cast<int>((row * 31U + i * 13U) % 21U) - 10) / 8.0F;
                std::array<float, rank * width> global{};
                for (unsigned part = 0; part < world; ++part) {
                    std::array<float, rank * width> local{};
                    for (unsigned r = 0; r < rank; ++r)
                        for (unsigned h = part * local_heads; h < (part + 1U) * local_heads; ++h)
                            for (unsigned d = 0; d < width; ++d)
                                local[r * width + d] += input[h * width + d] * u[h * rank + r];
                    for (unsigned i = 0; i < global.size(); ++i) global[i] += local[i];
                }
                auto partitioned = input;
                for (unsigned h = 0; h < heads; ++h)
                    for (unsigned d = 0; d < width; ++d)
                        for (unsigned r = 0; r < rank; ++r)
                            partitioned[h * width + d] += v[h * rank + r] * global[r * width + d];
                auto full = input, folded = input;
                residual_value_head_mix(full, heads, rank, v, u);
                apply_folded_value_head_mix_reference(folded, heads, matrix);
                core::round_bf16_reference(partitioned);
                core::round_bf16_reference(full);
                core::round_bf16_reference(folded);
                check(full == partitioned, "shards reduce the U-transpose latent before applying V");
                check(full == folded, "dyadic folded and partitioned head mixes agree");
            }
        }
    }
}

void packed_hca_head_mix_rows() {
    constexpr unsigned heads = 2, width = 128, rope_width = 2;
    constexpr std::array u{1.0F, 0.5F}, v{-0.5F, 0.25F};
    const auto matrix = fold_value_head_mix_reference(heads, 1, v, u, true);
    for (const unsigned tokens : {1U, 7U, 300U}) {
        for (unsigned row = 0; row < tokens; ++row) {
            const unsigned position = 3U + 11U * row;
            const RotaryReferenceRow rope(rope_width, position, 10000.0);
            std::array<float, heads * width> operand{};
            for (unsigned i = 0; i < operand.size(); ++i)
                operand[i] = static_cast<float>(static_cast<int>((i * 13U + row * 11U) % 49U) - 24) / 8.0F;
            auto actual = operand;
            apply_folded_value_head_mix_reference(actual, heads, matrix);
            compressed_mqa_output_input_reference(actual, width, rope, width);

            std::array<float, heads * width> expected{};
            for (unsigned d = 0; d < width; ++d) {
                expected[d] = operand[d] / 2.0F - operand[width + d] / 4.0F;
                expected[width + d] = operand[d] / 4.0F + 1.125F * operand[width + d];
            }
            const auto angle = static_cast<double>(position);
            const auto c = static_cast<float>(std::cos(angle)), sn = static_cast<float>(std::sin(angle));
            for (unsigned head = 0; head < heads; ++head) {
                auto group = std::span(expected).subspan(head * width, width);
                const float x = group[width - 2], y = group[width - 1];
                group[width - 2] = x * c + y * sn;
                group[width - 1] = y * c - x * sn;
                float peak = 1e-10F;
                for (const float z : group) peak = std::max(peak, std::fabs(z));
                const auto exponent = static_cast<int>(std::ceil(std::log2(static_cast<double>(peak) / 448.0)));
                const float scale = std::ldexp(1.0F, exponent);
                for (float& z : group) z = weight::scalar_float::e4m3(weight::scalar_float::e4m3_encode(z / scale)) * scale;
            }
            check(actual == expected, "one-pass head fold and inverse rotary reach block quantization in FP32");
        }
    }
}

void latent_attention() {
    const SelectedLatentShape shape{2, 1, 1, 1, 1};
    constexpr float weights[] = {3.0F, 5.0F, 1.0F, 7.0F};
    const auto projections = split_latent_projection(shape, weights);
    static constexpr float latent[] = {1.0F}, rotary[] = {4.0F}, epsilon = 1e-6F;
    std::array<float, 2> inverse;
    latent_key_inverse_rms_reference(shape, projections, latent, rotary, epsilon, inverse);
    near(inverse[0], 1.0F / std::sqrt(12.5F + epsilon), "first head has own norm across projected K and shared rotary");
    near(inverse[1], 1.0F / std::sqrt(8.5F + epsilon), "second head has distinct norm");

    constexpr float query[] = {0.0F, 0.0F, 0.0F, 0.0F}, sinks[] = {0.0F, 0.0F};
    constexpr std::uint32_t only[] = {1};
    std::array<float, 2> output;
    selected_latent_attention_reference(shape, projections, query, sinks, only, [&](std::size_t row) {
        check(row == 1, "unselected latent row is not read");
        return SelectedLatentRow{latent, rotary, inverse};
    }, output);
    near(output[0], 2.5F, "first value-plane weight after sink");
    near(output[1], 3.5F, "second value-plane weight after sink");

    const float scale_step = std::log(2.0F) * std::sqrt(2.0F);
    const float query_with_scores[] = {scale_step / (3.0F * inverse[0]), 0.0F,
                                      0.0F, scale_step / (4.0F * inverse[1])};
    selected_latent_attention_reference(shape, projections, query_with_scores, sinks, only,
        [&](std::size_t) { return SelectedLatentRow{latent, rotary, inverse}; }, output);
    near(output[0], 10.0F / 3.0F, "absorbed plain-key score changes attended weight");
    near(output[1], 14.0F / 3.0F, "separate rotary-key score changes attended weight");
    const auto execute = [](std::size_t count, const LatentHeadRange& work) {
        for (auto i = count; i > 0; --i) work(i - 1, i);
    };
    std::array<float, 2> split_inverse{}, split_output{};
    latent_key_inverse_rms_reference(shape, projections, latent, rotary, epsilon, split_inverse, execute);
    selected_latent_attention_reference(shape, projections, query_with_scores, sinks, only,
        [&](std::size_t) { return SelectedLatentRow{latent, rotary, split_inverse}; }, split_output, execute);
    check(inverse == split_inverse && output == split_output, "MLA ranges are independent");
}

void mla_history() {
    ernie::ReferenceShape s{};
    s.hidden = s.head_dim = s.o_rank = 2;
    s.heads = s.q_rank = s.o_groups = s.vha_rank = 1;
    ernie::AttentionHistory history;
    constexpr auto lora_rank = ernie::data::attention::mla::kv_lora_rank;
    constexpr auto nope_dim = ernie::data::attention::mla::qk_nope_head_dim;
    const auto read = [](std::string_view name, std::span<float> values, std::size_t) {
        std::ranges::fill(values, 0.0F);
        if (name.ends_with("layernorm.weight")) std::ranges::fill(values, 1.0F);
        if (name.ends_with("kv_b_proj.weight")) values[0] = values[nope_dim * lora_rank] = 1.0F;
    };
    float query_value = 0.0F;
    const auto project = [&](const std::string& name, std::span<const float> input, std::span<float> output, unsigned) {
        std::ranges::fill(output, 0.0F);
        if (name.ends_with("kv_a_proj_with_mqa.weight") || name.ends_with(".o_proj.weight"))
            output[0] = input.front();
        if (name.ends_with("q_b_proj.weight")) output[0] = query_value;
    };
    constexpr float first[] = {1.0F, 0.0F}, second[] = {0.0F, 0.0F};
    // Unit impulse normalized across 512 channels rounds to 22.625 in BF16.
    auto output = ernie::attention_reference(7, s, read, first, history, project);
    near(output[0], 22.625F / 4.0F, "MLA first row, value projection, sink and output gate");
    output = ernie::attention_reference(7, s, read, second, history, project);
    near(output[0], 22.625F / 6.0F, "MLA retains the previous latent in the causal prefix");

    // At position zero with one key, log(3) versus a zero sink puts three
    // quarters of the weight on the key. Per-head RMS would change that score.
    s.contracts.checkpoint_attention = true;
    query_value = std::log(3.0F) * 16.0F / 22.625F;
    ernie::AttentionHistory projected;
    output = ernie::attention_reference(7, s, read, first, projected, project);
    check(std::fabs(output[0] - 22.625F * 3.0F / 8.0F) <= 0.0625F,
          "MLA preserves the projected query and key magnitudes in the sink score");
}

} // namespace

int main() {
    try {
        pool_and_merge();
        hca_carry();
        repeated_hca_groups();
        packed_hca_binding();
        packed_hca_gate_rows();
        compressed_index();
        index_window_rows();
        index_leading_rotary();
        index_hadamard_rows();
        folded_value_mix();
        head_mix_partition_reference();
        packed_hca_head_mix_rows();
        latent_attention();
        mla_history();
    } catch (const std::exception& error) {
        std::cerr << error.what() << '\n';
        return 1;
    }
    return 0;
}
