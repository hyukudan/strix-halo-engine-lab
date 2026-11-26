#include "strix/ernie/attention.hpp"

#include "attn_core/hybrid_reference.hpp"
#include "attn_core/selected_latent_reference.hpp"
#include "attn_core/sparse_index_reference.hpp"
#include "core/hadamard_reference.hpp"

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
    const auto read = [](std::string_view name, std::span<float> values, std::size_t) {
        std::ranges::fill(values, 0.0F);
        if (name.ends_with("norm.weight")) std::ranges::fill(values, 1.0F);
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
    check(gated_index_top_k_reference(queries, weights, gates, 32, 3, 2, at) == std::vector<std::uint32_t>{0, 2},
          "index gate suppresses second head and lower position wins tie");
    check(gated_index_top_k_reference(queries, weights, fallback, 32, 3, 1, at) == std::vector<std::uint32_t>{1},
          "index retains strongest head when all gate logits are negative");
    check(gated_index_top_k_reference(queries, weights, gates, 32, 2, 1, at) == std::vector<std::uint32_t>{0},
          "future high-scoring key lies outside the causal prefix");
}

void latent_attention() {
    const SelectedLatentShape shape{2, 1, 1, 1, 1};
    constexpr float weights[] = {3.0F, 5.0F, 1.0F, 7.0F};
    const auto projections = split_latent_projection(shape, weights);
    constexpr float latent[] = {1.0F}, rotary[] = {4.0F}, epsilon = 1e-6F;
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
        if (name.ends_with("kv_b_proj.weight")) values[nope_dim * lora_rank] = 1.0F;
    };
    const auto project = [](const std::string& name, std::span<const float> input, std::span<float> output, unsigned) {
        std::ranges::fill(output, 0.0F);
        if (name.ends_with("kv_a_proj_with_mqa.weight") || name.ends_with(".o_proj.weight"))
            output[0] = input.front();
    };
    constexpr float first[] = {1.0F, 0.0F}, second[] = {0.0F, 0.0F};
    // Unit impulse normalized across 512 channels rounds to 22.625 in BF16.
    auto output = ernie::attention_reference(7, s, read, first, history, project);
    near(output[0], 22.625F / 4.0F, "MLA first row, value projection, sink and output gate");
    output = ernie::attention_reference(7, s, read, second, history, project);
    near(output[0], 22.625F / 6.0F, "MLA retains the previous latent in the causal prefix");
}

} // namespace

int main() {
    try {
        pool_and_merge();
        hca_carry();
        compressed_index();
        latent_attention();
        mla_history();
    } catch (const std::exception& error) {
        std::cerr << error.what() << '\n';
        return 1;
    }
    return 0;
}
