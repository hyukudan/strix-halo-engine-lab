#include "inkling2/inkling2_mini_dspark_checkpoint.hpp"
#include "attn_core/block_gqa_reference.hpp"
#include "attn_core/masks.hpp"
#include "core/bf16_reference.hpp"
#include "core/markov_head_reference.hpp"

#include <iostream>
#include <numeric>
#include <stdexcept>

namespace {

using strix::inkling2::MiniDSparkCheckpoint;
namespace weight = strix::weight;

void require(bool condition, const char* message) {
    if (!condition) throw std::runtime_error(message);
}

void validate_fixture() {
    const auto schema = MiniDSparkCheckpoint::schema();
    weight::TensorIndex index;
    index.container = schema.container;
    index.schema = schema.version;
    for (const auto& tensor : schema.tensors) {
        weight::RawTensor raw;
        raw.name = tensor.name;
        raw.wire_dtype = tensor.wire_dtype;
        raw.extents = tensor.extents;
        raw.payload.range.bytes = std::accumulate(raw.extents.begin(), raw.extents.end(),
                                                 std::uint64_t{2}, std::multiplies<>{});
        index.tensors.push_back(std::move(raw));
    }
    weight::TensorSchemaSummary summary;
    auto status = weight::validate_tensor_schema(index, schema, summary);
    require(static_cast<bool>(status), "DSpark schema rejected a complete inventory");
    require(summary.tensors == 54U && summary.bytes == 635795744U, "DSpark inventory count or size changed");

    for (auto& tensor : index.tensors) {
        if (tensor.name == "markov_head.rnn_proj.weight") {
            require(tensor.extents == std::vector<std::uint64_t>{768, 5120}, "unexpected recurrent projection shape");
            --tensor.extents.back();
        }
    }
    status = weight::validate_tensor_schema(index, schema, summary);
    require(status.code == weight::Error::unsupported_shape, "DSpark schema admitted a broken recurrent projection");
}

void check_reference_math() {
    namespace attn = strix::attn_core;
    namespace core = strix::core;
    constexpr auto local = attn::mask::block_range(1025U, MiniDSparkCheckpoint::block, MiniDSparkCheckpoint::window(0U));
    constexpr auto full = attn::mask::block_range(1025U, MiniDSparkCheckpoint::block, MiniDSparkCheckpoint::window(1U));
    static_assert(local.begin == 1U && local.end == 1033U && !local.contains(0U) && local.contains(1032U));
    static_assert(full.begin == 0U && full.end == local.end && !full.contains(1033U));
    static_assert(attn::mask::block_range(1024U, 8U, 1024U).begin == 0U);
    static_assert(attn::mask::block_range(1023U, 8U, 1024U).begin == 0U);
    static_assert(attn::mask::block_range(1048568U, 8U, 1024U).end == 1048576U);
    constexpr std::array coefficients{0.0F, 1.0F};
    constexpr std::array taps{5.0F, 12.0F, 3.0F, 4.0F};
    constexpr std::array gain{1.0F, 2.0F};
    const auto fused = core::mix_rms_norm_bf16_reference(taps, coefficients, gain, 1e-6F);
    require(fused[0] == 0.84765625F && fused[1] == 2.265625F, "select and normalize the fused tap");

    std::array<float, 128> rotary{};
    rotary[0] = 1.0F;
    attn::split_half_rotary_reference(rotary, 128, 1e6F, 1U);
    require(rotary[0] == std::cos(1.0F) && rotary[64] == std::sin(1.0F), "split-half rotary positions");

    // Both queries see the last of the shared keys. A caller may trim
    // the left side of that key range without changing which mask rows are visible.
    constexpr std::array q{0.0F, 0.0F};
    constexpr std::array k{0.0F, 0.0F, 0.0F, 0.0F};
    constexpr std::array v{4.0F, 0.0F, 0.0F, 1.0F};
    std::array<float, 2> output{};
    attn::bidirectional_gqa_reference(1U, 1U, 1U, q, k, v, output);
    require(output[0] == 1.25F && output[1] == 1.25F, "every query sees the future block row");
    attn::bidirectional_gqa_reference(1U, 1U, 1U, q, std::span(k).last(3), std::span(v).last(3), output);
    require(output[0] == 1.0F / 3.0F && output[1] == output[0], "common local attention boundary");

    std::array state{1.0F, 0.0F};
    std::array<float, 2> latent{};
    constexpr std::array affine{0.0F, 0.0F, 0.0F, 0.0F, 0.0F, 1.0F};
    core::bf16_markov_gate_reference(affine, state, latent);
    require(state == std::array{0.5F, 0.0F} && latent == std::array{0.0F, 0.76171875F},
            "BF16 state recurrence and independent output latent");
    constexpr std::array lm_row{1.0F, 2.0F};
    constexpr std::array fold_weight{1.0F, 3.0F, 2.0F, 4.0F};
    constexpr std::array fold_bias{3.0F, 4.0F};
    std::array<double, 2> workspace{};
    const auto offset = core::fold_markov_head_row_reference(lm_row, fold_weight, fold_bias, workspace, output);
    require(output == std::array{5.0F, 11.0F} && offset == 11.0F, "fold the low-rank output and bias");
    constexpr double midpoint = 1.00390625;
    require(core::markov_fold_bf16(midpoint) == 0x3f80U &&
            core::markov_fold_bf16(std::nextafter(midpoint, 2.0)) == 0x3f81U &&
            core::markov_fold_bf16(std::nextafter(-midpoint, -2.0)) == 0xbf81U,
            "round the completed FP64 fold directly to BF16, including either midpoint sign");
    constexpr std::array cancellation{1.0F, 1.0F, 1.0F};
    constexpr std::array ordered_weights{1.0e30F, -1.0e30F, 1.0F};
    std::array<double, 1> ordered_work{};
    std::array<float, 1> ordered_out{};
    require(core::fold_markov_head_row_reference(cancellation, ordered_weights, ordered_weights,
                                                 ordered_work, ordered_out) == 1.0F && ordered_out[0] == 1.0F,
            "fold weight and bias in the same ascending hidden-channel order");
}

}  // namespace

int main(int argc, char** argv) {
    try {
        validate_fixture();
        check_reference_math();
        if (argc == 2) {
            weight::TensorLookup lookup;
            const auto status = lookup.open_safetensors(argv[1], MiniDSparkCheckpoint::schema());
            if (!status) throw std::runtime_error(status.detail);
        }
    } catch (const std::exception& error) {
        std::cerr << error.what() << '\n';
        return 1;
    }
}
