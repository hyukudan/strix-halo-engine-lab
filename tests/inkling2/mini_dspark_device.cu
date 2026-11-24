#include "attn_core/block_gqa_reference.hpp"
#include "core/markov_head_reference.hpp"
#include "inkling2/inkling2_mini_resident_dspark.hpp"

#include <cmath>
#include <iostream>
#include <ranges>

namespace {

using namespace strix::inkling2;
namespace cr = strix::attn::cache_rows;
namespace ref = strix::attn_core;
namespace core = strix::core;
using D = MiniDSparkCheckpoint;

void require(bool condition, const char* diagnostic) {
    if (!condition) throw std::runtime_error(diagnostic);
}

template <class T>
void upload(std::span<const T> host, const mini_gpu::Buffer<T>& device, mini_gpu::Stream stream) {
    require(host.size() == device.span().size(), "fixture upload geometry");
    mini_gpu::copy(device.data(), host.data(), host.size_bytes(), mini_gpu::host_to_device, stream);
}

template <class T>
std::vector<T> download(const mini_gpu::Buffer<T>& device, mini_gpu::Stream stream) {
    std::vector<T> output(device.span().size());
    mini_gpu::copy(output.data(), device.data(), device.span().size_bytes(), mini_gpu::device_to_host, stream);
    mini_gpu::synchronize(stream);
    return output;
}

template <class Transform>
void transform(std::size_t count, Transform operation, mini_gpu::Stream stream) {
    mini_gpu::check(core::pointwise::elements(count, operation, stream), "run DSpark fixture transform");
}

void check_context_transforms(mini_gpu::Stream stream) {
    constexpr unsigned rows = 2U;
    constexpr auto width = MiniDSparkTapPlan::width;
    constexpr auto count = MiniDSparkTapPlan::count;
    std::vector<float> host_taps(rows * width), tap_image(rows * kHiddenWidth);
    for (unsigned row = 0; row < rows; ++row)
        for (unsigned tap = 0; tap < count; ++tap)
            for (unsigned i = 0; i < kHiddenWidth; ++i)
                host_taps[row * width + tap * kHiddenWidth + i] = static_cast<float>(1U + tap + row + i % 2U);
    mini_gpu::Buffer<float> input(tap_image.size()), output(tap_image.size());
    mini_gpu::Buffer<std::uint16_t> taps(host_taps.size());
    for (unsigned tap = 0; tap < count; ++tap) {
        for (unsigned row = 0; row < rows; ++row)
            std::ranges::copy(std::span(host_taps).subspan(row * width + tap * kHiddenWidth, kHiddenWidth),
                              tap_image.begin() + row * kHiddenWidth);
        upload<float>(tap_image, input, stream);
        transform(tap_image.size(), core::row_transforms::StoreBf16Slice{
            input.data(), taps.data() + tap * kHiddenWidth, kHiddenWidth, width}, stream);
        mini_gpu::synchronize(stream);
    }
    const auto bytes = download(taps, stream);
    require(std::ranges::equal(bytes, host_taps | std::views::transform(cr::encode)), "tap row stride/order");

    constexpr std::array<float, count> coefficients{0.125F, 0.25F, 0.5F, 0.125F};
    std::vector<float> host_gain(kHiddenWidth, 1.0F);
    std::vector<std::uint16_t> bits(kHiddenWidth, cr::encode(1.0F));
    mini_gpu::Buffer<float> fusion(coefficients.size());
    mini_gpu::Buffer<std::uint16_t> gain(bits.size());
    upload<float>(coefficients, fusion, stream);
    upload<std::uint16_t>(bits, gain, stream);
    mini_dspark_device::fuse(taps.data(), fusion.data(), gain.data(), output.data(), rows, stream);
    const auto fused = download(output, stream);
    for (unsigned row = 0; row < rows; ++row) {
        const auto expected = core::mix_rms_norm_bf16_reference(
            std::span(host_taps).subspan(row * width, width), coefficients, host_gain, D::epsilon);
        require(std::ranges::equal(std::span(fused).subspan(row * kHiddenWidth, kHiddenWidth), expected),
                "DSpark BF16 fusion differs from the oracle");
    }

    std::array<float, kHeadWidth> head{}, head_gain{};
    std::array<std::uint16_t, kHeadWidth> rotary_gain{};
    head_gain.fill(1.0F);
    rotary_gain.fill(cr::encode(1.0F));
    head.front() = 1.0F;
    mini_gpu::Buffer<float> rotary(kHeadWidth);
    mini_gpu::Buffer<std::uint16_t> gain_rotary(kHeadWidth);
    upload<float>(head, rotary, stream);
    upload<std::uint16_t>(rotary_gain, gain_rotary, stream);
    mini_dspark_device::norm_rotary(rotary.data(), gain_rotary.data(), 1U, 1U, 1U, stream);
    const auto rotated = download(rotary, stream);
    core::rms_norm_bf16_reference(head, head_gain, D::epsilon);
    ref::split_half_rotary_reference(head, kHeadWidth, D::rope_theta, 1U);
    core::round_bf16_reference(head);
    require(std::ranges::equal(rotated, head), "DSpark BF16 norm/RoPE differs from the oracle");
}

void check_attention(mini_gpu::Stream stream) {
    constexpr unsigned anchor = 1025U;
    constexpr auto block = D::block;
    const std::vector<float> host_q(block * kQueryWidth), host_k((anchor + block) * kKvWidth);
    std::vector<float> host_v(host_k.size());
    for (unsigned h = 0; h < kKvHeads; ++h) {
        host_v[h * kHeadWidth] = 4.0F;
        host_v[(anchor + block - 1U) * kKvWidth + h * kHeadWidth] = static_cast<float>(h + 1U);
    }
    mini_gpu::Buffer<float> q(host_q.size()), k(host_k.size()), v(host_v.size()), output(host_q.size());
    mini_gpu::Buffer<std::uint16_t> zeros(kHeadWidth);
    mini_gpu::Buffer<unsigned> failure(1U);
    upload<float>(host_q, q, stream);
    upload<float>(host_k, k, stream);
    upload<float>(host_v, v, stream);
    mini_gpu::clear(zeros.data(), zeros.span().size_bytes(), stream);
    for (const auto window : {0U, D::context_window}) {
        const auto wrapped = window != 0U;
        const mini_dspark_device::KvCache cache(wrapped ? window + block : anchor + block, wrapped);
        mini_gpu::clear(failure.data(), sizeof(unsigned), stream);
        cache.store(k.data(), v.data(), 0U, anchor, failure.data(), stream);
        mini_dspark_device::attend(window, anchor, cache, q.data(), k.data() + anchor * kKvWidth,
                                   v.data() + anchor * kKvWidth, output.data(), zeros.data(), failure.data(), stream);
        const auto actual = download(output, stream);
        require(download(failure, stream).front() == 0U, "device rejected DSpark GQA operands");
        const auto range = ref::mask::block_range(anchor, block, window);
        const auto offset = range.begin * kKvWidth;
        std::vector<float> expected(host_q.size());
        ref::bidirectional_gqa_reference(kQueryHeads, kKvHeads, kHeadWidth, host_q,
            std::span(host_k).subspan(offset), std::span(host_v).subspan(offset), expected);
        for (std::size_t i = 0; i < actual.size(); ++i)
            require(std::abs(actual[i] - expected[i]) < 2e-6F, "DSpark shared GQA differs from the oracle");
    }
}

void check_markov(mini_gpu::Stream stream) {
    constexpr unsigned rank = 2U;
    constexpr unsigned width = 3U;
    constexpr std::array lm{1.0F, 2.0F, 3.0F};
    constexpr std::array weight{1.0F, -2.0F, 4.0F, 1.0F, 0.5F, 2.0F};
    constexpr std::array bias{2.0F, 0.5F, -1.0F};
    const auto encode = [](const auto& values) {
        std::array<std::uint16_t, std::tuple_size_v<std::remove_cvref_t<decltype(values)>>> image{};
        std::ranges::transform(values, image.begin(), cr::encode);
        return image;
    };
    const auto lm_bits = encode(lm), bias_bits = encode(bias);
    const auto weight_bits = encode(weight);
    mini_gpu::Buffer<std::uint16_t> d_lm(lm.size()), d_weight(weight.size()), d_bias(bias.size()), fold(rank), offset(1U);
    upload<std::uint16_t>(lm_bits, d_lm, stream);
    upload<std::uint16_t>(weight_bits, d_weight, stream);
    upload<std::uint16_t>(bias_bits, d_bias, stream);
    using namespace core::markov_device;
    transform(rank + 1U, Fold{Bf16Matrix{d_lm.data(), width}, d_weight.data(), d_bias.data(),
                              fold.data(), offset.data(), width, rank}, stream);
    std::array<float, rank> expected_fold{}, expected_latent{};
    std::array<double, rank> work{};
    const auto expected_bias = core::fold_markov_head_row_reference(lm, weight, bias, work, expected_fold);
    require(std::ranges::equal(download(fold, stream), encode(expected_fold)) &&
            download(offset, stream).front() == cr::encode(expected_bias), "FP64 Markov fold differs from the oracle");

    using Q8 = strix::weight::Q8Storage;
    // The same logical row as `lm` (scale 0.5, codes 2, 4, 6), so both folds share one offset.
    std::array<std::uint8_t, Q8::bytes> packed{0x00U, 0x38U, 2U, 4U, 6U};
    std::array<float, Q8::elements> decoded{1.0F, 2.0F, 3.0F};
    std::array<float, Q8::elements * rank> padded_weight{};
    std::array<float, Q8::elements> padded_bias{};
    std::ranges::copy(weight, padded_weight.begin());
    std::ranges::copy(bias, padded_bias.begin());
    const auto packed_weight = encode(padded_weight);
    const auto packed_bias = encode(padded_bias);
    mini_gpu::Buffer<std::uint8_t> q8(packed.size());
    mini_gpu::Buffer<std::uint16_t> q8_weight(packed_weight.size()), q8_bias(packed_bias.size());
    upload<std::uint8_t>(packed, q8, stream);
    upload<std::uint16_t>(packed_weight, q8_weight, stream);
    upload<std::uint16_t>(packed_bias, q8_bias, stream);
    transform(rank + 1U, Fold{Q8Matrix{q8.data(), Q8::bytes}, q8_weight.data(), q8_bias.data(),
                              fold.data(), offset.data(), Q8::elements, rank}, stream);
    std::array<float, rank> expected_q8{};
    const auto q8_offset = core::fold_markov_head_row_reference(decoded, padded_weight, padded_bias, work, expected_q8);
    require(std::ranges::equal(download(fold, stream), encode(expected_q8)) &&
            download(offset, stream).front() == cr::encode(q8_offset), "Q8 Markov fold differs from the decoded oracle");
    require(q8_offset == expected_bias, "Markov fold fixture offsets must match");

    constexpr std::array context{0.0F, 0.0F, 0.25F, 0.0F, 2.0F, -1.0F};
    constexpr std::array recurrent{0.0F, 0.0F, -0.25F, 0.0F, -1.0F, 2.0F};
    constexpr std::array<float, 3U * rank> zero{};
    constexpr std::array init{1.0F, 0.0F};
    const auto zero_bits = encode(zero);
    mini_gpu::Buffer<float> context_dot(context.size()), recurrent_dot(recurrent.size()), state(rank), latent(rank);
    mini_gpu::Buffer<std::uint16_t> a_bias(zero.size());
    upload<float>(context, context_dot, stream);
    upload<float>(recurrent, recurrent_dot, stream);
    upload<float>(init, state, stream);
    upload<std::uint16_t>(zero_bits, a_bias, stream);
    transform(rank, Gate{context_dot.data(), recurrent_dot.data(), a_bias.data(), state.data(), latent.data(), rank}, stream);
    auto expected_state = init;
    std::array<float, 3U * rank> affine{};
    std::ranges::transform(context, recurrent, affine.begin(), std::plus<>{});
    core::bf16_markov_gate_reference(affine, expected_state, expected_latent);
    require(std::ranges::equal(download(state, stream), expected_state) &&
            std::ranges::equal(download(latent, stream), expected_latent), "Markov gate differs from the oracle");

    constexpr std::array base{0.3046875F}, correction{3.15625F};
    mini_gpu::Buffer<float> base_logits(1U), delta(1U);
    upload<float>(base, base_logits, stream);
    upload<float>(correction, delta, stream);
    transform(1U, ComposeHead{base_logits.data(), delta.data(), offset.data(), kLogitSoftcap}, stream);
    const auto round = [](float x) { return cr::decode(cr::encode(x)); };
    const auto combined = round(round(base.front()) + round(correction.front() + expected_bias));
    const auto expected_score = round(kLogitSoftcap * std::tanh(combined / kLogitSoftcap));
    require(download(base_logits, stream).front() == expected_score, "DSpark folded head rounding order");
}

}  // namespace

int main() {
    try {
        const mini_gpu::StreamOwner stream;
        check_context_transforms(stream.get());
        std::cout << "PASS DSpark taps, fusion and rotary\n";
        check_attention(stream.get());
        std::cout << "PASS DSpark local and global block GQA\n";
        check_markov(stream.get());
        std::cout << "PASS DSpark Markov fold, gate and composition\n";
    } catch (const std::exception& error) {
        std::cerr << error.what() << '\n';
        return 1;
    }
}
