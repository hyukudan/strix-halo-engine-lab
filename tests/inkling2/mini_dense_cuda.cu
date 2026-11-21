#include "inkling2/inkling2_mini_attention_prepare_cuda.hpp"

#include <algorithm>
#include <array>
#include <cstdlib>
#include <cstring>
#include <iostream>
#include <memory>
#include <span>
#include <utility>

namespace {

namespace mini = strix::inkling2;
namespace cc = strix::cuda_core;
using Role = mini::MiniAttentionProjection;

void require(cudaError_t status) {
    if (status == cudaSuccess) return;
    std::cerr << cudaGetErrorString(status) << '\n';
    std::exit(1);
}

template <class T>
struct Buffer {
    explicit Buffer(std::size_t size, T value = {}) {
        require(cudaMallocManaged(&data, size * sizeof(T)));
        std::fill_n(data, size, value);
    }
    ~Buffer() { require(cudaFree(data)); }
    Buffer(const Buffer&) = delete;
    Buffer& operator=(const Buffer&) = delete;
    T* data = nullptr;
};

bool check_norm_and_embedding() {
    constexpr unsigned width = mini::kHiddenWidth;
    Buffer<float> input(2U * width, 1.0F), output(2U * width);
    Buffer<std::uint16_t> gains(width, 0x4000U);
    Buffer<std::uint16_t> embed(2U * width, 0x3F80U);
    Buffer<std::uint32_t> ids(2U);
    ids.data[1] = 1U;
    std::fill_n(embed.data + width, width, std::uint16_t{0xBF80U});
    require(mini::enqueue_mini_cuda_rmsnorm(input.data, gains.data, output.data, 2U, nullptr));
    require(cudaDeviceSynchronize());
    if (output.data[0] != 2.0F || output.data[2U * width - 1U] != 2.0F) return false;
    require(mini::enqueue_mini_cuda_embeddings(embed.data, gains.data, ids.data, output.data, 2U, nullptr));
    require(cudaDeviceSynchronize());
    return output.data[0] == 2.0F && output.data[2U * width - 1U] == -2.0F;
}

bool check_attention_prepare_and_output() {
    using namespace mini;
    constexpr auto visits = mini_schedule();
    constexpr std::size_t role_count = static_cast<unsigned>(Role::output) + 1U;
    constexpr auto mode = cc::DenseArith::mma1;
    std::array<std::unique_ptr<Buffer<std::uint16_t>>, role_count> matrices;
    MiniCudaProjectionWeights weights;
    for (unsigned i = 0; i < role_count; ++i) {
        const auto spec = mini_attention_projection_spec(static_cast<Role>(i));
        matrices[i] = std::make_unique<Buffer<std::uint16_t>>(std::size_t{spec.input_width} * spec.output_width);
        weights.matrix[i] = matrices[i]->data;
    }
    const auto dump = [&](Role role) { return matrices[static_cast<unsigned>(role)]->data; };
    for (unsigned i = 0; i < kQueryWidth; ++i) dump(Role::query)[i * kHiddenWidth] = 0x4180U;
    for (unsigned i = 0; i < kQueryHeads * kRelativeWidth; ++i) dump(Role::relative)[i * kHiddenWidth] = 0x3F80U;
    for (unsigned i = 0; i < kKvWidth; ++i) dump(Role::key)[i * kHiddenWidth] = 0x3F80U;
    for (unsigned i = 0; i < kKvWidth; i += kHeadWidth) dump(Role::value)[i * kHiddenWidth] = 0x3F80U;
    for (unsigned head = 0; head < kKvHeads; ++head) {
        dump(Role::key_taps)[(head * kConvTaps + 1U) * kHiddenWidth] = 0x3F00U;
    }
    dump(Role::relative_bias)[3] = 0x3F00U;
    dump(Role::relative_bias)[kRelativeExtent + 100U] = 0xBF80U;
    dump(Role::sink)[0] = 0x3E80U;
    dump(Role::output)[0] = 0x3F80U;

    Buffer<std::uint16_t> q_gain(kQueryWidth, 0x3F80U), relative_gain(kQueryHeads * kRelativeWidth, 0x3F80U);
    Buffer<std::uint16_t> key_convolution(kKvWidth * kConvTaps), value_convolution(kKvWidth * kConvTaps);
    Buffer<std::uint16_t> key_dynamic(kKvWidth * kConvTaps);
    for (unsigned channel = 0; channel < kKvWidth; ++channel) {
        key_convolution.data[channel * kConvTaps + kConvTaps - 1U] = 0x3F80U;
        key_dynamic.data[channel] = 0x3F80U;
    }
    weights.query_gain = q_gain.data;
    weights.relative_gain = relative_gain.data;
    constexpr auto conv_elements = kKvWidth * kConvTaps;
    Buffer<float> key_lags(conv_elements), value_lags(conv_elements);
    weights.key_convolution = key_lags.data;
    weights.value_convolution = value_lags.data;
    const auto bind_convolution = [&](LayerVisit visit, const Buffer<std::uint16_t>& key) {
        const bool channel_taps = !is_full_attention(visit.physical_layer);
        require(strix::attn_core::enqueue_bind_convolution_weights(
            {key.data, conv_elements}, key_lags.data, kKvWidth, kConvTaps, channel_taps, nullptr));
        require(strix::attn_core::enqueue_bind_convolution_weights(
            {value_convolution.data, conv_elements}, value_lags.data, kKvWidth, kConvTaps, channel_taps, nullptr));
    };

    Buffer<float> qi(kHiddenWidth), ki(kHiddenWidth), vi(kHiddenWidth);
    qi.data[0] = 3.0F;
    ki.data[0] = 1.0F;
    vi.data[0] = 2.0F;
    const MiniCudaProjectionInputs inputs{qi.data, ki.data, vi.data};
    Buffer<float> raw_k(kKvWidth), raw_v(kKvWidth), dyn_k(kKvHeads * kConvTaps), dyn_v(kKvHeads * kConvTaps);
    Buffer<float> rel(kQueryHeads * kRelativeWidth);
    Buffer<float> kh((kConvTaps - 1U) * kKvWidth), vh((kConvTaps - 1U) * kKvWidth);
    const MiniCudaProjectionWorkspace work{raw_k.data, raw_v.data, dyn_k.data, dyn_v.data,
        rel.data, kh.data, vh.data};

    Buffer<float> q(kQueryWidth), k(kKvWidth), v(kKvWidth), rb(kQueryHeads * kRelativeExtent), sink(kQueryHeads), gate(kQueryWidth);
    const MiniCudaPreparedAttention projected{q.data, k.data, v.data, rb.data, sink.data, gate.data};
    bind_convolution(visits[6], key_convolution);
    require(enqueue_mini_cuda_attention_prepare(visits[6], 16383U, 1U, inputs, weights, work, projected, nullptr, mode));
    require(cudaDeviceSynchronize());
    if (q.data[0] != 2.0F || k.data[0] != 0.89453125F || v.data[0] != 11.3125F ||
        rb.data[3] != 1.0F || rb.data[100] != -2.0F || sink.data[0] != 0.75F) return false;
    if (kh.data[2U * kKvWidth] != 1.0F || vh.data[2U * kKvWidth] != 2.0F) return false;

    std::fill_n(kh.data + 2U * kKvWidth, kKvWidth, 4.0F);
    bind_convolution(visits[11], key_dynamic);
    require(enqueue_mini_cuda_attention_prepare(visits[11], 16383U, 1U, inputs, weights, work, projected, nullptr, mode));
    require(cudaDeviceSynchronize());
    if (q.data[0] != 2.140625F || k.data[0] != 1.3359375F ||
        rb.data[3] != 1.0703125F || rb.data[100] != -2.140625F) return false;

    Buffer<float> attended(kQueryWidth), gated(kQueryWidth), out(kHiddenWidth);
    attended.data[0] = 2.1F;
    require(enqueue_mini_cuda_attention_output(1U, weights.projection(Role::output), attended.data, gate.data,
                                              gated.data, out.data, nullptr, mode));
    require(cudaDeviceSynchronize());
    return out.data[0] == 1.046875F && out.data[kHiddenWidth - 1U] == 0.0F;
}

bool check_tall_attention_projection_bits(unsigned rows) {
    constexpr unsigned width = mini::kHiddenWidth;
    constexpr std::array projections{
        std::pair{Role::query, false}, std::pair{Role::relative, false}, std::pair{Role::sink, false},
        std::pair{Role::output_gate, false}, std::pair{Role::key, false}, std::pair{Role::value, false},
        std::pair{Role::key_taps, false}, std::pair{Role::value_taps, false}, std::pair{Role::key_taps, true},
    };
    Buffer<float> input(std::size_t{rows} * width);
    std::uint32_t seed = 0x13579BDFU;
    const auto next = [&] {
        seed = 1664525U * seed + 1013904223U;
        return seed;
    };
    for (std::size_t i = 0; i < std::size_t{rows} * width; ++i)
        input.data[i] = static_cast<float>(static_cast<int>(next() >> 8U) - 8388608) * 0x1p-23F;
    std::unique_ptr<cc::BigCtx, decltype(&cc::dense_big_ctx_destroy)> ctx(
        cc::dense_big_ctx_create(), cc::dense_big_ctx_destroy);
    if (!ctx) return false;

    for (const auto& [role, round_bf16] : projections) {
        const unsigned out = mini::mini_attention_projection_spec(role).output_width;
        const auto weight_elements = std::size_t{out} * width;
        const auto output_elements = std::size_t{rows} * out;
        Buffer<std::uint16_t> weight(weight_elements);
        for (std::size_t i = 0; i < weight_elements; ++i) {
            const auto bits = next();
            weight.data[i] = static_cast<std::uint16_t>(0x3C00U | ((bits >> 16U) & 0x8000U) |
                                                        ((bits >> 12U) & 0x01FFU));
        }
        Buffer<float> expected(output_elements);
        Buffer<float> actual(output_elements + 4U, -1234.5F);
        const cc::DenseWeight matrix{weight.data, cc::WeightFormat::bf16, out, width};
        if (cc::dense_big_supported(matrix)) {
            cc::BigAct act{};
            require(cc::dense_big_split(input.data, rows, width, cc::WeightFormat::bf16, 3, &act, nullptr, ctx.get()));
            require(cc::dense_big_gemm(matrix, act, expected.data, nullptr,
                                      round_bf16 ? cc::BigEpi::round_bf16 : cc::BigEpi::none, nullptr, ctx.get()));
        } else {
            require(strix::core::for_each_dense_chunk<cc::kMaxRows>(rows, cudaSuccess,
                [&](unsigned first, unsigned count) {
                    return cc::dense_rows(matrix, input.data + std::size_t{first} * width,
                        expected.data + std::size_t{first} * out, count, nullptr, nullptr, cc::DenseArith::fp32);
                }));
        }
        const std::array project{mini::MiniCudaDenseProjection{weight.data, out, actual.data, round_bf16}};
        require(mini::enqueue_mini_cuda_dense_rows(input.data, width, rows, project, nullptr, cc::DenseArith::mma3, ctx.get()));
        require(cudaDeviceSynchronize());
        if (std::memcmp(expected.data, actual.data, output_elements * sizeof(float)) ||
            !std::all_of(actual.data + output_elements, actual.data + output_elements + 4U,
                         [](float value) { return value == -1234.5F; })) {
            std::cerr << "CUDA Mini projection bits differ for role " << static_cast<unsigned>(role)
                      << ", rows " << rows << ", BF16 round " << round_bf16 << '\n';
            return false;
        }
    }
    return true;
}

}  // namespace

int main() {
    if (check_norm_and_embedding() && check_attention_prepare_and_output() &&
        check_tall_attention_projection_bits(33U) && check_tall_attention_projection_bits(2048U)) return 0;
    std::cerr << "CUDA Mini dense preparation differs from the reference fixture\n";
    return 1;
}
