#include "inkling2/inkling2_mini_attention_prepare_cuda.hpp"

#include <algorithm>
#include <array>
#include <cstdlib>
#include <iostream>
#include <memory>
#include <span>

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
    std::fill_n(embed.data + width, width, 0xBF80U);
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
    weights.key_convolution = key_convolution.data;
    weights.value_convolution = value_convolution.data;

    Buffer<float> qi(kHiddenWidth), ki(kHiddenWidth), vi(kHiddenWidth);
    qi.data[0] = 3.0F;
    ki.data[0] = 1.0F;
    vi.data[0] = 2.0F;
    const MiniCudaProjectionInputs inputs{qi.data, ki.data, vi.data};
    Buffer<float> raw_k(kKvWidth), raw_v(kKvWidth), dyn_k(kKvHeads * kConvTaps), dyn_v(kKvHeads * kConvTaps);
    Buffer<float> rel(kQueryHeads * kRelativeWidth), lags(kConvTaps * kKvWidth);
    Buffer<float> kh((kConvTaps - 1U) * kKvWidth), vh((kConvTaps - 1U) * kKvWidth);
    const MiniCudaProjectionWorkspace work{raw_k.data, raw_v.data, dyn_k.data, dyn_v.data,
        rel.data, lags.data, kh.data, vh.data};

    Buffer<float> q(kQueryWidth), k(kKvWidth), v(kKvWidth), rb(kQueryHeads * kRelativeExtent), sink(kQueryHeads), gate(kQueryWidth);
    const MiniCudaPreparedAttention projected{q.data, k.data, v.data, rb.data, sink.data, gate.data};
    require(enqueue_mini_cuda_attention_prepare(visits[6], 16383U, 1U, inputs, weights, work, projected, nullptr, mode));
    require(cudaDeviceSynchronize());
    if (q.data[0] != 2.0F || k.data[0] != 0.89453125F || v.data[0] != 11.3125F ||
        rb.data[3] != 1.0F || rb.data[100] != -2.0F || sink.data[0] != 0.75F) return false;
    if (kh.data[2U * kKvWidth] != 1.0F || vh.data[2U * kKvWidth] != 2.0F) return false;

    std::fill_n(kh.data + 2U * kKvWidth, kKvWidth, 4.0F);
    weights.key_convolution = key_dynamic.data;
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

}  // namespace

int main() {
    if (check_norm_and_embedding() && check_attention_prepare_and_output()) return 0;
    std::cerr << "CUDA Mini dense preparation differs from the reference fixture\n";
    return 1;
}
