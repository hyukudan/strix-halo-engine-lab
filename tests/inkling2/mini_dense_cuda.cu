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
    explicit Buffer(std::size_t count, T value = {}) : size(count) {
        require(cudaMallocManaged(&data, count * sizeof(T)));
        std::fill_n(data, count, value);
    }
    ~Buffer() { require(cudaFree(data)); }
    Buffer(const Buffer&) = delete;
    Buffer& operator=(const Buffer&) = delete;
    T* data = nullptr;
    std::size_t size = 0;
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

// The prepare backend's operand class: the raw entries keep FP32 rows (a value with a residual below BF16, such as
// 1 + 2^-10, keeps it: the bits of the three-plane chain), and a bf16_exact declaration over BF16-rounded rows gives
// the three-plane bits on one plane. Tall rows (the dense_big route), the q group's width.
bool check_declared_operand(unsigned rows) {
    constexpr unsigned width = mini::kHiddenWidth;
    const unsigned out = mini::mini_attention_projection_spec(Role::query).output_width;
    const auto inputs = std::size_t{rows} * width, outputs = std::size_t{rows} * out;
    Buffer<float> raw(inputs), rounded(inputs);
    std::uint32_t seed = 0x2468ACE1U;
    const auto next = [&] { seed = 1664525U * seed + 1013904223U; return seed; };
    for (std::size_t i = 0; i < inputs; ++i) {
        const float value = static_cast<float>(static_cast<int>(next() >> 8U) - 8388608) * 0x1p-23F;
        raw.data[i] = i % 7U == 0U ? 1.0F + 0x1p-10F : value;  // residuals below BF16 on a fixed stride
        std::uint32_t bits = 0;
        std::memcpy(&bits, &value, sizeof bits);
        bits = (bits + 0x7FFFU + ((bits >> 16U) & 1U)) & 0xFFFF0000U;  // round to nearest even BF16 (finite inputs)
        std::memcpy(&rounded.data[i], &bits, sizeof bits);
    }
    Buffer<std::uint16_t> weight(std::size_t{out} * width);
    for (std::size_t i = 0; i < std::size_t{out} * width; ++i) {
        const auto bits = next();
        weight.data[i] = static_cast<std::uint16_t>(0x3C00U | ((bits >> 16U) & 0x8000U) | ((bits >> 12U) & 0x01FFU));
    }
    std::unique_ptr<cc::BigCtx, decltype(&cc::dense_big_ctx_destroy)> ctx(cc::dense_big_ctx_create(), cc::dense_big_ctx_destroy);
    if (!ctx) return false;
    const cc::DenseWeight matrix{weight.data, cc::WeightFormat::bf16, out, width};
    if (!cc::dense_big_supported(matrix)) return false;
    const auto three_planes = [&](const float* input, float* result) {
        cc::BigAct act{};
        require(cc::dense_big_split(input, rows, width, cc::WeightFormat::bf16, 3, &act, nullptr, ctx.get()));
        require(cc::dense_big_gemm(matrix, act, result, nullptr, cc::BigEpi::none, nullptr, ctx.get()));
    };
    Buffer<float> raw_ref(outputs), rounded_ref(outputs), raw_default(outputs), raw_declared(outputs), rounded_declared(outputs);
    three_planes(raw.data, raw_ref.data);
    three_planes(rounded.data, rounded_ref.data);
    const mini::detail::MiniPreparedCudaBackend fp32{{nullptr}, cc::DenseArith::mma3, ctx.get()};
    auto declared = fp32;
    declared.operand = strix::core::Bf16Operand::bf16_exact;
    if (fp32.operand != strix::core::Bf16Operand::fp32) {
        std::cerr << "the raw prepare backend must default to FP32 operands\n";
        return false;
    }
    const auto linear = [&](const mini::detail::MiniPreparedCudaBackend& backend, const float* input, float* result) {
        const std::array view{mini::MiniAttentionLinear{weight.data, out, result, false}};
        require(backend.linear(input, width, rows, view));
        require(cudaDeviceSynchronize());
    };
    linear(fp32, raw.data, raw_default.data);
    linear(declared, raw.data, raw_declared.data);
    linear(declared, rounded.data, rounded_declared.data);
    const auto same = [&](const Buffer<float>& a, const Buffer<float>& b) { return !std::memcmp(a.data, b.data, outputs * sizeof(float)); };
    const bool ok = same(raw_default, raw_ref) && same(rounded_declared, rounded_ref) && !same(raw_declared, raw_ref);
    if (!ok)
        std::cerr << "CUDA Mini declared operand: rows " << rows << " raw FP32 == three planes " << same(raw_default, raw_ref)
                  << ", rounded one plane == three planes " << same(rounded_declared, rounded_ref)
                  << ", raw declared differs (residual dropped) " << !same(raw_declared, raw_ref) << '\n';
    return ok;
}

}  // namespace

// Three-plane reference of one BF16 projection on tall rows (the exact BF16 split, hi / mid / lo into one accumulator).
void three_plane_reference(cc::BigCtx* ctx, const std::uint16_t* weights, unsigned out, unsigned width, unsigned rows,
                           const float* input, float* result, cc::BigEpi epilogue = cc::BigEpi::none) {
    const cc::DenseWeight matrix{weights, cc::WeightFormat::bf16, out, width};
    if (!cc::dense_big_supported(matrix)) {
        std::cerr << "tall reference: matrix not supported by the big route\n";
        std::exit(1);
    }
    cc::BigAct act{};
    require(cc::dense_big_split(input, rows, width, cc::WeightFormat::bf16, 3, &act, nullptr, ctx));
    require(cc::dense_big_gemm(matrix, act, result, nullptr, epilogue, nullptr, ctx));
}

void fill_bf16_weights(Buffer<std::uint16_t>& weight, std::uint32_t& seed) {
    for (std::size_t i = 0; i < weight.size; ++i) {
        seed = 1664525U * seed + 1013904223U;
        weight.data[i] = static_cast<std::uint16_t>(0x3C00U | ((seed >> 16U) & 0x8000U) | ((seed >> 12U) & 0x01FFU));
    }
}

bool same_floats(const float* a, const float* b, std::size_t count) { return !std::memcmp(a, b, count * sizeof(float)); }

// The raw prepare entry on tall rows (above the short-route limit): it keeps FP32 operands, so FP32 rows with residuals
// below BF16 give the three-plane bits of the key and value projections. Forcing bf16_exact in the entry would drop
// the residuals (shown by the one-plane control) and fail this comparison.
bool check_tall_raw_prepare(unsigned rows) {
    using namespace mini;
    constexpr auto visit_index = 6U;  // a non-full-attention visit: the shared-input group
    constexpr auto visits = mini_schedule();
    constexpr std::size_t role_count = static_cast<unsigned>(Role::output) + 1U;
    std::uint32_t seed = 0x7A11E5U;
    std::array<std::unique_ptr<Buffer<std::uint16_t>>, role_count> matrices;
    MiniCudaProjectionWeights weights;
    for (unsigned i = 0; i < role_count; ++i) {
        const auto spec = mini_attention_projection_spec(static_cast<Role>(i));
        matrices[i] = std::make_unique<Buffer<std::uint16_t>>(std::size_t{spec.input_width} * spec.output_width);
        fill_bf16_weights(*matrices[i], seed);
        weights.matrix[i] = matrices[i]->data;
    }
    Buffer<std::uint16_t> q_gain(kQueryWidth, 0x3F80U), relative_gain(kQueryHeads * kRelativeWidth, 0x3F80U);
    weights.query_gain = q_gain.data;
    weights.relative_gain = relative_gain.data;
    constexpr auto conv_elements = kKvWidth * kConvTaps;
    Buffer<float> key_lags(conv_elements), value_lags(conv_elements);
    weights.key_convolution = key_lags.data;
    weights.value_convolution = value_lags.data;
    Buffer<std::uint16_t> taps(conv_elements, 0x3F80U);
    const bool channel_taps = !is_full_attention(visits[visit_index].physical_layer);
    require(strix::attn_core::enqueue_bind_convolution_weights({taps.data, conv_elements}, key_lags.data, kKvWidth,
                                                               kConvTaps, channel_taps, nullptr));
    require(strix::attn_core::enqueue_bind_convolution_weights({taps.data, conv_elements}, value_lags.data, kKvWidth,
                                                               kConvTaps, channel_taps, nullptr));

    const auto input_count = std::size_t{rows} * kHiddenWidth;
    Buffer<float> input(input_count);
    for (std::size_t i = 0; i < input_count; ++i) {
        seed = 1664525U * seed + 1013904223U;
        input.data[i] = i % 7U == 0U ? 1.0F + 0x1p-10F
                                      : static_cast<float>(static_cast<int>(seed >> 8U) - 8388608) * 0x1p-23F;
    }
    const MiniCudaProjectionInputs inputs{input.data, input.data, input.data};
    Buffer<float> raw_k(std::size_t{rows} * kKvWidth), raw_v(std::size_t{rows} * kKvWidth),
        dyn_k(std::size_t{rows} * kKvHeads * kConvTaps), dyn_v(std::size_t{rows} * kKvHeads * kConvTaps),
        rel(std::size_t{rows} * kQueryHeads * kRelativeWidth);
    Buffer<float> kh((kConvTaps - 1U) * kKvWidth), vh((kConvTaps - 1U) * kKvWidth);
    const MiniCudaProjectionWorkspace work{raw_k.data, raw_v.data, dyn_k.data, dyn_v.data, rel.data, kh.data, vh.data};
    Buffer<float> q(std::size_t{rows} * kQueryWidth), k(std::size_t{rows} * kKvWidth), v(std::size_t{rows} * kKvWidth),
        rb(std::size_t{rows} * kQueryHeads * kRelativeExtent), sink(std::size_t{rows} * kQueryHeads),
        gate(std::size_t{rows} * kQueryWidth);
    const MiniCudaPreparedAttention prepared{q.data, k.data, v.data, rb.data, sink.data, gate.data};

    std::unique_ptr<cc::BigCtx, decltype(&cc::dense_big_ctx_destroy)> ctx(cc::dense_big_ctx_create(), cc::dense_big_ctx_destroy);
    if (!ctx) return false;
    require(enqueue_mini_cuda_attention_prepare(visits[visit_index], 0U, rows, inputs, weights, work, prepared, nullptr,
                                                cc::DenseArith::mma3, ctx.get()));
    require(cudaDeviceSynchronize());

    const auto key_spec = mini_attention_projection_spec(Role::key), value_spec = mini_attention_projection_spec(Role::value);
    const auto key_elements = std::size_t{rows} * key_spec.output_width, value_elements = std::size_t{rows} * value_spec.output_width;
    Buffer<float> key_reference(key_elements), value_reference(value_elements), key_one_plane(key_elements);
    three_plane_reference(ctx.get(), weights.matrix[static_cast<unsigned>(Role::key)], key_spec.output_width, kHiddenWidth,
                          rows, input.data, key_reference.data);
    three_plane_reference(ctx.get(), weights.matrix[static_cast<unsigned>(Role::value)], value_spec.output_width,
                          kHiddenWidth, rows, input.data, value_reference.data);
    const std::array control{MiniCudaDenseProjection{weights.matrix[static_cast<unsigned>(Role::key)], key_spec.output_width,
                                                     key_one_plane.data, false}};
    require(enqueue_mini_cuda_dense_rows(input.data, kHiddenWidth, rows, control, nullptr, cc::DenseArith::mma3, ctx.get(),
                                         strix::core::Bf16Operand::bf16_exact));
    require(cudaDeviceSynchronize());
    const bool keys = same_floats(raw_k.data, key_reference.data, key_elements);
    const bool values = same_floats(raw_v.data, value_reference.data, value_elements);
    const bool control_differs = !same_floats(key_one_plane.data, key_reference.data, key_elements);
    if (!keys || !values || !control_differs)
        std::cerr << "CUDA Mini raw prepare, rows " << rows << ": keys == three planes " << keys << ", values == three planes "
                  << values << ", one-plane control differs " << control_differs << '\n';
    return keys && values && control_differs;
}

bool bf16_exact_rows(const float* values, std::size_t count) {
    for (std::size_t i = 0; i < count; ++i) {
        std::uint32_t bits = 0;
        std::memcpy(&bits, &values[i], sizeof bits);
        if (bits & 0xFFFFU) return false;
    }
    return true;
}

// The resident producers feeding the declared one-plane class. Input side: the rmsnorm_bf16 producer
// (enqueue_mini_cuda_rmsnorm) stores BF16-exact rows. Output side: the attention output entry's gated product is
// BF16-exact and its projection equals the rounded three-plane reference. In both cases the one-plane dispatch on the
// producer's own output has the three-plane bits.
bool check_resident_producers(unsigned rows) {
    using namespace mini;
    std::uint32_t seed = 0x51DEC0DEU;
    std::unique_ptr<cc::BigCtx, decltype(&cc::dense_big_ctx_destroy)> ctx(cc::dense_big_ctx_create(), cc::dense_big_ctx_destroy);
    if (!ctx) return false;
    const auto random_rows = [&](Buffer<float>& rows_buffer) {
        for (std::size_t i = 0; i < rows_buffer.size; ++i) {
            seed = 1664525U * seed + 1013904223U;
            rows_buffer.data[i] = static_cast<float>(static_cast<int>(seed >> 8U) - 8388608) * 0x1p-21F;
        }
    };
    const auto one_plane = [&](const float* input, unsigned width, const std::uint16_t* weight, unsigned out, float* result) {
        const std::array project{MiniCudaDenseProjection{weight, out, result, false}};
        require(enqueue_mini_cuda_dense_rows(input, width, rows, project, nullptr, cc::DenseArith::mma3, ctx.get(),
                                             strix::core::Bf16Operand::bf16_exact));
        require(cudaDeviceSynchronize());
    };
    // Input side.
    Buffer<float> raw(std::size_t{rows} * kHiddenWidth, 0.0F), normalised(std::size_t{rows} * kHiddenWidth);
    random_rows(raw);
    Buffer<std::uint16_t> gain(kHiddenWidth, 0x3F80U), weight(std::size_t{kQueryWidth} * kHiddenWidth);
    fill_bf16_weights(weight, seed);
    require(enqueue_mini_cuda_rmsnorm(raw.data, gain.data, normalised.data, rows, nullptr));
    require(cudaDeviceSynchronize());
    Buffer<float> reference(std::size_t{rows} * kQueryWidth), declared(std::size_t{rows} * kQueryWidth);
    three_plane_reference(ctx.get(), weight.data, kQueryWidth, kHiddenWidth, rows, normalised.data, reference.data);
    one_plane(normalised.data, kHiddenWidth, weight.data, kQueryWidth, declared.data);
    const bool producer = bf16_exact_rows(normalised.data, normalised.size) &&
                          same_floats(declared.data, reference.data, reference.size);
    // Output side.
    Buffer<float> attended(std::size_t{rows} * kQueryWidth), gate_logits(std::size_t{rows} * kQueryWidth),
        gated(std::size_t{rows} * kQueryWidth), out(std::size_t{rows} * kHiddenWidth),
        out_reference(std::size_t{rows} * kHiddenWidth), out_declared(std::size_t{rows} * kHiddenWidth);
    random_rows(attended);
    random_rows(gate_logits);
    Buffer<std::uint16_t> output_weight(std::size_t{kHiddenWidth} * kQueryWidth);
    fill_bf16_weights(output_weight, seed);
    require(enqueue_mini_cuda_attention_output(rows, output_weight.data, attended.data, gate_logits.data, gated.data,
                                              out.data, nullptr, cc::DenseArith::mma3, ctx.get()));
    require(cudaDeviceSynchronize());
    three_plane_reference(ctx.get(), output_weight.data, kHiddenWidth, kQueryWidth, rows, gated.data, out_reference.data,
                          cc::BigEpi::round_bf16);
    one_plane(gated.data, kQueryWidth, output_weight.data, kHiddenWidth, out_declared.data);
    // The entry's own output is the rounded reference; the declared one-plane dispatch (unrounded) equals the
    // unrounded three-plane bits, which the BF16 rounding of the reference then matches element for element.
    Buffer<float> unrounded(std::size_t{rows} * kHiddenWidth);
    three_plane_reference(ctx.get(), output_weight.data, kHiddenWidth, kQueryWidth, rows, gated.data, unrounded.data);
    const bool output = bf16_exact_rows(gated.data, gated.size) && same_floats(out.data, out_reference.data, out.size) &&
                        same_floats(out_declared.data, unrounded.data, unrounded.size);
    if (!producer || !output)
        std::cerr << "CUDA Mini resident producers, rows " << rows << ": input side " << producer << ", output side "
                  << output << '\n';
    return producer && output;
}

int main() {
    if (check_norm_and_embedding() && check_attention_prepare_and_output() &&
        check_tall_attention_projection_bits(33U) && check_tall_attention_projection_bits(2048U) &&
        check_declared_operand(2048U) && check_tall_raw_prepare(2048U) && check_resident_producers(2048U)) return 0;
    std::cerr << "CUDA Mini dense preparation differs from the reference fixture\n";
    return 1;
}
