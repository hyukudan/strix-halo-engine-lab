#include "microscaled_r1_cuda_candidate.cuh"
#include "microscaled_r1_cuda_fixture.cuh"

#include <algorithm>
#include <cstdlib>
#include <cstring>
#include <iostream>
#include <random>

namespace {

namespace r1 = strix::moe_core::microscaled_r1;
namespace fixture = strix::tests::microscaled_r1;
namespace packed = r1::packed;
using packed::MatrixFormat;

void check(cudaError_t status) {
    if (status == cudaSuccess) return;
    std::cerr << cudaGetErrorString(status) << '\n';
    std::exit(1);
}

template <class T>
struct Buffer {
    T* data{};
    std::size_t size;

    explicit Buffer(std::size_t n) : size(n) { check(cudaMallocManaged(&data, n * sizeof(T))); }
    ~Buffer() { check(cudaFree(data)); }
    Buffer(const Buffer&) = delete;
    Buffer& operator=(const Buffer&) = delete;

    void fill(T value) { std::fill_n(data, size, value); }
    bool same(const Buffer& other) const { return std::memcmp(data, other.data, size * sizeof(T)) == 0; }
};

template <MatrixFormat Format>
struct Matrices {
    using Operand = packed::PackedOperand<Format>;
    std::size_t code_stride, scale_stride;
    unsigned offset, rows, width;
    Buffer<std::uint8_t> codes, scales;

    Matrices(unsigned experts, unsigned rows_, unsigned width_, unsigned offset_, std::mt19937& rng)
        : code_stride(std::size_t{rows_} * width_ / (Operand::kNvfp4 ? 2U : 1U)),
          scale_stride(std::size_t{rows_} * width_ / Operand::kBlockColumns),
          offset(offset_), rows(rows_), width(width_), codes(experts * code_stride + offset),
          scales(experts * scale_stride + offset) {
        for (std::size_t i = 0; i < codes.size; ++i) {
            auto code = static_cast<std::uint8_t>(rng() & 0xffU);
            if (!Operand::kNvfp4 && (code & 0x7fU) == 0x7fU) --code;
            codes.data[i] = code;
        }
        for (std::size_t i = 0; i < scales.size; ++i) {
            scales.data[i] = static_cast<std::uint8_t>(Operand::kNvfp4 ? rng() % 57U : 117U + rng() % 5U);
        }
    }

    packed::PackedMatrix operator()(unsigned expert, packed::RowOrder order) const {
        return {codes.data + offset + expert * code_stride, scales.data + offset + expert * scale_stride,
                rows, width, static_cast<float>(expert + 5U) / 31.0F, order};
    }
};

template <MatrixFormat Format>
__global__ void compare_decode(unsigned* errors) {
    using Operand = packed::PackedOperand<Format>;
    const unsigned i = blockIdx.x * blockDim.x + threadIdx.x;
    const auto code = static_cast<std::uint8_t>(i);
    const auto scale = static_cast<std::uint8_t>(i >> 8U);
    const auto hi = static_cast<std::uint8_t>(255U - code);
    const std::uint8_t pair[] = {code, hi};
    const unsigned bits = code | (unsigned{hi} << 8U);
    if constexpr (Operand::kNvfp4) {
        if (r1::cuda_body::e2m1_half_pair_portable(bits) != r1::cuda_body::e2m1_half_pair(bits)) atomicOr(errors, 1U);
    }
    constexpr float outer_scales[]{0.0F, -0.0F, 13.0F / 31.0F, 0x1p-138F, 0x1p+124F};
    for (const float outer : outer_scales) {
        const Operand weights{{nullptr, nullptr, 0U, 0U, outer}};
        const unsigned actual = r1::cuda_body::decode(weights, bits, scale);
        const unsigned expected = weights.decode_pair(pair, Operand::decode_scale(scale));
        if (actual != expected) atomicOr(errors, 1U);
    }
}

template <MatrixFormat Format, class Input>
__global__ void compare_raw(packed::PackedMatrix matrix, unsigned columns, const Input* input, unsigned* errors) {
    constexpr unsigned tile = 2U;
    const unsigned col = blockIdx.x * tile;
    const unsigned width = static_cast<unsigned>(matrix.columns);
    float actual[tile], expected[tile];
    r1::cuda_body::dot_rows<Format, tile>(matrix, col, 0U, columns - col, input, width, threadIdx.x, actual);
    r1::detail::dot_rows<Format, tile>(matrix, col, 0U, columns - col, input, width, threadIdx.x, expected);
    for (unsigned i = 0; i < tile; ++i) {
        if (__float_as_uint(actual[i]) != __float_as_uint(expected[i])) atomicOr(errors, 2U);
    }
}

template <MatrixFormat Format, unsigned TopK>
bool routed(unsigned h, unsigned n, unsigned offset, cudaStream_t stream) {
    using ExpertWeights = strix::moe_core::device_grouped::ExpertWeights;
    using packed::RowOrder;
    std::mt19937 rng(0x9F204U);
    constexpr unsigned experts = TopK + 2U;
    const Matrices<Format> gate(experts, 2U * n, h, offset, rng);
    const Matrices<Format> down(experts, h, n, offset, rng);
    Buffer<ExpertWeights> matrices(experts);
    Buffer<unsigned> ids(TopK), status(1U), errors(1U);
    Buffer<float> weights(TopK), input(h + offset), output(h + 1U), reference(h + 1U);
    Buffer<std::uint16_t> activation(std::size_t{TopK} * n + 1U), ref_activation(activation.size);
    for (unsigned e = 0; e < experts; ++e) {
        matrices.data[e] = {gate(e, RowOrder::interleaved_halves), down(e, RowOrder::contiguous)};
    }
    for (unsigned slot = 0; slot < TopK; ++slot) {
        ids.data[slot] = TopK - slot;
        weights.data[slot] = static_cast<float>(slot + 1U) / 29.0F;
    }
    ids.data[0] = ids.data[TopK - 1U];
    for (unsigned i = 0; i < h; ++i) {
        input.data[i + offset] = static_cast<float>(static_cast<int>(rng() % 173U) - 86) / 191.0F;
    }

    errors.fill(0U);
    compare_raw<Format><<<2U, 32U, 0U, stream>>>(gate(0U, RowOrder::interleaved_halves), 3U, input.data + offset,
                                                 errors.data);
    check(cudaGetLastError());
    check(cudaStreamSynchronize(stream));
    const fixture::RoutedCase route{matrices.data, ids.data, weights.data, status.data, input.data + offset,
                                    experts, h, n};
    for (const unsigned scenario : {0U, 1U, 2U}) {
        constexpr std::uint16_t activation_sentinel = 0x7654U;
        constexpr float output_sentinel = -0x1p+120F;
        status.data[0] = static_cast<unsigned>(scenario == 1U);
        if (scenario == 2U) ids.data[TopK - 1U] = experts;
        activation.fill(activation_sentinel);
        ref_activation.fill(activation_sentinel);
        output.fill(output_sentinel);
        reference.fill(output_sentinel);
        check(fixture::launch_reference<Format, TopK>(route, ref_activation.data, reference.data, stream));
        check(r1::launch<Format, TopK>(route.bind<r1::Args>(activation.data, output.data), stream));
        check(cudaStreamSynchronize(stream));
        if (!activation.same(ref_activation) || !output.same(reference) ||
            activation.data[activation.size - 1U] != activation_sentinel || output.data[h] != output_sentinel ||
            (scenario == 0U && (activation.data[0] == activation_sentinel || output.data[0] == output_sentinel)) ||
            (scenario != 0U && output.data[0] != output_sentinel)) {
            std::cerr << "R1 launch differs at H=" << h << " I=" << n << " offset=" << offset
                      << " top_k=" << TopK << " scenario=" << scenario << '\n';
            return false;
        }
    }

    compare_raw<Format><<<2U, 32U, 0U, stream>>>(down(0U, RowOrder::contiguous), 3U, activation.data, errors.data);
    check(cudaGetLastError());
    check(cudaStreamSynchronize(stream));
    return *errors.data == 0U;
}

template <MatrixFormat Format>
bool run(cudaStream_t stream) {
    using Operand = packed::PackedOperand<Format>;
    constexpr unsigned block = static_cast<unsigned>(Operand::kBlockColumns);
    Buffer<unsigned> errors(1U);
    errors.fill(0U);
    compare_decode<Format><<<256U, 256U, 0U, stream>>>(errors.data);
    check(cudaGetLastError());
    check(cudaStreamSynchronize(stream));
    if (*errors.data != 0U || !routed<Format, 8U>(2304U, 1152U, 0U, stream) ||
        !routed<Format, 9U>(256U + block, 128U + block, 0U, stream) ||
        !routed<Format, 8U>(256U + block, 128U + block, 1U, stream) ||
        !routed<Format, 8U>(block, block, 0U, stream)) {
        std::cerr << "K3c R1 CUDA " << (Operand::kNvfp4 ? "NVFP4" : "MXFP8") << " bitwise: FAIL\n";
        return false;
    }
    std::cout << "K3c R1 CUDA " << (Operand::kNvfp4 ? "NVFP4" : "MXFP8")
              << " decoder, FP32 dots, activation and output bitwise: PASS\n";
    return true;
}

} // namespace

int main() {
    cudaStream_t stream{};
    check(cudaStreamCreate(&stream));
    const bool fp4 = run<MatrixFormat::nvfp4>(stream);
    const bool fp8 = run<MatrixFormat::mxfp8>(stream);
    check(cudaStreamDestroy(stream));
    return fp4 && fp8 ? 0 : 1;
}
