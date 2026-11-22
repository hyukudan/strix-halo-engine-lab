#include "exec/catalog_cuda_decode.hpp"
#include "strix/device/runtime.hpp"
#include "strix/weight/formats/scalar_float.hpp"

#include <algorithm>
#include <bit>
#include <iostream>
#include <stdexcept>
#include <string_view>

namespace {

namespace decode = strix::exec::cuda_decode;
namespace graph = strix::graph;
using strix::device::native::check;

template <class T>
class Image {
public:
    explicit Image(std::size_t count) : count_(count) {
        check(cudaMallocManaged(&data_, count * sizeof(T)), "allocate fixture image");
    }
    ~Image() { (void)cudaFree(data_); }
    Image(const Image&) = delete;
    Image& operator=(const Image&) = delete;
    T* data() const { return data_; }
    std::span<T> values() const { return {data_, count_}; }

private:
    T* data_ = nullptr;
    std::size_t count_;
};

void submit(const auto& args) {
    const auto kernel = decode::bind(args);
    kernel.fn(kernel.args, kernel.grid, nullptr);
}

void compare(const Image<float>& a, const Image<float>& b, std::string_view what) {
    const auto same_bits = [](float lhs, float rhs) {
        return std::bit_cast<std::uint32_t>(lhs) == std::bit_cast<std::uint32_t>(rhs);
    };
    if (!std::ranges::equal(a.values(), b.values(), same_bits)) throw std::runtime_error(std::string(what));
}

void dense(decode::Bf16DenseMode mode, graph::Form form, unsigned outputs, std::string_view name,
           unsigned second_outputs = 0U) {
    constexpr unsigned in = 256;
    constexpr float cap = 7.0F;
    const auto second_offset = outputs + 16U;
    Image<std::uint16_t> weights(in * (outputs + second_outputs));
    Image<float> input(in), left(second_offset + second_outputs + 16U), right(left.values().size());
    std::ranges::fill(left.values(), -123.0F);
    std::ranges::fill(right.values(), -123.0F);
    for (unsigned i = 0; i < in; ++i) input.data()[i] = static_cast<float>(static_cast<int>(i % 31U) - 15) / 11.0F;
    for (unsigned i = 0; i < in * (outputs + second_outputs); ++i) {
        const auto value = static_cast<float>(static_cast<int>((i * 13U + i / in) % 71U) - 35) / 31.0F;
        weights.data()[i] = strix::weight::scalar_float::bf16_encode(value);
    }
    decode::Bf16DenseGroup head{input.data(), in, {{{weights.data(), outputs, left.data()}}}, 1U, mode};
    if (second_outputs != 0U) {
        head.projections[1] = {weights.data() + in * outputs, second_outputs, left.data() + second_offset};
        head.count = 2U;
    }
    submit(head);
    if (form == graph::Form::softcap) submit(decode::Bf16Softcap{left.data(), outputs, cap});
    else submit(decode::Bf16Round{left.data(), outputs});
    auto fused = head;
    fused.projections[0].output = right.data();
    if (second_outputs != 0U) {
        submit(decode::Bf16Round{left.data() + second_offset, second_outputs});
        fused.projections[1].output = right.data() + second_offset;
    }
    if (mode == decode::Bf16DenseMode::mma3) {
        fused.mode = decode::Bf16DenseMode::mma3_round;
        for (unsigned i = 0; i < fused.count; ++i) fused.projections[i].round_bf16 = true;
        submit(fused);
    } else {
        submit(decode::Bf16Fp32Epilogue{fused, form, cap});
    }
    check(cudaDeviceSynchronize(), "complete dense fixture");
    compare(left, right, name);
    std::cout << "EPI " << name << " PASS\n";
}

void convolution() {
    constexpr unsigned channels = 49, taps = 4, heads = 7;
    constexpr float canary = -456.0F;
    for (const bool dynamic : {false, true}) {
        Image<float> input(channels), weights(channels * taps), delta(heads * taps);
        Image<float> left_history(channels * taps + 16U), right_history(channels * taps + 16U);
        Image<float> left(channels + 16U), right(channels + 16U);
        std::ranges::fill(left.values(), canary);
        std::ranges::fill(right.values(), canary);
        for (std::size_t i = 0; i < left_history.values().size(); ++i)
            left_history.data()[i] = right_history.data()[i] = static_cast<float>(i % 31U) / 29.0F;
        for (std::size_t i = 0; i < weights.values().size(); ++i)
            weights.data()[i] = static_cast<float>(static_cast<int>(i % 37U) - 18) / 19.0F;
        for (std::size_t i = 0; i < delta.values().size(); ++i) delta.data()[i] = static_cast<float>(i % 11U) / 13.0F;
        for (std::size_t i = 0; i < input.values().size(); ++i) input.data()[i] = static_cast<float>(i % 17U) / 7.0F;
        const strix::core::ShortConvolution plan{channels, taps, dynamic ? heads : 0U, 1.0F, 1.25F};
        const decode::CausalConvolution head{plan, input.data(), weights.data(), dynamic ? delta.data() : nullptr,
                                            left_history.data(), left.data(), false};
        submit(head);
        submit(decode::Bf16Round{left.data(), channels});
        auto fused = head;
        fused.output = right.data();
        fused.history = right_history.data();
        fused.round_bf16 = true;
        submit(decode::RoundedCausalConvolution{fused});
        check(cudaDeviceSynchronize(), "complete convolution fixture");
        compare(left, right, "convolution output");
        compare(left_history, right_history, "convolution history");
    }
    std::cout << "EPI convolution-round PASS\n";
}

}  // namespace

int main() {
    try {
        dense(decode::Bf16DenseMode::fp32, graph::Form::store_bf16, 24U, "fp32-round");
        dense(decode::Bf16DenseMode::fp32, graph::Form::softcap, 41U, "fp32-softcap");
        dense(decode::Bf16DenseMode::mma3, graph::Form::store_bf16, 64U, "mma3-round");
        dense(decode::Bf16DenseMode::mma3, graph::Form::store_bf16, 64U, "mma3-group-round", 32U);
        convolution();
    } catch (const std::exception& error) {
        std::cerr << error.what() << '\n';
        return 1;
    }
}
