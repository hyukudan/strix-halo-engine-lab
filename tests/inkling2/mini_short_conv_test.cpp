#include "core/short_convolution.hpp"

#include <array>
#include <cmath>
#include <cstddef>
#include <span>
#include <stdexcept>

int main() {
    using strix::core::ShortConvolution;
    using strix::core::causal_short_convolution;

    constexpr std::size_t channels = 4;
    constexpr std::size_t taps = 4;
    constexpr std::size_t groups = 2;
    constexpr std::size_t rows = 5;
    const std::array<float, rows * channels> input{
         1.0F,  2.0F,  3.0F,  4.0F,
         5.0F,  6.0F,  7.0F,  8.0F,
         9.0F, 10.0F, 11.0F, 12.0F,
        13.0F, 14.0F, 15.0F, 16.0F,
        17.0F, 18.0F, 19.0F, 20.0F,
    };
    const std::array<float, taps * channels> weights{
         2.0F,  3.0F,  4.0F,  5.0F,
         6.0F,  7.0F,  8.0F,  9.0F,
        10.0F, 11.0F, 12.0F, 13.0F,
        14.0F, 15.0F, 16.0F, 17.0F,
    };
    const std::array<float, rows * groups * taps> dynamic_weights{
         1.0F, 0.0F, 0.0F, 0.0F,  -2.0F,  0.0F, 0.0F, 0.0F,
        -2.0F, 3.0F, 0.0F, 0.0F,   2.0F, -1.0F, 0.0F, 0.0F,
         0.0F, 0.0F, 0.0F, 0.0F,   0.0F,  0.0F, 0.0F, 0.0F,
         0.0F, 0.0F, 0.0F, 0.0F,   0.0F,  0.0F, 0.0F, 0.0F,
         0.0F, 0.0F, 0.0F, 1.0F,   0.0F,  0.0F, 0.0F, 2.0F,
    };
    using History = std::array<float, (taps - 1) * channels>;
    constexpr History final_history{
         9.0F, 10.0F, 11.0F, 12.0F,
        13.0F, 14.0F, 15.0F, 16.0F,
        17.0F, 18.0F, 19.0F, 20.0F,
    };

    const float static_divisor = std::sqrt(5.0F);
    const ShortConvolution static_plan{channels, taps, 0, 1.0F, static_divisor};
    History static_history{};
    std::array<float, rows * channels> static_out{};
    causal_short_convolution(static_plan, input, weights, {}, static_history, static_out);
    if (static_out[0] != 3.0F / static_divisor || static_out[3] != 24.0F / static_divisor ||
        static_out[4] != 21.0F / static_divisor || static_out[12] != 157.0F / static_divisor ||
        static_out[19] != 556.0F / static_divisor || static_history != final_history) return 1;

    const ShortConvolution dynamic_plan{channels, taps, groups, 1.0F, std::sqrt(9.0F)};
    History dynamic_history{};
    std::array<float, rows * channels> dynamic_out{};
    causal_short_convolution(dynamic_plan, input, weights, dynamic_weights, dynamic_history, dynamic_out);
    if (dynamic_out[0] != 4.0F / 3.0F || dynamic_out[1] != 10.0F / 3.0F ||
        dynamic_out[2] != 9.0F / 3.0F || dynamic_out[3] != 16.0F / 3.0F ||
        dynamic_out[4] != 14.0F / 3.0F || dynamic_out[5] != 32.0F / 3.0F ||
        dynamic_out[6] != 70.0F / 3.0F || dynamic_out[7] != 96.0F / 3.0F ||
        dynamic_out[19] != 572.0F / 3.0F || dynamic_history != final_history) return 2;

    // Two prompt rows, two more prompt rows, one decode row. Replay the suffix
    // from the original prefix to exercise the contents and ordering of the state.
    History chunked_history{};
    std::array<float, rows * channels> chunked_out{};
    const auto run = [&](std::size_t begin, std::size_t count) {
        causal_short_convolution(
            dynamic_plan, std::span(input).subspan(begin * channels, count * channels), weights,
            std::span(dynamic_weights).subspan(begin * groups * taps, count * groups * taps), chunked_history,
            std::span(chunked_out).subspan(begin * channels, count * channels));
    };
    run(0, 2);
    const History prefix_state = chunked_history;
    run(2, 2);
    run(4, 1);
    if (chunked_out != dynamic_out || chunked_history != dynamic_history) return 3;
    chunked_history = prefix_state;
    run(2, 3);
    if (chunked_out != dynamic_out || chunked_history != dynamic_history) return 4;

    try {
        const ShortConvolution invalid{channels, taps, 3};
        causal_short_convolution(invalid, input, weights, dynamic_weights, chunked_history, chunked_out);
        return 5;
    } catch (const std::invalid_argument&) {
        if (chunked_out != dynamic_out || chunked_history != dynamic_history) return 6;
    }

    const ShortConvolution single_tap{1, 1};
    const std::array<float, 2> tiny_input{-2.0F, 3.0F};
    const std::array<float, 1> tiny_weights{2.0F};
    std::array<float, 2> tiny_out{};
    causal_short_convolution(single_tap, tiny_input, tiny_weights, {}, {}, tiny_out);
    return tiny_out == std::array{-4.0F, 6.0F} ? 0 : 7;
}
