#include "emulator.hpp"
#include "gemma4/prefill/emulator.hpp"
#include <cfenv>

namespace strix::gemma4::exl3 {
std::vector<float> emulate_prefill(LinearView w, std::span<const float> input, unsigned rows) {
    validate(w);
    if (!rows || rows > prefill::chunk_rows || input.size() != std::size_t(rows) * w.inputs || std::fegetround() != FE_TONEAREST)
        throw std::invalid_argument("EXL3 prefill emulator geometry/RNE");
    std::vector<float> transformed(input.size()), out(std::size_t(rows) * w.outputs);
    input_transform(w, input.data(), rows, transformed.data());
    for (unsigned o = 0; o < w.outputs; ++o)
        for (unsigned r = 0; r < rows; ++r) {
            float sum = 0.0F;
            for (unsigned k0 = 0; k0 < w.inputs; k0 += 16) {
                std::array<std::uint16_t, 16> a{}, b{};
                for (unsigned e = 0; e < 16; ++e) {
                    (void)prefill::half_bits(transformed[std::size_t(r) * w.inputs + k0 + e], a[e]);
                    (void)prefill::half_bits(rotated_at(w, k0 + e, o), b[e]);
                }
                sum = prefill::instruction(a, b, sum);
            }
            out[std::size_t(r) * w.outputs + o] = sum;
        }
    output_transform(w, out.data(), rows);
    return out;
}
}
