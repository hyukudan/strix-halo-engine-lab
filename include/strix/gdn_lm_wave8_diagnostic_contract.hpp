#pragma once

#include <cstdint>
#include <limits>
#include <stdexcept>

namespace strix::gdn_lm_wave8_diagnostic {
enum class Kind { gdn_bf16, lm_bf16 };
enum class Arm { reference, prefetch1, wave8 };
struct Shape { std::uint32_t width = 0, outputs = 0; };
struct Span { std::uintptr_t address = 0; std::uint64_t bytes = 0; };
struct Launch { std::uint32_t grid_x = 0, block_x = 0; };

// Diagnostic bounds cover the real GDN/head shapes and small tail fixtures.
// All outputs are FP32; gdn_bf16 rounds its result to BF16 before FP32 storage.
inline Launch validate(Shape shape, Span input, Span weights, Span output,
                       Kind kind, Arm arm) {
    if (!shape.width || shape.width > 10240U || shape.width % 32U ||
        !shape.outputs || shape.outputs > 248320U)
        throw std::invalid_argument("wave8 diagnostic shape outside admitted bounds");
    if (kind != Kind::gdn_bf16 && kind != Kind::lm_bf16)
        throw std::invalid_argument("wave8 diagnostic kind is invalid");
    if (arm != Arm::reference && arm != Arm::prefetch1 && arm != Arm::wave8)
        throw std::invalid_argument("wave8 diagnostic arm is invalid");
    const auto check = [](Span view, std::uint64_t required) {
        if (!view.address || view.address % 16U || view.bytes != required ||
            view.bytes > std::numeric_limits<std::uintptr_t>::max() - view.address)
            throw std::invalid_argument("wave8 diagnostic view has invalid alignment or extent");
    };
    check(input, static_cast<std::uint64_t>(shape.width) * 2U);
    check(weights, static_cast<std::uint64_t>(shape.width / 32U) * 34U * shape.outputs);
    check(output, static_cast<std::uint64_t>(shape.outputs) * 4U);
    const auto overlap = [](Span a, Span b) {
        return a.address < b.address + b.bytes && b.address < a.address + a.bytes;
    };
    if (overlap(output, input) || overlap(output, weights))
        throw std::invalid_argument("wave8 diagnostic output aliases a read-only view");
    const auto waves = arm == Arm::wave8 ? 8U : 1U;
    return {(shape.outputs + waves - 1U) / waves, 32U * waves};
}
} // namespace strix::gdn_lm_wave8_diagnostic
