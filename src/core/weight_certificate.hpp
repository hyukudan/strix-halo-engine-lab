#pragma once

#include <algorithm>
#include <cmath>
#include <bit>
#include <cstdint>
#include <cstddef>
#include <limits>
#include <memory_resource>
#include <new>
#include <span>
#include <vector>

namespace strix::core {

// Finite-weight certificate of the exact one-plane BF16 operand schedule (see exact_bf16_planes in
// core/dense_row_plan.hpp). The schedule is bit-identical to the three-plane split only while every product, partial
// sum and BF16 rounding stays finite: a non-finite weight meets the split's zero planes as 0 x inf, and an overflowing
// sum or rounding leaves the split's residual non-finite. The certificate is the largest weight magnitude (+infinity
// when any element is not finite), taken over the very bytes a backend stages; it is a host check made once at load.

// Largest |w| of an FP32 weight tensor, +infinity when an element is not finite.
inline float finite_magnitude_max(std::span<const float> values) {
    float top = 0.0F;
    for (const float value : values) {
        if (!std::isfinite(value)) return std::numeric_limits<float>::infinity();
        top = std::max(top, std::fabs(value));
    }
    return top;
}

// The same over a BF16 tensor (raw bits): an all-ones exponent field is infinity or NaN.
inline float finite_magnitude_max_bf16(std::span<const std::uint16_t> bits) {
    std::uint32_t top = 0U;
    for (const std::uint16_t value : bits) {
        if ((value & 0x7F80U) == 0x7F80U) return std::numeric_limits<float>::infinity();
        top = std::max<std::uint32_t>(top, value & 0x7FFFU);
    }
    // Magnitude bits are monotonic in the value, so the largest pattern is the largest magnitude.
    const std::uint32_t widened = top << 16U;
    return std::bit_cast<float>(widened);
}

// The same certificate over BF16 bytes the host cannot read in place (device-resident after upload): read in bounded
// chunks through ONE small scratch, so a certificate adds no allocation that scales with the tensors. Backends and models
// share this scan; a model only chooses which tensors to feed it and how to bound the result.
// Lack of scratch is a refusal, never an error: `available()` is false and every scan reports +infinity (the FP32
// class stays selected), while an exception thrown by `read` (a device copy failing) propagates unchanged.
class Bf16StagedScan {
public:
    static constexpr std::size_t kChunkElements = std::size_t{1} << 18U;  // 512 KiB of BF16

    explicit Bf16StagedScan(std::pmr::memory_resource* memory = std::pmr::get_default_resource(),
                            std::size_t chunk_elements = kChunkElements)
        : scratch_(memory) {
        try {
            scratch_.resize(chunk_elements);
        } catch (const std::bad_alloc&) {
            scratch_ = std::pmr::vector<std::uint16_t>(memory);
        }
    }
    [[nodiscard]] bool available() const { return !scratch_.empty(); }

    // Largest magnitude of `count` staged elements. `read(offset, host)` fills `host` with elements
    // [offset, offset + host.size()) of the tensor.
    template <class Read>
    [[nodiscard]] float magnitude_max(std::size_t count, Read&& read) {
        if (!available()) return std::numeric_limits<float>::infinity();
        float top = 0.0F;
        for (std::size_t offset = 0; offset < count; offset += scratch_.size()) {
            const auto host = std::span(scratch_).first(std::min(scratch_.size(), count - offset));
            read(offset, host);
            top = std::max(top, finite_magnitude_max_bf16(host));
            if (!std::isfinite(top)) break;
        }
        return top;
    }

private:
    std::pmr::vector<std::uint16_t> scratch_;
};

// Largest sum magnitude (also bound of every partial sum) must stay below 2^126 so that no product, partial sum or BF16
// rounding of either schedule leaves the finite range. `operand_l1_max` is the largest L1 norm of one operand vector
// (the caller derives it from its normalisation, and the operands are assumed finite); false for a non-finite bound.
inline bool one_plane_sum_bounded(float operand_l1_max, float weight_max) {
    constexpr float limit = 0x1p126F;
    return std::isfinite(operand_l1_max) && std::isfinite(weight_max) && operand_l1_max * weight_max < limit;
}

}  // namespace strix::core
