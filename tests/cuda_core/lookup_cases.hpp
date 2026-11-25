#pragma once

#include "core/lookup_plan.hpp"

#include <array>
#include <bit>
#include <cmath>
#include <cstring>
#include <limits>
#include <stdexcept>
#include <vector>

namespace lookup_test {

namespace lookup = strix::cuda::core::lookup;
namespace exec = strix::exec;
using lookup::Form;

inline constexpr std::uint32_t kVocabulary = 32U;
inline constexpr std::array<std::uint32_t, 7> kIds{31U, 0U, 8U, 3U, 17U, 8U, 16U};
inline constexpr auto kRows = static_cast<std::uint32_t>(kIds.size());
inline constexpr float kTensorScale = 0.137F;

inline void require(bool condition, const char* message) {
    if (!condition) throw std::runtime_error(message);
}

template <class T>
void equal_bytes(const std::vector<T>& got, const std::vector<T>& expected, const char* message) {
    require(got.size() == expected.size() &&
            std::memcmp(got.data(), expected.data(), got.size() * sizeof(T)) == 0, message);
}

template <class T>
T sentinel() {
    if constexpr (sizeof(T) == 4U) return std::bit_cast<T>(0x5a5a5a5aU);
    else return static_cast<T>(0x5a5aU);
}

inline exec::StepBlock make_step(const std::array<std::uint32_t, kRows>& ids) {
    exec::StepBlock step{};
    step.h.rows = kRows - 1U;
    step.h.valid_mask = 0x7fU ^ (1U << 1U);
    for (std::uint32_t i = 0; i < kRows; ++i) {
        step.row[i].token = ids[i];
        step.row[i].position = 400U + i;
        step.row[i].slot = kRows - i;
        step.row[i].flags = i == 3U ? 0U : static_cast<std::uint32_t>(exec::step_valid);
    }
    step.row[1].token = step.row[3].token = step.row[6].token = 0xffffffffU;
    return step;
}

inline bool active(std::uint32_t row) { return row != 1U && row != 3U && row != 6U; }

// Independent scalar reference for the host and fixture. The native CUDA
// half widening uses the canonical all-mantissa-ones FP32 NaN.
inline float half(std::uint16_t bits, bool payload = false) {
    const auto mantissa = bits & 1023U;
    const auto exponent = (bits >> 10U) & 31U;
    const bool negative = (bits & 0x8000U) != 0U;
    if (exponent == 31U && mantissa != 0U) {
        const auto sign = negative ? 0x80000000U : 0U;
        return std::bit_cast<float>(payload ? sign | 0x7f800000U | (mantissa << 13U) : 0x7fffffffU);
    }
    const float value = exponent == 31U ? std::numeric_limits<float>::infinity() :
        exponent == 0U ? std::ldexp(static_cast<float>(mantissa), -24) :
        std::ldexp(static_cast<float>(1024U + mantissa), static_cast<int>(exponent) - 25);
    return negative ? -value : value;
}

inline std::uint16_t bf16(float value, bool canonical_nan) {
    if (canonical_nan && std::isnan(value)) return 0x7fc0U;
    const auto bits = std::bit_cast<std::uint32_t>(value);
    const auto top = bits >> 16U, low = bits & 0xffffU;
    return static_cast<std::uint16_t>(top + (low > 0x8000U || (low == 0x8000U && (top & 1U))));
}

inline std::vector<std::uint16_t> plain16(std::uint32_t width, bool f16) {
    constexpr std::array<std::uint16_t, 11> b{0x0000, 0x8000, 0x0001, 0xff80, 0x7fa1, 0xffc2, 0x3f80,
                                             0x3f81, 0x7f7f, 0x807f, 0xbf90};
    constexpr std::array<std::uint16_t, 11> h{0x0000, 0x8000, 0x0001, 0xfc00, 0x7d45, 0xfe23, 0x3c00,
                                             0x3c04, 0x7bff, 0x83ff, 0xbc74};
    std::vector<std::uint16_t> result(std::size_t{kVocabulary} * width);
    for (std::uint32_t r = 0; r < kVocabulary; ++r)
        for (std::uint32_t i = 0; i < width; ++i)
            result[std::size_t{r} * width + i] = (f16 ? h : b)[(i + r * 7U + i / b.size()) % b.size()];
    return result;
}

inline std::vector<std::uint8_t> q8(std::uint32_t width) {
    constexpr std::array<std::uint16_t, 7> scales{0x3c04, 0x3c0c, 0x0001, 0x8000, 0x7bff, 0xbc00, 0x3555};
    std::vector<std::uint8_t> result(std::size_t{kVocabulary} * (width / 32U) * 34U);
    for (std::size_t block = 0; block < result.size() / 34U; ++block) {
        auto* bytes = result.data() + 34U * block;
        const auto scale = scales[block % scales.size()];
        bytes[0] = static_cast<std::uint8_t>(scale);
        bytes[1] = static_cast<std::uint8_t>(scale >> 8U);
        for (unsigned i = 0; i < 32U; ++i) bytes[2U + i] = static_cast<std::uint8_t>(block * 31U + i * 59U);
        bytes[2] = 1U;
        bytes[3] = 0xffU;
    }
    return result;
}

// Build row-major expected NVFP4 values while writing records in physical
// [tile][half][lane][group][pair] order, independently of the gather offsets.
struct TileData { std::vector<std::uint8_t> bytes; std::vector<float> values; };

inline TileData tile_data(std::uint32_t width) {
    constexpr std::array<float, 8> magnitudes{0.0F, 0.5F, 1.0F, 1.5F, 2.0F, 3.0F, 4.0F, 6.0F};
    constexpr std::array<std::uint8_t, 8> scales{0x38U, 0x01U, 0x31U, 0x7eU, 0x00U, 0x42U, 0x08U, 0x24U};
    TileData data;
    data.values.resize(std::size_t{kVocabulary} * width);
    for (unsigned tile = 0; tile < kVocabulary / 16U; ++tile) {
        for (unsigned chunk = 0; chunk < width / 128U; ++chunk) {
            std::array<std::uint8_t, 1152> record{};
            std::size_t at = 0;
            for (unsigned h = 0; h < 2U; ++h)
                for (unsigned lane = 0; lane < 32U; ++lane)
                    for (unsigned group = 0; group < 8U; ++group) {
                        const auto row = tile * 16U + h * 8U + lane / 4U;
                        const auto scale = scales[(row + chunk * 3U + group) % scales.size()];
                        record[1024U + (lane / 4U) * 16U + h * 8U + group] = scale;
                        const auto mantissa = scale % 8U, exponent = scale / 8U;
                        const float gain = exponent == 0U ? std::ldexp(static_cast<float>(mantissa), -9) :
                            std::ldexp(static_cast<float>(8U + mantissa), static_cast<int>(exponent) - 10);
                        for (unsigned pair = 0; pair < 2U; ++pair, ++at) {
                            for (unsigned nibble = 0; nibble < 2U; ++nibble) {
                                const auto col = chunk * 128U + group * 16U + (lane % 4U) * 2U + pair * 8U + nibble;
                                const auto code = (row * 5U + col * 7U + col / 16U + col / 128U) % 16U;
                                record[at] |= static_cast<std::uint8_t>(code << (nibble * 4U));
                                const auto m = magnitudes[code % 8U];
                                data.values[std::size_t{row} * width + col] = ((code >= 8U ? -m : m) * gain) * kTensorScale;
                            }
                        }
                    }
            data.bytes.insert(data.bytes.end(), record.begin(), record.end());
        }
    }
    return data;
}

inline float plain_value(Form form, const void* table, std::size_t i) {
    if (form == Form::f32) return static_cast<const float*>(table)[i];
    const auto bits = static_cast<const std::uint16_t*>(table)[i];
    return form == Form::bf16_to_f32 ? std::bit_cast<float>(std::uint32_t{bits} << 16U) :
        half(bits, form == Form::f16_payload_to_f32);
}

inline float q8_value(const void* table, std::size_t i) {
    const auto* p = static_cast<const std::uint8_t*>(table) + 34U * (i / 32U);
    const auto bits = static_cast<std::uint16_t>(p[0] + 256U * p[1]);
    const int code = p[2U + i % 32U];
    const auto scale = half(bits);
    if (std::isnan(scale) || (std::isinf(scale) && code == 0)) return std::bit_cast<float>(0x7fffffffU);
    return scale * static_cast<float>(code >= 128 ? code - 256 : code);
}

template <class T>
std::vector<T> reference(Form form, const void* table, std::uint32_t width, std::uint32_t copies,
                         const std::vector<std::uint32_t>& ids, bool use_step = false) {
    std::vector<T> result(ids.size() * width * copies, sentinel<T>());
    for (std::size_t r = 0; r < ids.size(); ++r) {
        if (use_step && !active(static_cast<std::uint32_t>(r))) continue;
        auto id = ids[r];
        if (id >= kVocabulary) {
            if (form != Form::q8_0_to_bf16_row_zero) continue;
            id = 0U;
        }
        for (std::uint32_t copy = 0; copy < copies; ++copy) {
            for (std::uint32_t col = 0; col < width; ++col) {
                const auto input = std::size_t{id} * width + col, output = (r * copies + copy) * width + col;
                if constexpr (sizeof(T) == 2U) {
                    result[output] = bf16(q8_value(table, input), form == Form::q8_0_to_bf16);
                } else {
                    result[output] = form == Form::q8_0_to_f32 ? q8_value(table, input) : plain_value(form, table, input);
                }
            }
        }
    }
    return result;
}

}  // namespace lookup_test
