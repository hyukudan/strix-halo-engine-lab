#include "strix/ple_lookup.hpp"

#include <array>
#include <bit>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <cstring>
#include <iostream>
#include <limits>
#include <span>
#include <stdexcept>
#include <vector>

namespace {
int failures = 0;
void expect(bool condition, const char* message) {
    if (!condition) { std::cerr << "FAIL: " << message << '\n'; ++failures; }
}

// Mathematical binary16 and nearest-even BF16 references: no production
// dequantizer or bit-add rounding helper supplies the expected values.
double reference_scale(std::uint16_t bits) {
    const int exponent = (bits >> 10U) & 31U;
    const int fraction = bits & 1023U;
    const double magnitude = exponent == 0
        ? std::ldexp(static_cast<double>(fraction), -24)
        : std::ldexp(static_cast<double>(1024 + fraction), exponent - 25);
    return std::copysign(magnitude, (bits & 0x8000U) ? -1.0 : 1.0);
}
std::uint16_t reference_bf16(double value) {
    if (value == 0) return std::signbit(value) ? 0x8000U : 0U;
    int exponent = 0;
    const double scaled = std::frexp(std::fabs(value), &exponent) * 256.0;
    auto integer = static_cast<unsigned int>(std::floor(scaled));
    const double fraction = scaled - static_cast<double>(integer);
    if (fraction > 0.5 || (fraction == 0.5 && (integer & 1U))) ++integer;
    const auto rounded = static_cast<float>(std::copysign(
        std::ldexp(static_cast<double>(integer), exponent - 8), value));
    return static_cast<std::uint16_t>(std::bit_cast<std::uint32_t>(rounded) >> 16U);
}
constexpr std::array<int, 16> values{-127, -104, -83, -65, -49, -35, -22, -10,
                                    1, 13, 25, 38, 53, 69, 89, 113};
void write_block(std::byte* block, std::uint16_t scale, unsigned int rotation = 0) {
    block[0] = static_cast<std::byte>(scale & 255U);
    block[1] = static_cast<std::byte>(scale >> 8U);
    for (unsigned int i = 0; i < 16; ++i) {
        block[2 + i] = static_cast<std::byte>(
            ((i + rotation) % 16U) | (((15U - i + rotation) % 16U) << 4U));
    }
}
strix::ManifestTensor manifest(std::uint64_t width, std::uint64_t rows, std::uint64_t bytes) {
    strix::ManifestTensor result;
    result.name = "per_layer_token_embd.weight";
    result.role = strix::TensorRole::ngram_table;
    result.ggml_type = 20;
    result.dimensions = {width, rows};
    result.byte_size = bytes;
    return result;
}
strix::RuntimeTensorBinding mapping(std::vector<std::byte>& bytes) {
    strix::RuntimeTensorBinding result;
    result.storage = strix::RuntimeTensorStorage::file_mapped;
    result.address = reinterpret_cast<std::uintptr_t>(bytes.data());
    result.bytes = bytes.size();
    result.allocation_generation = 1;
    return result;
}
template<class Exception, class Action> bool rejects(Action action) {
    try { action(); } catch (const Exception&) { return true; }
    return false;
}
}

int main() {
    {
        // Every finite binary16 scale, every codebook entry and both nibble
        // planes, including signed zero, subnormals and BF16 ties/carries.
        std::vector<std::byte> table(65536U * 18U);
        for (unsigned int bits = 0; bits < 65536U; ++bits)
            write_block(table.data() + bits * 18U, static_cast<std::uint16_t>(bits));
        auto tensor = manifest(32, 65536, table.size());
        auto runtime = mapping(table);
        strix::FileMappedPleLookupBackend backend;
        std::array<std::uint64_t, 1> row{};
        std::array<std::uint16_t, 32> output{};
        strix::PleLookupRequest request{row, 64, &tensor, &runtime};
        bool all_equal = true;
        for (unsigned int bits = 0; bits < 65536U; ++bits) {
            if ((bits & 0x7c00U) == 0x7c00U) continue;
            row[0] = bits;
            const auto handle = backend.start(request);
            backend.finish(handle, std::as_writable_bytes(std::span(output)));
            for (unsigned int element = 0; element < 32U; ++element) {
                const auto code = element < 16U ? element : 31U - element;
                const auto expected = reference_bf16(reference_scale(static_cast<std::uint16_t>(bits)) * values[code]);
                all_equal = all_equal && output[element] == expected;
            }
        }
        expect(all_equal, "all finite IQ4_NL scales/codes match mathematical BF16 nearest-even");
        for (const auto bits : {0x7c00U, 0x7e00U, 0xfc00U, 0xfe01U}) {
            row[0] = bits;
            const auto handle = backend.start(request);
            expect(rejects<std::invalid_argument>([&] {
                backend.finish(handle, std::as_writable_bytes(std::span(output)));
            }), "nonfinite IQ4_NL scales fail closed");
            expect(rejects<std::logic_error>([&] { (void)backend.start(request); }),
                   "failed decode retains its active handle until cancellation");
            backend.cancel(handle);
        }
    }
    {
        constexpr std::uint64_t width = 160, rows = 3, stored_row = 90, output_row = 320;
        std::vector<std::byte> table(rows * stored_row);
        constexpr std::array<std::uint16_t, 5> scales{0x3c01, 0xbc01, 0x0001, 0x7bff, 0x3555};
        for (unsigned int row = 0; row < rows; ++row)
            for (unsigned int block = 0; block < 5; ++block)
                write_block(table.data() + row * stored_row + block * 18, scales[block], row);
        auto tensor = manifest(width, rows, table.size());
        auto runtime = mapping(table);
        std::array<std::uint64_t, 3> order{2, 0, 2};
        strix::PleLookupRequest request{order, output_row, &tensor, &runtime};
        strix::FileMappedPleLookupBackend backend;
        std::vector<std::byte> guarded(order.size() * output_row + 2, std::byte{0x5a});
        const auto handle = backend.start(request);
        auto destination = std::span(guarded).subspan(1, order.size() * output_row);
        expect(rejects<std::invalid_argument>([&] { backend.finish(handle, destination.first(destination.size() - 1)); }),
               "undersized BF16 destination rejected before gather");
        backend.finish(handle, destination);
        bool equal = true;
        for (unsigned int index = 0; index < order.size(); ++index) {
            for (unsigned int column = 0; column < width; ++column) {
                const auto element = column % 32U;
                const auto code = ((element < 16U ? element : 31U - element) + order[index]) % 16U;
                std::uint16_t word = 0;
                std::memcpy(&word, destination.data() + index * output_row + column * 2, sizeof(word));
                equal = equal && word == reference_bf16(reference_scale(scales[column / 32U]) * values[code]);
            }
        }
        expect(equal, "five-block rows use compressed stride, order and duplicate row IDs");
        expect(guarded.front() == std::byte{0x5a} && guarded.back() == std::byte{0x5a},
               "unaligned byte destination has intact guard bytes");
        auto invalid = tensor;
        request.table_manifest = &invalid;
        invalid.dimensions[0] = 159; request.row_bytes = 318;
        expect(rejects<std::invalid_argument>([&] { (void)backend.start(request); }), "partial IQ4_NL block rejected");
        invalid = tensor; request.row_bytes = stored_row;
        expect(rejects<std::invalid_argument>([&] { (void)backend.start(request); }), "stored bytes cannot replace activation bytes");
        request.row_bytes = output_row; invalid.ggml_type = 8;
        expect(rejects<std::invalid_argument>([&] { (void)backend.start(request); }), "unsupported storage encoding rejected");
        invalid = tensor; invalid.dimensions.push_back(2);
        expect(rejects<std::invalid_argument>([&] { (void)backend.start(request); }), "non-row geometry rejected");
        invalid = tensor; invalid.byte_size -= 1;
        expect(rejects<std::invalid_argument>([&] { (void)backend.start(request); }), "truncated manifest payload rejected");
        invalid = tensor; runtime.bytes -= 1;
        expect(rejects<std::invalid_argument>([&] { (void)backend.start(request); }), "truncated mapping rejected");
        runtime.bytes += 1; order[0] = rows;
        expect(rejects<std::out_of_range>([&] { (void)backend.start(request); }), "out-of-range row rejected");
        order[0] = 2; runtime.address = std::numeric_limits<std::uintptr_t>::max() - 1;
        expect(rejects<std::overflow_error>([&] { (void)backend.start(request); }), "host pointer range overflow rejected");
    }
    if (!failures) std::cout << "IQ4_NL PLE mathematical and lifecycle checks passed\n";
    return failures ? 1 : 0;
}
