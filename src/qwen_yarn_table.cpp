#include "strix/startup_markers.hpp"
#include "strix/qwen_yarn_table.hpp"
#include "strix/model_content_hash.hpp"
#include <algorithm>
#include <bit>
#include <cfenv>
#include <cmath>
#include <istream>
#include <limits>
#include <numbers>
#include <ostream>
#include <stdexcept>

namespace strix {
namespace {
static_assert(sizeof(float) == 4 && sizeof(double) == 8 &&
    std::numeric_limits<float>::is_iec559 && std::numeric_limits<double>::is_iec559);
void range(const RotaryDescriptor& d, std::uint32_t first, std::uint32_t end) {
    if (first > end || end > d.effective_max()) throw std::invalid_argument("YaRN row range exceeds effective maximum");
}
void rounding() {
    if (std::fegetround() != FE_TONEAREST) throw std::invalid_argument("YaRN generator requires RNE rounding");
}
YarnRow row(const RotaryDescriptor& d, const YarnFrequencies& f, std::uint32_t p) {
    YarnRow result{};
    const double a = d.attention_factor();
    for (std::size_t i = 0; i < result.size(); ++i) {
        const double angle = static_cast<double>(p) * f.omega[i];
        result[i] = {static_cast<float>(a * std::cos(angle)), static_cast<float>(a * std::sin(angle))};
    }
    return result;
}
void append(std::vector<std::byte>& bytes, std::uint64_t value, unsigned size) {
    for (unsigned i = 0; i < size; ++i) bytes.push_back(static_cast<std::byte>((value >> (8U*i)) & 255U));
}
std::vector<std::byte> encode(std::span<const YarnRow> rows) {
    std::vector<std::byte> bytes;
    bytes.reserve(rows.size() * 256U);
    for (const auto& r : rows) for (const auto& pair : r) {
        append(bytes, std::bit_cast<std::uint32_t>(pair.cosine), 4);
        append(bytes, std::bit_cast<std::uint32_t>(pair.sine), 4);
    }
    return bytes;
}
std::vector<std::byte> header(const RotaryDescriptor& d, std::uint32_t first, std::uint32_t end) {
    std::vector<std::byte> bytes;
    for (const char c : std::string_view("STYARN01")) bytes.push_back(static_cast<std::byte>(c));
    append(bytes, 1, 4); // schema
    append(bytes, 0x01020304U, 4); // explicit little-endian marker
    const auto descriptor = d.canonical_bytes();
    append(bytes, descriptor.size(), 4);
    append(bytes, first, 4);
    append(bytes, end - first, 4);
    append(bytes, static_cast<std::uint64_t>(end - first) * 256U, 8);
    bytes.insert(bytes.end(), descriptor.begin(), descriptor.end());
    return bytes;
}
void write(std::ostream& out, std::span<const std::byte> bytes) {
    out.write(reinterpret_cast<const char*>(bytes.data()), static_cast<std::streamsize>(bytes.size()));
    if (!out) throw std::runtime_error("YaRN table write failed");
}
void read(std::istream& in, std::span<std::byte> bytes) {
    in.read(reinterpret_cast<char*>(bytes.data()), static_cast<std::streamsize>(bytes.size()));
    if (!in) throw std::invalid_argument("truncated YaRN table");
}
}
YarnFrequencies qwen_yarn_frequencies(const RotaryDescriptor& d) {
    if (d.mode() != RotaryMode::yarn) throw std::invalid_argument("native mode has no host rotary frequencies");
    rounding();
    const auto correction = [](double beta) {
        return 64.0 * std::log(262144.0 / (2.0 * std::numbers::pi * beta)) / (2.0 * std::log(10000000.0));
    };
    YarnFrequencies result{};
    result.low = std::max(0, static_cast<int>(std::floor(correction(d.beta_fast))));
    result.high = std::min(63, static_cast<int>(std::ceil(correction(d.beta_slow))));
    const double width = result.low == result.high ? 0.001 : static_cast<double>(result.high - result.low);
    for (std::size_t i = 0; i < result.omega.size(); ++i) {
        const double ramp = std::clamp((static_cast<double>(i) - result.low) / width, 0.0, 1.0);
        const double u = std::pow(d.theta, -2.0 * static_cast<double>(i) / 64.0);
        result.omega[i] = u * (1.0 - ramp) + (u / d.factor()) * ramp;
    }
    return result;
}
YarnRow qwen_yarn_row(const RotaryDescriptor& d, std::uint32_t p) {
    if (d.mode() != RotaryMode::yarn) throw std::invalid_argument("native mode generates no rotary table");
    if (p >= d.effective_max()) throw std::invalid_argument("YaRN position exceeds effective maximum");
    return row(d, qwen_yarn_frequencies(d), p);
}
void generate_qwen_yarn_rows(const RotaryDescriptor& d, std::uint32_t first,
    std::uint32_t end, const YarnRowsConsumer& consume, std::size_t chunk_rows) {
    if (d.mode() == RotaryMode::native) return;
    range(d, first, end);
    if (!chunk_rows || chunk_rows > 4096) throw std::invalid_argument("YaRN staging requires 1..4096 rows");
    const auto frequencies = qwen_yarn_frequencies(d);
    std::vector<YarnRow> rows(std::min<std::size_t>(chunk_rows, end - first));
    while (first < end) {
        const auto count = static_cast<std::uint32_t>(std::min<std::size_t>(rows.size(), end - first));
        { startup::Phase marker("rotary_generate", first);
          for (std::uint32_t i = 0; i < count; ++i) rows[i] = row(d, frequencies, first + i); }
        consume(first, std::span<const YarnRow>(rows.data(), count));
        first += count;
    }
}
void write_qwen_yarn_table(std::ostream& out, const RotaryDescriptor& d,
    std::uint32_t first, std::uint32_t end, std::size_t chunk_rows) {
    if (d.mode() == RotaryMode::native) return;
    Sha256Fast hash;
    generate_qwen_yarn_rows(d, first, end, [&](auto, auto rows) { hash.update(encode(rows)); }, chunk_rows);
    const auto digest = hash.finalize();
    write(out, header(d, first, end));
    write(out, std::as_bytes(std::span(digest.bytes)));
    generate_qwen_yarn_rows(d, first, end, [&](auto, auto rows) { write(out, encode(rows)); }, chunk_rows);
}
void validate_qwen_yarn_table(std::istream& in, const RotaryDescriptor& d,
    std::uint32_t first, std::uint32_t end) {
    if (d.mode() == RotaryMode::native) {
        if (in.peek() != std::char_traits<char>::eof()) throw std::invalid_argument("native mode has no table");
        return;
    }
    range(d, first, end);
    const auto expected = header(d, first, end);
    std::vector<std::byte> actual(expected.size());
    read(in, actual);
    if (actual != expected) throw std::invalid_argument("YaRN table header/descriptor/endianness mismatch");
    ModelContentIdentity digest;
    read(in, std::as_writable_bytes(std::span(digest.bytes)));
    Sha256Fast hash;
    std::array<std::byte, 65536> buffer{};
    std::uint64_t remaining = static_cast<std::uint64_t>(end - first) * 256U;
    while (remaining) {
        const auto count = static_cast<std::size_t>(std::min<std::uint64_t>(remaining, buffer.size()));
        const auto bytes = std::span(buffer).first(count);
        read(in, bytes);
        for (std::size_t i = 0; i < count; i += 4) {
            // Both NaN and infinity have an all-ones binary32 exponent.
            const auto upper = std::to_integer<unsigned>(bytes[i+3]);
            const auto lower = std::to_integer<unsigned>(bytes[i+2]);
            if ((upper & 0x7fU) == 0x7fU && (lower & 0x80U))
                throw std::invalid_argument("non-finite YaRN table coefficient");
        }
        hash.update(bytes); remaining -= count;
    }
    if (hash.finalize() != digest) throw std::invalid_argument("YaRN table SHA256 mismatch");
    if (in.peek() != std::char_traits<char>::eof()) throw std::invalid_argument("YaRN table has trailing bytes");
}
} // namespace strix
