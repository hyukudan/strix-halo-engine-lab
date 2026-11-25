// strix.sp4t end to end on the host: the calibrated pack carries SP4 entries with their tensor header (round trip,
// identity, refusals), the resident plan hands the base to the consumers as the outer exponent, and the device
// operand of the small-M schedule (core/f16_tile_gemm_small.hpp Sp4Operand) decodes to the host reference weights:
// FP16 table operand x FP32 group multiplier x 2^base == sp::decode, for every code, scale byte and accepted base.
#include "core/lowbit_f16_staging.hpp"
#include "strix/weight/host_lowbit.hpp"
#include "strix/weight/host_lowbit_gptq.hpp"
#include "strix/weight/host_q8.hpp"
#include "strix/weight/lowbit_pack.hpp"
#include "strix/weight/resident_slices.hpp"

#include <cmath>
#include <cstdint>
#include <cstring>
#include <filesystem>
#include <fstream>
#include <iostream>
#include <random>
#include <span>
#include <string>
#include <vector>

namespace {

namespace wei = strix::weight;
namespace sp = strix::weight::sp;
namespace lf = strix::core::lowbit_f16;
using S = sp::SP4Storage;

int failures = 0;

void check(bool ok, const char* what) {
    if (ok) return;
    ++failures;
    std::cerr << "FAIL: " << what << '\n';
}

// v_perm_b32 on the host: byte i of the result from selector byte i (0..3 = s1 bytes, 4..7 = s0 bytes, 12 = 0x00).
std::uint32_t perm(std::uint32_t s0, std::uint32_t s1, std::uint32_t sel) {
    const std::uint64_t pool = (std::uint64_t{s0} << 32U) | s1;
    std::uint32_t out = 0;
    for (unsigned i = 0; i < 4U; ++i) {
        const unsigned s = (sel >> (8U * i)) & 0xFFU;
        const unsigned byte = s < 8U ? static_cast<unsigned>((pool >> (8U * s)) & 0xFFU) : s == 12U ? 0U : 0xFFU;
        out |= static_cast<std::uint32_t>(byte) << (8U * i);
    }
    return out;
}

// Host copy of lowbit_f16::sign_index_pairs (same selectors).
void sign_index_pairs(const lf::SignIndexTable& t, std::uint32_t w, std::uint32_t* out) {
    const std::uint32_t even = w & 0x0F0F0F0FU, odd = (w >> 4U) & 0x0F0F0F0FU;
    for (unsigned j = 0; j < 4U; ++j) {
        const std::uint32_t x = perm(odd, even, ((4U + j) << 24U) | 0x000C0000U | (j << 8U) | 0x0CU);
        std::uint32_t magnitude = perm(t.hi_hi, t.hi_lo, (x & 0x07000700U) | 0x000C000CU);
        if ((t.lo_lo | t.lo_hi) != 0U) magnitude |= perm(t.lo_hi, t.lo_lo, ((x >> 8U) & 0x00070007U) | 0x0C000C00U);
        out[j] = magnitude | ((x & 0x08000800U) << 4U);
    }
}

float f16(std::uint16_t bits) { return strix::core::onfly::f16_bits_to_f32(bits); }

// Sp4Operand::scale: (byte + 127 * 8) << 20 as FP32 bits.
float group_multiplier(std::uint8_t byte) {
    const std::uint32_t bits = (std::uint32_t{byte} + 127U * 8U) << 20U;
    float v;
    std::memcpy(&v, &bits, sizeof v);
    return v;
}

void operand_matches_reference() {
    constexpr auto table = lf::sign_index_table(sp::default_table<4>());
    // Every pair of codes in one byte, through the device selectors, gives the exact FP16 of +-table[index].
    for (unsigned lo = 0; lo < 16U; ++lo)
        for (unsigned hi = 0; hi < 16U; ++hi) {
            const std::uint32_t byte = lo | (hi << 4U);
            const std::uint32_t word = byte | (byte << 8U) | (byte << 16U) | (byte << 24U);
            std::uint32_t pairs[4];
            sign_index_pairs(table, word, pairs);
            for (unsigned j = 0; j < 4U; ++j)
                for (unsigned half = 0; half < 2U; ++half) {
                    const unsigned code = half == 0U ? lo : hi;
                    const float want = (code & 8U ? -1.0F : 1.0F) * static_cast<float>(sp::default_table<4>()[code & 7U]);
                    const float got = f16(static_cast<std::uint16_t>(pairs[j] >> (16U * half)));
                    if (std::memcmp(&got, &want, sizeof got) != 0) {
                        check(false, "sign-index operand differs from the table value");
                        return;
                    }
                }
        }
    // Group multiplier x 2^base == the reference scale value, every byte, every accepted base.
    for (int base = sp::kMinBase; base <= sp::kMaxBase; ++base)
        for (unsigned b = 0; b < 256U; ++b) {
            const float got = group_multiplier(static_cast<std::uint8_t>(b)) * std::ldexp(1.0F, base);
            const float want = sp::scale_value(static_cast<std::uint8_t>(b), base);
            if (std::memcmp(&got, &want, sizeof got) != 0) {
                check(false, "group multiplier x 2^base differs from sp::scale_value");
                return;
            }
        }
}

struct Coded {
    std::vector<std::uint8_t> bf16, bytes;
    std::shared_ptr<const wei::Source> source;
    wei::RawTensor raw;
    sp::Header header;
};

Coded make(std::uint64_t rows, std::uint64_t columns, std::uint32_t seed) {
    Coded c;
    std::mt19937 rng(seed);
    std::normal_distribution<float> dist(0.0F, 0.02F);
    std::vector<float> w(static_cast<std::size_t>(rows * columns));
    for (auto& v : w) {
        v = dist(rng);
        std::uint32_t bits;
        std::memcpy(&bits, &v, sizeof bits);
        bits &= 0xFFFF0000U;  // exact BF16 source
        std::memcpy(&v, &bits, sizeof v);
    }
    c.bf16.resize(w.size() * 2U);
    for (std::size_t i = 0; i < w.size(); ++i) {
        std::uint32_t bits;
        std::memcpy(&bits, &w[i], sizeof bits);
        c.bf16[2U * i] = static_cast<std::uint8_t>((bits >> 16U) & 0xFFU);
        c.bf16[2U * i + 1U] = static_cast<std::uint8_t>(bits >> 24U);
    }
    c.source = wei::memory_source(std::as_bytes(std::span(c.bf16)));
    c.raw.wire_dtype = "BF16";
    c.raw.shape.rank = 2;
    c.raw.shape.dims = {rows, columns, 0, 0};
    c.raw.payload = {c.source, {0, c.bf16.size()}};
    c.header = sp::make_header<S>(w.data(), rows, columns, sp::default_table<4>());
    c.bytes.resize(static_cast<std::size_t>(rows * S::row_bytes(columns)));
    check(static_cast<bool>(wei::lowbit::gptq::encode_rows_sp<S>(c.header, w.data(), rows, columns, c.bytes.data())),
          "encode SP4");
    sp::mark_foldable<S>(c.header, c.bytes.data(), rows, columns);
    return c;
}

wei::Digest256 fingerprint(const wei::RawTensor& raw) {
    wei::Digest256 d{};
    check(static_cast<bool>(wei::digest_slice(raw.payload, d)), "fingerprint");
    return d;
}

std::vector<char> slurp(const std::string& path) {
    std::ifstream in(path, std::ios::binary);
    return {std::istreambuf_iterator<char>(in), {}};
}
std::string spill(const std::string& path, const std::vector<char>& bytes) {
    std::ofstream(path, std::ios::binary | std::ios::trunc).write(bytes.data(), static_cast<std::streamsize>(bytes.size()));
    return path;
}

void pack_round_trip(const std::string& dir) {
    auto head = make(4, 256, 7);
    const wei::Digest256 checkpoint = wei::digest_bytes(std::as_bytes(std::span(head.bf16)));
    const auto path = dir + "/sp4.lp";
    {
        wei::LowbitPackWriter writer(checkpoint);
        writer.add("unembed.weight", wei::kSP4TWeightStorage, 4, 256, fingerprint(head.raw), head.bytes, head.header);
        check(static_cast<bool>(writer.write(path)), "write SP4 pack");
    }
    wei::LowbitPack pack;
    check(static_cast<bool>(wei::LowbitPack::open(path, checkpoint, pack)), "open SP4 pack");
    if (pack.entries().size() != 1U) {
        check(false, "SP4 entry count");
        return;
    }
    const auto& e = pack.entries()[0];
    check(e.storage == wei::kSP4TWeightStorage && e.sp_header && e.sp_header->base == head.header.base &&
              e.sp_header->table == sp::default_table<4>() && e.sp_header->flags == head.header.flags,
          "SP4 header round trip");
    std::optional<wei::Slice> slice;
    std::int32_t outer = 99;
    check(static_cast<bool>(pack.find("unembed.weight", wei::kSP4TWeightStorage, head.raw, slice, &outer)) && slice &&
              outer == head.header.base,
          "find returns the slice and the base");
    if (slice) {
        std::vector<std::uint8_t> got(static_cast<std::size_t>(slice->range.bytes));
        check(static_cast<bool>(wei::read_slice(*slice, std::as_writable_bytes(std::span(got)))) && got == head.bytes,
              "SP4 payload round trip");
    }
    // The resident plan carries the base to the matrix view; a q4_g64 storage request for the entry is refused.
    if (slice) {
        auto plan = wei::ResidentSlicePlan::make({{"head", &head.raw, wei::kSP4TWeightStorage, 0U, slice, outer}}, 64U);
        std::vector<std::uint8_t> storage(plan.bytes() + 64U);
        auto* aligned = storage.data() + (64U - reinterpret_cast<std::uintptr_t>(storage.data()) % 64U) % 64U;
        const auto slices = wei::ResidentSlices::bind(plan, std::span<const std::uint8_t>(aligned, plan.bytes()));
        const auto view = slices.matrix("head");
        check(view.storage == wei::kSP4TWeightStorage && view.outer_exponent == head.header.base &&
                  view.row_bytes == S::row_bytes(256) && view.rows == 4U && view.columns == 256U,
              "resident view carries the outer exponent");
        bool refused = false;
        try {
            static_cast<void>(wei::ResidentSlicePlan::make({{"head", &head.raw, std::nullopt, 0U, std::nullopt, 3}}, 64U));
        } catch (const std::invalid_argument&) {
            refused = true;
        }
        check(refused, "an outer exponent without a prepacked matrix is refused");
    }
    std::optional<wei::Slice> wrong;
    check(!pack.find("unembed.weight", wei::kQ4G64WeightStorage, head.raw, wrong), "storage mismatch refused");
    // Load-time conversion has no SP4 encoder.
    wei::Slice converted{};
    check(!wei::prepare_bf16_matrix_slice(head.raw.payload, head.raw.shape, wei::kSP4TWeightStorage, converted),
          "no load-time SP4 encoder");

    // Writer refusals: no header, a fitted (non-fixed) table, a header on a q4_g64 entry.
    const auto refused_write = [&](wei::FormatId storage, std::optional<sp::Header> header, std::uint64_t bytes) {
        wei::LowbitPackWriter writer(checkpoint);
        writer.add("unembed.weight", storage, 4, 256, fingerprint(head.raw), std::vector<std::uint8_t>(bytes), header);
        return !writer.write(dir + "/refused.lp");
    };
    check(refused_write(wei::kSP4TWeightStorage, std::nullopt, head.bytes.size()), "SP4 without header refused");
    auto fitted = head.header;
    fitted.table = {1, 3, 6, 9, 12, 16, 22, 28};
    check(refused_write(wei::kSP4TWeightStorage, fitted, head.bytes.size()), "fitted table refused for strix.sp4t");
    check(refused_write(wei::kQ4G64WeightStorage, head.header, head.bytes.size()), "header on q4_g64 refused");

    // Reader refusals: the header field edited (identity covers it), a missing field, a non-hex digit.
    const auto file = slurp(path);
    const std::string text(file.begin(), file.end());
    const auto line = text.find("tensor unembed.weight strix.sp4t 1 ");
    const auto eol = text.find('\n', line);
    const auto field = text.rfind(' ', eol) + 1U;
    check(line != std::string::npos && eol - field == 32U, "SP4 entry line has the header field");
    if (line == std::string::npos || eol - field != 32U) return;
    wei::LowbitPack scratch;
    auto other_table = file;
    other_table[field + 2U * 9U + 1U] = other_table[field + 2U * 9U + 1U] == '4' ? '5' : '4';  // table[1] 3 -> 4 / 5
    check(!wei::LowbitPack::open(spill(dir + "/t.lp", other_table), checkpoint, scratch), "edited table refused");
    auto bad_hex = file;
    bad_hex[field] = 'g';
    check(!wei::LowbitPack::open(spill(dir + "/h.lp", bad_hex), checkpoint, scratch), "non-hex header refused");
    auto no_field = file;
    no_field[field - 1U] = '\n';  // truncates the line to 9 fields (and breaks the framing): refused either way
    check(!wei::LowbitPack::open(spill(dir + "/n.lp", no_field), checkpoint, scratch), "header-less SP4 line refused");
    // A q4 entry keeps its nine-field line (packs written before this storage keep their bytes and identity).
    {
        wei::LowbitPackWriter writer(checkpoint);
        std::vector<std::uint8_t> q4(static_cast<std::size_t>(4U * wei::Q4G64Storage::row_bytes(256)));
        writer.add("unembed.weight", wei::kQ4G64WeightStorage, 4, 256, fingerprint(head.raw), q4);
        check(static_cast<bool>(writer.write(dir + "/q4.lp")), "write q4 pack");
        const auto q4file = slurp(dir + "/q4.lp");
        const std::string q4text(q4file.begin(), q4file.end());
        const auto at = q4text.find("tensor unembed.weight strix.q4_g64 1 ");
        const auto end = q4text.find('\n', at);
        unsigned spaces = 0;
        for (auto i = at; i < end; ++i) spaces += q4text[i] == ' ' ? 1U : 0U;
        check(at != std::string::npos && spaces == 8U, "q4 entry line keeps nine fields");
    }
}

}  // namespace

int main() {
    const auto dir = std::filesystem::temp_directory_path() / "strix-sp4-pack-test";
    std::filesystem::create_directories(dir);
    try {
        operand_matches_reference();
        pack_round_trip(dir.string());
    } catch (const std::exception& e) {
        ++failures;
        std::cerr << "exception: " << e.what() << '\n';
    }
    std::error_code ec;
    std::filesystem::remove_all(dir, ec);
    if (failures != 0) return 1;
    std::cout << "weight sp4 pack tests passed\n";
    return 0;
}
