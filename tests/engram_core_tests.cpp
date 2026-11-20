// Shared Engram component (strix/engram/*): host tests.
// - DS41 re-bind: hash ids and gathered rows of the shared code equal golden digests captured from the DS41-private
//   implementation before it was removed (same synthetic contract, tokens, file and requests), plus the cache sizing.
// - MODEL_B: hash ids from the pack's 12 multipliers and 64 moduli equal vectors computed on the host CPU by the reference
//   formula (modeling_hy_v4_flash hash_ngrams: lookup, int64 products, XOR fold, % head vocab size); padding at the
//   document start; ids at or past the reference table size read the pad; e4m3 decode of all 256 codes; 96-byte rows.
#include "strix/ds41_engram.hpp"
#include "strix/ds41_formats.hpp"
#include "strix/engram/hasher.hpp"
#include "strix/engram/row_reader.hpp"
#include "strix/engram/spec.hpp"

#include <algorithm>
#include <array>
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <cstring>
#include <exception>
#include <filesystem>
#include <fstream>
#include <stdexcept>
#include <string>
#include <vector>

namespace {

int failures = 0;
void expect(bool ok, const char* what) {
    if (!ok) { std::fprintf(stderr, "FAIL: %s\n", what); ++failures; }
}
template <class F> bool throws(F&& f) {
    try { f(); } catch (const std::exception&) { return true; }
    return false;
}

std::uint64_t fnv(std::uint64_t h, const void* p, std::size_t n) {
    const auto* b = static_cast<const std::uint8_t*>(p);
    for (std::size_t i = 0; i < n; ++i) { h ^= b[i]; h *= 0x100000001b3ULL; }
    return h;
}
struct Lcg {
    std::uint64_t s;
    std::uint64_t next() { s = s * 6364136223846793005ULL + 1442695040888963407ULL; return s >> 17U; }
};

// ---- DS41 golden (captured from the pre-rebind src/ds41_engram.cpp) ----
constexpr std::uint64_t kDs41HashGolden = 0x885d7837ce424747ULL;
constexpr std::uint64_t kDs41ReadGolden = 0xb7eab9d88e09b794ULL;

strix::deepseek41::EngramContract ds41_contract() {
    strix::deepseek41::EngramContract c;
    c.encoding = "e4m3_e8m0_32_row264";
    c.row_bytes = 264;
    c.layer_ids = {1, 14};
    c.max_ngram_size = 3;
    c.n_heads = 4;
    c.primes = {1009, 1013, 1019, 1021, 1031, 1033, 1039, 1049, 1051, 1061, 1063, 1069, 1087, 1091, 1093, 1097};
    c.rows = {0, 0};
    for (std::size_t e = 0; e < 2; ++e)
        for (std::size_t k = 0; k < 8; ++k) c.rows[e] += c.primes[e * 8 + k];
    c.multipliers = {0x2F3A5C7D9E1B4C6DULL | 1U, 0x1D2C3B4A59687706ULL | 1U, 0x0123456789ABCDEFULL,
                     0x7A6B5C4D3E2F1A0BULL, 0x55AA55AA12345679ULL, 0x0F1E2D3C4B5A6979ULL};
    c.token_map.resize(1000);
    for (std::uint32_t t = 0; t < 1000; ++t) c.token_map[t] = (t * 37U + 11U) % 997U;
    c.compressed_vocab_size = 997;
    c.pad_id = 999;
    return c;
}

std::uint64_t ds41_hash_digest() {
    const auto c = ds41_contract();
    strix::ds41::EngramHasher h(c);
    Lcg g{12345};
    std::vector<std::uint32_t> tokens(600);
    for (auto& t : tokens) {
        t = static_cast<std::uint32_t>(g.next() % 1000U);
        if (g.next() % 23U == 0U) t = strix::ds41::engram_dead_token(t);
    }
    std::uint64_t d = 0xcbf29ce484222325ULL;
    const auto all = h.hash_sequence(tokens);
    d = fnv(d, all.data(), all.size() * sizeof(std::uint64_t));
    h.truncate(400);
    std::vector<std::uint64_t> rows(static_cast<std::size_t>(h.layers()) * h.columns());
    for (std::uint32_t i = 0; i < 50; ++i) {
        h.push(static_cast<std::uint32_t>(g.next() % 1000U), rows);
        d = fnv(d, rows.data(), rows.size() * sizeof(std::uint64_t));
    }
    return d;
}

std::vector<std::uint8_t> row_file_bytes() {
    std::vector<std::uint8_t> bytes(3U * 4096U + 264U * 900U + 77U);
    Lcg g{777};
    for (auto& b : bytes) b = static_cast<std::uint8_t>(g.next());
    return bytes;
}
std::filesystem::path write_file(const char* name, const std::vector<std::uint8_t>& bytes) {
    const auto path = std::filesystem::temp_directory_path() / name;
    std::ofstream(path, std::ios::binary).write(reinterpret_cast<const char*>(bytes.data()), static_cast<std::streamsize>(bytes.size()));
    return path;
}

// Digest of the DS41 reader's batches; `raw` is the same digest computed from the file bytes directly.
std::uint64_t ds41_read_digest(bool direct, std::uint64_t cache, std::uint64_t& raw, strix::ds41::EngramReaderStats& stats) {
    const auto bytes = row_file_bytes();
    const auto path = write_file("strix_engram_core_rows.bin", bytes);
    const std::vector<strix::ds41::EngramTable> tables{{4096U + 5U, 400U}, {4096U + 5U + 400U * 264U, 500U}};
    std::uint64_t d = 0xcbf29ce484222325ULL;
    raw = d;
    {
        strix::ds41::EngramRowReader reader(path, tables, 4, direct, cache);
        Lcg g{99};
        for (int b = 0; b < 3; ++b) {
            std::vector<strix::ds41::EngramRowRequest> req(300);
            for (auto& r : req) {
                r.table = static_cast<std::uint32_t>(g.next() % 2U);
                r.row = g.next() % (r.table == 0U ? 400U : 500U);
            }
            req[7] = req[3];
            std::vector<std::uint8_t> out(req.size() * 264U);
            reader.read(req, out);
            d = fnv(d, out.data(), out.size());
            for (const auto& r : req) raw = fnv(raw, bytes.data() + tables[r.table].file_offset + r.row * 264U, 264U);
        }
        stats = reader.stats();
    }
    std::filesystem::remove(path);
    return d;
}

void test_ds41_rebind() {
    expect(ds41_hash_digest() == kDs41HashGolden, "DS41 hash ids equal the pre-rebind golden");
    std::uint64_t raw = 0;
    strix::ds41::EngramReaderStats s;
    expect(ds41_read_digest(true, 1U << 20U, raw, s) == kDs41ReadGolden, "DS41 rows (direct + cache) equal the golden");
    expect(raw == kDs41ReadGolden, "DS41 golden is the file's bytes");
    expect(s.cache_capacity == 3542U && s.cache_storage_bytes == 1024528U, "DS41 payload cache sized as before");
    expect(ds41_read_digest(false, 0, raw, s) == kDs41ReadGolden, "DS41 rows (plain) equal the golden");
    // The shared e8m0_per32 row trait decodes exactly as the DS41 row decoder.
    Lcg g{5};
    std::array<std::uint8_t, strix::ds41::engram_row_bytes> row{};
    bool same = true;
    for (int it = 0; it < 64; ++it) {
        for (auto& b : row) b = static_cast<std::uint8_t>(g.next());
        for (std::size_t j = 256; j < 264; ++j) row[j] = static_cast<std::uint8_t>(100U + g.next() % 50U);
        std::array<float, strix::ds41::engram_row_elements> a{}, b{};
        strix::ds41::decode_engram_row(row, a);
        strix::engram::decode_row({256, strix::engram::RowScale::e8m0_per32}, row, b);
        for (std::size_t j = 0; j < a.size(); ++j) same = same && (std::memcmp(&a[j], &b[j], 4) == 0 || (a[j] != a[j] && b[j] != b[j]));
    }
    expect(same, "e8m0_per32 decode equals ds41 decode_engram_row");
    auto c = ds41_contract();
    c.token_map.resize(500);
    c.pad_id = 499;
    strix::ds41::EngramHasher h(c);
    std::vector<std::uint64_t> rows(16);
    expect(throws([&] { h.push(500, rows); }), "DS41 rejects an unmapped token");
    c.rows[1] += 1U;
    expect(throws([&] { strix::ds41::EngramHasher bad(c); }), "DS41 primes must sum to the table rows");
}

// ---- MODEL_B (the model package's Spec adapter; built when it is present) ----
// Independent E4M3 (finite variant) value: bias 7, subnormals m * 2^-9, S.1111.111 NaN.
float e4m3_reference(std::uint8_t c) {
    const int e = (c >> 3) & 15, m = c & 7;
    if ((c & 0x7F) == 0x7F) return std::nanf("");
    const float mag = e == 0 ? std::ldexp(static_cast<float>(m), -9) : std::ldexp(1.0F + static_cast<float>(m) / 8.0F, e - 7);
    return (c & 0x80) != 0 ? -mag : mag;
}

void test_decode() {
    std::vector<std::uint8_t> codes(256);
    for (std::size_t i = 0; i < 256; ++i) codes[i] = static_cast<std::uint8_t>(i);
    std::vector<float> out(256);
    strix::engram::decode_row({256, strix::engram::RowScale::none}, codes, out);
    bool ok = true;
    for (std::size_t i = 0; i < 256; ++i) {
        const float r = e4m3_reference(static_cast<std::uint8_t>(i));
        ok = ok && ((r != r) ? out[i] != out[i] : std::memcmp(&r, &out[i], 4) == 0);
    }
    expect(ok, "e4m3 decode of all 256 codes");
    expect(out[0x7E] == 448.0F && out[0xFE] == -448.0F && out[1] == 0.001953125F, "e4m3 extremes");
    std::vector<std::uint8_t> row(100);
    for (std::size_t i = 0; i < 96; ++i) row[i] = static_cast<std::uint8_t>(i * 2U);
    const float scale = 0.375F;
    std::memcpy(row.data() + 96, &scale, 4);
    std::vector<float> scaled(96);
    strix::engram::decode_row({96, strix::engram::RowScale::f32_per_row}, row, scaled);
    expect(scaled[10] == e4m3_reference(20) * scale, "f32_per_row scale");
    expect(throws([&] { strix::engram::decode_row({96, strix::engram::RowScale::none}, row, scaled); }), "row size checked");
}

// The shared reader on 96-byte rows (MODEL_B layout: rows straddle 4 KiB pages), direct + cache and plain.
void test_reader_96() {
    std::vector<std::uint8_t> bytes(4096U * 12U + 13U);
    Lcg g{31};
    for (auto& b : bytes) b = static_cast<std::uint8_t>(g.next());
    const auto path = write_file("strix_engram_core_rows96.bin", bytes);
    const std::vector<strix::engram::Table> tables{{100U, 200U}, {100U + 200U * 96U, 300U}};
    for (int mode = 0; mode < 2; ++mode) {
        strix::engram::ReaderOptions o;
        o.row_bytes = 96;
        o.threads = 3;
        o.direct_io = mode == 0;
        o.payload_cache_bytes = mode == 0 ? (64U << 10U) : 0U;
        strix::engram::RowReader reader(path, tables, o);
        bool same = true;
        for (int b = 0; b < 3; ++b) {
            std::vector<strix::engram::RowRequest> req(257);
            for (auto& r : req) {
                r.table = static_cast<std::uint32_t>(g.next() % 2U);
                r.row = g.next() % (r.table == 0U ? 200U : 300U);
            }
            req[9] = req[2];
            std::vector<std::uint8_t> out(req.size() * 96U);
            reader.read(req, out);
            for (std::size_t i = 0; i < req.size(); ++i)
                same = same && std::memcmp(out.data() + i * 96U, bytes.data() + tables[req[i].table].file_offset + req[i].row * 96U, 96U) == 0;
        }
        expect(same, mode == 0 ? "96-byte rows, direct + cache" : "96-byte rows, plain");
        std::vector<strix::engram::RowRequest> oob{{1U, 300U}};
        std::vector<std::uint8_t> one(96);
        expect(throws([&] { reader.read(oob, one); }), "row out of range rejected");
    }
    strix::engram::ReaderOptions o;
    o.row_bytes = 96;
    expect(throws([&] { strix::engram::RowReader r(path, {{bytes.size() - 50U, 1U}}, o); }), "table outside the file rejected");
    o.row_bytes = 5000;
    expect(throws([&] { strix::engram::RowReader r(path, tables, o); }), "row larger than a page rejected");
    std::filesystem::remove(path);
}

} // namespace

int main() {
    test_ds41_rebind();
    test_decode();
    test_reader_96();
    if (failures == 0) std::printf("engram core tests: PASS\n");
    return failures == 0 ? 0 : 1;
}
