// Shared Engram component (strix/engram/*): host tests.
// - DS41 re-bind: hash ids and gathered rows of the shared code equal golden digests captured from the DS41-private
//   implementation before it was removed (same synthetic contract, tokens, file and requests), plus the cache sizing.
// - MODEL_B: hash ids from the pack's 12 multipliers and 64 moduli equal vectors computed on the host CPU by the reference
//   formula (modeling_hy_v4_flash hash_ngrams: lookup, int64 products, XOR fold, % head vocab size); padding at the
//   document start; ids at or past the reference table size read the pad; e4m3 decode of all 256 codes; 96-byte rows.
// - Read-ahead: blocks read on the worker equal RowReader::read of the same requests byte for byte (direct and plain),
//   in any wait order; a failed block rethrows on wait and does not stop the blocks after it; release frees.
#include "strix/ds41_engram.hpp"
#include "strix/ds41_formats.hpp"
#include "strix/engram/hasher.hpp"
#include "strix/engram/read_ahead.hpp"
#include "strix/engram/row_reader.hpp"
#include "strix/engram/spec.hpp"
#if defined(STRIX_ENGRAM_HY4)
#include "strix/hy4/engram_spec.hpp"
#endif

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
    expect(throws([&] { h.rows_at(std::vector<std::uint32_t>{500}, 0, 0, std::span<std::uint64_t>(rows.data(), h.columns())); }),
           "DS41 rows_at rejects an unmapped token");
    {
        // Stateless rows_at equals the stateful push at every position and layer (dead positions included) and
        // leaves the Hasher's history untouched.
        strix::ds41::EngramHasher sh(ds41_contract());
        Lcg r{777};
        std::vector<std::uint32_t> tokens(300);
        for (auto& t : tokens) {
            t = static_cast<std::uint32_t>(r.next() % 1000U);
            if (r.next() % 11U == 0U) t = strix::ds41::engram_dead_token(t);
        }
        const auto all = sh.hash_sequence(tokens);
        const std::size_t per = static_cast<std::size_t>(sh.layers()) * sh.columns();
        std::vector<std::uint64_t> one(sh.columns());
        bool equal_rows = true;
        for (std::size_t t = 0; t < tokens.size(); ++t)
            for (std::uint32_t e = 0; e < sh.layers(); ++e) {
                sh.rows_at(tokens, t, e, one);
                equal_rows = equal_rows && std::equal(one.begin(), one.end(), all.begin() + static_cast<std::ptrdiff_t>(t * per + e * sh.columns()));
            }
        expect(equal_rows, "DS41 rows_at equals push (every position, every layer, dead positions)");
        expect(sh.positions() == tokens.size(), "rows_at does not touch the history");
    }
    c.rows[1] += 1U;
    expect(throws([&] { strix::ds41::EngramHasher bad(c); }), "DS41 primes must sum to the table rows");
}

// ---- HY4 (the model package's Spec adapter; built when it is present) ----
#if defined(STRIX_ENGRAM_HY4)
const std::vector<std::int64_t> kHy4Mult{81853575799317LL, 39690085502883LL, 24229915018165LL, 19457948045719LL, 44729556510419LL, 84779352922315LL, 72974230057307LL, 21530602121221LL, 55998057243097LL, 29128509562787LL, 16745155128523LL, 22474588302733LL};
const std::vector<std::uint64_t> kHy4Moduli{
    10312007ULL, 10312021ULL, 10312039ULL, 10312069ULL, 10312091ULL, 10312121ULL, 10312123ULL, 10312147ULL,
    10312177ULL, 10312187ULL, 10312207ULL, 10312213ULL, 10312229ULL, 10312241ULL, 10312249ULL, 10312259ULL,
    10312277ULL, 10312283ULL, 10312301ULL, 10312343ULL, 10312373ULL, 10312397ULL, 10312409ULL, 10312453ULL,
    10312457ULL, 10312483ULL, 10312507ULL, 10312513ULL, 10312531ULL, 10312543ULL, 10312567ULL, 10312571ULL,
    10312613ULL, 10312619ULL, 10312633ULL, 10312657ULL, 10312681ULL, 10312699ULL, 10312723ULL, 10312747ULL,
    10312759ULL, 10312789ULL, 10312811ULL, 10312823ULL, 10312867ULL, 10312873ULL, 10312889ULL, 10312903ULL,
    10312909ULL, 10312919ULL, 10312927ULL, 10312957ULL, 10312979ULL, 10313011ULL, 10313021ULL, 10313057ULL,
    10313063ULL, 10313117ULL, 10313123ULL, 10313129ULL, 10313161ULL, 10313179ULL, 10313189ULL, 10313201ULL,
};
constexpr std::uint32_t kVocab = 120832, kTable = 120818, kPad = 120002;

strix::hy4::EngramPackMeta hy4_meta() {
    strix::hy4::EngramPackMeta m;
    m.layers = {1, 11, 21, 31};
    m.ngram_size = 3;
    m.heads_per_ngram = 8;
    m.row_elements = 96;
    m.layer_multipliers = kHy4Mult;
    m.head_vocab_sizes = kHy4Moduli;
    std::uint64_t sum = 0;
    for (const auto v : kHy4Moduli) { m.head_offsets.push_back(sum); sum += v; }
    m.pad_token_id = kPad;
    // Synthetic map shaped like the pack's: valid ids -> < 92620, the 14 tail ids map to themselves (as in the GGUF).
    m.token_map.resize(kVocab);
    for (std::uint32_t t = 0; t < kVocab; ++t)
        m.token_map[t] = static_cast<std::int32_t>(t < kTable ? (std::uint64_t{t} * 7919U + 13U) % 92620U : t);
    return m;
}

// Reference vectors: rows within each layer's table, [position][layer][order][head], tokens {120000, 5, 77, 120831}.
const std::vector<std::uint64_t> kHy4Expected{
    // position 0, token 120000
    838358ULL, 12854141ULL, 29219319ULL, 34968805ULL,
    44044266ULL, 59295288ULL, 67222983ULL, 75570692ULL,
    91434672ULL, 96268256ULL, 108240096ULL, 123155526ULL,
    126607944ULL, 138118662ULL, 147094377ULL, 163645724ULL,
    4043117ULL, 14392292ULL, 29984293ULL, 31206861ULL,
    46139439ULL, 55030062ULL, 69800500ULL, 77816244ULL,
    82735764ULL, 102933384ULL, 110942494ULL, 118171657ULL,
    129716157ULL, 134116244ULL, 153568009ULL, 158572683ULL,
    7394443ULL, 11540794ULL, 27781786ULL, 35465053ULL,
    50716066ULL, 58741329ULL, 71765748ULL, 82043906ULL,
    82725396ULL, 101496543ULL, 105810640ULL, 118149020ULL,
    124203336ULL, 135459163ULL, 147599154ULL, 160096747ULL,
    8680520ULL, 18427504ULL, 21835468ULL, 34240402ULL,
    45132088ULL, 58550244ULL, 71425405ULL, 73481184ULL,
    86455967ULL, 96306142ULL, 104794624ULL, 114991084ULL,
    126025997ULL, 142624968ULL, 145883682ULL, 164307190ULL,
    // position 1, token 5
    8058700ULL, 19542669ULL, 25641148ULL, 38199919ULL,
    44346865ULL, 59801788ULL, 70410395ULL, 76609672ULL,
    82924395ULL, 97160290ULL, 105392539ULL, 119924422ULL,
    134015039ULL, 142839122ULL, 151574356ULL, 160546344ULL,
    8696847ULL, 16908896ULL, 25376546ULL, 38471987ULL,
    43587745ULL, 52797670ULL, 67014498ULL, 72751080ULL,
    90542932ULL, 99332645ULL, 107598629ULL, 113971619ULL,
    133113640ULL, 140737898ULL, 145719745ULL, 161260949ULL,
    3569703ULL, 17572642ULL, 22165918ULL, 33749313ULL,
    44124856ULL, 54980947ULL, 63242059ULL, 80607274ULL,
    83012208ULL, 100324756ULL, 103632137ULL, 120968099ULL,
    133953895ULL, 135894727ULL, 154470199ULL, 155773574ULL,
    9011981ULL, 14627882ULL, 27882266ULL, 40382488ULL,
    42610976ULL, 60426113ULL, 62972741ULL, 80083302ULL,
    92545805ULL, 97361200ULL, 107471585ULL, 117663600ULL,
    125271638ULL, 137017197ULL, 151881309ULL, 163829634ULL,
    // position 2, token 77
    1355737ULL, 13130834ULL, 27956652ULL, 33958781ULL,
    48609757ULL, 58190500ULL, 68844551ULL, 79850452ULL,
    89743644ULL, 93627432ULL, 113050221ULL, 123557357ULL,
    130089815ULL, 142134700ULL, 149147716ULL, 164504136ULL,
    388384ULL, 18478853ULL, 27645769ULL, 31701355ULL,
    47730908ULL, 61362703ULL, 69479182ULL, 74192403ULL,
    89835449ULL, 95666762ULL, 104820514ULL, 122429346ULL,
    129524536ULL, 137395176ULL, 154485807ULL, 161710542ULL,
    9320885ULL, 13198923ULL, 27025149ULL, 38591231ULL,
    46535612ULL, 55595559ULL, 64935646ULL, 80965825ULL,
    82954417ULL, 95635823ULL, 107677505ULL, 119954315ULL,
    132106660ULL, 139422440ULL, 150307431ULL, 162433066ULL,
    4072374ULL, 15595589ULL, 27636481ULL, 33255484ULL,
    43296897ULL, 58018349ULL, 68272170ULL, 79888880ULL,
    89404706ULL, 95807100ULL, 108077750ULL, 120368458ULL,
    131254202ULL, 137628065ULL, 151560259ULL, 155976589ULL,
    // position 3, token 120831
    6080894ULL, 15668532ULL, 24381318ULL, 31340220ULL,
    44757741ULL, 59392607ULL, 70573425ULL, 76345610ULL,
    87153690ULL, 96978143ULL, 106396362ULL, 118524006ULL,
    130287788ULL, 144312387ULL, 150246442ULL, 160266523ULL,
    2450425ULL, 13576738ULL, 26111788ULL, 31737052ULL,
    42023572ULL, 59906418ULL, 68628200ULL, 79111905ULL,
    82543375ULL, 95492649ULL, 107611694ULL, 114791330ULL,
    129789913ULL, 139494942ULL, 145824218ULL, 154732106ULL,
    4183005ULL, 17853019ULL, 28119083ULL, 39791388ULL,
    44675050ULL, 59029394ULL, 64923344ULL, 74340864ULL,
    92334986ULL, 102977272ULL, 108199068ULL, 116732238ULL,
    127755726ULL, 142428334ULL, 154105862ULL, 164385215ULL,
    8642094ULL, 15588991ULL, 28250428ULL, 39153827ULL,
    42739983ULL, 60825121ULL, 70437963ULL, 72265122ULL,
    89353686ULL, 97172192ULL, 108482356ULL, 120047370ULL,
    131028122ULL, 140390327ULL, 146582602ULL, 154947748ULL,
};

void test_hy4() {
    const auto meta = hy4_meta();
    const auto spec = strix::hy4::engram_spec(meta);
    expect(spec.columns() == 16U && spec.layers() == 4U && spec.row.bytes() == 96U, "HY4 geometry");
    expect(strix::engram::layer_row_base(spec, 1) == 164994380U && strix::engram::layer_row_base(spec, 3) + strix::engram::layer_rows(spec, 3) == 660006416U,
           "HY4 flat table: pack head_offsets[16] and the 660,006,416 rows");
    expect(strix::engram::products_fit_int64(spec), "HY4 products fit int64");
    strix::engram::Hasher h(spec);
    const std::vector<std::uint32_t> tokens{120000, 5, 77, 120831};
    const auto rows = h.hash_sequence(tokens);
    expect(rows == kHy4Expected, "HY4 hash ids equal the reference formula");
    // Document start: position 0 reads the pad for t-1 and t-2, i.e. equals the token after two pad tokens.
    const auto first = h.hash_sequence(std::vector<std::uint32_t>{77});
    const auto padded = h.hash_sequence(std::vector<std::uint32_t>{kPad, kPad, 77});
    expect(std::equal(first.begin(), first.end(), padded.begin() + 128), "HY4 n-gram padding at document start");
    // Ids at or past the reference table size read the pad, even where the pack's map has its own entry.
    const auto a = h.hash_sequence(std::vector<std::uint32_t>{5, kTable});
    const auto b = h.hash_sequence(std::vector<std::uint32_t>{5, kVocab - 1U});
    const auto c = h.hash_sequence(std::vector<std::uint32_t>{5, kVocab + 7U});
    const auto p = h.hash_sequence(std::vector<std::uint32_t>{5, kPad});
    expect(a == p && b == p && c == p, "HY4 ids >= table size map to the pad");
    // Incremental push equals the sequence hash; truncate rewinds.
    h.reset();
    std::vector<std::uint64_t> one(64);
    for (const auto t : tokens) h.push(t, one);
    expect(std::equal(one.begin(), one.end(), rows.begin() + 192), "HY4 push equals hash_sequence");
    h.truncate(2);
    h.push(77, one);
    expect(std::equal(one.begin(), one.end(), rows.begin() + 128), "HY4 truncate rewinds the history");
    // Stateless rows_at (the Hy4 forwards' entry): equal to push per layer; the 14 ids 120,818..120,831 read the
    // pad's compressed id at every n-gram slot.
    std::vector<std::uint64_t> col(16), pad_col(16);
    bool same = true;
    for (std::size_t t = 0; t < tokens.size(); ++t)
        for (std::uint32_t e = 0; e < 4U; ++e) {
            h.rows_at(tokens, t, e, col);
            same = same && std::equal(col.begin(), col.end(), rows.begin() + static_cast<std::ptrdiff_t>(t * 64U + e * 16U));
        }
    expect(same, "HY4 rows_at equals hash_sequence");
    bool unmapped_pad = true;
    for (std::uint32_t id = kTable; id < kVocab; ++id)
        for (std::size_t slot = 0; slot < 3U; ++slot)
            for (std::uint32_t e = 0; e < 4U; ++e) {
                std::vector<std::uint32_t> w{77, 5, 120000}, wp = w;
                w[2U - slot] = id;
                wp[2U - slot] = kPad;
                h.rows_at(w, 2, e, col);
                h.rows_at(wp, 2, e, pad_col);
                unmapped_pad = unmapped_pad && col == pad_col;
            }
    expect(unmapped_pad, "HY4 ids 120,818..120,831 read the pad at every n-gram slot");
    // Bad packs.
    auto bad = meta;
    bad.layer_multipliers[3] += 1;
    expect(throws([&] { (void)strix::hy4::engram_spec(bad); }), "HY4 even multiplier rejected");
    bad = meta;
    bad.head_offsets[17] += 1U;
    expect(throws([&] { (void)strix::hy4::engram_spec(bad); }), "HY4 head offsets checked");
    bad = meta;
    bad.token_map[3] = 0x7FFFFFFF;
    expect(throws([&] { (void)strix::hy4::engram_spec(bad); }), "HY4 product overflow rejected");
    bad = meta;
    bad.head_vocab_sizes.pop_back();
    expect(throws([&] { (void)strix::hy4::engram_spec(bad); }), "HY4 moduli count checked");
}

#endif  // STRIX_ENGRAM_HY4
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

void test_read_ahead() {
    std::vector<std::uint8_t> bytes(4096U * 40U + 7U);
    Lcg g{47};
    for (auto& b : bytes) b = static_cast<std::uint8_t>(g.next());
    const auto path = write_file("strix_engram_core_read_ahead.bin", bytes);
    const std::vector<strix::engram::Table> tables{{64U, 900U}, {64U + 900U * 96U, 700U}};
    for (int mode = 0; mode < 2; ++mode) {
        strix::engram::ReaderOptions o;
        o.row_bytes = 96;
        o.threads = 2;
        o.direct_io = mode == 0;
        strix::engram::RowReader reader(path, tables, o);
        std::vector<std::vector<strix::engram::RowRequest>> blocks(6);
        for (std::size_t k = 0; k < blocks.size(); ++k) {
            blocks[k].resize(k == 3U ? 0U : 64U + 37U * k);  // block 3 empty
            for (auto& r : blocks[k]) {
                r.table = static_cast<std::uint32_t>(g.next() % 2U);
                r.row = g.next() % (r.table == 0U ? 900U : 700U);
            }
            if (blocks[k].size() > 5U) blocks[k][5] = blocks[k][1];  // a duplicate inside the block
        }
        blocks[4].push_back({1U, 700U});  // out of range: block 4 fails, block 5 is still read
        std::vector<std::vector<std::uint8_t>> direct(blocks.size());
        for (std::size_t k = 0; k < blocks.size(); ++k)
            if (k != 4U) {
                direct[k].resize(blocks[k].size() * 96U);
                reader.read(blocks[k], direct[k]);
            }
        strix::engram::ReadAhead ahead(reader);
        std::vector<std::uint64_t> tickets;
        for (const auto& b : blocks) tickets.push_back(ahead.submit(b));
        expect(tickets.front() == 0U && tickets.back() == blocks.size() - 1U, "read-ahead tickets in submission order");
        bool same = true;
        for (const std::size_t k : {std::size_t{5}, std::size_t{0}, std::size_t{2}, std::size_t{1}, std::size_t{3}}) {
            const auto got = ahead.wait(tickets[k]);
            same = same && got.size() == direct[k].size() &&
                   (got.empty() || std::memcmp(got.data(), direct[k].data(), got.size()) == 0);
        }
        expect(same, mode == 0 ? "read-ahead bytes equal RowReader::read (direct)" : "read-ahead bytes equal RowReader::read (plain)");
        expect(throws([&] { (void)ahead.wait(tickets[4]); }), "read-ahead rethrows the failed block");
        expect(ahead.held() == blocks.size(), "read-ahead holds every unreleased block");
        for (const auto t : tickets) ahead.release(t);
        expect(ahead.held() == 0U, "read-ahead release frees");
        expect(throws([&] { (void)ahead.wait(tickets[0]); }), "read-ahead rejects a released ticket");
        const auto st = ahead.stats();
        expect(st.blocks == blocks.size() - 1U, "read-ahead counts the read blocks (empty block not queued)");
        // destruction with blocks still queued: finishes them, then joins
        strix::engram::ReadAhead drain(reader);
        for (std::size_t k = 0; k < 3U; ++k) (void)drain.submit(blocks[k]);
    }
    std::filesystem::remove(path);
}

} // namespace

int main() {
    test_ds41_rebind();
#if defined(STRIX_ENGRAM_HY4)
    test_hy4();
#endif
    test_decode();
    test_reader_96();
    test_read_ahead();
    if (failures == 0) std::printf("engram core tests: PASS\n");
    return failures == 0 ? 0 : 1;
}
