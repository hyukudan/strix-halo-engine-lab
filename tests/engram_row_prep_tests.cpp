// Shared Engram row preparation contract (include/strix/engram/row_prep.hpp): host reference of both forms.
//   decode        == engram::decode_row (spec.cpp) for every RowScale, all 256 codes;
//   head_rms_bf16 == the model-B oracle row statements (src/modelb/ref/modelb_host_ref.cpp Impl::engram: x = bf(e4m3),
//                    rms_norm with ds41::chunked_dot, eps 2^-7), transcribed here with the non-template chunked_dot.
#include "strix/ds41_layer.hpp"
#include "strix/ds41_scalar_codecs.hpp"
#include "strix/engram/row_prep.hpp"
#include "strix/engram/spec.hpp"

#include <cstdint>
#include <cstdio>
#include <cstring>
#include <stdexcept>
#include <vector>

namespace {

int failures = 0;
void check(bool ok, const char* what) {
    if (!ok) {
        ++failures;
        std::printf("FAIL %s\n", what);
    }
}
std::uint32_t bits(float x) {
    std::uint32_t u;
    std::memcpy(&u, &x, sizeof u);
    return u;
}
struct Lcg {
    std::uint64_t s;
    std::uint32_t next() {
        s = s * 6364136223846793005ULL + 1442695040888963407ULL;
        return static_cast<std::uint32_t>(s >> 33U);
    }
    float uniform(float lo, float hi) { return lo + (hi - lo) * static_cast<float>(next() & 0xFFFFFFU) / 16777216.0F; }
};

// Rows whose codes walk all 256 values (NaN codes 0x7f / 0xff excluded unless `nan`).
std::vector<std::uint8_t> make_rows(const strix::engram::RowFormat& f, std::uint32_t count, Lcg& g, bool nan) {
    std::vector<std::uint8_t> rows(static_cast<std::size_t>(count) * f.bytes());
    for (std::uint32_t i = 0; i < count; ++i) {
        std::uint8_t* r = rows.data() + static_cast<std::size_t>(i) * f.bytes();
        for (std::uint32_t j = 0; j < f.elements; ++j) {
            auto c = static_cast<std::uint8_t>((i * 37U + j * 11U + (g.next() & 3U)) & 0xFFU);
            if (!nan && (c & 0x7FU) == 0x7FU) c = static_cast<std::uint8_t>(c ^ 1U);
            r[j] = c;
        }
        if (f.scale == strix::engram::RowScale::e8m0_per32)
            for (std::uint32_t b = 0; b < f.elements / 32U; ++b) r[f.elements + b] = static_cast<std::uint8_t>(110U + (g.next() % 30U));
        if (f.scale == strix::engram::RowScale::f32_per_row) {
            const float s = g.uniform(0.001F, 2.0F);
            std::memcpy(r + f.elements, &s, sizeof s);
        }
    }
    return rows;
}

void decode_case(strix::engram::RowScale scale, std::uint32_t elements) {
    using namespace strix::engram;
    Lcg g{elements * 7U + static_cast<std::uint32_t>(scale)};
    const RowFormat f{elements, scale};
    const std::uint32_t count = 40;
    const auto rows = make_rows(f, count, g, false);
    std::vector<float> out(static_cast<std::size_t>(count) * elements), want(elements);
    RowPrepView v;
    v.form = RowPrepForm::decode;
    v.fmt = f;
    v.rows = rows.data();
    v.row_count = count;
    v.out = out.data();
    row_prep_reference(v);
    bool same = true;
    for (std::uint32_t i = 0; i < count; ++i) {
        decode_row(f, std::span<const std::uint8_t>(rows.data() + static_cast<std::size_t>(i) * f.bytes(), f.bytes()), want);
        for (std::uint32_t j = 0; j < elements; ++j) same = same && bits(out[static_cast<std::size_t>(i) * elements + j]) == bits(want[j]);
    }
    check(same, "decode form == engram::decode_row (bitwise)");
}

// The oracle's statements (modelb_host_ref.cpp: bf, rms_norm with dot = ds41 chunked dot, eps), for one row.
void oracle_row(const std::uint8_t* codes, std::uint32_t n, const float* w, float eps, float* y) {
    std::vector<float> row(n);
    for (std::uint32_t j = 0; j < n; ++j) row[j] = strix::ds41::round_to_bf16(strix::ds41::codec::e4m3(codes[j]));
    const float r = 1.0F / strix::ds41::declared_sqrt(strix::ds41::chunked_dot(row.data(), row.data(), n) / static_cast<float>(n) + eps);
    for (std::uint32_t j = 0; j < n; ++j) y[j] = strix::ds41::round_to_bf16(w[j] * strix::ds41::round_to_bf16(row[j] * r));
}

void head_rms_case(std::uint32_t elements, std::uint32_t heads, std::uint32_t tokens, bool zero_row) {
    using namespace strix::engram;
    Lcg g{elements * 131U + heads};
    const RowFormat f{elements, RowScale::none};
    const std::uint32_t count = tokens * heads;
    auto rows = make_rows(f, count, g, false);
    if (zero_row) std::memset(rows.data(), 0, f.bytes());  // all-zero codes: r = 1 / sqrt(eps), output +-0
    std::vector<float> w(static_cast<std::size_t>(heads) * elements);
    for (auto& x : w) x = strix::ds41::round_to_bf16(g.uniform(-2.0F, 2.0F));
    std::vector<float> out(static_cast<std::size_t>(count) * elements), want(elements);
    RowPrepView v;
    v.form = RowPrepForm::head_rms_bf16;
    v.fmt = f;
    v.rows = rows.data();
    v.row_count = count;
    v.out = out.data();
    v.norm_weight = w.data();
    v.heads = heads;
    v.eps = 0.0078125F;
    row_prep_reference(v);
    bool same = true, bf16 = true;
    for (std::uint32_t i = 0; i < count; ++i) {
        oracle_row(rows.data() + static_cast<std::size_t>(i) * f.bytes(), elements, w.data() + static_cast<std::size_t>(i % heads) * elements,
                   v.eps, want.data());
        for (std::uint32_t j = 0; j < elements; ++j) {
            const float o = out[static_cast<std::size_t>(i) * elements + j];
            same = same && bits(o) == bits(want[j]);
            bf16 = bf16 && (bits(o) & 0xFFFFU) == 0U;
        }
    }
    check(same, "head_rms_bf16 == oracle row statements (bitwise)");
    check(bf16, "head_rms_bf16 outputs are BF16 values");
}

void validation() {
    using namespace strix::engram;
    RowPrepView v;
    v.form = RowPrepForm::head_rms_bf16;
    v.fmt = RowFormat{96U, RowScale::none};
    v.heads = 16U;
    v.row_count = 32U;
    check(row_prep_invalid(v) == nullptr, "model-B view (96 x 16 heads) is supported");
    v.row_count = 33U;
    check(row_prep_invalid(v) != nullptr, "partial token rejected");
    v.row_count = 32U;
    v.fmt.elements = 704U;
    check(row_prep_invalid(v) != nullptr, "more than one lane-dot chunk rejected");
    v.fmt = RowFormat{96U, RowScale::none};
    v.heads = 0U;
    check(row_prep_invalid(v) != nullptr, "heads 0 rejected");
    v.form = RowPrepForm::decode;
    v.fmt = RowFormat{48U, RowScale::e8m0_per32};
    check(row_prep_invalid(v) != nullptr, "e8m0_per32 with a partial block rejected");
    bool threw = false;
    try {
        row_prep_reference(v);
    } catch (const std::invalid_argument&) {
        threw = true;
    }
    check(threw, "reference throws on an unsupported view");
}

}  // namespace

int main() {
    using strix::engram::RowScale;
    decode_case(RowScale::e8m0_per32, 256U);  // DS41 rows (264 B)
    decode_case(RowScale::none, 96U);         // model-B rows (96 B)
    decode_case(RowScale::f32_per_row, 64U);
    head_rms_case(96U, 16U, 9U, false);  // model-B geometry
    head_rms_case(96U, 16U, 2U, true);
    head_rms_case(40U, 3U, 5U, false);   // a row width that is not a multiple of the wave
    validation();
    if (failures) return 1;
    std::printf("PASS engram row prep (decode == decode_row, head_rms_bf16 == oracle statements)\n");
    return 0;
}
