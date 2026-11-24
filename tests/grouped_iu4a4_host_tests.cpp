// Host invariants of the grouped rotated A4W4 expert class (moe_core/grouped_iu4a4_contract.hpp):
//  1. iu4a4_rot::paired_column is a bijection of [0, 2 rows) and tiles 2 c / 2 c + 1 are gate / up of outputs 16 c ..;
//  2. ExpertSource::value == the BF16 operand of the EXACT tile (PackedOperand::load_tile16 + decode_bf16), NVFP4 and
//     MXFP8, contiguous and interleaved row order;
//  3. a paired image column n == the plain image column paired_column(n) of the same matrix (codes and step).
#include "moe_core/grouped_iu4a4_contract.hpp"

#include <cstdint>
#include <cstdio>
#include <cstring>
#include <random>
#include <vector>

namespace {
namespace gi = strix::moe_core::grouped_iu4a4;
namespace rot = strix::core::iu4a4_rot;
namespace packed = strix::weight::microscaled;

static_assert(gi::image_code_bytes(128U, 32U) == 2048U);
static_assert(gi::image_step_floats(128U, 32U) == 32U);

int fails = 0;
void expect(bool ok, const char* what) {
    if (!ok) {
        std::printf("FAIL %s\n", what);
        ++fails;
    }
}
std::uint32_t bits(float v) {
    std::uint32_t b;
    std::memcpy(&b, &v, 4);
    return b;
}

void paired_order(unsigned rows) {
    std::vector<unsigned> seen(2U * rows, 0U);
    bool tiles = true;
    for (unsigned n = 0; n < 2U * rows; ++n) {
        const unsigned c = rot::paired_column(n, rows);
        if (c < 2U * rows) ++seen[c];
        const unsigned t = n / 16U, j = n % 16U;
        tiles = tiles && c == (t % 2U) * rows + 16U * (t / 2U) + j;
    }
    bool bijective = true;
    for (const unsigned s : seen) bijective = bijective && s == 1U;
    expect(bijective && tiles, "paired_column order");
}

template <packed::MatrixFormat F>
void decode_and_pair(packed::RowOrder order, unsigned seed) {
    constexpr bool nv = F == packed::MatrixFormat::nvfp4;
    const unsigned rows = 64U, in = 256U, code_row = nv ? in / 2U : in, scale_row = in / (nv ? 16U : 32U);
    std::mt19937 g(seed);
    std::vector<std::uint8_t> codes(std::size_t(rows) * code_row), scales(std::size_t(rows) * scale_row);
    for (auto& c : codes) c = static_cast<std::uint8_t>(g() & 0xffU);
    for (auto& s : scales) s = static_cast<std::uint8_t>(nv ? 0x20U + g() % 32U : 119U + g() % 5U);
    const packed::PackedMatrix m{codes.data(), scales.data(), rows, in, nv ? 0.75F : 1.0F, order};
    const packed::PackedOperand<F> op{m};
    const gi::ExpertSource<F> plain{m, 0U}, paired{m, rows / 2U};
    unsigned bad = 0;
    for (unsigned r = 0; r < rows; ++r)
        for (unsigned k0 = 0; k0 < in; k0 += 16U) {
            std::uint32_t pairs[8];
            op.decode_bf16(op.load_tile16(r, k0), pairs);
            for (unsigned j = 0; j < 16U; ++j) {
                const std::uint32_t want = ((j & 1U) != 0U ? pairs[j / 2U] >> 16U : pairs[j / 2U] & 0xffffU) << 16U;
                const float got = plain.value(r, k0 + j);
                bad += bits(got) != want ? 1U : 0U;
            }
        }
    expect(bad == 0U, nv ? "nvfp4 value == EXACT operand" : "mxfp8 value == EXACT operand");
    // 3: paired column n == plain column paired_column(n).
    std::vector<std::uint8_t> pc(gi::image_code_bytes(in, rows), 0), qc(gi::image_code_bytes(in, rows), 0);
    unsigned bad_pair = 0;
    for (unsigned b = 0; b < in / rot::kBlock; ++b)
        for (unsigned n = 0; n < rows; ++n) {
            const unsigned src = rot::paired_column(n, rows / 2U);
            const float sp = paired.requantise(n, b, rows, pc.data()), sq = plain.requantise(src, b, rows, qc.data());
            bad_pair += bits(sp) != bits(sq);
            for (unsigned i = 0; i < rot::kBlock; ++i)
                bad_pair += rot::pack_code(pc.data(), b * rot::kBlock + i, n, rows) != rot::pack_code(qc.data(), b * rot::kBlock + i, src, rows);
        }
    expect(bad_pair == 0U, "paired image column == plain image column");
}

// Host admission of a pass (refusal): a valid pass is admitted; each defect alone refuses it. Addresses are never read.
void admission() {
    const auto at = [](std::uintptr_t a) { return reinterpret_cast<std::uint8_t*>(a); };
    const unsigned hidden = 256U, inter = 128U, rows = 97U, assignments = 291U;
    gi::LayerImages images;
    const auto view = [&](gi::ExpertImage& v, unsigned in, unsigned n) {
        v.codes = at(0x100000U);
        v.steps = reinterpret_cast<float*>(at(0x200000U));
        v.code_bytes = gi::image_code_bytes(in, n);
        v.step_floats = gi::image_step_floats(in, n);
        v.experts = 5U;
    };
    view(images.gate_up, hidden, 2U * inter);
    view(images.down, inter, hidden);
    images.gate_up_codes = at(0x300000U);
    images.gate_up_stride = 4096U;
    images.down_codes = at(0x400000U);
    images.down_stride = 4096U;
    gi::Pass ok;
    ok.tokens = reinterpret_cast<const float*>(at(0x500000U));
    ok.rows = rows;
    ok.assignments = assignments;
    ok.hidden = hidden;
    ok.intermediate = inter;
    ok.assignment_rows = reinterpret_cast<const std::uint32_t*>(at(0x600000U));
    ok.gate_up = reinterpret_cast<const gi::TileRows*>(at(0x700000U));
    ok.down = reinterpret_cast<const gi::TileRows*>(at(0x800000U));
    ok.tiles = 7U;
    ok.max_tile_rows = 128U;
    ok.index_base = reinterpret_cast<const std::uint16_t*>(at(0x900000U));
    ok.h = reinterpret_cast<float*>(at(0xa00000U));
    ok.act = at(0xb00000U);
    ok.act_size = gi::act_bytes(rows, assignments, hidden, inter);
    ok.images = &images;
    ok.fault = reinterpret_cast<unsigned*>(at(0xc00000U));
    ok.y = reinterpret_cast<float*>(at(0xd00000U));
    expect(gi::refusal(ok) == nullptr, "a valid pass is admitted");
    const auto refused = [&](const char* what, auto&& defect) {
        gi::Pass p = ok;
        gi::LayerImages im = images;
        p.images = &im;
        defect(p, im);
        expect(gi::refusal(p) != nullptr, what);
    };
    using P = gi::Pass&;
    using I = gi::LayerImages&;
    refused("zero rows", [](P p, I) { p.rows = 0U; });
    refused("fewer assignments than rows", [](P p, I) { p.assignments = p.rows - 1U; });
    refused("assignments past the quantiser grid", [](P p, I) { p.assignments = 0x7ffffff8U; });
    refused("missing tokens", [](P p, I) { p.tokens = nullptr; });
    refused("missing index base", [](P p, I) { p.index_base = nullptr; });
    refused("missing images", [](P p, I) { p.images = nullptr; });
    refused("missing verdict word", [](P p, I) { p.fault = nullptr; });
    refused("missing down destination", [](P p, I) { p.y = nullptr; });
    refused("tiles past the validation table", [](P p, I) { p.tiles = gi::kMaxPassTiles + 1U; });
    refused("intermediate past the one-CTA quantiser", [](P p, I) { p.intermediate = gi::kMaxIntermediate + 128U; });
    refused("zero tiles", [](P p, I) { p.tiles = 0U; });
    refused("zero max tile rows", [](P p, I) { p.max_tile_rows = 0U; });
    refused("max tile rows above assignments", [](P p, I) { p.max_tile_rows = p.assignments + 1U; });
    refused("hidden off the 128-input block", [](P p, I) { p.hidden = 200U; });
    refused("unaligned activation workspace", [](P p, I) { p.act += 128; });
    refused("activation workspace one byte short", [](P p, I) { --p.act_size; });
    refused("default (empty) images", [](P, I im) { im = gi::LayerImages{}; });
    refused("gate/up image without codes", [](P, I im) { im.gate_up.codes = nullptr; });
    refused("down image without steps", [](P, I im) { im.down.steps = nullptr; });
    refused("image without expert slots", [](P, I im) { im.down.experts = 0U; });
    refused("gate/up code span one byte short", [](P, I im) { --im.gate_up.code_bytes; });
    refused("down step span one float short", [](P, I im) { --im.down.step_floats; });
    refused("missing code base", [](P, I im) { im.down_codes = nullptr; });
    refused("zero code stride", [](P, I im) { im.gate_up_stride = 0U; });
    // Slot rule: every expert slot on a 16-byte boundary, the whole image addressable.
    refused("gate/up code span one byte long (expert 1 off 16 B)", [](P, I im) { ++im.gate_up.code_bytes; });
    refused("down step span one float long (expert 1 off 16 B)", [](P, I im) { ++im.down.step_floats; });
    refused("gate/up codes off 16 B", [](P, I im) { im.gate_up.codes += 8; });
    refused("down steps off 16 B", [](P, I im) { im.down.steps += 1; });
    refused("image extent past the address space", [](P, I im) { im.gate_up.experts = ~0U; im.gate_up.code_bytes = ~std::size_t{0} / 2U & ~std::size_t{15}; });
    {
        gi::Pass p = ok;
        gi::LayerImages im = images;
        p.images = &im;
        im.gate_up.code_bytes += 16U;  // a longer slot on the 16-byte rule is admitted
        expect(gi::refusal(p) == nullptr, "a 16-byte-aligned longer slot is admitted");
    }
}
}  // namespace

int main() {
    admission();
    paired_order(16U);
    paired_order(64U);
    paired_order(1152U);
    decode_and_pair<packed::MatrixFormat::nvfp4>(packed::RowOrder::contiguous, 1U);
    decode_and_pair<packed::MatrixFormat::nvfp4>(packed::RowOrder::interleaved_halves, 2U);
    decode_and_pair<packed::MatrixFormat::mxfp8>(packed::RowOrder::contiguous, 3U);
    std::printf("%s\n", fails == 0 ? "ALL PASS" : "FAIL");
    return fails == 0 ? 0 : 1;
}
