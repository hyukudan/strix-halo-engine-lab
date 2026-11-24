// Host test of the grouped W4A4 expert tile (moe_core/grouped_w4a4.hpp, sm_120 maps in arch/sm120/grouped_w4a4_layout.hpp).
// 1. Admission: the Inkling-class shapes are admitted, misaligned / oversize / ragged ones refused.
// 2. Fragment simulation: the kernel's stage copy plan, fragment register reads and accumulator map, fed through the PTX ISA
//    semantics of mma.sync m16n8k64 kind::mxf4nvf4 scale_vec::4X (A row-major: a_i = row g + 8 (i & 1), k 8 t + 32 (i >> 1) + j;
//    B: b_i = column g, k 8 t + 32 i + j; scale of A row m from lane 4 (m % 8) + m / 8, of B column n from lane 4 n; byte b of
//    a scale word = inputs 16 b ..), give every live (row, output) exactly once, equal (double, same k order) to the scalar
//    product of the canonical operands; padding rows are zero and never stored.
#include "moe_core/grouped_w4a4.hpp"
#include "cuda/core/arch/sm120/grouped_w4a4_layout.hpp"

#include <cmath>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <random>
#include <vector>

namespace {

namespace gw = strix::moe_core::grouped_w4a4;
namespace lay = strix::cuda::core::grouped_w4a4::sm120;
namespace packed = strix::weight::microscaled;

int failures = 0;
void check(bool ok, const char* what) {
    if (!ok) {
        ++failures;
        std::printf("FAIL %s\n", what);
    }
}

double e2m1(unsigned nibble) {
    static constexpr double mag[8] = {0.0, 0.5, 1.0, 1.5, 2.0, 3.0, 4.0, 6.0};
    return (nibble & 8U) ? -mag[nibble & 7U] : mag[nibble & 7U];
}
double e4m3(std::uint8_t code) { return static_cast<double>(strix::weight::scalar_float::e4m3(code)); }
unsigned nibble(const std::uint8_t* row, unsigned k) { return (row[k / 2U] >> (4U * (k & 1U))) & 15U; }
// Bytes [offset, offset + n) of the simulated stage; an out-of-stage offset is a map defect.
std::uint8_t* slot(std::vector<std::uint8_t>& stage, std::size_t offset, std::size_t n) {
    if (offset > stage.size() || n > stage.size() - offset) {
        std::printf("FAIL stage offset %zu + %zu outside %zu bytes\n", offset, n, stage.size());
        std::exit(1);
    }
    return stage.data() + offset;
}

void admission() {
    alignas(16) static std::uint8_t codes[2304 * 1152];
    alignas(16) static std::uint8_t scales[2304 * 144];
    alignas(16) static std::uint16_t input[64 * 2304];
    alignas(16) static float output[64 * 2304];
    const packed::PackedMatrix w{codes, scales, 2304, 2304, 0.5F, packed::RowOrder::contiguous};
    gw::TileRows tile{w, input, output, 2304, 2304, 64};
    check(gw::fits_tile(tile, 2304, 2304), "gate-up 2304 -> 2304 admitted");
    const packed::PackedMatrix down_w{codes, scales, 2304, 1152, 0.5F, packed::RowOrder::contiguous};
    gw::TileRows down{down_w, input, output, 1152, 2304, 64};
    check(gw::fits_tile(down, 1152, 2304), "down 1152 -> 2304 admitted");
    gw::TileRows big = tile;
    big.rows = 128;
    check(gw::fits_tile(big, 2304, 2304), "128 rows admitted (one CTA)");
    big.rows = 129;
    check(!gw::fits_tile(big, 2304, 2304), "129 rows refused");
    gw::TileRows ragged = tile;
    ragged.weights.columns = 2320;
    ragged.input_row = 2320;
    check(!gw::fits_tile(ragged, 2320, 2304), "inputs % 128 != 0 refused");
    gw::TileRows misaligned = tile;
    misaligned.weights.block_scales = scales + 4;
    check(!gw::fits_tile(misaligned, 2304, 2304), "scale rows off 8 bytes refused");
    check(!gw::fits_tile(tile, 2304, 2304 + 64), "outputs % 128 != 0 refused");

    alignas(256) static std::uint8_t image_bytes[gw::image_layout(512, 2304).bytes];
    const auto image = gw::make_image(image_bytes, 512, 2304);
    check(gw::fits_image(image, 2304) && gw::fits_image(gw::make_image(image_bytes, 512, 1152), 1152), "image admitted");
    check(image.code_row == 1152 && image.scale_row == 144, "image strides");
    gw::Args args{{&tile, 1, 2304, 2304, nullptr, nullptr, 64}, input, image};
    check(gw::fits(args), "args admitted");
    args.tiles.max_tile_rows = 128;
    check(gw::fits(args), "max_tile_rows of a 128-row plan admitted");
    args.tiles.max_tile_rows = 129;
    check(!gw::fits(args), "max_tile_rows above the plan tile refused");
}

struct Case {
    unsigned outputs, inputs, rows, first;
    packed::RowOrder order;
};

void simulate(const Case& c, std::mt19937& rng) {
    const unsigned out_total = c.outputs, k_total = c.inputs, image_rows = c.first + c.rows + 3U;
    std::vector<std::uint8_t> w_codes(std::size_t{out_total} * k_total / 2U), w_scales(std::size_t{out_total} * k_total / 16U);
    std::vector<std::uint8_t> x_codes(std::size_t{image_rows} * k_total / 2U), x_scales(std::size_t{image_rows} * k_total / 16U);
    std::uniform_int_distribution<int> byte(0, 255), scale(0x20, 0x50);  // positive finite E4M3 scales
    for (auto& b : w_codes) b = static_cast<std::uint8_t>(byte(rng));
    for (auto& b : x_codes) b = static_cast<std::uint8_t>(byte(rng));
    for (auto& b : w_scales) b = static_cast<std::uint8_t>(scale(rng));
    for (auto& b : x_scales) b = static_cast<std::uint8_t>(scale(rng));
    const unsigned w_code_row = k_total / 2U, w_scale_row = k_total / 16U;
    std::vector<int> seen(std::size_t{c.rows} * out_total, 0);
    const unsigned rows = c.rows < lay::kRows ? c.rows : lay::kRows, rows8 = (rows + 7U) & ~7U, rows16 = (rows + 15U) & ~15U;
    constexpr unsigned kAcc = 4U * 4U * 4U;  // acc[mt][nt][z] per lane
    for (unsigned block = 0; block < out_total / lay::kOutputs; ++block) {
        const unsigned first_output = block * lay::kOutputs;
        std::vector<double> acc(std::size_t{8} * 32U * kAcc, 0.0);  // [warp][lane][mt nt z]
        std::vector<int> touched(acc.size(), 0);
        std::vector<std::uint8_t> smem(lay::kStageBytes, 0xA5);  // one stage slot; bytes never copied are flagged when read
        std::vector<std::uint8_t> written(lay::kStageBytes, 0);
        for (unsigned stage = 0; stage < k_total / lay::kStageInputs; ++stage) {
            for (unsigned thread = 0; thread < lay::kThreads; ++thread) {
                for (unsigned j = 0; j < lay::kCodeChunksPerThread; ++j) {
                    const auto ch = lay::code_chunk(thread, j);
                    if (ch.activation && ch.row >= rows16) continue;  // not issued: never read
                    const std::size_t off = lay::stage_offset(ch.activation, ch.row, 16U * ch.part);
                    std::uint8_t* dst = slot(smem, off, 16);
                    std::memset(written.data() + off, 1, 16);
                    if (ch.activation) {
                        if (ch.row < rows)
                            std::memcpy(dst, x_codes.data() + (c.first + ch.row) * std::size_t{w_code_row} +
                                                 lay::kCodeBytes * stage + 16U * ch.part, 16);
                        else std::memset(dst, 0, 16);
                    } else {
                        const auto row = packed::physical_row(c.order, out_total, first_output + ch.row);
                        std::memcpy(dst, w_codes.data() + row * w_code_row + lay::kCodeBytes * stage + 16U * ch.part, 16);
                    }
                }
                if (!lay::has_scale_chunk(thread)) continue;
                const auto ch = lay::scale_chunk(thread);
                if (ch.activation && ch.row >= rows8) continue;
                const std::size_t off = lay::stage_offset(ch.activation, ch.row, lay::kScaleOffset);
                std::uint8_t* dst = slot(smem, off, 8);
                std::memset(written.data() + off, 1, 8);
                if (ch.activation) {
                    if (ch.row < rows)
                        std::memcpy(dst, x_scales.data() + (c.first + ch.row) * std::size_t{w_scale_row} +
                                             lay::kScaleBytes * stage, 8);
                    else std::memset(dst, 0, 8);
                } else {
                    const auto row = packed::physical_row(c.order, out_total, first_output + ch.row);
                    std::memcpy(dst, w_scales.data() + row * w_scale_row + lay::kScaleBytes * stage, 8);
                }
            }
            bool stale = false;
            const auto word = [&](unsigned offset) {
                std::uint32_t v;
                std::memcpy(&v, slot(smem, offset, 4), 4);
                for (unsigned i = 0; i < 4U; ++i) stale = stale || written[offset + i] == 0;
                return v;
            };
            // ldmatrix m8n8 x4 b16: lane l receives, in register j, word l % 4 of the 16-byte row addressed by lane 8 j + l / 4.
            const auto ldsm = [&](const unsigned (&addr)[32], std::uint32_t (&r)[32][4]) {
                for (unsigned l = 0; l < 32U; ++l)
                    for (unsigned j = 0; j < 4U; ++j) {
                        const unsigned row_addr = addr[8U * j + l / 4U];
                        if (row_addr % 16U != 0U) check(false, "ldmatrix row address 16-byte aligned");
                        r[l][j] = word(row_addr + 4U * (l % 4U));
                    }
            };
            for (unsigned warp = 0; warp < 8U; ++warp) {
                const unsigned live = lay::live_blocks(warp, rows);
                if (live == 0U) continue;
                for (unsigned kb = 0; kb < 2U; ++kb) {
                    std::uint32_t a[4][32][4], b[2][32][4], sa[32][4], sb[32][4];
                    for (unsigned mt = 0; mt < 4U; ++mt) {
                        unsigned addr[32];
                        for (unsigned l = 0; l < 32U; ++l)
                            addr[l] = lay::stage_offset(false, lay::ldsm_a_row(warp, mt, l), lay::ldsm_a_byte(kb, l));
                        ldsm(addr, a[mt]);
                    }
                    for (unsigned p = 0; p < 2U; ++p) {
                        if (2U * p >= live) continue;
                        unsigned addr[32];
                        for (unsigned l = 0; l < 32U; ++l)
                            addr[l] = lay::stage_offset(true, lay::ldsm_b_row(warp, p, l), lay::ldsm_b_byte(kb, l));
                        ldsm(addr, b[p]);
                    }
                    for (unsigned l = 0; l < 32U; ++l) {
                        for (unsigned mt = 0; mt < 4U; ++mt)
                            sa[l][mt] = word(lay::stage_offset(false, lay::a_scale_row(warp, mt, l), lay::scale_byte(kb)));
                        for (unsigned nt = 0; nt < live; ++nt)
                            sb[l][nt] = word(lay::stage_offset(true, lay::b_scale_row(warp, nt, l), lay::scale_byte(kb)));
                    }
                    // ISA semantics of one m16n8k64 per (mt, nt < live): element maps of the fragments.
                    for (unsigned mt = 0; mt < 4U; ++mt) {
                        for (unsigned nt = 0; nt < live; ++nt) {
                            const auto breg = [&](unsigned l, unsigned i) { return b[nt / 2U][l][2U * (nt % 2U) + i]; };
                            double d[16][8] = {};
                            for (unsigned m = 0; m < 16U; ++m) {
                                for (unsigned n = 0; n < 8U; ++n) {
                                    double sum = 0.0;
                                    for (unsigned k = 0; k < 64U; ++k) {
                                        const unsigned la = 4U * (m % 8U) + k % 32U / 8U, ia = (m / 8U) + 2U * (k / 32U);
                                        const unsigned lb = 4U * n + k % 32U / 8U, ib = k / 32U;
                                        const unsigned na = (a[mt][la][ia] >> (4U * (k % 8U))) & 15U;
                                        const unsigned nbits = (breg(lb, ib) >> (4U * (k % 8U))) & 15U;
                                        const double scale_a = e4m3(static_cast<std::uint8_t>(sa[4U * (m % 8U) + m / 8U][mt] >> (8U * (k / 16U))));
                                        const double scale_b = e4m3(static_cast<std::uint8_t>(sb[4U * n][nt] >> (8U * (k / 16U))));
                                        sum += e2m1(na) * scale_a * (e2m1(nbits) * scale_b);
                                    }
                                    d[m][n] = sum;
                                }
                            }
                            for (unsigned lane = 0; lane < 32U; ++lane)
                                for (unsigned z = 0; z < 4U; ++z) {
                                    const std::size_t i = (std::size_t{warp} * 32U + lane) * kAcc + (mt * 4U + nt) * 4U + z;
                                    acc[i] += d[lane / 4U + 8U * (z >> 1U)][2U * (lane % 4U) + (z & 1U)];
                                    ++touched[i];
                                }
                        }
                    }
                }
            }
            if (stale) check(false, "every fragment / scale word read was copied (or zero-filled) in a stage");
        }
        // Epilogue map vs the scalar product of the canonical operands (same k order: 64-blocks ascending, k ascending).
        for (unsigned warp = 0; warp < 8U; ++warp) {
            for (unsigned lane = 0; lane < 32U; ++lane) {
                for (unsigned mt = 0; mt < 4U; ++mt) {
                    for (unsigned nt = 0; nt < 4U; ++nt) {
                        for (unsigned z = 0; z < 4U; ++z) {
                            const unsigned row = lay::acc_row(warp, nt, lane, z);
                            const unsigned o = first_output + lay::acc_output(warp, mt, lane, z);
                            const std::size_t i = (std::size_t{warp} * 32U + lane) * kAcc + (mt * 4U + nt) * 4U + z;
                            const double got = acc[i];
                            if (row >= rows) {
                                if (got != 0.0) check(false, "padding row accumulates zero");
                                continue;
                            }
                            if (touched[i] != static_cast<int>(2U * (k_total / lay::kStageInputs))) {
                                check(false, "a live row's accumulator receives one MMA per 64-input block");
                                return;
                            }
                            ++seen[std::size_t{row} * out_total + o];
                            const auto prow = packed::physical_row(c.order, out_total, o);
                            const std::uint8_t* wr = w_codes.data() + prow * w_code_row;
                            const std::uint8_t* xr = x_codes.data() + (c.first + row) * std::size_t{w_code_row};
                            double want = 0.0;
                            for (unsigned blk = 0; blk < k_total / 64U; ++blk) {
                                double part = 0.0;
                                for (unsigned k = 64U * blk; k < 64U * blk + 64U; ++k)
                                    part += e2m1(nibble(wr, k)) * e4m3(w_scales[prow * w_scale_row + k / 16U]) *
                                            (e2m1(nibble(xr, k)) * e4m3(x_scales[(c.first + row) * std::size_t{w_scale_row} + k / 16U]));
                                want += part;
                            }
                            if (got != want) {
                                std::printf("  row %u output %u got %.17g want %.17g\n", row, o, got, want);
                                check(false, "fragment product equals the canonical product");
                                return;
                            }
                        }
                    }
                }
            }
        }
    }
    for (int s : seen)
        if (s != 1) {
            check(false, "every live (row, output) produced exactly once");
            return;
        }
}

}  // namespace

int main() {
    admission();
    std::mt19937 rng(20261122U);
    const Case cases[] = {{128, 256, 64, 0, packed::RowOrder::contiguous},
                          {256, 384, 37, 5, packed::RowOrder::interleaved_halves},
                          {128, 128, 1, 11, packed::RowOrder::contiguous},
                          {256, 256, 33, 2, packed::RowOrder::contiguous},
                          {128, 256, 128, 3, packed::RowOrder::interleaved_halves},
                          {128, 128, 97, 0, packed::RowOrder::contiguous},
                          {256, 128, 72, 9, packed::RowOrder::contiguous},
                          {128, 256, 8, 4, packed::RowOrder::contiguous},
                          {256, 128, 17, 6, packed::RowOrder::interleaved_halves}};
    for (const auto& c : cases) simulate(c, rng);
    if (failures != 0) {
        std::printf("grouped_w4a4_host_test: %d failure(s)\n", failures);
        return 1;
    }
    std::printf("grouped_w4a4_host_test: PASS (admission + fragment simulation, %zu cases)\n", std::size(cases));
    return 0;
}
