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
    big.rows = 65;
    check(!gw::fits_tile(big, 2304, 2304), "65 rows refused");
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
    check(!gw::fits(args), "max_tile_rows above the tile refused");
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
    const packed::PackedMatrix w{w_codes.data(), w_scales.data(), out_total, k_total, 1.0F, c.order};
    const unsigned w_code_row = k_total / 2U, w_scale_row = k_total / 16U;
    std::vector<int> seen(std::size_t{c.rows} * out_total, 0);

    for (unsigned block = 0; block < out_total / lay::kOutputs; ++block) {
        const unsigned first_output = block * lay::kOutputs;
        std::vector<double> acc(std::size_t{8} * 32U * 32U, 0.0);  // [warp][lane][mb nb z]
        for (unsigned stage = 0; stage < k_total / lay::kStageInputs; ++stage) {
            std::vector<std::uint8_t> smem(lay::kStageBytes, 0xA5);  // poison: every read byte must come from a copy
            for (unsigned thread = 0; thread < lay::kThreads; ++thread) {
                for (unsigned j = 0; j < lay::kCodeChunksPerThread; ++j) {
                    const auto ch = lay::code_chunk(thread, j);
                    std::uint8_t* dst = smem.data() + lay::stage_offset(ch.activation, ch.row, 16U * ch.part);
                    if (ch.activation) {
                        if (ch.row < c.rows)
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
                std::uint8_t* dst = smem.data() + lay::stage_offset(ch.activation, ch.row, lay::kScaleOffset);
                if (ch.activation) {
                    if (ch.row < c.rows)
                        std::memcpy(dst, x_scales.data() + (c.first + ch.row) * std::size_t{w_scale_row} +
                                             lay::kScaleBytes * stage, 8);
                    else std::memset(dst, 0, 8);
                } else {
                    const auto row = packed::physical_row(c.order, out_total, first_output + ch.row);
                    std::memcpy(dst, w_scales.data() + row * w_scale_row + lay::kScaleBytes * stage, 8);
                }
            }
            const auto word = [&](unsigned offset) {
                std::uint32_t v;
                std::memcpy(&v, smem.data() + offset, 4);
                return v;
            };
            for (unsigned warp = 0; warp < 8U; ++warp) {
                for (unsigned kb = 0; kb < 2U; ++kb) {
                    std::uint32_t a[32][2][4], sa[32][2], b[32][4][2], sb[32][4];
                    for (unsigned lane = 0; lane < 32U; ++lane) {
                        for (unsigned mb = 0; mb < 2U; ++mb) {
                            for (unsigned i = 0; i < 4U; ++i)
                                a[lane][mb][i] = word(lay::stage_offset(false, lay::a_row(warp, mb, lane, i), lay::a_byte(kb, lane, i)));
                            sa[lane][mb] = word(lay::stage_offset(false, lay::a_scale_row(warp, mb, lane), lay::scale_byte(kb)));
                        }
                        for (unsigned nb = 0; nb < 4U; ++nb) {
                            const unsigned r = lay::b_row(warp, nb, lane);
                            b[lane][nb][0] = word(lay::stage_offset(true, r, lay::b_byte(kb, lane, 0U)));
                            b[lane][nb][1] = word(lay::stage_offset(true, r, lay::b_byte(kb, lane, 1U)));
                            sb[lane][nb] = word(lay::stage_offset(true, r, lay::scale_byte(kb)));
                        }
                    }
                    // ISA semantics of one m16n8k64 per (mb, nb): element maps of the fragments.
                    for (unsigned mb = 0; mb < 2U; ++mb) {
                        for (unsigned nb = 0; nb < 4U; ++nb) {
                            double d[16][8] = {};
                            for (unsigned m = 0; m < 16U; ++m) {
                                for (unsigned n = 0; n < 8U; ++n) {
                                    double sum = 0.0;
                                    for (unsigned k = 0; k < 64U; ++k) {
                                        const unsigned la = 4U * (m % 8U) + k % 32U / 8U, ia = (m / 8U) + 2U * (k / 32U);
                                        const unsigned lb = 4U * n + k % 32U / 8U, ib = k / 32U;
                                        const unsigned na = (a[la][mb][ia] >> (4U * (k % 8U))) & 15U;
                                        const unsigned nbits = (b[lb][nb][ib] >> (4U * (k % 8U))) & 15U;
                                        const double scale_a = e4m3(static_cast<std::uint8_t>(sa[4U * (m % 8U) + m / 8U][mb] >> (8U * (k / 16U))));
                                        const double scale_b = e4m3(static_cast<std::uint8_t>(sb[4U * n][nb] >> (8U * (k / 16U))));
                                        sum += e2m1(na) * scale_a * (e2m1(nbits) * scale_b);
                                    }
                                    d[m][n] = sum;
                                }
                            }
                            for (unsigned lane = 0; lane < 32U; ++lane)
                                for (unsigned z = 0; z < 4U; ++z)
                                    acc[(std::size_t{warp} * 32U + lane) * 32U + (mb * 4U + nb) * 4U + z] +=
                                        d[lane / 4U + 8U * (z >> 1U)][2U * (lane % 4U) + (z & 1U)];
                        }
                    }
                }
            }
        }
        // Epilogue map vs the scalar product of the canonical operands (same k order: 64-blocks ascending, k ascending).
        for (unsigned warp = 0; warp < 8U; ++warp) {
            for (unsigned lane = 0; lane < 32U; ++lane) {
                for (unsigned mb = 0; mb < 2U; ++mb) {
                    for (unsigned nb = 0; nb < 4U; ++nb) {
                        for (unsigned z = 0; z < 4U; ++z) {
                            const unsigned row = lay::acc_row(warp, nb, lane, z);
                            const unsigned o = first_output + lay::acc_output(warp, mb, lane, z);
                            const double got = acc[(std::size_t{warp} * 32U + lane) * 32U + (mb * 4U + nb) * 4U + z];
                            if (row >= c.rows) {
                                if (got != 0.0) check(false, "padding row accumulates zero");
                                continue;
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
                          {256, 256, 33, 2, packed::RowOrder::contiguous}};
    for (const auto& c : cases) simulate(c, rng);
    if (failures != 0) {
        std::printf("grouped_w4a4_host_test: %d failure(s)\n", failures);
        return 1;
    }
    std::printf("grouped_w4a4_host_test: PASS (admission + fragment simulation, %zu cases)\n", std::size(cases));
    return 0;
}
