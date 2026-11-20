// Host-only: contract dense-iu4a4-rot-v1 (src/core/iu4a4_rot.hpp).
//  1. activation rule: codes in [-7, 7], u in [1, 255], the row's largest group has u = 255, |y - code * step| <=
//     step / 2, a zero row gives base 0 and codes 0, a NaN / Inf row gives a non-finite base (the output propagates it);
//  2. Abel form: lead * C_before + sum e_g * C_after == sum u_g * P_g exactly (int64 check), on random and worst-case
//     codes, and the bounds the kernel relies on hold (|C| < 2^23, |t| < 2^24);
//  3. the bit-defined dot == an independent FP64 evaluation of sum_b wstep_b * t_b * base within 1e-6 relative;
//  4. rotation: W R . R x == W . x in FP64 before quantisation (R = fwht128, symmetric orthogonal), and the requantised
//     pack round-trips: |W R - code * step| <= step / 2 per element;
//  5. quality of the integer-relative scales against the FP32 per-group rule of a4_tiles (same rotation, same groups):
//     squared error ratio <= 1.03 on Gaussian rows with outlier channels (printed).
#include "core/iu4a4_rot.hpp"

#include <cmath>
#include <cstdint>
#include <cstdio>
#include <random>
#include <vector>

namespace {
int failures = 0;
void check(bool ok, const char* why) {
    if (!ok) {
        std::fprintf(stderr, "FAIL: %s\n", why);
        ++failures;
    }
}
namespace rot = strix::core::iu4a4_rot;
namespace im = strix::core::integer_math;
namespace a4 = strix::core::a4_tiles;

std::vector<float> gaussian_row(std::mt19937& g, unsigned in, bool outliers) {
    std::normal_distribution<float> nd(0.0f, 1.0f);
    std::vector<float> r(in);
    for (unsigned k = 0; k < in; ++k) r[k] = nd(g);
    if (outliers)
        for (unsigned k = 7; k < in; k += 211U) r[k] *= 40.0f;  // a few heavy channels, as in LLM activations
    return r;
}
std::uint16_t f16_bits(float v) { return strix::core::onfly::f32_to_f16_bits(v); }
}  // namespace

int main() {
    constexpr unsigned in = 1024, outputs = 32;
    std::mt19937 gen(12345U);
    {  // 1
        std::vector<float> y = gaussian_row(gen, in, true);
        std::vector<std::int8_t> codes(in);
        std::vector<std::uint8_t> u(in / rot::kGroupA);
        std::vector<float> rotated = y;
        for (unsigned b = 0; b < in; b += rot::kBlock) a4::fwht128(rotated.data() + b);
        const float base = rot::quantise_row(y.data(), in, codes.data(), u.data());
        bool codes_ok = true, u_ok = true, bound_ok = true, same_rotation = true;
        int umax = 0;
        for (unsigned k = 0; k < in; ++k) {
            codes_ok = codes_ok && codes[k] >= -7 && codes[k] <= 7;
            same_rotation = same_rotation && rotated[k] == y[k];
            const float step = base * static_cast<float>(u[k / rot::kGroupA]);
            bound_ok = bound_ok && std::fabs(y[k] - static_cast<float>(codes[k]) * step) <= step * 0.5f * (1.0f + 1e-6f);
        }
        for (const std::uint8_t v : u) {
            u_ok = u_ok && v >= 1U;
            umax = v > umax ? v : umax;
        }
        check(codes_ok, "1: codes in [-7, 7]");
        check(u_ok && umax == rot::kUMax, "1: u in [1, 255] and the largest group has u = 255");
        check(bound_ok, "1: |y - code * step| <= step / 2");
        check(same_rotation, "1: the row is rotated in place by fwht128 per block");
        std::vector<float> z(in, 0.0f);
        const float bz = rot::quantise_row(z.data(), in, codes.data(), u.data());
        bool zero_ok = bz == 0.0f;
        for (unsigned k = 0; k < in; ++k) zero_ok = zero_ok && codes[k] == 0;
        check(zero_ok, "1: a zero row gives base 0 and codes 0");
        std::vector<float> bad = gaussian_row(gen, in, false);
        bad[300] = std::nanf("");
        const float bn = rot::quantise_row(bad.data(), in, codes.data(), u.data());
        check(!(std::fabs(bn) <= 3.4e38f), "1: a NaN row gives a non-finite base");
    }
    {  // 2
        for (int trial = 0; trial < 2000; ++trial) {
            std::uniform_int_distribution<int> cd(-7, 7), ud(1, rot::kUMax);
            const bool worst = trial < 2;
            std::uint8_t u[rot::kGroupsPerBlock];
            std::int64_t p[rot::kGroupsPerBlock], want = 0;
            for (unsigned g = 0; g < rot::kGroupsPerBlock; ++g) {
                u[g] = static_cast<std::uint8_t>(worst ? rot::kUMax : ud(gen));
                p[g] = 0;
                for (unsigned i = 0; i < rot::kGroupA; ++i) {
                    const int a = worst ? 7 : cd(gen), w = worst ? (trial == 0 ? 7 : -7) : cd(gen);
                    p[g] += a * w;
                }
                want += std::int64_t{u[g]} * p[g];
            }
            const std::int32_t c0 = worst ? static_cast<std::int32_t>((rot::kMaxInputs - rot::kBlock) * 49U) * (trial == 0 ? 1 : -1)
                                          : std::uniform_int_distribution<std::int32_t>(-4000000, 4000000)(gen);
            const rot::BlockCoef bc = rot::block_coef(u);
            std::int64_t c = c0, t = std::int64_t{bc.lead} * c;
            bool fits = true;
            for (unsigned g = 0; g < rot::kGroupsPerBlock; ++g) {
                c += p[g];
                fits = fits && (c < 0 ? -c : c) < (std::int64_t{1} << 23);
                t += std::int64_t{bc.e[g]} * c;
            }
            check(t == want, "2: Abel form == sum u_g P_g");
            check(fits && (t < 0 ? -t : t) < (std::int64_t{1} << 24), "2: |C| < 2^23 and |t| < 2^24");
            if (t != want) break;
        }
    }
    // A pack (IU4 g64, the shared layout) for 3 and 4: random codes, FP16 scales.
    std::vector<std::uint8_t> pack(std::size_t(in) * outputs / 2U, 0U);
    std::vector<std::uint16_t> scales(std::size_t(outputs) * (in / 64U));
    std::vector<double> wdeq(std::size_t(outputs) * in);
    {
        std::uniform_int_distribution<int> cd(-8, 7);
        std::uniform_real_distribution<float> sd(0.002f, 0.03f);
        for (auto& s : scales) s = f16_bits(sd(gen));
        for (unsigned n = 0; n < outputs; ++n)
            for (unsigned k = 0; k < in; ++k) {
                const int q = cd(gen);
                rot::put_code(pack.data(), k, n, outputs, q);
                wdeq[std::size_t(n) * in + k] = double(q) * double(strix::core::onfly::f16_bits_to_f32(scales[std::size_t(n) * (in / 64U) + k / 64U]));
            }
        bool rt = true;
        for (unsigned n = 0; n < outputs && rt; ++n)
            for (unsigned k = 0; k < in; ++k) rt = rt && double(rot::pack_code(pack.data(), k, n, outputs)) *
                                                             double(strix::core::onfly::f16_bits_to_f32(scales[std::size_t(n) * (in / 64U) + k / 64U])) ==
                                                         wdeq[std::size_t(n) * in + k];
        check(rt, "pack: put_code / pack_code round trip");
    }
    std::vector<std::uint8_t> rpack(pack.size(), 0U);
    std::vector<float> wstep(std::size_t(outputs) * (in / rot::kBlock));
    for (unsigned n = 0; n < outputs; ++n)
        for (unsigned b = 0; b < in / rot::kBlock; ++b) rot::requantise_block(pack.data(), scales.data(), in, outputs, n, b, rpack.data(), wstep.data());
    {  // 4
        bool rq = true;
        double worst_rel = 0.0;
        const std::vector<float> x = gaussian_row(gen, in, true);
        for (unsigned n = 0; n < outputs; ++n) {
            std::vector<float> wr(in), xr(x);
            for (unsigned k = 0; k < in; ++k) wr[k] = static_cast<float>(wdeq[std::size_t(n) * in + k]);
            double plain = 0.0, rotated = 0.0, mag = 0.0;
            for (unsigned k = 0; k < in; ++k) plain += wdeq[std::size_t(n) * in + k] * double(x[k]), mag += std::fabs(wdeq[std::size_t(n) * in + k] * double(x[k]));
            for (unsigned b = 0; b < in; b += rot::kBlock) a4::fwht128(wr.data() + b), a4::fwht128(xr.data() + b);
            for (unsigned k = 0; k < in; ++k) {
                rotated += double(wr[k]) * double(xr[k]);
                const float step = wstep[std::size_t(n) * (in / rot::kBlock) + k / rot::kBlock];
                rq = rq && std::fabs(wr[k] - static_cast<float>(rot::pack_code(rpack.data(), k, n, outputs)) * step) <= step * 0.5f * (1.0f + 1e-6f);
            }
            worst_rel = std::fmax(worst_rel, std::fabs(plain - rotated) / mag);
        }
        check(worst_rel < 1e-5, "4: W R . R x == W . x before quantisation");
        check(rq, "4: requantised pack |W R - code * step| <= step / 2");
    }
    {  // 3
        bool ok = true;
        for (int r = 0; r < 8; ++r) {
            std::vector<float> y = gaussian_row(gen, in, r % 2 == 1);
            std::vector<std::int8_t> codes(in);
            std::vector<std::uint8_t> u(in / rot::kGroupA);
            const float base = rot::quantise_row(y.data(), in, codes.data(), u.data());
            for (unsigned n = 0; n < outputs; ++n) {
                const float got = rot::dot(codes.data(), u.data(), base, rpack.data(), wstep.data(), in, outputs, n);
                double want = 0.0, mag = 0.0;
                for (unsigned b = 0; b < in / rot::kBlock; ++b) {
                    std::int64_t t = 0;
                    for (unsigned g = 0; g < rot::kGroupsPerBlock; ++g) {
                        std::int64_t p = 0;
                        for (unsigned i = 0; i < rot::kGroupA; ++i) {
                            const unsigned k = b * rot::kBlock + g * rot::kGroupA + i;
                            p += codes[k] * rot::pack_code(rpack.data(), k, n, outputs);
                        }
                        t += std::int64_t{u[b * rot::kGroupsPerBlock + g]} * p;
                    }
                    const double term = double(t) * double(wstep[std::size_t(n) * (in / rot::kBlock) + b]);
                    want += term, mag += std::fabs(term);
                }
                want *= double(base), mag *= double(base);
                ok = ok && std::fabs(double(got) - want) <= 1e-6 * mag + 1e-30;
            }
        }
        check(ok, "3: bit-defined dot == FP64 sum_b wstep_b t_b base");
    }
    {  // 5
        double e_rel = 0.0, e_flt = 0.0;
        for (int r = 0; r < 64; ++r) {
            const std::vector<float> x = gaussian_row(gen, in, r % 2 == 0);
            std::vector<float> y = x, f = x;
            std::vector<std::int8_t> codes(in);
            std::vector<std::uint8_t> u(in / rot::kGroupA);
            const float base = rot::quantise_row(y.data(), in, codes.data(), u.data());  // y = rotated
            for (unsigned b = 0; b < in; b += rot::kBlock) a4::fwht128(f.data() + b);
            for (unsigned k = 0; k < in; ++k) {
                const double qi = double(codes[k]) * double(base * static_cast<float>(u[k / rot::kGroupA]));
                e_rel += (qi - double(y[k])) * (qi - double(y[k]));
            }
            for (unsigned g = 0; g < in; g += rot::kGroupA) {
                std::vector<float> grp(f.begin() + g, f.begin() + g + rot::kGroupA);
                a4::dequant_group(grp.data(), rot::kGroupA);
                for (unsigned i = 0; i < rot::kGroupA; ++i) e_flt += (double(grp[i]) - double(f[g + i])) * (double(grp[i]) - double(f[g + i]));
            }
        }
        const double ratio = e_rel / e_flt;
        std::printf("integer-relative / FP32 group scale squared error ratio %.4f\n", ratio);
        check(ratio <= 1.03, "5: integer-relative scales cost <= 3 % squared error");
    }
    if (failures == 0) std::printf("iu4a4_rot host tests PASS (%s)\n", rot::kIdentity);
    return failures == 0 ? 0 : 1;
}
