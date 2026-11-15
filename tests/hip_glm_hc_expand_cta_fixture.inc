// Fixture-free gate for the expand CTA placement. Included after hc_special.
// Every payload is compared by memcmp, including NaNs, signed zeros and both guards.
#include <cmath>
__global__ void hc_cta_sum2_kernel(const float* y0, const float* y1, float* sum, std::size_t count) {
    const std::size_t i = static_cast<std::size_t>(blockIdx.x) * 256 + threadIdx.x;
    if (i < count) sum[i] = y0[i] + y1[i];
}

// Word-level mismatch classes for a guarded payload (guards are never payload-tolerant).
struct HcCtaDiff { std::size_t nan_payload = 0, other = 0, first_other = 0; };
HcCtaDiff hc_cta_classify(const std::vector<float>& got, const std::vector<float>& want, std::size_t guard = 17) {
    HcCtaDiff d;
    for (std::size_t i = 0; i < got.size() && i < want.size(); ++i) {
        if (std::memcmp(&got[i], &want[i], sizeof(float)) == 0) continue;
        const bool payload = i >= guard && i + guard < got.size() && std::isnan(got[i]) && std::isnan(want[i]);
        if (payload) { ++d.nan_payload; continue; }
        if (d.other++ == 0) d.first_other = i;
    }
    if (got.size() != want.size()) ++d.other;
    return d;
}
std::size_t g_hc_cta_nan_payload_words = 0, g_hc_cta_nan_payload_cases = 0;

void test_hc_expand_cta(std::mt19937& g, bool timing) {
    constexpr unsigned n = kHc * kHidden;
    constexpr std::size_t guard = 17;
    check(P::hc_expand_cta_rows() == 1, "expand CTA gate requires original wrapper (EXPAND_ROWS unset or 1)");
    if (failures) return;
    check(P::hc_expand_cta_resources(), "expand CTA resources: launchable, resident, no reported local bytes");
    if (failures) return;
    for (unsigned R : {1U, 2U, 3U, 7U, 2048U, 4096U, 8192U}) {
        const std::size_t nf = static_cast<std::size_t>(R) * n, ny = static_cast<std::size_t>(R) * kHidden;
        const unsigned stresses = !timing && R < 2048 ? 5 : 1;
        for (unsigned stress = 0; stress < stresses; ++stress) {
            // Bench replays the exact finite case admitted by the complete preceding gate.
            g.seed(20261025U + R * 5U + stress);
            auto hx = stress == 1 ? hc_stress_input(R, g) : rnd(nf, g, 1.0F);
            auto hy = rnd(ny, g, 1.0F), hq = rnd(ny, g, 1.0F);
            auto hm = rnd(static_cast<std::size_t>(R) * kHcMix, g, 0.2F);
            if (stress == 2) hy[(R - 1) * static_cast<std::size_t>(kHidden) + 5] = std::numeric_limits<float>::infinity();
            if (stress >= 3) {
                for (auto& v : hx) v = hc_special(g);
                for (auto& v : hy) v = hc_special(g);
                for (auto& v : hq) v = hc_special(g);
                std::fill(hm.begin(), hm.end(), 0.0F);
                for (unsigned r = 0; r < R; ++r) {
                    float* m = hm.data() + static_cast<std::size_t>(r) * kHcMix;
                    for (unsigned i = 0; i < kHc; ++i) {
                        m[kHc + i] = 1.0F;
                        m[2 * kHc + i * kHc + i] = 1.0F;
                    }
                    if (stress == 4)
                        for (unsigned d = 0; d < kHidden; ++d) {
                            const std::size_t xi = static_cast<std::size_t>(r) * n + d;
                            const std::size_t yi = static_cast<std::size_t>(r) * kHidden + d;
                            if (r % 2 == 0) { hy[yi] = -hx[xi]; hq[yi] = -0.0F; }
                            else hx[xi] = 3.0e37F;
                        }
                }
            }
            float *X = up(hx), *Y = up(hy), *Q = up(hq), *mix = up(hm), *sum = poison<float>(ny);
            float *ref = poison<float>(nf + 2 * guard), *refrr = poison<float>(R + 2 * guard);
            float *out = poison<float>(nf + 2 * guard), *rr = poison<float>(R + 2 * guard);
            unsigned *flag = zalloc<unsigned>(1), *refflag = zalloc<unsigned>(1);
            const unsigned seed = 0x40U;  // OR must preserve unrelated bits
            float* host_q = nullptr;
            float* host_q_device = nullptr;
            // Exercise the production mapped RX view too, without changing arithmetic.
            if (stress == 0) {
                hok(hipHostMalloc(reinterpret_cast<void**>(&host_q), ny * sizeof(float),
                                  hipHostMallocMapped | hipHostMallocCoherent), "CTA mapped RX allocation");
                std::memcpy(host_q, hq.data(), ny * sizeof(float));
                hok(hipHostGetDevicePointer(reinterpret_cast<void**>(&host_q_device), host_q, 0), "CTA mapped RX view");
            }
            auto reset = [&] {
                hok(hipMemset(out, 0xFF, (nf + 2 * guard) * sizeof(float)), "CTA reset output and guards");
                hok(hipMemset(rr, 0xFF, (R + 2 * guard) * sizeof(float)), "CTA reset rr and guards");
                hok(hipMemcpy(flag, &seed, sizeof(seed), hipMemcpyHostToDevice), "CTA reset flag");
            };
            hipEvent_t start{}, stop{};
            if (timing) { hok(hipEventCreate(&start), "CTA event"); hok(hipEventCreate(&stop), "CTA event"); }
            for (unsigned mem = 0; mem < (stress == 0 ? 2U : 1U); ++mem) {
                const float* q = mem ? host_q_device : Q;
                for (unsigned pair = 0; pair < 2; ++pair) {
                    hc_cta_sum2_kernel<<<static_cast<unsigned>((ny + 255) / 256), 256>>>(Y, q, sum, ny);
                    hok(hipGetLastError(), "CTA sum2 reference");
                    hok(hipMemcpy(refflag, &seed, sizeof(seed), hipMemcpyHostToDevice), "CTA reference flag");
                    P::hc_expand_norm_rows(X, mix, pair ? sum : Y, ref + guard, refrr + guard, refflag, R, nullptr);
                    const auto ref_x = down(ref, nf + 2 * guard), ref_r = down(refrr, R + 2 * guard);
                    const unsigned ref_f = down(refflag, 1)[0];
                    check((ref_f & seed) == seed && (stress != 2 || ref_f == (seed | 1U)), "expand CTA reference flag stress");
                    for (unsigned h = 0; h < 4; ++h) {
                        const std::string label = "expand CTA R=" + std::to_string(R) + " stress=" + std::to_string(stress) +
                            " pair=" + std::to_string(pair) + " hoist=" + std::to_string(h) + " mapped=" + std::to_string(mem);
                        auto run = [&](unsigned cta_rows, bool seam) {
                            if (seam && !pair) {
                                hc_cta_sum2_kernel<<<static_cast<unsigned>((ny + 255) / 256), 256>>>(Y, q, sum, ny);
                                hok(hipGetLastError(), "CTA sum2 seam");
                            }
                            P::hc_expand_norm_rows_cta(cta_rows, h, X, mix, seam && !pair ? sum : Y,
                                                       pair ? q : nullptr, out + guard, rr + guard, flag, R, nullptr);
                        };
                        // Each candidate must equal the original single-row kernel, not just its sibling.
                        for (unsigned cta_rows : {1U, 2U}) {
                            reset();
                            run(cta_rows, false);
                            const auto got_x = down(out, nf + 2 * guard), got_r = down(rr, R + 2 * guard);
                            const unsigned got_f = down(flag, 1)[0];
                            const bool strict = bits_eq(got_x, ref_x) && bits_eq(got_r, ref_r) && got_f == ref_f;
                            const std::string what = label + " rows/CTA=" + std::to_string(cta_rows);
                            if (h < 3 || strict) {
                                check(strict, what + " strict payload/guards/flag");
                                continue;
                            }
                            // hoist 3 is compiled separately: register allocation may commute an fadd / fmul whose two
                            // operands are both NaN, which changes only the NaN payload. Gate: every non-NaN word, every NaN
                            // position and the flag identical; the payload-only words are counted and reported.
                            const auto dx = hc_cta_classify(got_x, ref_x), dr = hc_cta_classify(got_r, ref_r);
                            g_hc_cta_nan_payload_words += dx.nan_payload + dr.nan_payload;
                            ++g_hc_cta_nan_payload_cases;
                            if (dx.other + dr.other != 0 || got_f != ref_f)
                                std::printf("hc-expand-diff %s: x nan-payload %zu other %zu first-other %zu; rr nan-payload %zu other %zu; flag %u/%u\n",
                                            what.c_str(), dx.nan_payload, dx.other, dx.first_other, dr.nan_payload, dr.other, got_f, ref_f);
                            check(dx.other == 0 && dr.other == 0 && got_f == ref_f, what + " bitwise except NaN payloads, guards/flag strict");
                        }
                        // Timed single seam includes sum2; regenerate its independent original-kernel reference.
                        if (timing && stress == 0 && failures == 0) {
                            if (!pair) P::hc_expand_norm_rows(X, mix, sum, ref + guard, refrr + guard, refflag, R, nullptr);
                            const auto seam_x = down(ref, nf + 2 * guard), seam_r = down(refrr, R + 2 * guard);
                            for (unsigned cta_rows : {1U, 2U}) {
                                reset(); run(cta_rows, true);
                                check(bits_eq(down(out, nf + 2 * guard), seam_x) && bits_eq(down(rr, R + 2 * guard), seam_r) &&
                                          down(flag, 1)[0] == down(refflag, 1)[0], label + " seam strict before timing");
                            }
                            if (failures) continue;
                            std::vector<float> times[2];
                            const unsigned repeats = R < 4096 ? 8 : 2;
                            run(1, true); run(2, true);  // warm both organizations
                            for (unsigned round = 0; round < 4; ++round)
                                for (unsigned cta_rows : {1U, 2U, 2U, 1U}) {
                                    hok(hipEventRecord(start, nullptr), "CTA start");
                                    for (unsigned rep = 0; rep < repeats; ++rep) run(cta_rows, true);
                                    hok(hipEventRecord(stop, nullptr), "CTA stop");
                                    times[cta_rows - 1].push_back(ms_between(start, stop) / static_cast<float>(repeats));
                                }
                            for (auto& t : times) std::sort(t.begin(), t.end());
                            const float a = (times[0][3] + times[0][4]) * 0.5F, b = (times[1][3] + times[1][4]) * 0.5F;
                            std::printf("hc-expand-ABBA R=%u mapped=%u path=%s hoist=%u rows1_ms=%.6f rows2_ms=%.6f delta_pct=%.3f samples=8 repeats=%u\n",
                                        R, mem, pair ? "pair" : "sum2+single", h, a, b, 100.0F * (b / a - 1.0F), repeats);
                            std::fflush(stdout);
                            // Restore the untimed reference for the next hoist variant.
                            if (!pair) P::hc_expand_norm_rows(X, mix, Y, ref + guard, refrr + guard, refflag, R, nullptr);
                        }
                    }
                }
            }
            // A null flag is supported even for nonfinite odd-tail rows.
            P::hc_expand_norm_rows(X, mix, Y, ref + guard, refrr + guard, nullptr, R, nullptr);
            reset();
            P::hc_expand_norm_rows_cta(2, 1, X, mix, Y, nullptr, out + guard, rr + guard, nullptr, R, nullptr);
            check(bits_eq(down(out, nf + 2 * guard), down(ref, nf + 2 * guard)) &&
                      bits_eq(down(rr, R + 2 * guard), down(refrr, R + 2 * guard)), "expand CTA null flag strict");
            // The candidate must leave all read-only inputs intact.
            check(bits_eq(down(X, nf), hx) && bits_eq(down(Y, ny), hy) && bits_eq(down(Q, ny), hq) &&
                      bits_eq(down(mix, hm.size()), hm), "expand CTA inputs unchanged");
            if (host_q) check(std::memcmp(host_q, hq.data(), ny * sizeof(float)) == 0, "expand CTA mapped RX unchanged");
            if (timing) { hok(hipEventDestroy(start), "CTA event"); hok(hipEventDestroy(stop), "CTA event"); }
            if (host_q) hok(hipHostFree(host_q), "CTA mapped RX free");
            free_all();
        }
    }
    for (unsigned cta_rows : {1U, 2U})
        for (unsigned h = 0; h < 4; ++h)
            P::hc_expand_norm_rows_cta(cta_rows, h, nullptr, nullptr, nullptr, nullptr, nullptr, nullptr, nullptr, 0, nullptr);
    for (unsigned invalid : {0U, 3U, 512U}) {
        bool rejected = false;
        try { P::hc_expand_norm_rows_cta(invalid, 0, nullptr, nullptr, nullptr, nullptr, nullptr, nullptr, nullptr, 0, nullptr); }
        catch (const std::invalid_argument&) { rejected = true; }
        check(rejected, "expand CTA rejects invalid row selector " + std::to_string(invalid));
    }
    bool rejected = false;
    try { P::hc_expand_norm_rows_cta(2, 4, nullptr, nullptr, nullptr, nullptr, nullptr, nullptr, nullptr, 0, nullptr); }
    catch (const std::invalid_argument&) { rejected = true; }
    check(rejected, "expand CTA rejects invalid hoist");
    std::printf("hc-expand hoist3 NaN-payload-only words %zu over %zu non-strict cases\n", g_hc_cta_nan_payload_words,
                g_hc_cta_nan_payload_cases);
    hok(hipGetLastError(), "expand CTA zero rows");
}
