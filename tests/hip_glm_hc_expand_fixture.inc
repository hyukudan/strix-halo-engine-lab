// Fixture-free gate for the fused mHC expand + norm: the shared row expansion (STRIX_GLM_PF_HC_EXPAND_BW) and the one-row
// fallback against the original single-row kernel. Included after hc_special.
// Every payload is compared by memcmp, including NaNs, signed zeros and both guards.
__global__ void hc_ref_sum2_kernel(const float* y0, const float* y1, float* sum, std::size_t count) {
    const std::size_t i = static_cast<std::size_t>(blockIdx.x) * 256 + threadIdx.x;
    if (i < count) sum[i] = y0[i] + y1[i];
}

void test_hc_expand(std::mt19937& g, bool timing) {
    constexpr unsigned n = kHc * kHidden;
    constexpr std::size_t guard = 17;
    for (unsigned R : {1U, 2U, 3U, 7U, 4096U, 8192U}) {
        const std::size_t nf = static_cast<std::size_t>(R) * n, ny = static_cast<std::size_t>(R) * kHidden;
        const unsigned stresses = !timing && R < 4096 ? 5 : 1;
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
                                  hipHostMallocMapped | hipHostMallocCoherent), "expand mapped RX allocation");
                std::memcpy(host_q, hq.data(), ny * sizeof(float));
                hok(hipHostGetDevicePointer(reinterpret_cast<void**>(&host_q_device), host_q, 0), "expand mapped RX view");
            }
            auto reset = [&] {
                hok(hipMemset(out, 0xFF, (nf + 2 * guard) * sizeof(float)), "expand reset output and guards");
                hok(hipMemset(rr, 0xFF, (R + 2 * guard) * sizeof(float)), "expand reset rr and guards");
                hok(hipMemcpy(flag, &seed, sizeof(seed), hipMemcpyHostToDevice), "expand reset flag");
            };
            hipEvent_t start{}, stop{};
            if (timing) { hok(hipEventCreate(&start), "expand event"); hok(hipEventCreate(&stop), "expand event"); }
            for (unsigned mem = 0; mem < (stress == 0 ? 2U : 1U); ++mem) {
                const float* q = mem ? host_q_device : Q;
                for (unsigned pair = 0; pair < 2; ++pair) {
                    hc_ref_sum2_kernel<<<static_cast<unsigned>((ny + 255) / 256), 256>>>(Y, q, sum, ny);
                    hok(hipGetLastError(), "expand sum2 reference");
                    hok(hipMemcpy(refflag, &seed, sizeof(seed), hipMemcpyHostToDevice), "expand reference flag");
                    P::hc_expand_norm_rows_h(0, X, mix, pair ? sum : Y, ref + guard, refrr + guard, refflag, R, nullptr);
                    const auto ref_x = down(ref, nf + 2 * guard), ref_r = down(refrr, R + 2 * guard);
                    const unsigned ref_f = down(refflag, 1)[0];
                    const std::string base = "expand R=" + std::to_string(R) + " stress=" + std::to_string(stress) +
                        " pair=" + std::to_string(pair) + " mapped=" + std::to_string(mem);
                    check((ref_f & seed) == seed && (stress != 2 || ref_f == (seed | 1U)), base + " reference flag stress");
                    auto same = [&] {
                        return bits_eq(down(out, nf + 2 * guard), ref_x) && bits_eq(down(rr, R + 2 * guard), ref_r) &&
                               down(flag, 1)[0] == ref_f;
                    };
                    auto run_bw = [&] {
                        return P::hc_expand_bw_rows(X, mix, Y, pair ? q : nullptr, out + guard, rr + guard, flag, R, nullptr);
                    };
                    auto run_row = [&](unsigned h) {
                        if (pair) P::hc_sum_expand_norm_rows_h(h, X, mix, Y, q, out + guard, rr + guard, flag, R, nullptr);
                        else P::hc_expand_norm_rows_h(h, X, mix, Y, out + guard, rr + guard, flag, R, nullptr);
                    };
                    // The one-row fallback (every hoist control) and the shared row expansion must each equal the
                    // original single-row kernel on the materialised sum.
                    for (unsigned h = 0; h < 3; ++h) {
                        reset();
                        run_row(h);
                        check(same(), base + " one-row hoist=" + std::to_string(h) + " strict payload/guards/flag");
                    }
                    reset();
                    check(run_bw() && same(), base + " bw strict payload/guards/flag");
                    check(!P::hc_expand_bw_rows(X, mix, Y, nullptr, X, rr + guard, flag, R, nullptr), base + " bw refuses Xo aliasing X");
                    // Admission must be fail-closed: optional pair input may be null, not the
                    // nonempty kernel's required buffers. None of these attempts may launch.
                    check(!P::hc_expand_bw_rows(nullptr, mix, Y, nullptr, out + guard, rr + guard, flag, R, nullptr),
                          base + " bw refuses missing X");
                    check(!P::hc_expand_bw_rows(X, nullptr, Y, nullptr, out + guard, rr + guard, flag, R, nullptr),
                          base + " bw refuses missing mix");
                    check(!P::hc_expand_bw_rows(X, mix, nullptr, nullptr, out + guard, rr + guard, flag, R, nullptr),
                          base + " bw refuses missing y");
                    check(!P::hc_expand_bw_rows(X, mix, Y, nullptr, nullptr, rr + guard, flag, R, nullptr),
                          base + " bw refuses missing Xo");
                    check(!P::hc_expand_bw_rows(X, mix, Y, nullptr, out + guard, nullptr, flag, R, nullptr),
                          base + " bw refuses missing norm output");
                    if (timing && stress == 0 && failures == 0) {
                        // ABBA: one-row kernel at the production hoist vs the shared row expansion, same inputs.
                        const unsigned h = P::hc_expand_hoist();
                        std::vector<float> times[2];
                        const unsigned repeats = R < 4096 ? 8 : 2;
                        run_row(h);
                        (void)run_bw();  // warm both organizations
                        for (unsigned round = 0; round < 4; ++round)
                            for (unsigned arm : {0U, 1U, 1U, 0U}) {
                                hok(hipEventRecord(start, nullptr), "expand start");
                                for (unsigned rep = 0; rep < repeats; ++rep) {
                                    if (arm) (void)run_bw();
                                    else run_row(h);
                                }
                                hok(hipEventRecord(stop, nullptr), "expand stop");
                                times[arm].push_back(ms_between(start, stop) / static_cast<float>(repeats));
                            }
                        for (auto& t : times) std::sort(t.begin(), t.end());
                        const float a = (times[0][3] + times[0][4]) * 0.5F, b = (times[1][3] + times[1][4]) * 0.5F;
                        std::printf("hc-expand-ABBA R=%u mapped=%u path=%s hoist=%u row_ms=%.6f bw_ms=%.6f delta_pct=%.3f samples=8 repeats=%u\n",
                                    R, mem, pair ? "pair" : "single", h, a, b, 100.0F * (b / a - 1.0F), repeats);
                        std::fflush(stdout);
                    }
                }
            }
            // A null flag is supported even for nonfinite odd-tail rows.
            P::hc_expand_norm_rows_h(0, X, mix, Y, ref + guard, refrr + guard, nullptr, R, nullptr);
            reset();
            check(P::hc_expand_bw_rows(X, mix, Y, nullptr, out + guard, rr + guard, nullptr, R, nullptr) &&
                      bits_eq(down(out, nf + 2 * guard), down(ref, nf + 2 * guard)) &&
                      bits_eq(down(rr, R + 2 * guard), down(refrr, R + 2 * guard)), "expand bw null flag strict");
            // The candidates must leave all read-only inputs intact.
            check(bits_eq(down(X, nf), hx) && bits_eq(down(Y, ny), hy) && bits_eq(down(Q, ny), hq) &&
                      bits_eq(down(mix, hm.size()), hm), "expand inputs unchanged");
            if (host_q) check(std::memcmp(host_q, hq.data(), ny * sizeof(float)) == 0, "expand mapped RX unchanged");
            if (timing) { hok(hipEventDestroy(start), "expand event"); hok(hipEventDestroy(stop), "expand event"); }
            if (host_q) hok(hipHostFree(host_q), "expand mapped RX free");
            free_all();
        }
    }
    P::hc_expand_norm_rows(nullptr, nullptr, nullptr, nullptr, nullptr, nullptr, 0, nullptr);  // zero rows: no launch
    hok(hipGetLastError(), "expand zero rows");
}
