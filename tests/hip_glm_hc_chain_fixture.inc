// Fixture-free gate + timing for the v2 mHC chain shape (rows per CTA x outputs per CTA). Included after the expand
// CTA fixture. Every shape must equal the 32-row x 24-output reference by memcmp (guards included), raw and normed
// (F = X * rr on load) forms; timing is ABAB over the shapes on the normed form (the product sites).
void test_hc_chain(std::mt19937& g, bool timing) {
    constexpr unsigned n = kHc * kHidden;
    constexpr std::size_t guard = 17;
    struct Shape { unsigned tile; bool cols8; };
    const Shape shapes[] = {{32, false}, {64, false}, {16, false}, {8, false}, {32, true}, {16, true}, {8, true}};
    for (unsigned R : {1U, 7U, 32U, 2048U, 4096U}) {
        g.seed(20261115U + R);
        const std::size_t nf = static_cast<std::size_t>(R) * n, nz = static_cast<std::size_t>(R) * kHcMix;
        auto hx = R <= 32 ? hc_stress_input(R, g) : rnd(nf, g, 1.0F);
        auto hfn = rnd(static_cast<std::size_t>(kHcMix) * n, g, 0.05F);
        std::vector<float> hr(R);
        std::uniform_real_distribution<float> ur(0.25F, 4.0F);
        for (auto& v : hr) v = ur(g);
        const float *X = up(hx), *fn = up(hfn), *rr = up(hr);
        float* ref = poison<float>(nz + 2 * guard);
        float* out = poison<float>(nz + 2 * guard);
        for (unsigned fx = 0; fx < 2; ++fx) {
            hok(hipMemset(ref, 0xFF, (nz + 2 * guard) * sizeof(float)), "chain reset ref");
            P::hc_chain_rows_shape(32, false, fx != 0, X, fn, ref + guard, R, rr, nullptr);
            const auto want = down(ref, nz + 2 * guard);
            for (const Shape& sh : shapes) {
                hok(hipMemset(out, 0xFF, (nz + 2 * guard) * sizeof(float)), "chain reset out");
                P::hc_chain_rows_shape(sh.tile, sh.cols8, fx != 0, X, fn, out + guard, R, rr, nullptr);
                check(bits_eq(down(out, nz + 2 * guard), want),
                      "hc chain R=" + std::to_string(R) + " tile=" + std::to_string(sh.tile) + " cols=" + (sh.cols8 ? "8" : "24") +
                          " normed=" + std::to_string(fx) + " strict payload/guards");
            }
        }
        if (timing && R >= 2048 && failures == 0) {
            hipEvent_t start{}, stop{};
            hok(hipEventCreate(&start), "chain event");
            hok(hipEventCreate(&stop), "chain event");
            constexpr unsigned kShapes = sizeof(shapes) / sizeof(shapes[0]), repeats = 8, rounds = 6;
            std::vector<float> times[kShapes];
            for (const Shape& sh : shapes) P::hc_chain_rows_shape(sh.tile, sh.cols8, true, X, fn, out + guard, R, rr, nullptr);
            for (unsigned round = 0; round < rounds; ++round)
                for (unsigned k = 0; k < kShapes; ++k) {
                    const unsigned i = round % 2 ? kShapes - 1 - k : k;  // ABBA order across rounds
                    hok(hipEventRecord(start, nullptr), "chain start");
                    for (unsigned rep = 0; rep < repeats; ++rep)
                        P::hc_chain_rows_shape(shapes[i].tile, shapes[i].cols8, true, X, fn, out + guard, R, rr, nullptr);
                    hok(hipEventRecord(stop, nullptr), "chain stop");
                    times[i].push_back(ms_between(start, stop) / static_cast<float>(repeats));
                }
            for (unsigned i = 0; i < kShapes; ++i) {
                std::sort(times[i].begin(), times[i].end());
                std::printf("hc-chain R=%u tile=%u cols=%u median_ms=%.6f min_ms=%.6f vs_ref_pct=%.2f samples=%u repeats=%u\n", R,
                            shapes[i].tile, shapes[i].cols8 ? 8U : 24U, (times[i][2] + times[i][3]) * 0.5F, times[i][0],
                            100.0F * ((times[i][2] + times[i][3]) / (times[0][2] + times[0][3]) - 1.0F), rounds, repeats);
            }
            std::fflush(stdout);
            hok(hipEventDestroy(start), "chain event");
            hok(hipEventDestroy(stop), "chain event");
        }
        free_all();
    }
    std::printf("hc-chain-rows auto: R=1 -> %u, R=2048 -> %u, R=4096 -> %u\n", P::hc_chain_rows(1), P::hc_chain_rows(2048),
                P::hc_chain_rows(4096));
    bool rejected = false;
    try { P::hc_chain_rows_shape(4, false, false, nullptr, nullptr, nullptr, 1, nullptr, nullptr); }
    catch (const std::invalid_argument&) { rejected = true; }
    check(rejected, "hc chain rejects invalid tile");
}
