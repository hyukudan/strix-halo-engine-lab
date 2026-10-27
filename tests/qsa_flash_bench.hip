// qpf2k/attn microbench (standalone; build: amdclang++ -O3 -std=gnu++20 --offload-arch=gfx1151 -Iinclude -x hip, the engine TU's flags).
// Times the QSA prefill part-2 kernels of one layer-chunk (T rows from position P) on synthetic data, the
// way enqueue_qsa_known_rows runs them for 64-row groups (pool, block scores, selection, batched union +
// attend), and checks the qpf2k kernels bitwise against the production ones (tests/qsa_flash_prod.inc).
#include "qsa_flash_prod.inc"
using namespace strix;
static void check(hipError_t e, const char* what) { if (e != hipSuccess) { std::fprintf(stderr, "%s: %s\n", what, hipGetErrorString(e)); std::exit(2); } }
template <class T> static T* dalloc(std::size_t n) { void* p = nullptr; check(hipMalloc(&p, n * sizeof(T)), "hipMalloc"); return static_cast<T*>(p); }
static std::uint16_t host_bf16(float v) { std::uint32_t b; std::memcpy(&b, &v, 4); return static_cast<std::uint16_t>((b + 0x7fffU + ((b >> 16U) & 1U)) >> 16U); }
static float host_from_bf16(std::uint16_t h) { std::uint32_t b = static_cast<std::uint32_t>(h) << 16U; float v; std::memcpy(&v, &b, 4); return v; }

int main(int argc, char** argv) {
    std::uint64_t P = 32768, T = 8192; unsigned reps = 3; std::string what = "all"; unsigned seed = 1; float corr = 0.0F;
    for (int i = 1; i + 1 < argc; i += 2) {
        const std::string a = argv[i];
        if (a == "--pos") P = std::strtoull(argv[i + 1], nullptr, 10);
        else if (a == "--rows") T = std::strtoull(argv[i + 1], nullptr, 10);
        else if (a == "--reps") reps = static_cast<unsigned>(std::atoi(argv[i + 1]));
        else if (a == "--what") what = argv[i + 1];
        else if (a == "--seed") seed = static_cast<unsigned>(std::atoi(argv[i + 1]));
        else if (a == "--corr") corr = std::strtof(argv[i + 1], nullptr); // query rows as a random walk: q_r = c q_{r-1} + sqrt(1-c^2) noise
    }
    if (T % 64U) { std::fprintf(stderr, "rows must be a multiple of 64\n"); return 2; }
    constexpr unsigned ratio = 4, topk = 512, dim = 128, heads = 4, maxsel = 2051, aheads = 24, kvh = 2, hd = 256;
    constexpr std::uint64_t score_stride = 65536; // block_scores floats per row (262144 / 4)
    const std::uint64_t L = P + T, blocks_max = L / ratio, groups = T / 64U;
    // fast pseudo-normal (sum of four uniforms), deterministic per seed
    struct Fast { std::uint64_t x; float operator()() { float t = 0.0F; for (int i = 0; i < 4; ++i) { x ^= x << 13; x ^= x >> 7; x ^= x << 17;
        t += static_cast<float>(x >> 40) * (1.0F / 16777216.0F); } return (t - 2.0F) * 1.7320508F; } } nd{0x9e3779b97f4a7c15ULL * (seed + 1U)};
    // raw index keys (BF16) of every token; index-k norm weight; index queries (BF16 values) per row.
    std::vector<std::uint16_t> raw(L * dim);
    for (auto& x : raw) x = host_bf16(nd());
    std::vector<std::uint16_t> knorm(dim);
    for (auto& x : knorm) x = host_bf16(1.0F + 0.1F * nd());
    std::vector<float> q(T * heads * dim);
    for (auto& x : q) x = host_from_bf16(host_bf16(nd()));
    if (corr > 0.0F) {
        const float c = corr, e = std::sqrt(1.0F - corr * corr);
        std::vector<float> prev(heads * dim);
        for (std::uint64_t r = 0; r < T; ++r)
            for (unsigned i = 0; i < heads * dim; ++i) {
                const float v = r ? c * prev[i] + e * nd() : nd();
                prev[i] = v; q[r * heads * dim + i] = host_from_bf16(host_bf16(v));
            }
    }
    // Edge rows (every 64-row group): values outside the fused-form guard (tiny, huge, subnormal, inf, NaN, -0).
    {
        const float edge[] = {std::ldexp(1.0F, -110), std::ldexp(1.0F, 124), 1e-40F, std::numeric_limits<float>::infinity(),
            std::numeric_limits<float>::quiet_NaN(), -0.0F, std::ldexp(1.0F, -60), std::ldexp(1.5F, 63)};
        for (std::uint64_t g = 0; g < T / 64U; ++g)
            for (unsigned e = 0; e < 8U; ++e) q[(g * 64U + 5U + 7U * e) * heads * dim + (e * 37U) % (heads * dim)] = host_from_bf16(host_bf16(edge[e]));
        // row 3 of every 8th group: all-zero query (every score 0: the selection's candidate set overflows)
        for (std::uint64_t g = 0; g < T / 64U; g += 8U) std::fill_n(q.begin() + (g * 64U + 3U) * heads * dim, heads * dim, 0.0F);
    }
    auto* d_raw = dalloc<std::uint16_t>(raw.size()); check(hipMemcpy(d_raw, raw.data(), raw.size() * 2, hipMemcpyHostToDevice), "h2d");
    auto* d_knorm = dalloc<std::uint16_t>(dim); check(hipMemcpy(d_knorm, knorm.data(), dim * 2, hipMemcpyHostToDevice), "h2d");
    auto* d_q = dalloc<float>(q.size()); check(hipMemcpy(d_q, q.data(), q.size() * 4, hipMemcpyHostToDevice), "h2d");
    auto* d_pool = dalloc<float>(blocks_max * dim);
    auto* d_ranges = dalloc<uint2>(blocks_max);
    auto* d_scores = dalloc<float>(T * score_stride);
    auto* d_scores2 = dalloc<float>(T * score_stride);
    auto* d_sel = dalloc<std::uint32_t>(T * maxsel);
    auto* d_sel2 = dalloc<std::uint32_t>(T * maxsel);
    constexpr std::uint32_t max_stride = 65536U / qsa_flash::kGroup;
    auto* d_max = dalloc<float>(T * max_stride); // per-group maxima (the engine reuses one group's)
    hipStream_t s; check(hipStreamCreate(&s), "stream");
    hipEvent_t e0, e1; check(hipEventCreate(&e0), "event"); check(hipEventCreate(&e1), "event");
    const auto timed = [&](auto&& body) { float best = 1e30F, sum = 0; for (unsigned r = 0; r < reps; ++r) {
        check(hipEventRecord(e0, s), "rec"); body(); check(hipEventRecord(e1, s), "rec"); check(hipEventSynchronize(e1), "sync"); float ms = 0; check(hipEventElapsedTime(&ms, e0, e1), "elapsed");
        best = std::min(best, ms); sum += ms; } check(hipGetLastError(), "launch"); return std::make_pair(best, sum / reps); };
    // Per-group views as enqueue_qsa_known_rows builds them.
    std::vector<QsaRowsScoreViews<64>> sv(groups), sv2(groups);
    std::vector<QsaRowsSelectViews<64>> lv(groups), lv2(groups);
    std::vector<std::uint32_t> scored(groups);
    for (std::uint64_t g = 0; g < groups; ++g) {
        std::uint32_t sb = 0;
        for (unsigned r = 0; r < 64U; ++r) {
            const std::uint64_t row = g * 64U + r, pos = P + row, visible = pos + 1, complete = visible / ratio;
            sv[g].index_query[r] = d_q + row * heads * dim;
            sv[g].block_scores[r] = d_scores + row * score_stride;
            sv[g].blocks[r] = complete > topk ? static_cast<std::uint32_t>(complete) : 0U;
            sb = std::max(sb, sv[g].blocks[r]);
            lv[g].block_scores[r] = sv[g].block_scores[r]; lv[g].selected[r] = d_sel + row * maxsel; lv[g].visible[r] = static_cast<std::uint32_t>(visible);
        }
        sv2[g] = sv[g];
        for (unsigned r = 0; r < 64U; ++r) sv2[g].block_scores[r] = d_scores2 + (g * 64U + r) * score_stride;
        lv2[g] = lv[g];
        for (unsigned r = 0; r < 64U; ++r) { lv2[g].block_scores[r] = sv2[g].block_scores[r]; lv2[g].selected[r] = d_sel2 + (g * 64U + r) * maxsel; }
        scored[g] = sb;
    }
    std::printf("P %llu T %llu L %llu groups %llu blocks_max %llu pairs %.4e\n", (unsigned long long)P, (unsigned long long)T,
        (unsigned long long)L, (unsigned long long)groups, (unsigned long long)blocks_max,
        [&] { double p = 0; for (auto& v : sv) for (unsigned r = 0; r < 64U; ++r) p += v.blocks[r]; return p; }());
    // Pool: production re-pools all blocks the first group needs, then the increments.
    const auto pool_all = [&] {
        std::uint32_t valid = 0;
        for (std::uint64_t g = 0; g < groups; ++g) if (scored[g] > valid) {
            hipLaunchKernelGGL(HIP_KERNEL_NAME(block_pool_kernel<SmallWeightKind::bf16>), dim3(scored[g] - valid), dim3(dim), 2 * dim * 4, s,
                d_raw, valid, dim, ratio, d_knorm, 1e-6F, 64U, 1e7F, d_pool);
            hipLaunchKernelGGL(HIP_KERNEL_NAME(qsa_flash::key_range_kernel<>), dim3((scored[g] - valid + 255U) / 256U), dim3(256), 0, s, d_pool, valid, scored[g] - valid, d_ranges);
            valid = scored[g];
        }
    };
    { auto t = timed(pool_all); std::printf("pool          best %8.3f ms  mean %8.3f\n", t.first, t.second); }
    const auto parked = [&] {
        for (std::uint64_t g = 0; g < groups; ++g) if (scored[g]) {
            const auto waves = static_cast<std::uint64_t>(64U / 4U) * ((scored[g] + 31U) / 32U);
            hipLaunchKernelGGL(HIP_KERNEL_NAME(qsa_index::block_score_parked_kernel<64U, 4U>), dim3(static_cast<std::uint32_t>((waves + 7U) / 8U)), dim3(256), 0, s,
                sv[g], d_pool, scored[g], dim);
        }
    };
    const auto flash_one = [&]<unsigned RowsPerCta, unsigned W, bool Lds = false, bool Fence = true, unsigned U = 1U>(std::uint64_t g, float* maxima) {
        const unsigned tiles = (scored[g] + 31U) / 32U;
        hipLaunchKernelGGL(HIP_KERNEL_NAME(qsa_flash::score_kernel<64U, RowsPerCta, W, false, Lds, Fence, U>), dim3((tiles + W - 1U) / W, 64U / RowsPerCta), dim3(W * 32U), 0, s,
            sv2[g], d_pool, scored[g], dim, maxima, max_stride, d_ranges, nullptr);
    };
    const auto flash = [&]<unsigned RowsPerCta, unsigned W, bool Lds = false, bool Fence = true, unsigned U = 1U>() {
        for (std::uint64_t g = 0; g < groups; ++g) if (scored[g]) flash_one.template operator()<RowsPerCta, W, Lds, Fence, U>(g, nullptr);
    };
    const auto key_ranges_all = [&] {
        std::uint32_t valid = 0;
        for (std::uint64_t g = 0; g < groups; ++g) if (scored[g] > valid) {
            hipLaunchKernelGGL(HIP_KERNEL_NAME(qsa_flash::key_range_kernel<>), dim3((scored[g] - valid + 255U) / 256U), dim3(256), 0, s, d_pool, valid, scored[g] - valid, d_ranges);
            valid = scored[g];
        }
    };
    // qpf2k pair: scores + maxima, then the candidate selection, per group (as the engine interleaves them)
    const auto flash_pair = [&] {
        for (std::uint64_t g = 0; g < groups; ++g) {
            if (scored[g]) flash_one.template operator()<qsa_flash::kScoreRows, qsa_flash::kScoreWaves>(g, d_max + g * 64U * max_stride);
            hipLaunchKernelGGL(HIP_KERNEL_NAME(qsa_flash::select_kernel<64U>), dim3(64), dim3(kSelectThreads), 0, s, lv2[g], d_max + g * 64U * max_stride, max_stride, ratio, topk);
        }
    };
    const auto flash_scores_max = [&] {
        for (std::uint64_t g = 0; g < groups; ++g) if (scored[g]) flash_one.template operator()<qsa_flash::kScoreRows, qsa_flash::kScoreWaves>(g, d_max + g * 64U * max_stride);
    };
    const auto select_all = [&] {
        for (std::uint64_t g = 0; g < groups; ++g)
            hipLaunchKernelGGL(HIP_KERNEL_NAME(select_rows_kernel<64U>), dim3(64), dim3(kSelectThreads), 0, s, lv[g], ratio, topk);
    };
    pool_all(); check(hipStreamSynchronize(s), "sync");
    if (what == "all" || what == "score") {
        auto t = timed(parked); std::printf("score parked  best %8.3f ms  mean %8.3f\n", t.first, t.second);
        const auto cmp = [&](const char* name) {
            check(hipStreamSynchronize(s), "sync");
            std::vector<float> a(score_stride), b(score_stride);
            std::uint64_t differ = 0, n = 0;
            for (std::uint64_t row = 0; row < T; row += 1) {
                const auto blocks = sv[row / 64U].blocks[row % 64U];
                if (!blocks) continue;
                check(hipMemcpy(a.data(), d_scores + row * score_stride, blocks * 4ULL, hipMemcpyDeviceToHost), "d2h");
                check(hipMemcpy(b.data(), d_scores2 + row * score_stride, blocks * 4ULL, hipMemcpyDeviceToHost), "d2h");
                for (std::uint32_t i = 0; i < blocks; ++i) differ += std::memcmp(&a[i], &b[i], 4) != 0;
                n += blocks;
            }
            std::printf("  check %-14s %llu of %llu scores differ\n", name, (unsigned long long)differ, (unsigned long long)n);
            check(hipMemset(d_scores2, 0xff, T * score_stride * 4), "memset");
            return differ;
        };
        check(hipMemset(d_scores2, 0xff, T * score_stride * 4), "memset");
        std::uint64_t bad = 0;
        { auto t2 = timed([&] { flash.template operator()<64U, 8U>(); }); std::printf("score flash64x8 best %8.3f ms  mean %8.3f\n", t2.first, t2.second); bad += cmp("flash64x8"); }
        { auto t2 = timed([&] { flash.template operator()<16U, 8U>(); }); std::printf("score flash16x8 best %8.3f ms  mean %8.3f\n", t2.first, t2.second); bad += cmp("flash16x8"); }
        { auto t2 = timed([&] { flash.template operator()<8U, 4U>(); }); std::printf("score flash8x4  best %8.3f ms  mean %8.3f\n", t2.first, t2.second); bad += cmp("flash8x4"); }
        { auto t2 = timed([&] { flash.template operator()<32U, 8U>(); }); std::printf("score flash32x8 best %8.3f ms  mean %8.3f\n", t2.first, t2.second); bad += cmp("flash32x8"); }
        { auto t2 = timed([&] { flash.template operator()<16U, 4U>(); }); std::printf("score flash16x4 best %8.3f ms  mean %8.3f\n", t2.first, t2.second); bad += cmp("flash16x4"); }
        { auto t2 = timed([&] { flash.template operator()<16U, 8U, true>(); }); std::printf("score lds16x8  best %8.3f ms  mean %8.3f\n", t2.first, t2.second); bad += cmp("lds16x8"); }
        { auto t2 = timed([&] { flash.template operator()<8U, 8U, true>(); }); std::printf("score lds8x8   best %8.3f ms  mean %8.3f\n", t2.first, t2.second); bad += cmp("lds8x8"); }
        { auto t2 = timed([&] { flash.template operator()<16U, 8U, true, false>(); }); std::printf("score lds16x8nf best %8.3f ms  mean %8.3f\n", t2.first, t2.second); bad += cmp("lds16x8nf"); }
        { auto t2 = timed([&] { flash.template operator()<16U, 4U, true, false>(); }); std::printf("score lds16x4nf best %8.3f ms  mean %8.3f\n", t2.first, t2.second); bad += cmp("lds16x4nf"); }
        { auto t2 = timed([&] { flash.template operator()<16U, 8U, true, false, 2U>(); }); std::printf("score lds16x8u2 best %8.3f ms  mean %8.3f\n", t2.first, t2.second); bad += cmp("lds16x8u2"); }
        { auto t2 = timed([&] { flash.template operator()<16U, 8U, false, true, 2U>(); }); std::printf("score sgpr16x8u2 best %8.3f ms  mean %8.3f\n", t2.first, t2.second); bad += cmp("sgpr16x8u2"); }
        { auto t2 = timed(key_ranges_all); std::printf("key ranges    best %8.3f ms  mean %8.3f\n", t2.first, t2.second); }
        { auto t2 = timed(flash_scores_max); std::printf("score flash+max best %8.3f ms  mean %8.3f\n", t2.first, t2.second); bad += cmp("flash+max"); }
        if (bad) { std::printf("FAIL flash scores differ\n"); return 1; }
    }
    if (what == "all" || what == "select") {
        parked(); check(hipStreamSynchronize(s), "sync");
        // selection overwrites selected scores with -inf: time one pass on fresh scores per rep
        float best = 1e30F;
        for (unsigned r = 0; r < reps; ++r) {
            parked(); check(hipStreamSynchronize(s), "sync");
            hipEventRecord(e0, s); select_all(); hipEventRecord(e1, s); hipEventSynchronize(e1);
            float ms = 0; hipEventElapsedTime(&ms, e0, e1); best = std::min(best, ms);
        }
        std::printf("select        best %8.3f ms\n", best);
        { auto t = timed([&] { parked(); select_all(); }); std::printf("score+select  best %8.3f ms  mean %8.3f (parked)\n", t.first, t.second); }
        { auto t = timed(flash_pair); std::printf("score+select  best %8.3f ms  mean %8.3f (qpf2k)\n", t.first, t.second); }
        // exactness: production parked + select_rows vs qpf2k scorer + select_kernel, fresh scores on both sides
        parked(); select_all(); flash_pair(); check(hipStreamSynchronize(s), "sync");
        {
            std::uint64_t sel_differ = 0, score_differ = 0, rows_checked = 0;
            std::vector<std::uint32_t> a(maxsel), b(maxsel);
            std::vector<float> x(score_stride), y(score_stride);
            for (std::uint64_t row = 0; row < T; ++row) {
                const std::uint64_t visible = P + row + 1, complete = visible / ratio;
                const auto count = complete <= topk ? visible : topk * ratio + (visible - complete * ratio);
                check(hipMemcpy(a.data(), d_sel + row * maxsel, count * 4, hipMemcpyDeviceToHost), "d2h");
                check(hipMemcpy(b.data(), d_sel2 + row * maxsel, count * 4, hipMemcpyDeviceToHost), "d2h");
                sel_differ += std::memcmp(a.data(), b.data(), count * 4) != 0;
                const auto blocks = sv[row / 64U].blocks[row % 64U];
                if (blocks) {
                    check(hipMemcpy(x.data(), d_scores + row * score_stride, blocks * 4ULL, hipMemcpyDeviceToHost), "d2h");
                    check(hipMemcpy(y.data(), d_scores2 + row * score_stride, blocks * 4ULL, hipMemcpyDeviceToHost), "d2h");
                    score_differ += std::memcmp(x.data(), y.data(), blocks * 4ULL) != 0;
                }
                ++rows_checked;
            }
            std::printf("  check select  %llu of %llu rows: selections differ in %llu, post-selection scores differ in %llu\n",
                (unsigned long long)rows_checked, (unsigned long long)rows_checked, (unsigned long long)sel_differ, (unsigned long long)score_differ);
            if (sel_differ || score_differ) { std::printf("FAIL selection differs\n"); return 1; }
        }
        {
            float best2 = 1e30F;
            for (unsigned r = 0; r < reps; ++r) {
                flash_scores_max(); check(hipStreamSynchronize(s), "sync");
                hipEventRecord(e0, s);
                for (std::uint64_t g = 0; g < groups; ++g)
                    hipLaunchKernelGGL(HIP_KERNEL_NAME(qsa_flash::select_kernel<64U>), dim3(64), dim3(kSelectThreads), 0, s, lv2[g], d_max + g * 64U * max_stride, max_stride, ratio, topk);
                hipEventRecord(e1, s); hipEventSynchronize(e1);
                float ms = 0; hipEventElapsedTime(&ms, e0, e1); best2 = std::min(best2, ms);
            }
            std::printf("select qpf2k  best %8.3f ms \n", best2);
        }
    }
    if (what == "all" || what == "attend") {
        // Selections from the scores (fresh), then union + attend over all groups in one launch pair.
        parked(); select_all(); check(hipStreamSynchronize(s), "sync");
        std::vector<std::uint16_t> kv(L * kvh * hd * 2);
        for (auto& x : kv) x = host_bf16(0.5F * nd());
        auto* d_kv = dalloc<std::uint16_t>(kv.size()); check(hipMemcpy(d_kv, kv.data(), kv.size() * 2, hipMemcpyHostToDevice), "h2d");
        std::vector<float> qg(T * aheads * hd * 2);
        for (auto& x : qg) x = nd();
        auto* d_qg = dalloc<float>(qg.size()); check(hipMemcpy(d_qg, qg.data(), qg.size() * 4, hipMemcpyHostToDevice), "h2d");
        auto* d_scr = dalloc<float>(T * aheads * maxsel);
        auto* d_out = dalloc<float>(T * aheads * hd);
        std::vector<qsa_fast::Views<64U>> views(groups);
        for (std::uint64_t g = 0; g < groups; ++g)
            for (unsigned r = 0; r < 64U; ++r) {
                const std::uint64_t row = g * 64U + r, visible = P + row + 1;
                views[g].q_gate[r] = d_qg + row * aheads * hd * 2; views[g].selected[r] = d_sel + row * maxsel;
                views[g].scratch[r] = d_scr + row * aheads * maxsel; views[g].output[r] = d_out + row * aheads * hd;
                const std::uint64_t complete = visible / ratio;
                views[g].count[r] = static_cast<std::uint32_t>(complete <= topk ? visible : topk * ratio + (visible - complete * ratio));
                views[g].visible[r] = static_cast<std::uint32_t>(visible);
            }
        qsa_fast::Views<64U>* d_views = dalloc<qsa_fast::Views<64U>>(groups);
        check(hipMemcpy(d_views, views.data(), groups * sizeof(qsa_fast::Views<64U>), hipMemcpyHostToDevice), "h2d");
        constexpr unsigned TR = 4U, G = 3U, HG = G * (16U / TR);
        const auto union_at = 1U * aheads * qsa_fast::kPartial;
        const auto n = static_cast<std::uint32_t>(groups);
        { auto t = timed([&] { hipLaunchKernelGGL(HIP_KERNEL_NAME(qsa_fast::union_multi_kernel<64U, TR>), dim3(n * (64U / TR)), dim3(256), 0, s, d_views, union_at); });
          std::printf("union         best %8.3f ms  mean %8.3f\n", t.first, t.second); }
        { auto t = timed([&] { hipLaunchKernelGGL(HIP_KERNEL_NAME(qsa_fast::attend_multi_kernel<64U, TR, G>), dim3(n * (64U / TR), kvh * (aheads / kvh / HG), 1U),
              dim3((2U * G + 4U) * 32U), 0, s, d_views, d_kv, kvh, aheads, union_at); });
          std::printf("attend        best %8.3f ms  mean %8.3f\n", t.first, t.second); }
        {   // qpf2k pipelined attend (one barrier per chunk): outputs bitwise == the production attend
            check(hipStreamSynchronize(s), "sync");
            const std::size_t outn = T * aheads * hd;
            std::vector<float> ref(outn), got(outn);
            check(hipMemcpy(ref.data(), d_out, outn * 4, hipMemcpyDeviceToHost), "d2h");
            check(hipMemset(d_out, 0xA5, outn * 4), "memset");
            auto t = timed([&] { hipLaunchKernelGGL(HIP_KERNEL_NAME(qsa_fast::attend_pipe_multi_kernel<64U, TR, G>), dim3(n * (64U / TR), kvh * (aheads / kvh / HG), 1U),
                dim3((2U * G + 4U) * 32U), 0, s, d_views, d_kv, kvh, aheads, union_at); });
            std::printf("attend pipe   best %8.3f ms  mean %8.3f\n", t.first, t.second);
            check(hipStreamSynchronize(s), "sync");
            check(hipMemcpy(got.data(), d_out, outn * 4, hipMemcpyDeviceToHost), "d2h");
            std::size_t differ = 0;
            for (std::size_t i = 0; i < outn; ++i) differ += std::memcmp(&ref[i], &got[i], 4) != 0;
            std::printf("  check attend pipe %zu of %zu outputs differ\n", differ, outn);
            // load depth variants (the same arithmetic; D = chunks in flight per loader)
            const auto variant = [&](const char* name, auto&& launch) {
                check(hipMemset(d_out, 0xA5, outn * 4), "memset");
                auto tv = timed(launch);
                check(hipStreamSynchronize(s), "sync");
                check(hipMemcpy(got.data(), d_out, outn * 4, hipMemcpyDeviceToHost), "d2h");
                std::size_t dv = 0;
                for (std::size_t i = 0; i < outn; ++i) dv += std::memcmp(&ref[i], &got[i], 4) != 0;
                std::printf("attend %-7s best %8.3f ms  mean %8.3f  (%zu outputs differ)\n", name, tv.first, tv.second, dv);
                return dv;
            };
            const dim3 grid(n * (64U / TR), kvh * (aheads / kvh / HG), 1U), block((2U * G + 4U) * 32U);
            differ += variant("D2", [&] { hipLaunchKernelGGL(HIP_KERNEL_NAME(qsa_fast::attend_multi_kernel<64U, TR, G, 2U>), grid, block, 0, s, d_views, d_kv, kvh, aheads, union_at); });
            differ += variant("D4", [&] { hipLaunchKernelGGL(HIP_KERNEL_NAME(qsa_fast::attend_multi_kernel<64U, TR, G, 4U>), grid, block, 0, s, d_views, d_kv, kvh, aheads, union_at); });
            {   // I on the production BF16 attend: group-major order (+ dynamic LDS to cap workgroups per WGP), per-group launches
                const unsigned ys = grid.y;
                const dim3 g1(grid.x * ys);
                for (const unsigned pad : {0U, 5000U}) {
                    char name[32]; std::snprintf(name, sizeof name, "bford%u", pad / 1000U);
                    differ += variant(name, [&] { hipLaunchKernelGGL(HIP_KERNEL_NAME(qsa_fast::attend_order_kernel<64U, TR, G, 3U>), g1, block, pad, s, d_views, d_kv, kvh, aheads, union_at, ys); });
                }
                differ += variant("bfpergr", [&] { for (std::uint64_t g = 0; g < groups; ++g)
                    hipLaunchKernelGGL(HIP_KERNEL_NAME(qsa_fast::attend_order_kernel<64U, TR, G, 3U>), dim3(16U * ys), block, 0, s, d_views + g, d_kv, kvh, aheads, union_at, ys); });
            }
            differ += variant("share", [&] { hipLaunchKernelGGL(HIP_KERNEL_NAME(qsa_fast::attend_share_multi_kernel<64U, TR, G, 3U>), grid, block, 0, s, d_views, d_kv, kvh, aheads, union_at); });
            {   // F: FP16 shadow of the KV (one conversion per call) + attend without per-chunk conversion
                auto* d_kv16 = dalloc<std::uint16_t>(kv.size());
                const std::uint64_t words = kv.size() / 2U;
                const auto convert = [&] { hipLaunchKernelGGL(qsa_fast::kv_to_f16_kernel, dim3(static_cast<std::uint32_t>((words / 4U + 255U) / 256U + 1U)), dim3(256), 0, s,
                    reinterpret_cast<const unsigned*>(d_kv), reinterpret_cast<unsigned*>(d_kv16), 0ULL, words); };
                auto tc = timed(convert);
                std::printf("kv_to_f16     best %8.3f ms  mean %8.3f  (%.1f MB, whole KV of L tokens)\n", tc.first, tc.second, words * 8.0 / 1e6);
                differ += variant("shadow", [&] { hipLaunchKernelGGL(HIP_KERNEL_NAME(qsa_fast::attend_shadow_multi_kernel<64U, TR, G, 3U>), grid, block, 0, s, d_views, d_kv16, kvh, aheads, union_at); });
                differ += variant("shshare", [&] { hipLaunchKernelGGL(HIP_KERNEL_NAME(qsa_fast::attend_shshare_multi_kernel<64U, TR, G, 3U>), grid, block, 0, s, d_views, d_kv16, kvh, aheads, union_at); });
                differ += variant("shadOcc", [&] { hipLaunchKernelGGL(HIP_KERNEL_NAME(qsa_fast::attend_shadow_occ_multi_kernel<64U, TR, G, 2U>), grid, block, 0, s, d_views, d_kv16, kvh, aheads, union_at); });
                differ += variant("shpipe", [&] { hipLaunchKernelGGL(HIP_KERNEL_NAME(qsa_fast::attend_shpipe_multi_kernel<64U, TR, G, 3U>), grid, block, 0, s, d_views, d_kv16, kvh, aheads, union_at); });
                differ += variant("shpipeD2", [&] { hipLaunchKernelGGL(HIP_KERNEL_NAME(qsa_fast::attend_shpipe_multi_kernel<64U, TR, G, 2U>), grid, block, 0, s, d_views, d_kv16, kvh, aheads, union_at); });
                {   // I: group-major order (+ dynamic LDS to cap workgroups per WGP)
                    const unsigned ys = grid.y;
                    const dim3 g1(grid.x * ys);
                    for (const unsigned pad : {0U, 5000U}) {
                        char name[32]; std::snprintf(name, sizeof name, "order%u", pad / 1024U);
                        differ += variant(name, [&] { hipLaunchKernelGGL(HIP_KERNEL_NAME(qsa_fast::attend_shadow_order_kernel<64U, TR, G, 3U>), g1, block, pad, s, d_views, d_kv16, kvh, aheads, union_at, ys); });
                    }
                    // per-group launches of the shadow attend (the ablation's 1-group-at-a-time order, batched kernel)
                    differ += variant("pergroup", [&] { for (std::uint64_t g = 0; g < groups; ++g)
                        hipLaunchKernelGGL(HIP_KERNEL_NAME(qsa_fast::attend_shadow_order_kernel<64U, TR, G, 3U>), dim3(16U * ys), block, 0, s, d_views + g, d_kv16, kvh, aheads, union_at, ys); });
                }
                differ += variant("shadD2", [&] { hipLaunchKernelGGL(HIP_KERNEL_NAME(qsa_fast::attend_shadow_multi_kernel<64U, TR, G, 2U>), grid, block, 0, s, d_views, d_kv16, kvh, aheads, union_at); });
            }
            differ += variant("pipeD2", [&] { hipLaunchKernelGGL(HIP_KERNEL_NAME(qsa_fast::attend_pipe_multi_kernel<64U, TR, G, 2U>), grid, block, 0, s, d_views, d_kv, kvh, aheads, union_at); });
            if (differ) { std::printf("FAIL attend variant differs\n"); return 1; }
            // Root-cause ablations of the production attend (per-group attend_kernel, splits 1): debug 0 = as production,
            // 1 = every slot loads token 0 (cache-resident gather), 2 = loaders only (compute skipped), 3 = both.
            for (unsigned debug = 0; debug < 4U; ++debug) {
                auto tv = timed([&] {
                    for (std::uint64_t g = 0; g < groups; ++g)
                        hipLaunchKernelGGL(HIP_KERNEL_NAME(qsa_fast::attend_kernel<64U, TR, G, 3U, false>), dim3(64U / TR, kvh * (aheads / kvh / HG), 1U),
                            block, 0, s, views[g], d_kv, kvh, aheads, union_at, 1U, debug);
                });
                std::printf("attend ablation debug %u  best %8.3f ms  mean %8.3f%s\n", debug, tv.first, tv.second,
                    debug == 0 ? " (per-group launches)" : debug == 1 ? " (token 0 only: no gather misses)" : debug == 2 ? " (loaders only)" : " (loaders only, token 0)");
            }
        }
        // union statistics: padded blocks per tile vs 512+1
        std::vector<std::uint32_t> head(4);
        double ublocks = 0; unsigned tiles = 0;
        for (std::uint64_t row = 0; row < T; row += TR) {
            check(hipMemcpy(head.data(), d_scr + row * aheads * maxsel + union_at, 16, hipMemcpyDeviceToHost), "d2h");
            ublocks += head[0]; ++tiles;
        }
        std::printf("union blocks per 4-row tile mean %.1f (per-row selection %.1f blocks)\n", ublocks / tiles,
            std::min<double>((P + T / 2.0) / ratio, topk + 1.0));
    }
    std::printf("DONE\n");
    return 0;
}
