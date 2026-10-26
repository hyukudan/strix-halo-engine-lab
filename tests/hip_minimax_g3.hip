// minimax-m3 gate G3: layers 0..N-1 of MiniMax-M3.1-Flash (EXL3 pack) on ONE node, teacher-forced prefill of a real
// prompt, compared layer by layer with the float64 oracle of tools/minimax/cpu_layer_reference.py (--gold-dir).
// Not bitwise (MSA selection is discrete, see docs/minimax-msa.md): declared thresholds below.
//
// Device pieces: EXL3 linears through strix-trellis (C ABI 1, strix_gemm), index projections (F16 weights) and the
// router (F32 weights) with simple FP32 kernels, Gemma RMS norms, MSA through the minimax decode kernels in chunks of
// 8 rows, MoE grouped by expert (top-4 routing on the host, the reference's order and tie rule).
//
// usage: strix-hip-minimax-g3 MODEL_DIR GOLD_DIR LAYERS [chain|reset]
#include "minimax_gate_common.hpp"

#include <fcntl.h>
#include <sys/mman.h>
#include <sys/stat.h>
#include <unistd.h>

#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <fstream>
#include <map>
#include <memory>
#include <numeric>
#include <sstream>
#include <stdexcept>
#include <string>
#include <variant>
#include <vector>

namespace {

// Declared G3 thresholds (fixed before the first device run).
// chain: the device must not be worse than twice the CPU production-numerics chain of the same layer
//        (h{L}_cpuprod.f32): rel_l2 <= 2 x cpu + 1e-3 and token p99 <= 2 x cpu + 5e-3;
// reset (each layer fed the oracle's input): rel_l2 <= 3e-3 and token p99 <= 2e-2.
constexpr double kMaxRelL2Reset = 3e-3, kMaxRelTokenP99Reset = 2e-2;
constexpr double kMinSelectionAgree = 0.95;  // non-trivial rows (index head x token past 16 blocks) with the oracle's block set
constexpr double kMinRouterAgree = 0.98;     // tokens with the oracle's exact expert set

}  // namespace

int main(int argc, char** argv) {
    if (argc < 4) {
        std::fprintf(stderr, "usage: %s MODEL_DIR GOLD_DIR LAYERS [chain|reset]\n", argv[0]);
        return 2;
    }
    try {
        const std::string model = argv[1], gold = argv[2];
        const unsigned n_layers = static_cast<unsigned>(std::atoi(argv[3]));
        const bool reset = argc > 4 && std::string(argv[4]) == "reset";
        Pack pack(model);
        Trellis tl;
        std::vector<std::uint32_t> tokens;
        {
            std::ifstream f(gold + "/tokens.u32", std::ios::binary | std::ios::ate);
            tokens.resize(static_cast<std::size_t>(f.tellg()) / 4);
            f.seekg(0);
            f.read(reinterpret_cast<char*>(tokens.data()), static_cast<std::streamsize>(tokens.size() * 4));
        }
        const std::size_t T = tokens.size();
        const std::uint32_t max_blocks = static_cast<std::uint32_t>(msa::block_count(T));
        std::printf("G3 %s: T %zu layers %u blocks %u\n", reset ? "reset" : "chain", T, n_layers, max_blocks);
        // Embedding rows (BF16 -> F32).
        std::vector<float> h_host(T * kHidden);
        {
            const Tensor emb = pack.get(kP + "embed_tokens.weight");
            for (std::size_t t = 0; t < T; ++t)
                for (unsigned i = 0; i < kHidden; ++i) {
                    std::uint16_t b;
                    std::memcpy(&b, emb.data + (static_cast<std::size_t>(tokens[t]) * kHidden + i) * 2, 2);
                    h_host[t * kHidden + i] = msa::bf16_to_float(b);
                }
        }
        float* h = dupload(h_host);
        float* hn = dalloc<float>(T * kHidden);
        float* q = dalloc<float>(T * kQ);
        float* k = dalloc<float>(T * kKV);
        float* v = dalloc<float>(T * kKV);
        float* iq = dalloc<float>(T * kIdxQ);
        float* ik = dalloc<float>(T * 128);
        float* attn = dalloc<float>(T * kQ);
        float* y = dalloc<float>(T * kHidden);
        float* big1 = dalloc<float>(T * kDense);
        float* big2 = dalloc<float>(T * kDense);
        float* xe = dalloc<float>(T * kHidden);
        float* logits = dalloc<float>(T * kExperts);
        auto* kc = dalloc<std::uint16_t>(T * kKV);
        auto* vc = dalloc<std::uint16_t>(T * kKV);
        auto* ikc = dalloc<std::uint16_t>(T * 128);
        float* scores = dalloc<float>(8ULL * 4 * max_blocks);
        auto* idx = dalloc<std::uint32_t>(8ULL * 4 * msa::kTopK);
        auto* cnt = dalloc<std::uint32_t>(8ULL * 4);
        float* partial = dalloc<float>(8ULL * 64 * msa::kMaxChunks * 130);
        std::vector<msa::RopeRow> rope(T);
        std::vector<std::uint64_t> pos(T);
        for (std::size_t t = 0; t < T; ++t) {
            pos[t] = t;
            rope[t] = msa::rope_row(t, 1e7);
        }
        auto* d_rope = dupload(rope);
        auto* d_pos = dupload(pos);
        auto* rows_d = dalloc<std::uint32_t>(T);
        auto* wts_d = dalloc<float>(T);
        bool pass = true;
        for (unsigned L = 0; L < n_layers; ++L) {
            const auto t0 = std::chrono::steady_clock::now();
            if (reset && L > 0) {
                const auto g = read_f32(gold + "/h" + std::to_string(L - 1) + ".f32", T * kHidden);
                hip_ok(hipMemcpy(h, g.data(), g.size() * 4, hipMemcpyHostToDevice), "reset h");
            }
            const std::string a = kP + "layers." + std::to_string(L) + ".self_attn.";
            const std::string lp = kP + "layers." + std::to_string(L) + ".";
            // ---- attention
            float* w_in = dupload(pack.floats(lp + "input_layernorm.weight"));
            hipLaunchKernelGGL(rmsnorm_gemma_kernel, dim3(static_cast<unsigned>(T)), dim3(256), 0, nullptr, h, w_in, hn, kHidden);
            auto lq = tl.prepare(pack, a + "q_proj"), lk = tl.prepare(pack, a + "k_proj"), lv = tl.prepare(pack, a + "v_proj");
            tl.gemm(lq, hn, q, T);
            tl.gemm(lk, hn, k, T);
            tl.gemm(lv, hn, v, T);
            float* wiq = dupload(transpose_f(pack.floats(a + "index_q_proj.weight"), kIdxQ, kHidden));
            float* wik = dupload(transpose_f(pack.floats(a + "index_k_proj.weight"), 128, kHidden));
            hipLaunchKernelGGL(dense_t_kernel, dim3(static_cast<unsigned>(T), kIdxQ / 128), dim3(128), 0, nullptr, hn, wiq, iq, kHidden, kIdxQ);
            hipLaunchKernelGGL(dense_t_kernel, dim3(static_cast<unsigned>(T), 1), dim3(128), 0, nullptr, hn, wik, ik, kHidden, 128U);
            float* norms[4];
            const char* names[4] = {"q_norm", "k_norm", "index_q_norm", "index_k_norm"};
            for (int i = 0; i < 4; ++i) norms[i] = dupload(pack.floats(a + names[i] + ".weight"));
            mh::PrepView pv{static_cast<std::uint32_t>(T), 64, 4, 4, q, k, v, iq, ik, norms[0], norms[1], norms[2], norms[3],
                            d_rope, d_pos, kc, vc, ikc};
            mh::enqueue_prep(pv, nullptr);
            std::vector<std::uint8_t> sel_dev(4 * T * ((max_blocks + 7) / 8), 0);
            std::vector<std::uint32_t> hidx(8 * 4 * msa::kTopK), hcnt(8 * 4);
            for (std::size_t r0 = 0; r0 < T; r0 += 8) {
                const std::uint32_t rows = static_cast<std::uint32_t>(std::min<std::size_t>(8, T - r0));
                mh::ScoreView sv{rows, 4, max_blocks, iq + r0 * kIdxQ, ikc, d_pos + r0, scores};
                mh::SelectView sl{rows, 4, max_blocks, d_pos + r0, scores, idx, cnt};
                mh::AttentionView av{rows, 4, q + r0 * kQ, kc, vc, d_pos + r0, idx, cnt, partial, attn + r0 * kQ};
                mh::enqueue_scores(sv, nullptr);
                mh::enqueue_select(sl, nullptr);
                mh::enqueue_attention(av, nullptr);
                hip_ok(hipMemcpy(hidx.data(), idx, hidx.size() * 4, hipMemcpyDeviceToHost), "idx");
                hip_ok(hipMemcpy(hcnt.data(), cnt, hcnt.size() * 4, hipMemcpyDeviceToHost), "cnt");
                for (std::uint32_t r = 0; r < rows; ++r)
                    for (unsigned hd = 0; hd < 4; ++hd)
                        for (std::uint32_t i = 0; i < hcnt[r * 4 + hd]; ++i) {
                            const std::uint32_t b = hidx[(r * 4 + hd) * msa::kTopK + i];
                            sel_dev[(hd * T + r0 + r) * ((max_blocks + 7) / 8) + b / 8] |= static_cast<std::uint8_t>(0x80U >> (b % 8));
                        }
            }
            auto lo = tl.prepare(pack, a + "o_proj");
            tl.gemm(lo, attn, y, T);
            hipLaunchKernelGGL(add_kernel, dim3(grid(T * kHidden)), dim3(256), 0, nullptr, h, y, T * kHidden);
            for (auto* l : {&lq, &lk, &lv, &lo}) tl.release(*l);
            for (float* p : {w_in, wiq, wik, norms[0], norms[1], norms[2], norms[3]}) hip_ok(hipFree(p), "free");
            // ---- FFN
            float* w_post = dupload(pack.floats(lp + "post_attention_layernorm.weight"));
            hipLaunchKernelGGL(rmsnorm_gemma_kernel, dim3(static_cast<unsigned>(T)), dim3(256), 0, nullptr, h, w_post, hn, kHidden);
            hip_ok(hipMemset(y, 0, T * kHidden * 4), "zero y");
            double router_agree = -1;
            const bool moe = L >= 3;
            auto mlp = [&](const std::string& pre, float* xin, std::size_t n, unsigned inter, float* out) {
                auto g = tl.prepare(pack, pre + "gate_proj"), u = tl.prepare(pack, pre + "up_proj"), dn = tl.prepare(pack, pre + "down_proj");
                tl.gemm(g, xin, big1, n);
                tl.gemm(u, xin, big2, n);
                hipLaunchKernelGGL(swiglu_kernel, dim3(grid(n * inter)), dim3(256), 0, nullptr, big1, big2, big1, n * inter);
                tl.gemm(dn, big1, out, n);
                for (auto* l : {&g, &u, &dn}) tl.release(*l);
            };
            if (!moe) {
                mlp(lp + "mlp.", hn, T, kDense, y);
            } else {
                const std::string b = lp + "block_sparse_moe.";
                float* wr = dupload(transpose_f(pack.floats(b + "gate.weight"), kExperts, kHidden));
                hipLaunchKernelGGL(dense_t_kernel, dim3(static_cast<unsigned>(T), 1), dim3(128), 0, nullptr, hn, wr, logits, kHidden, kExperts);
                std::vector<float> lg(T * kExperts);
                hip_ok(hipMemcpy(lg.data(), logits, lg.size() * 4, hipMemcpyDeviceToHost), "logits");
                const auto bias = pack.floats(b + "e_score_correction_bias");
                std::vector<std::vector<std::uint32_t>> rows(kExperts);
                std::vector<std::vector<float>> wts(kExperts);
                std::vector<std::uint8_t> ids(T * kTopE);
                for (std::size_t t = 0; t < T; ++t) {
                    float s[kExperts], c[kExperts];
                    for (unsigned e = 0; e < kExperts; ++e) {
                        s[e] = 1.0F / (1.0F + std::exp(-lg[t * kExperts + e]));
                        c[e] = s[e] + bias[e];
                    }
                    std::uint32_t order[kExperts];
                    std::iota(order, order + kExperts, 0U);
                    std::partial_sort(order, order + kTopE, order + kExperts, [&](std::uint32_t x, std::uint32_t z) {
                        return c[x] != c[z] ? c[x] > c[z] : x < z;
                    });
                    float sum = 0.0F;
                    for (unsigned i = 0; i < kTopE; ++i) sum += s[order[i]];
                    std::uint32_t sorted[kTopE];
                    for (unsigned i = 0; i < kTopE; ++i) {
                        rows[order[i]].push_back(static_cast<std::uint32_t>(t));
                        wts[order[i]].push_back(s[order[i]] / (sum + 1e-20F) * kRouteScale);
                        sorted[i] = order[i];
                    }
                    std::sort(sorted, sorted + kTopE);
                    for (unsigned i = 0; i < kTopE; ++i) ids[t * kTopE + i] = static_cast<std::uint8_t>(sorted[i]);
                }
                {
                    std::vector<std::uint8_t> gold_ids(T * kTopE);
                    std::ifstream f(gold + "/router" + std::to_string(L) + ".u8", std::ios::binary);
                    f.read(reinterpret_cast<char*>(gold_ids.data()), static_cast<std::streamsize>(gold_ids.size()));
                    std::size_t agree = 0;
                    for (std::size_t t = 0; t < T; ++t)
                        agree += std::memcmp(ids.data() + t * kTopE, gold_ids.data() + t * kTopE, kTopE) == 0 ? 1U : 0U;
                    router_agree = static_cast<double>(agree) / static_cast<double>(T);
                }
                for (unsigned e = 0; e < kExperts; ++e) {
                    if (rows[e].empty()) continue;
                    const std::size_t n = rows[e].size();
                    hip_ok(hipMemcpy(rows_d, rows[e].data(), n * 4, hipMemcpyHostToDevice), "rows");
                    hip_ok(hipMemcpy(wts_d, wts[e].data(), n * 4, hipMemcpyHostToDevice), "wts");
                    hipLaunchKernelGGL(gather_kernel, dim3(static_cast<unsigned>(n)), dim3(256), 0, nullptr, hn, rows_d, xe, kHidden);
                    const std::string pe = b + "experts." + std::to_string(e) + ".";
                    auto w1 = tl.prepare(pack, pe + "w1"), w3 = tl.prepare(pack, pe + "w3"), w2 = tl.prepare(pack, pe + "w2");
                    tl.gemm(w1, xe, big1, n);
                    tl.gemm(w3, xe, big2, n);
                    hipLaunchKernelGGL(swiglu_kernel, dim3(grid(n * kExpertI)), dim3(256), 0, nullptr, big1, big2, big1, n * kExpertI);
                    tl.gemm(w2, big1, xe, n);
                    hipLaunchKernelGGL(scatter_add_kernel, dim3(static_cast<unsigned>(n)), dim3(256), 0, nullptr, xe, rows_d, wts_d, y, kHidden);
                    for (auto* l : {&w1, &w3, &w2}) tl.release(*l);
                }
                float* sh = dalloc<float>(T * kHidden);
                mlp(b + "shared_experts.", hn, T, kExpertI, sh);
                hipLaunchKernelGGL(add_kernel, dim3(grid(T * kHidden)), dim3(256), 0, nullptr, y, sh, T * kHidden);
                hip_ok(hipFree(sh), "free");
                hip_ok(hipFree(wr), "free");
            }
            hipLaunchKernelGGL(add_kernel, dim3(grid(T * kHidden)), dim3(256), 0, nullptr, h, y, T * kHidden);
            hip_ok(hipFree(w_post), "free");
            hip_ok(hipDeviceSynchronize(), "layer sync");
            hip_ok(hipMemcpy(h_host.data(), h, h_host.size() * 4, hipMemcpyDeviceToHost), "h");
            // ---- compare
            const auto ref = read_f32(gold + "/h" + std::to_string(L) + ".f32", T * kHidden);
            const auto cpu = read_f32(gold + "/h" + std::to_string(L) + "_cpuprod.f32", T * kHidden);
            const Metrics m = compare(h_host, ref, T, kHidden), mc = compare(cpu, ref, T, kHidden);
            std::vector<std::uint8_t> sel_gold(sel_dev.size());
            {
                std::ifstream f(gold + "/sel" + std::to_string(L) + ".u8", std::ios::binary);
                f.read(reinterpret_cast<char*>(sel_gold.data()), static_cast<std::streamsize>(sel_gold.size()));
            }
            const std::size_t stride = (max_blocks + 7) / 8;
            std::size_t sel_agree = 0;
            for (std::size_t row = 0; row < 4 * T; ++row)
                sel_agree += std::memcmp(sel_dev.data() + row * stride, sel_gold.data() + row * stride, stride) == 0 ? 1U : 0U;
            std::size_t nt_rows = 0, nt_agree = 0;
            for (std::size_t hd = 0; hd < 4; ++hd)
                for (std::size_t t = msa::kTopK * msa::kBlock; t < T; ++t) {
                    const std::size_t row = hd * T + t;
                    ++nt_rows;
                    nt_agree += std::memcmp(sel_dev.data() + row * stride, sel_gold.data() + row * stride, stride) == 0 ? 1U : 0U;
                }
            const double sel_all = static_cast<double>(sel_agree) / (4.0 * static_cast<double>(T));
            const double sel_frac = nt_rows ? static_cast<double>(nt_agree) / static_cast<double>(nt_rows) : 1.0;
            const bool close = reset ? (m.rel_l2 <= kMaxRelL2Reset && m.tok_p99 <= kMaxRelTokenP99Reset)
                                     : (m.rel_l2 <= 2 * mc.rel_l2 + 1e-3 && m.tok_p99 <= 2 * mc.tok_p99 + 5e-3);
            const bool ok = close && sel_frac >= kMinSelectionAgree &&
                            (router_agree < 0 || router_agree >= kMinRouterAgree);
            pass &= ok;
            const double secs = std::chrono::duration<double>(std::chrono::steady_clock::now() - t0).count();
            std::printf("{\"mode\": \"%s\", \"layer\": %u, \"device_vs_oracle\": {\"rel_l2\": %.3e, \"tok_p50\": %.3e, \"tok_p99\": %.3e, "
                        "\"tok_max\": %.3e}, \"cpuprod_vs_oracle\": {\"rel_l2\": %.3e, \"tok_p99\": %.3e}, \"selection_agree_all\": %.4f, \"selection_agree_nontrivial\": %.4f, "
                        "\"router_agree\": %.4f, \"seconds\": %.1f, \"pass\": %s}\n",
                        reset ? "reset" : "chain", L, m.rel_l2, m.tok_p50, m.tok_p99, m.tok_max, mc.rel_l2, mc.tok_p99, sel_all, sel_frac,
                        router_agree, secs, ok ? "true" : "false");
            std::fflush(stdout);
        }
        std::printf("G3 %s: %s\n", reset ? "reset" : "chain", pass ? "PASS" : "FAIL");
        return pass ? 0 : 1;
    } catch (const std::exception& e) {
        std::fprintf(stderr, "G3 error: %s\n", e.what());
        return 2;
    }
}
