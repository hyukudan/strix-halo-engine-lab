// minimax-m3 gate G4: the whole MiniMax-M3.1-Flash (EXL3 pack, 60 layers + lm_head) on the two-node pair in tensor
// parallel, teacher-forced prefill of a real prompt; rank 0 writes the logits of the scored rows, which
// tools/minimax/g4_score.py compares with the exact-path oracle (KL / top-1, qualification notes).
//
// Per rank r (of 2): query heads 32r..32r+31, KV heads 2r, 2r+1 and index heads 2r, 2r+1 (index key replicated);
// q/k/v output halves, o_proj input half; dense MLP, shared expert and every routed expert split on the intermediate
// dimension (gate/up output halves, down input half); router replicated; lm_head split by vocabulary at H128 blocks
// (782 / 781). Two exchanges per layer (after o_proj, after the FFN) of FP32 partial rows, summed as part0 + part1 on
// both ranks so the residual stays bitwise identical; the head logits are gathered in row chunks.
//
// usage: strix-hip-minimax-g4 MODEL_DIR ORACLE_DIR RANK TRANSPORT OUT_DIR [RUNS]
//   TRANSPORT: usb4-auto | tcp-listen:PORT | tcp-connect:HOST:PORT
//   G4_LAYERS=N (smoke, N < 60) with G4_GOLD=DIR (cpu_layer_reference --gold-dir of the same tokens): stop after N layers
//   and compare the residual with the oracle's h{N-1}.f32 (and the CPU production chain), without the head.
#include "minimax_gate_common.hpp"

#include "strix/dist/communicator.hpp"
#include "strix/hip_minimax_moe.hpp"

#include <chrono>
#include <numeric>

namespace {

constexpr unsigned kWorld = 2;
constexpr unsigned kQLocal = kQ / kWorld, kKVLocal = kKV / kWorld, kIdxLocal = kIdxQ / kWorld;
constexpr unsigned kDenseLocal = kDense / kWorld, kExpertLocal = kExpertI / kWorld;
constexpr std::uint64_t kVocab = 200064, kVocabBlocks0 = 782, kVocabPad = kVocabBlocks0 * 128;  // rank 0's share
constexpr std::size_t kGatherRows = 256;

std::uint64_t fnv(const float* p, std::size_t n) {
    std::uint64_t h = 14695981039346656037ULL;
    const auto* b = reinterpret_cast<const std::uint8_t*>(p);
    for (std::size_t i = 0; i < n * 4; ++i) h = (h ^ b[i]) * 1099511628211ULL;
    return h;
}

std::unique_ptr<strix::dist::Transport> open_transport(const std::string& spec, std::uint32_t rank, std::string& receipt) {
    const std::chrono::seconds limit(120);
    const auto colon = spec.find(':');
    const std::string kind = spec.substr(0, colon), rest = colon == std::string::npos ? "" : spec.substr(colon + 1);
    if (kind == "tcp-listen") return strix::dist::tcp_listen(static_cast<std::uint16_t>(std::stoul(rest)), limit);
    if (kind == "tcp-connect") {
        const auto c2 = rest.rfind(':');
        return strix::dist::tcp_connect(rest.substr(0, c2), static_cast<std::uint16_t>(std::stoul(rest.substr(c2 + 1))), limit);
    }
    if (kind == "usb4-auto") {
        strix::dist::Usb4AutoOptions o;
        o.minimax_compat = true;
        o.rank = rank;
        o.timeout = limit;
        strix::dist::Usb4Topology topology;
        auto t = strix::dist::usb4_host_auto(o, &topology);
        receipt = topology.receipt();
        return t;
    }
    throw std::invalid_argument("unknown transport " + spec);
}

struct Ctx {
    std::uint32_t rank;
    std::size_t T;
    Pack& pack;
    Trellis& tl;
    strix::dist::Communicator& comm;
    float* remote;  // receive buffer, T x kHidden
    float* part;    // local partial, T x kHidden
};

// Exchange of the ranks' partial rows, then the fused post-exchange kernel (the decode design, article item 3):
// h += part0 + part1 (fixed order: the same bits on both ranks), y = Gemma norm of the next sublayer, and the router
// top-4 when router_w is set.
void exchange_post(Ctx& c, float* h, std::uint32_t layer, std::uint32_t which, const float* norm_w, float* y,
                   const float* router_w, const float* bias, std::uint32_t* ids, float* weights) {
    strix::dist::Tag tag;
    tag.kind = static_cast<std::uint32_t>(strix::dist::TagKind::half_sum);
    tag.layer = layer;
    tag.request = which;
    c.comm.exchange(nullptr, c.part, c.remote, c.T * kHidden * sizeof(float), tag);
    const float* p0 = c.rank == 0 ? c.part : c.remote;
    const float* p1 = c.rank == 0 ? c.remote : c.part;
    strix::minimax::moe::hip::enqueue_post_exchange(
        {static_cast<std::uint32_t>(c.T), h, p0, p1, norm_w, y, router_w, bias, ids, weights}, nullptr);
}

}  // namespace

int main(int argc, char** argv) {
    if (argc < 6) {
        std::fprintf(stderr, "usage: %s MODEL_DIR ORACLE_DIR RANK TRANSPORT OUT_DIR [RUNS]\n", argv[0]);
        return 2;
    }
    try {
        const std::string model = argv[1], oracle = argv[2], out_dir = argv[5];
        const auto rank = static_cast<std::uint32_t>(std::stoul(argv[3]));
        const int runs = argc > 6 ? std::atoi(argv[6]) : 1;
        if (rank >= kWorld) throw std::invalid_argument("rank must be 0 or 1");
        std::string receipt;
        auto transport = open_transport(argv[4], rank, receipt);
        if (!receipt.empty()) std::printf("transport: %s\n", receipt.c_str());
        auto read_u32 = [](const std::string& path) {
            std::ifstream f(path, std::ios::binary | std::ios::ate);
            if (!f) throw std::runtime_error("read " + path);
            std::vector<std::uint32_t> v(static_cast<std::size_t>(f.tellg()) / 4);
            f.seekg(0);
            f.read(reinterpret_cast<char*>(v.data()), static_cast<std::streamsize>(v.size() * 4));
            return v;
        };
        const auto tokens = read_u32(oracle + "/tokens.u32");
        const auto rows = read_u32(oracle + "/rows.u32");
        const std::size_t T = tokens.size(), S = rows.size();
        const std::uint32_t max_blocks = static_cast<std::uint32_t>(msa::block_count(T));
        strix::dist::Group group;
        group.minimax_compat = true;
        group.rank = rank;
        group.world = kWorld;
        group.plan = 0x4d4d5833ULL;  // "MMX3"
        strix::dist::Communicator comm(group, std::move(transport),
                                       std::max<std::size_t>(T * kHidden, kGatherRows * kVocabPad) * sizeof(float));
        Pack pack(model);
        Trellis tl;
        std::printf("G4 rank %u: T %zu scored rows %zu, transport %s\n", rank, T, S, comm.transport_name().c_str());
        float* h = dalloc<float>(T * kHidden);
        float* hn = dalloc<float>(T * kHidden);
        float* q = dalloc<float>(T * kQLocal);
        float* k = dalloc<float>(T * kKVLocal);
        float* v = dalloc<float>(T * kKVLocal);
        float* iq = dalloc<float>(T * kIdxLocal);
        float* ik = dalloc<float>(T * 128);
        float* attn = dalloc<float>(T * kQLocal);
        float* big1 = dalloc<float>(T * kDenseLocal);
        float* big2 = dalloc<float>(T * kDenseLocal);
        float* xe = dalloc<float>(T * kHidden);
        float* ye = dalloc<float>(T * kHidden);
        auto* ids_d = dalloc<std::uint32_t>(T * kTopE);
        float* wts4_d = dalloc<float>(T * kTopE);
        float* part = dalloc<float>(T * kHidden);
        float* remote = dalloc<float>(T * kHidden);
        auto* kc = dalloc<std::uint16_t>(T * kKVLocal);
        auto* vc = dalloc<std::uint16_t>(T * kKVLocal);
        auto* ikc = dalloc<std::uint16_t>(T * 128);
        float* scores = dalloc<float>(8ULL * 2 * max_blocks);
        auto* idx = dalloc<std::uint32_t>(8ULL * 2 * msa::kTopK);
        auto* cnt = dalloc<std::uint32_t>(8ULL * 2);
        float* partial = dalloc<float>(8ULL * 32 * msa::kMaxChunks * 130);
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
        Ctx ctx{rank, T, pack, tl, comm, remote, part};
        std::vector<float> emb_host(T * kHidden), h_host(T * kHidden);  // embedding rows stay intact across runs
        {
            const Tensor emb = pack.get(kP + "embed_tokens.weight");
            for (std::size_t t = 0; t < T; ++t)
                for (unsigned i = 0; i < kHidden; ++i) {
                    std::uint16_t b;
                    std::memcpy(&b, emb.data + (static_cast<std::size_t>(tokens[t]) * kHidden + i) * 2, 2);
                    emb_host[t * kHidden + i] = msa::bf16_to_float(b);
                }
        }
        const unsigned n_layers = std::getenv("G4_LAYERS") != nullptr ? static_cast<unsigned>(std::atoi(std::getenv("G4_LAYERS"))) : 60U;
        const std::string gold = std::getenv("G4_GOLD") != nullptr ? std::getenv("G4_GOLD") : "";
        if (n_layers == 0 || n_layers > 60 || (n_layers < 60 && gold.empty())) throw std::invalid_argument("G4_LAYERS needs 1..60 and G4_GOLD");
        std::uint64_t first_logits_fnv = 0, first_residual_fnv = 0;
        bool deterministic = true;
        for (int run = 0; run < runs; ++run) {
            const auto t_run = std::chrono::steady_clock::now();
            hip_ok(hipMemcpy(h, emb_host.data(), emb_host.size() * 4, hipMemcpyHostToDevice), "h");
            for (unsigned L = 0; L < n_layers; ++L) {
                const std::string lp = kP + "layers." + std::to_string(L) + ".", a = lp + "self_attn.";
                // ---- attention (rank's heads)
                if (L == 0) {
                    float* w_in = dupload(pack.floats(lp + "input_layernorm.weight"));
                    hipLaunchKernelGGL(rmsnorm_gemma_kernel, dim3(static_cast<unsigned>(T)), dim3(256), 0, nullptr, h, w_in, hn, kHidden);
                    hip_ok(hipDeviceSynchronize(), "norm0");
                    hip_ok(hipFree(w_in), "free");
                }
                auto lq = tl.prepare_out_slice(pack, a + "q_proj", rank * kQLocal, kQLocal);
                auto lk = tl.prepare_out_slice(pack, a + "k_proj", rank * kKVLocal, kKVLocal);
                auto lv = tl.prepare_out_slice(pack, a + "v_proj", rank * kKVLocal, kKVLocal);
                tl.gemm(lq, hn, q, T);
                tl.gemm(lk, hn, k, T);
                tl.gemm(lv, hn, v, T);
                const auto wiq_all = pack.floats(a + "index_q_proj.weight");  // [512][6144]
                const std::vector<float> wiq_mine(wiq_all.begin() + rank * kIdxLocal * kHidden,
                                                  wiq_all.begin() + (rank + 1) * kIdxLocal * kHidden);
                float* wiq = dupload(transpose_f(wiq_mine, kIdxLocal, kHidden));
                float* wik = dupload(transpose_f(pack.floats(a + "index_k_proj.weight"), 128, kHidden));
                hipLaunchKernelGGL(dense_t_kernel, dim3(static_cast<unsigned>(T), kIdxLocal / 128), dim3(128), 0, nullptr, hn, wiq, iq, kHidden, kIdxLocal);
                hipLaunchKernelGGL(dense_t_kernel, dim3(static_cast<unsigned>(T), 1), dim3(128), 0, nullptr, hn, wik, ik, kHidden, 128U);
                float* norms[4];
                const char* names[4] = {"q_norm", "k_norm", "index_q_norm", "index_k_norm"};
                for (int i = 0; i < 4; ++i) norms[i] = dupload(pack.floats(a + names[i] + ".weight"));
                mh::PrepView pv{static_cast<std::uint32_t>(T), 32, 2, 2, q, k, v, iq, ik, norms[0], norms[1], norms[2], norms[3],
                                d_rope, d_pos, kc, vc, ikc};
                mh::enqueue_prep(pv, nullptr);
                for (std::size_t r0 = 0; r0 < T; r0 += 8) {
                    const auto n = static_cast<std::uint32_t>(std::min<std::size_t>(8, T - r0));
                    mh::enqueue_scores({n, 2, max_blocks, iq + r0 * kIdxLocal, ikc, d_pos + r0, scores}, nullptr);
                    mh::enqueue_select({n, 2, max_blocks, d_pos + r0, scores, idx, cnt}, nullptr);
                    mh::enqueue_attention({n, 2, q + r0 * kQLocal, kc, vc, d_pos + r0, idx, cnt, partial, attn + r0 * kQLocal}, nullptr);
                }
                auto lo = tl.prepare_in_slice(pack, a + "o_proj", rank * kQLocal, kQLocal);
                tl.gemm(lo, attn, part, T);
                const bool moe_layer = L >= 3;
                const std::string b = lp + "block_sparse_moe.";
                float* w_post = dupload(pack.floats(lp + "post_attention_layernorm.weight"));
                float* wr = moe_layer ? dupload(pack.floats(b + "gate.weight")) : nullptr;  // [128][6144] FP32
                float* wb = moe_layer ? dupload(pack.floats(b + "e_score_correction_bias")) : nullptr;
                exchange_post(ctx, h, L, 0, w_post, hn, wr, wb, ids_d, wts4_d);
                for (auto* l : {&lq, &lk, &lv, &lo}) tl.release(*l);
                for (float* p : {wiq, wik, norms[0], norms[1], norms[2], norms[3]}) hip_ok(hipFree(p), "free");
                // ---- FFN (intermediate halves)
                // gate/up output halves and the down input half of one MLP (names: gate, up, down).
                auto mlp = [&](const std::string& pre, const char* const* nm, float* xin, std::size_t n, unsigned inter_local, float* out) {
                    auto g = tl.prepare_out_slice(pack, pre + nm[0], rank * inter_local, inter_local);
                    auto u = tl.prepare_out_slice(pack, pre + nm[1], rank * inter_local, inter_local);
                    auto dn = tl.prepare_in_slice(pack, pre + nm[2], rank * inter_local, inter_local);
                    tl.gemm(g, xin, big1, n);
                    tl.gemm(u, xin, big2, n);
                    hipLaunchKernelGGL(swiglu_kernel, dim3(grid(n * inter_local)), dim3(256), 0, nullptr, big1, big2, big1, n * inter_local);
                    tl.gemm(dn, big1, out, n);
                    for (auto* l : {&g, &u, &dn}) tl.release(*l);
                };
                static const char* const kMlpNames[3] = {"gate_proj", "up_proj", "down_proj"};
                static const char* const kExpertNames[3] = {"w1", "w3", "w2"};
                if (!moe_layer) {
                    mlp(lp + "mlp.", kMlpNames, hn, T, kDenseLocal, part);
                } else {
                    // Routes from the fused post-exchange kernel (ids/weights per row), grouped by expert on the host.
                    std::vector<std::uint32_t> ids(T * kTopE);
                    std::vector<float> w4(T * kTopE);
                    hip_ok(hipMemcpy(ids.data(), ids_d, ids.size() * 4, hipMemcpyDeviceToHost), "route ids");
                    hip_ok(hipMemcpy(w4.data(), wts4_d, w4.size() * 4, hipMemcpyDeviceToHost), "route weights");
                    std::vector<std::vector<std::uint32_t>> erows(kExperts);
                    std::vector<std::vector<float>> ewts(kExperts);
                    for (std::size_t t = 0; t < T; ++t)
                        for (unsigned i = 0; i < kTopE; ++i) {
                            erows[ids[t * kTopE + i]].push_back(static_cast<std::uint32_t>(t));
                            ewts[ids[t * kTopE + i]].push_back(w4[t * kTopE + i]);
                        }
                    hip_ok(hipMemset(part, 0, T * kHidden * 4), "zero part");
                    // Batch invariance (article rule 11): each token's sum runs over ITS experts in ascending expert
                    // id, independent of which other tokens share the call.
                    for (unsigned e = 0; e < kExperts; ++e) {
                        if (erows[e].empty()) continue;
                        const std::size_t n = erows[e].size();
                        hip_ok(hipMemcpy(rows_d, erows[e].data(), n * 4, hipMemcpyHostToDevice), "rows");
                        hip_ok(hipMemcpy(wts_d, ewts[e].data(), n * 4, hipMemcpyHostToDevice), "wts");
                        hipLaunchKernelGGL(gather_kernel, dim3(static_cast<unsigned>(n)), dim3(256), 0, nullptr, hn, rows_d, xe, kHidden);
                        mlp(b + "experts." + std::to_string(e) + ".", kExpertNames, xe, n, kExpertLocal, ye);
                        hipLaunchKernelGGL(scatter_add_kernel, dim3(static_cast<unsigned>(n)), dim3(256), 0, nullptr, ye, rows_d, wts_d, part, kHidden);
                    }
                    mlp(b + "shared_experts.", kMlpNames, hn, T, kExpertLocal, xe);
                    hipLaunchKernelGGL(add_kernel, dim3(grid(T * kHidden)), dim3(256), 0, nullptr, part, xe, T * kHidden);
                }
                // Next sublayer's norm: the next layer's input norm, or the final norm after layer 59.
                float* w_next = dupload(pack.floats(L + 1 < 60 ? kP + "layers." + std::to_string(L + 1) + ".input_layernorm.weight"
                                                                 : kP + "norm.weight"));
                exchange_post(ctx, h, L, 1, w_next, hn, nullptr, nullptr, ids_d, wts4_d);
                hip_ok(hipDeviceSynchronize(), "post");
                for (float* p : {w_post, wr, wb, w_next})
                    if (p != nullptr) hip_ok(hipFree(p), "free");
                hip_ok(hipDeviceSynchronize(), "layer");
                if (L % 10 == 9) std::printf("rank %u run %d: layer %u done, %.1f s\n", rank, run, L,
                                             std::chrono::duration<double>(std::chrono::steady_clock::now() - t_run).count());
                std::fflush(stdout);
            }
            hip_ok(hipMemcpy(h_host.data(), h, h_host.size() * 4, hipMemcpyDeviceToHost), "h last");
            const std::uint64_t rf = fnv(h_host.data(), h_host.size());
            if (run == 0) first_residual_fnv = rf;
            deterministic &= rf == first_residual_fnv;
            std::printf("rank %u run %d: residual after layer %u fnv %016llx\n", rank, run, n_layers - 1,
                        static_cast<unsigned long long>(rf));
            if (n_layers < 60) {  // smoke: compare with the oracle's residual, no head
                const auto ref = read_f32(gold + "/h" + std::to_string(n_layers - 1) + ".f32", T * kHidden);
                const auto cpu = read_f32(gold + "/h" + std::to_string(n_layers - 1) + "_cpuprod.f32", T * kHidden);
                const Metrics m = compare(h_host, ref, T, kHidden), mc = compare(cpu, ref, T, kHidden);
                std::printf("rank %u run %d: layer %u pair vs oracle rel_l2 %.3e tok_p50 %.3e tok_p99 %.3e tok_max %.3e "
                            "(cpu prod rel_l2 %.3e tok_p99 %.3e) %.1f s\n", rank, run, n_layers - 1, m.rel_l2, m.tok_p50,
                            m.tok_p99, m.tok_max, mc.rel_l2, mc.tok_p99,
                            std::chrono::duration<double>(std::chrono::steady_clock::now() - t_run).count());
                std::fflush(stdout);
                continue;
            }
            // ---- final norm + lm_head (vocabulary half), gathered on rank 0 in row chunks
            std::vector<std::uint32_t> rows_v(rows.begin(), rows.end());
            hip_ok(hipMemcpy(rows_d, rows_v.data(), S * 4, hipMemcpyHostToDevice), "scored rows");
            hipLaunchKernelGGL(gather_kernel, dim3(static_cast<unsigned>(S)), dim3(256), 0, nullptr, hn, rows_d, ye, kHidden);
            const std::uint64_t v0 = rank == 0 ? 0 : kVocabPad, vn = rank == 0 ? kVocabPad : kVocab - kVocabPad;
            auto head = tl.prepare_out_slice(pack, "language_model.lm_head", v0, vn);
            std::vector<float> full(rank == 0 ? S * kVocab : 0);
            float* lg_local = dalloc<float>(kGatherRows * kVocabPad);
            float* lg_packed = dalloc<float>(kGatherRows * kVocabPad);
            float* lg_remote = dalloc<float>(kGatherRows * kVocabPad);
            for (std::size_t r0 = 0; r0 < S; r0 += kGatherRows) {
                const std::size_t n = std::min(kGatherRows, S - r0);
                hip_ok(hipMemset(lg_local, 0, kGatherRows * kVocabPad * 4), "zero");
                // Rows of width vn written with stride kVocabPad: gemm into a packed temp, then a strided copy.
                tl.gemm(head, ye + r0 * kHidden, lg_packed, n);
                hip_ok(hipMemcpy2D(lg_local, kVocabPad * 4, lg_packed, vn * 4, vn * 4, n, hipMemcpyDeviceToDevice), "pad");
                strix::dist::Tag tag;
                tag.kind = static_cast<std::uint32_t>(strix::dist::TagKind::raw_slots);
                tag.layer = 60;
                tag.request = r0;
                comm.exchange(nullptr, lg_local, lg_remote, kGatherRows * kVocabPad * 4, tag);
                if (rank == 0) {
                    std::vector<float> a(n * kVocabPad), b(n * kVocabPad);
                    hip_ok(hipMemcpy(a.data(), lg_local, a.size() * 4, hipMemcpyDeviceToHost), "a");
                    hip_ok(hipMemcpy(b.data(), lg_remote, b.size() * 4, hipMemcpyDeviceToHost), "b");
                    for (std::size_t i = 0; i < n; ++i) {
                        std::memcpy(&full[(r0 + i) * kVocab], &a[i * kVocabPad], kVocabPad * 4);
                        std::memcpy(&full[(r0 + i) * kVocab + kVocabPad], &b[i * kVocabPad], (kVocab - kVocabPad) * 4);
                    }
                }
            }
            tl.release(head);
            for (float* p : {lg_local, lg_remote, lg_packed}) hip_ok(hipFree(p), "free");
            const double secs = std::chrono::duration<double>(std::chrono::steady_clock::now() - t_run).count();
            if (rank == 0) {
                bool finite = true;
                for (float f : full) finite &= std::isfinite(f);
                const std::uint64_t lf = fnv(full.data(), full.size());
                if (run == 0) {
                    first_logits_fnv = lf;
                    std::ofstream o(out_dir + "/logits_pair.f32", std::ios::binary);
                    o.write(reinterpret_cast<const char*>(full.data()), static_cast<std::streamsize>(full.size() * 4));
                } else {
                    deterministic &= lf == first_logits_fnv;
                }
                std::printf("rank 0 run %d: logits fnv %016llx finite %s, %.1f s\n", run, static_cast<unsigned long long>(lf),
                            finite ? "yes" : "NO", secs);
            } else {
                std::printf("rank 1 run %d: %.1f s\n", run, secs);
            }
            std::fflush(stdout);
        }
        if (runs > 1) std::printf("rank %u G4 determinism (%d runs, residual%s): %s\n", rank, runs, n_layers == 60 ? " + logits" : "",
                                  deterministic ? "IDENTICAL" : "DIFFER");
        const auto& st = comm.stats();
        std::printf("rank %u: exchanges %llu, %.1f MB sent, %.2f s in exchange\n", rank,
                    static_cast<unsigned long long>(st.exchanges), static_cast<double>(st.bytes_sent) / 1e6, st.seconds);
        return 0;
    } catch (const std::exception& e) {
        std::fprintf(stderr, "G4 error: %s\n", e.what());
        return 2;
    }
}
