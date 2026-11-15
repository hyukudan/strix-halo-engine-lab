// CudaEngine bitwise tests on a small synthetic qwen35 model (device generated weights):
//  1. serial decode == forward_rows(R) + commit(k) chains (random R and k, also k = 0 and k < R), logits and hidden, bf16 and q8_0
//  2. CUDA graph on == graph off
//  3. MTP: rows in one call == rows one at a time
// Home: specific (qwen35 engine test).
#include <algorithm>
#include <cstdio>
#include <cstring>
#include <random>
#include <vector>

#include "strix/qwen35/cuda_engine.hpp"

namespace q = strix::qwen35;
static int g_fail = 0;
#define CHECK(c, ...)                         \
    do {                                      \
        if (!(c)) {                           \
            std::printf("FAIL: " __VA_ARGS__); \
            std::printf("\n");                \
            ++g_fail;                         \
        }                                     \
    } while (0)

static q::Config small_cfg() {
    q::Config c;
    c.layers = 8;
    c.hidden = 512;
    c.intermediate = 1024;
    c.vocab = 2048;
    c.heads = 6;
    c.kv_heads = 2;
    c.head_dim = 256;
    c.rotary_dim = 64;
    c.gdn_key_heads = 2;
    c.gdn_value_heads = 6;
    c.mtp_layers = 1;
    return c;
}

struct Rec {
    std::vector<float> logits, hidden;  // [pos][V], [pos][H]
};

static void run_serial(q::CudaEngine& E, const std::vector<std::uint32_t>& tok, Rec& r) {
    const auto& c = E.config();
    E.reset();
    r.logits.assign(tok.size() * c.vocab, 0);
    r.hidden.assign(tok.size() * c.hidden, 0);
    for (std::size_t i = 0; i < tok.size(); ++i) {
        const auto o = E.forward_rows(&tok[i], 1, q::LogitRows::all);
        E.read_logits(o, 0, &r.logits[i * c.vocab]);
        E.read_hidden(o.hidden_post, 1, &r.hidden[i * c.hidden]);
        E.commit(1);
    }
}

static void run_chain(q::CudaEngine& E, const std::vector<std::uint32_t>& tok, Rec& r, std::mt19937& rng, bool mark) {
    const auto& c = E.config();
    E.reset();
    r.logits.assign(tok.size() * c.vocab, 0);
    r.hidden.assign(tok.size() * c.hidden, 0);
    std::size_t pos = 0;
    int zero_k = 0, partial = 0, full = 0;
    std::vector<float> lg(c.vocab), hd(32 * c.hidden);
    while (pos < tok.size()) {
        const std::uint32_t R = std::min<std::size_t>(1 + rng() % 32, tok.size() - pos);
        const auto o = E.forward_rows(&tok[pos], R, q::LogitRows::all);
        E.read_hidden(o.hidden_post, R, hd.data());
        for (std::uint32_t i = 0; i < R; ++i) {
            E.read_logits(o, i, lg.data());
            std::copy(lg.begin(), lg.end(), r.logits.begin() + (pos + i) * c.vocab);
            std::copy(hd.begin() + i * c.hidden, hd.begin() + (i + 1) * c.hidden, r.hidden.begin() + (pos + i) * c.hidden);
        }
        std::uint32_t k = static_cast<std::uint32_t>(rng() % (R + 1));
        if (rng() % 4 == 0) k = R;
        if (k == 0 && rng() % 3 != 0) k = 1;
        zero_k += k == 0;
        partial += k > 0 && k < R;
        full += k == R;
        E.commit(k);
        pos += k;
        // the next window re-feeds the rejected rows: record only rows that will not be overwritten later is unnecessary,
        // later windows overwrite the same positions with identical bits when the engine is exact.
    }
    if (mark) std::printf("  chain: %d full, %d partial, %d zero commits\n", full, partial, zero_k);
}

static bool same(const std::vector<float>& a, const std::vector<float>& b) {
    return a.size() == b.size() && std::memcmp(a.data(), b.data(), a.size() * 4) == 0;
}

int main() {
    // variants: 0 bf16, 1 q8_0, 2 nvfp4 onfly (all 2-D incl. lm_head, ssm beta/alpha, multi launches), 3 nvfp4 onfly lm_head bf16
    for (int var = 0; var < 4; ++var) {
        const auto pack = var == 1 ? q::PackFormat::q8_0 : q::PackFormat::bf16;
        std::printf("variant %d: %s%s\n", var, var == 0 ? "bf16" : var == 1 ? "q8_0" : "nvfp4 onfly", var == 3 ? " lm_head bf16" : "");
        q::EngineOptions opt;
        opt.max_positions = 512;
        opt.quiet = true;
        q::Config cfg = small_cfg();
        if (var >= 2) {
            opt.turbo_weights = q::TurboWeights::onfly;
            opt.turbo_lm_head = var == 2;
            cfg.gdn_value_heads = 16;  // ssm beta/alpha out = 16: NVFP4-eligible, exercises the 4-job multi launch
        }
        auto E = q::CudaEngine::synthetic(cfg, pack, opt, 7);
        if (var >= 2) CHECK(E->nvfp4_tensors() > 40, "nvfp4 tensors converted");
        std::mt19937 rng(99);
        std::vector<std::uint32_t> tok(160);
        for (auto& t : tok) t = rng() % 2048;
        Rec ser, ch, chg;
        run_serial(*E, tok, ser);
        bool finite = true;
        for (float f : ser.logits) finite = finite && std::isfinite(f);
        CHECK(finite, "serial logits finite");
        run_chain(*E, tok, ch, rng, true);
        CHECK(same(ser.logits, ch.logits), "chain logits differ from serial");
        CHECK(same(ser.hidden, ch.hidden), "chain hidden differs from serial");
        E->set_graphs(false);
        run_chain(*E, tok, chg, rng, false);
        CHECK(same(ser.logits, chg.logits), "graph-off chain logits differ from serial");
        E->set_graphs(true);
        if (var >= 2) {  // multi-launch lever off must be bitwise equal too (separate GEMV launches)
            opt.multi_gemv = false;
            auto E2 = q::CudaEngine::synthetic(cfg, pack, opt, 7);
            Rec s2;
            run_serial(*E2, tok, s2);
            CHECK(same(ser.logits, s2.logits), "multi-launch off differs");
        }
        if (var >= 2) {  // multi-launch lever off must be bitwise equal too (separate GEMV launches)
            opt.multi_gemv = false;
            auto E2 = q::CudaEngine::synthetic(cfg, pack, opt, 7);
            Rec s2;
            run_serial(*E2, tok, s2);
            CHECK(same(ser.logits, s2.logits), "multi-launch off differs");
        }
        // MTP: one call with rows == the same rows one at a time
        if (E->has_mtp()) {
            const auto& c = E->config();
            E->reset();
            std::vector<std::uint32_t> t8(tok.begin(), tok.begin() + 9);
            const auto o = E->forward_rows(t8.data(), 8, q::LogitRows::none);
            std::vector<float> hid(8 * c.hidden);
            E->read_hidden(o.hidden_post, 8, hid.data());
            float* dh = nullptr;
            cudaMalloc(reinterpret_cast<void**>(&dh), hid.size() * 4);
            cudaMemcpy(dh, hid.data(), hid.size() * 4, cudaMemcpyHostToDevice);
            std::vector<std::uint32_t> nxt(t8.begin() + 1, t8.begin() + 9);
            const auto m8 = E->mtp_forward(nxt.data(), dh, 8, 0);
            std::vector<float> l8(8 * c.vocab), l1(8 * c.vocab);
            for (int r = 0; r < 8; ++r) E->read_logits(q::RowsOut{m8.logits}, r, &l8[r * c.vocab]);
            for (int r = 0; r < 8; ++r) {
                const auto m1 = E->mtp_forward(&nxt[r], dh + r * c.hidden, 1, r);
                E->read_logits(q::RowsOut{m1.logits}, 0, &l1[r * c.vocab]);
            }
            CHECK(same(l8, l1), "mtp 8 rows != 8 single rows");
            cudaFree(dh);
        }
    }
    std::printf(g_fail ? "FAILED (%d)\n" : "PASS\n", g_fail);
    return g_fail ? 1 : 0;
}
