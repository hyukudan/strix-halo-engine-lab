// G3 (a') independent replay (design notes section 4): a separately scheduled reference pipeline for one
// node (world 1) over layers [0, N): every Tier A op on the host emulation (src/glm/host), every EXL3 linear through a
// plain strix_gemv call on its own prepared weights (Tier B reference), its own buffers and caches. Compares, bit for
// bit, the engine's G3 dump (strix-glm5-rank --g3 TOK --layers N --out DUMP): the 4-stream residual after layer N-1
// for every token, then every persistent state buffer of slot 0 (same layout as Engine::state_bytes).
// usage: strix-glm5-g3-replay MODEL_DIR TOKENS.u32 N DUMP CAPACITY
#include "glm/engine/glm_lib.hpp"
#include "glm/host/glm_host.hpp"

#include <cstdio>
#include <cstring>
#include <fstream>
#include <string>
#include <vector>

using namespace strix::glm;
namespace H = strix::glm::host;
namespace E = strix::glm::engine;
using namespace strix::glm::engine::detail;

namespace {

const std::string kP = "model.language_model.";

struct Raw {  // a plain tensor's bytes on the host
    std::vector<std::uint8_t> b;
    H::Mat mat(H::WType t, unsigned out, unsigned in) const { return H::Mat{b.data(), t, out, in}; }
    H::Vec vec(H::WType t, unsigned n) const { return H::Vec{b.data(), t, n}; }
    const float* f() const { return reinterpret_cast<const float*>(b.data()); }
    const std::uint16_t* u16() const { return reinterpret_cast<const std::uint16_t*>(b.data()); }
};

class Replay {
public:
    Replay(const std::string& dir, unsigned layers, unsigned cap) : pack_(dir), lib_(pack_, false), n_(layers), cap_(cap) {
        hip_ok(hipMalloc(&din_, 16384 * 4), "din");
        hip_ok(hipMalloc(&dout_, 32768 * 4), "dout");
        for (unsigned l = 0; l < n_; ++l) load(l);
        lib_.release_staging();
    }
    // One token through layers [0, N); returns the 4-stream residual.
    std::vector<float> token(std::uint32_t tok, unsigned t) {
        std::vector<float> X(kHc * kHidden);
        const auto er = raw_rows(kP + "embed_tokens.weight", tok, 1);
        for (unsigned c = 0; c < kHc; ++c)
            for (unsigned d = 0; d < kHidden; ++d) X[c * kHidden + d] = bf16_to_float(er.u16()[d]);
        for (auto& L : layers_) {
            X = site(L, X, t, true);
            X = site(L, X, t, false);
        }
        return X;
    }
    // Slot-0 state in Engine::state_bytes layout.
    std::vector<std::uint8_t> state() const {
        std::vector<std::uint8_t> out;
        auto put = [&](const void* p, std::size_t n) {
            const auto* b = static_cast<const std::uint8_t*>(p);
            out.insert(out.end(), b, b + n);
        };
        for (const auto& L : layers_) {
            if (L.kda) {
                put(L.kst.S.data(), L.kst.S.size() * 4);
                put(L.kst.conv.data(), L.kst.conv.size() * 4);
            } else {
                auto pad = [&](const std::vector<std::uint16_t>& v, std::size_t n) {
                    std::vector<std::uint16_t> w(v);
                    w.resize(n, 0);
                    put(w.data(), n * 2);
                };
                pad(L.dc.latent, static_cast<std::size_t>(cap_) * kKvLora);
                pad(L.dc.ikey, static_cast<std::size_t>(cap_) * kIdxDim);
                pad(L.dc.igate, static_cast<std::size_t>(cap_) * kIdxDim);
                std::vector<float> p(L.dc.pooled);
                p.resize(static_cast<std::size_t>(cap_ / kKpool) * kIdxDim, 0.0F);
                put(p.data(), p.size() * 4);
            }
        }
        return out;
    }

private:
    struct Lay {
        unsigned index = 0;
        bool kda = false, moe = false;
        Raw fn[2], base[2], scale[2], in_norm, post_norm;
        // KDA
        E::Linear qkv[3], o;
        Raw f_a, f_b, g_a, g_b, b, conv, A_log, dt, o_norm;
        H::KdaState kst;
        // DSA
        E::Linear q_a, kv_a, q_b, wq_b;
        Raw q_a_norm, kv_a_norm, kv_b, wk, gate, wp, knw, knb, ape;
        H::DsaCache dc;
        // FFN
        Raw router, rbias;
        std::vector<E::Mlp> experts;
        E::Mlp shared, dense;
    };
    Raw raw(const std::string& name) {
        const TensorInfo& t = pack_.info(name);
        Raw r;
        r.b.resize(t.bytes);
        pack_.read(t, 0, t.bytes, r.b.data());
        return r;
    }
    Raw raw_rows(const std::string& name, std::uint64_t r0, std::uint64_t n) {
        const TensorInfo& t = pack_.info(name);
        const std::uint64_t rb = t.bytes / t.shape[0];
        Raw r;
        r.b.resize(n * rb);
        pack_.read(t, r0 * rb, n * rb, r.b.data());
        return r;
    }
    void load(unsigned l) {
        Lay L;
        L.index = l;
        L.kda = is_kda_layer(l);
        L.moe = is_moe_layer(l);
        const std::string p = kP + "layers." + std::to_string(l) + ".", a = p + "self_attn.";
        for (int s = 0; s < 2; ++s) {
            const std::string n = p + (s == 0 ? "hc_attn_" : "hc_ffn_");
            L.fn[s] = raw(n + "fn");
            L.base[s] = raw(n + "base");
            L.scale[s] = raw(n + "scale");
        }
        L.in_norm = raw(p + "input_layernorm.weight");
        L.post_norm = raw(p + "post_attention_layernorm.weight");
        if (L.kda) {
            for (unsigned c = 0; c < 3; ++c) L.qkv[c] = lib_.out_slice(a + "qkv_proj", c * kKdaQkv, kKdaQkv);
            L.o = lib_.whole(a + "o_proj");
            L.f_a = raw(a + "f_a_proj.weight");
            L.f_b = raw(a + "f_b_proj.weight");
            L.g_a = raw(a + "g_a_proj.weight");
            L.g_b = raw(a + "g_b_proj.weight");
            L.b = raw(a + "b_proj.weight");
            L.conv = raw(a + "conv1d.weight");
            L.A_log = raw(a + "A_log");
            L.dt = raw(a + "dt_bias");
            L.o_norm = raw(a + "o_norm.weight");
            L.kst = H::KdaState{kKdaHeads, std::vector<float>(static_cast<std::size_t>(kKdaHeads) * kKdaDim * kKdaDim, 0.0F),
                                std::vector<float>(static_cast<std::size_t>(3) * kKdaQkv * 3, 0.0F)};
        } else {
            L.q_a = lib_.whole(a + "q_a_proj");
            L.kv_a = lib_.whole(a + "kv_a_proj_with_mqa");
            L.q_b = lib_.whole(a + "q_b_proj");
            L.wq_b = lib_.whole(a + "indexer.wq_b");
            L.o = lib_.whole(a + "o_proj");
            L.q_a_norm = raw(a + "q_a_layernorm.weight");
            L.kv_a_norm = raw(a + "kv_a_layernorm.weight");
            L.kv_b = raw(a + "kv_b_proj.weight");
            L.wk = raw(a + "indexer.wk.weight");
            L.gate = raw(a + "indexer.index_kpool_compress_gate");
            L.wp = raw(a + "indexer.weights_proj.weight");
            L.knw = raw(a + "indexer.k_norm.weight");
            L.knb = raw(a + "indexer.k_norm.bias");
            L.ape = raw(a + "indexer.index_kpool_compress_ape");
        }
        const std::string m = p + "mlp.";
        auto mlp = [&](const std::string& q) { return E::Mlp{lib_.whole(q + "gate_proj"), lib_.whole(q + "up_proj"), lib_.whole(q + "down_proj")}; };
        if (L.moe) {
            L.router = raw(m + "gate.weight");
            L.rbias = raw(m + "gate.e_score_correction_bias");
            for (unsigned e = 0; e < kExperts; ++e) L.experts.push_back(mlp(m + "experts." + std::to_string(e) + "."));
            L.shared = mlp(m + "shared_experts.");
        } else {
            L.dense = mlp(m);
        }
        layers_.push_back(std::move(L));
    }
    std::vector<float> gemv(const E::Linear& l, const std::vector<float>& x) {
        hip_ok(hipMemcpy(din_, x.data(), x.size() * 4, hipMemcpyHostToDevice), "in");
        lib_.run(l, din_, dout_, 1, nullptr);
        std::vector<float> y(l.raw.output);
        hip_ok(hipMemcpy(y.data(), dout_, y.size() * 4, hipMemcpyDeviceToHost), "out");
        return y;
    }
    std::vector<float> mlp(const E::Mlp& m, const std::vector<float>& x) {
        const auto g = gemv(m.gate, x), u = gemv(m.up, x);
        std::vector<float> act(g.size());
        for (std::size_t i = 0; i < g.size(); ++i) act[i] = H::swiglu(g[i], u[i]);
        return gemv(m.down, act);
    }
    std::vector<float> site(Lay& L, const std::vector<float>& X, unsigned t, bool attn) {
        const int s = attn ? 0 : 1;
        const H::HcWeights hw{H::Mat{L.fn[s].b.data(), H::WType::F32, kHcMix, kHc * kHidden}, L.base[s].f(), L.scale[s].f()};
        const H::HcMix mix = H::hc_mix(X, hw);
        std::vector<float> x(kHidden), xn(kHidden), y(kHidden);
        H::hc_collapse(X, mix, x);
        const H::Vec nw = (attn ? L.in_norm : L.post_norm).vec(H::WType::BF16, kHidden);
        H::rmsnorm(x, &nw, kEps, xn);
        if (attn && L.kda) {
            std::vector<float> qkv;
            for (unsigned c = 0; c < 3; ++c) {
                const auto v = gemv(L.qkv[c], xn);
                qkv.insert(qkv.end(), v.begin(), v.end());
            }
            std::vector<float> fa(kKdaDim), ga(kKdaDim), f(kKdaQkv), g(kKdaQkv), b(kKdaHeads), o(kKdaQkv);
            H::dense(xn, L.f_a.mat(H::WType::F16, kKdaDim, kHidden), fa);
            H::dense(fa, L.f_b.mat(H::WType::F16, kKdaQkv, kKdaDim), f);
            H::dense(xn, L.g_a.mat(H::WType::F16, kKdaDim, kHidden), ga);
            H::dense(ga, L.g_b.mat(H::WType::F16, kKdaQkv, kKdaDim), g);
            H::dense(xn, L.b.mat(H::WType::F16, kKdaHeads, kHidden), b);
            const H::KdaWeights kw{L.conv.u16(), L.A_log.f(), L.dt.f(), L.o_norm.vec(H::WType::BF16, kKdaDim)};
            H::kda_step(L.kst, H::KdaInputs{qkv, f, g, b}, kw, o);
            y = gemv(L.o, o);
        } else if (attn) {
            const auto qa = gemv(L.q_a, xn);
            std::vector<float> qr(kQLora), lat(kKvLora);
            const H::Vec qn = L.q_a_norm.vec(H::WType::BF16, kQLora), kn = L.kv_a_norm.vec(H::WType::BF16, kKvLora);
            H::rmsnorm(qa, &qn, kEps, qr);
            const auto q = gemv(L.q_b, qr);
            H::rmsnorm(gemv(L.kv_a, xn), &kn, kEps, lat);
            const auto qI = gemv(L.wq_b, qr);
            const H::IndexerWeights iw{L.wk.mat(H::WType::F16, kIdxDim, kHidden), L.gate.mat(H::WType::F16, kIdxDim, kHidden),
                                       L.wp.mat(H::WType::F16, kIdxHeads, kHidden), L.knw.vec(H::WType::F16, kIdxDim),
                                       L.knb.vec(H::WType::F16, kIdxDim), L.ape.f()};
            H::dsa_append(L.dc, lat, xn, iw);
            const auto tok = H::indexer_select(L.dc, t, qI, xn, iw);
            std::vector<float> o(static_cast<std::size_t>(kMlaHeads) * kMlaDim);
            H::mla_attention(L.dc, tok, q, L.kv_b.mat(H::WType::F16, kMlaHeads * 2 * kMlaDim, kKvLora), kMlaHeads, o);
            y = gemv(L.o, o);
        } else if (L.moe) {
            const H::Route r = H::route(xn, L.router.mat(H::WType::F16, kExperts, kHidden), L.rbias.f());
            std::vector<float> eo(static_cast<std::size_t>(kTopE) * kHidden);
            for (unsigned k = 0; k < kTopE; ++k) {
                const auto v = mlp(L.experts[r.id[k]], xn);
                std::copy(v.begin(), v.end(), eo.begin() + static_cast<std::ptrdiff_t>(k * kHidden));
            }
            H::moe_combine(r, eo, mlp(L.shared, xn), y);
        } else {
            y = mlp(L.dense, xn);
        }
        std::vector<float> Xo(kHc * kHidden);
        H::hc_expand(X, mix, y, Xo);
        H::check_finite(Xo, "C1 residual");
        return Xo;
    }
    PackReader pack_;
    Lib lib_;
    unsigned n_, cap_;
    std::vector<Lay> layers_;
    float *din_ = nullptr, *dout_ = nullptr;
};

}  // namespace

int main(int argc, char** argv) {
    try {
        if (argc < 6) {
            std::fprintf(stderr, "usage: %s MODEL_DIR TOKENS.u32 N DUMP CAPACITY\n", argv[0]);
            return 2;
        }
        std::ifstream tf(argv[2], std::ios::binary | std::ios::ate);
        std::vector<std::uint32_t> tokens(static_cast<std::size_t>(tf.tellg()) / 4);
        tf.seekg(0);
        tf.read(reinterpret_cast<char*>(tokens.data()), static_cast<std::streamsize>(tokens.size() * 4));
        const unsigned n = static_cast<unsigned>(std::stoul(argv[3])), cap = static_cast<unsigned>(std::stoul(argv[5]));
        std::ifstream df(argv[4], std::ios::binary | std::ios::ate);
        if (!df) throw std::runtime_error("no engine dump");
        std::vector<float> dump(static_cast<std::size_t>(df.tellg()) / 4);
        df.seekg(0);
        df.read(reinterpret_cast<char*>(dump.data()), static_cast<std::streamsize>(dump.size() * 4));
        if (dump.size() != tokens.size() * kHc * kHidden) throw std::runtime_error("dump size != tokens x 4 x 4096");  // fail closed
        Replay rp(argv[1], n, cap);
        std::size_t bad = 0;
        for (std::size_t t = 0; t < tokens.size(); ++t) {
            const auto X = rp.token(tokens[t], static_cast<unsigned>(t));
            const bool same = std::memcmp(X.data(), dump.data() + t * kHc * kHidden, X.size() * 4) == 0;
            if (!same) {
                if (bad < 5) std::printf("MISMATCH residual at token %zu\n", t);
                ++bad;
            }
        }
        const auto st = rp.state();
        std::ifstream sf(std::string(argv[4]) + ".state", std::ios::binary | std::ios::ate);
        std::vector<std::uint8_t> est(sf ? static_cast<std::size_t>(sf.tellg()) : 0);
        if (sf) {
            sf.seekg(0);
            sf.read(reinterpret_cast<char*>(est.data()), static_cast<std::streamsize>(est.size()));
        }
        const bool state_same = est.size() == st.size() && std::memcmp(est.data(), st.data(), st.size()) == 0;
        std::printf("%s G3 replay: %zu tokens, layers %u: residual mismatches %zu; state %s (%zu B)\n",
                    bad == 0 && state_same ? "PASS" : "FAIL", tokens.size(), n, bad, state_same ? "identical" : "DIFFERENT", st.size());
        return bad == 0 && state_same ? 0 : 1;
    } catch (const std::exception& e) {
        std::fprintf(stderr, "error: %s\n", e.what());
        return 1;
    }
}
