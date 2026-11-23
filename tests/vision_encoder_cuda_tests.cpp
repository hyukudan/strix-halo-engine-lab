// Device self-test of the CUDA vision tower: synthetic towers on CudaEncoder against the host FP32 reference.
// Gate (per row of the hidden state and of every tap): cosine mean >= 0.9995, min >= 0.998, relative L2 <= 2 %
// (the HIP runner's gate; the CUDA tower is FP32 end to end so it lands far inside it). Also: determinism over two
// encodes, the DenseOps linear and the shared merge_project graph against the host reference backend.
// Needs a GPU; run it before any family adapter's ticket. Exit 0 = PASS.
#include "strix/vision/embed_cache.hpp"
#include "strix/vision/encoder_cuda.hpp"

#include "vision_encoder_toy.hpp"

#include <algorithm>
#include <cmath>
#include <cstdio>
#include <exception>
#include <string>
#include <vector>

using namespace strix::vision;
using vision_toy::make_grid;
using vision_toy::make_tower;
using vision_toy::Tower;

namespace {

struct Score {
    double cos_mean = 0.0, cos_min = 1.0, rel_l2 = 0.0;
    bool pass() const { return cos_mean >= 0.9995 && cos_min >= 0.998 && rel_l2 <= 0.02; }
};
Score score(const std::vector<float>& got, const std::vector<float>& want, unsigned rows, unsigned width) {
    Score s;
    if (got.size() != want.size() || got.size() != std::size_t{rows} * width) return {0.0, 0.0, 1.0};
    double err = 0.0, ref = 0.0;
    for (unsigned r = 0; r < rows; ++r) {
        double ab = 0.0, aa = 0.0, bb = 0.0;
        for (unsigned k = 0; k < width; ++k) {
            const double a = static_cast<double>(got[std::size_t{r} * width + k]), b = static_cast<double>(want[std::size_t{r} * width + k]);
            if (!std::isfinite(a)) return {0.0, 0.0, 1.0};
            ab += a * b;
            aa += a * a;
            bb += b * b;
            err += (a - b) * (a - b);
        }
        const double c = aa > 0.0 && bb > 0.0 ? ab / std::sqrt(aa * bb) : 0.0;
        s.cos_mean += c / rows;
        s.cos_min = std::min(s.cos_min, c);
        ref += bb;
    }
    s.rel_l2 = ref > 0.0 ? std::sqrt(err / ref) : 1.0;
    return s;
}

struct Case {
    const char* name;
    EncoderConfig cfg;
    unsigned gh, gw, tiles;
};

EncoderConfig tower(unsigned width, unsigned heads, unsigned mlp, EncoderConfig::Norm norm, EncoderConfig::Mlp act, EncoderConfig::Pos pos,
                    unsigned patch) {
    EncoderConfig c;
    c.layers = 3;
    c.width = width;
    c.heads = heads;
    c.mlp_width = mlp;
    c.norm = norm;
    c.mlp = act;
    c.pos = pos;
    c.pos_grid = c.has_learned() ? 6U : 0U;
    c.patch = patch;
    c.qkv_bias = c.out_bias = true;
    c.mlp_bias = act != EncoderConfig::Mlp::swiglu;
    return c;
}

bool run(const Case& k, std::uint64_t seed) {
    Tower t = make_tower(k.cfg, seed, 48);
    const PatchGrid g = make_grid(k.cfg, k.gh, k.gw, seed + 1U, k.tiles);
    const PositionInputs p = default_positions(k.cfg, t.w, g);
    const EncoderOutput want = reference_encode(k.cfg, t.w, g, p);
    CudaEncoderOptions opt;
    opt.mlp_workspace_bytes = std::size_t{1} << 20U;  // several MLP chunks
    CudaEncoder enc(k.cfg, opt);
    enc.load(t.w);
    const EncoderOutput got = enc.encode(g, p);
    const EncoderOutput twice = enc.encode(g, p);
    bool ok = got.hidden == twice.hidden && got.taps.size() == want.taps.size();  // deterministic, workspace reuse
    Score worst = score(got.hidden, want.hidden, g.rows, k.cfg.width);
    ok = ok && worst.pass();
    for (std::size_t i = 0; ok && i < want.taps.size(); ++i) {
        const Score s = score(got.taps[i], want.taps[i], g.rows, k.cfg.width);
        ok = s.pass();
        if (s.cos_min < worst.cos_min) worst = s;
    }
    // projector linear + cache path on the device backend
    vision_toy::ToyAdapter adapter(t);
    ReferenceEncoder ref(k.cfg, t.w);
    Digest d{};
    const CachedEncode dev = encode_cached(adapter, enc, t.w, nullptr, d, g), host = encode_cached(adapter, ref, t.w, nullptr, d, g);
    std::vector<float> a(dev.rows->bf16.size()), b(a.size());
    for (std::size_t i = 0; i < a.size(); ++i) {
        a[i] = from_bf16(dev.rows->bf16[i]);
        b[i] = from_bf16(host.rows->bf16[i]);
    }
    const Score proj = score(a, b, g.rows, 48);
    ok = ok && proj.pass();
    std::printf("%-22s rows %4u  cos mean %.6f min %.6f  relL2 %.5f  proj min %.6f  dev %.1f MiB  %s\n", k.name, g.rows, worst.cos_mean,
                worst.cos_min, worst.rel_l2, proj.cos_min, static_cast<double>(enc.device_bytes()) / 1048576.0, ok ? "PASS" : "FAIL");
    return ok;
}

// The shared DenseOps::merge_project graph (2x2 merge, two linears, activation) on the CUDA linear against the host one.
bool merge_project_case() {
    using C = EncoderConfig;
    auto cfg = tower(128, 2, 512, C::Norm::layernorm, C::Mlp::gelu, C::Pos::learned_interp, 1);
    Tower t = make_tower(cfg, 603);
    constexpr unsigned ratio = 2, p = 96, out = 40;
    std::vector<float> w1(std::size_t{p} * cfg.width * ratio * ratio), w2(std::size_t{out} * p), b1(p);
    vision_toy::Rng rng{604};
    for (float& v : w1) v = rng.next() * 0.03125F;
    for (float& v : w2) v = rng.next() * 0.25F;
    for (float& v : b1) v = rng.next();
    MergeProjector m;
    m.ratio = ratio;
    m.w1 = Tensor{w1.data(), Tensor::Type::f32, p, std::size_t{cfg.width} * ratio * ratio};
    m.b1 = Tensor{b1.data(), Tensor::Type::f32, p, 1};
    m.w2 = Tensor{w2.data(), Tensor::Type::f32, out, p};
    CudaEncoder enc(cfg);
    enc.load(t.w);
    ReferenceEncoder ref(cfg, t.w);
    const PatchGrid g = make_grid(cfg, 6U, 8U, 605);
    const PositionInputs pos = default_positions(cfg, t.w, g);
    const EncoderOutput h = reference_encode(cfg, t.w, g, pos);
    bool ok = true;
    double cmin = 1.0;
    for (const bool row_major : {false, true}) {
        m.row_major = row_major;
        const std::size_t tokens = m.tokens(6U, 8U);
        std::vector<float> want(tokens * out), got(want.size());
        ref.merge_project(h, 6U, 8U, m, want.data());
        enc.merge_project(h, 6U, 8U, m, got.data());
        const Score s = score(got, want, static_cast<unsigned>(tokens), out);
        cmin = std::min(cmin, s.cos_min);
        ok = ok && s.pass();
    }
    std::printf("%-22s merge 2x2 -> %u -> %u  cos min %.6f  %s\n", "merge-project", p, out, cmin, ok ? "PASS" : "FAIL");
    return ok;
}

}  // namespace

int main() {
    using C = EncoderConfig;
    std::vector<Case> cases;
    // The Inkling-class tower in miniature: LayerNorm eps 1e-5, GELU, learned interpolated positions, embed_norm on.
    Case ink{"ln-gelu-learned-embnorm", tower(128, 2, 512, C::Norm::layernorm, C::Mlp::gelu, C::Pos::learned_interp, 14), 9, 11, 1};
    ink.cfg.embed_norm = true;
    ink.cfg.patch_bias = false;
    ink.cfg.eps = 1e-5F;
    ink.cfg.qkv_bias = ink.cfg.out_bias = false;
    cases.push_back(ink);
    Case ink2{"ln-gelu-embnorm-final", tower(128, 2, 512, C::Norm::layernorm, C::Mlp::gelu, C::Pos::learned_interp, 14), 7, 10, 1};
    ink2.cfg.embed_norm = true;
    ink2.cfg.final_norm = true;
    ink2.cfg.eps = 1e-5F;
    cases.push_back(ink2);
    cases.push_back({"d64-rms-swiglu-rope", tower(128, 2, 352, C::Norm::rmsnorm, C::Mlp::swiglu, C::Pos::rope2d, 14), 13, 13, 1});
    cases.push_back({"d64-rms-swiglu-tiles", tower(128, 2, 352, C::Norm::rmsnorm, C::Mlp::swiglu, C::Pos::rope2d, 14), 12, 12, 3});
    Case deep{"d72-ln-tanh-both-taps", tower(144, 2, 538, C::Norm::layernorm, C::Mlp::gelu_tanh, C::Pos::rope2d_plus_learned, 16), 10, 14, 1};
    deep.cfg.deepstack_taps = {0, 2};
    deep.cfg.final_norm = true;
    cases.push_back(deep);
    cases.push_back({"d72-tiled-tail", deep.cfg, 3, 5, 3});
    Case clamp{"d64-swiglu-limit-adj", tower(256, 4, 512, C::Norm::rmsnorm, C::Mlp::swiglu, C::Pos::rope2d, 14), 20, 17, 1};
    clamp.cfg.swiglu_limit = 1.0F;
    clamp.cfg.swiglu_gate_first = false;
    clamp.cfg.rope_pairing = C::RopePairing::adjacent;
    cases.push_back(clamp);
    bool ok = true;
    try {
        std::printf("backend %s\n", CudaEncoder::kArithmetic);
        std::uint64_t seed = 101;
        for (const auto& k : cases) {
            seed += 7U;
            ok = run(k, seed) && ok;
        }
        ok = merge_project_case() && ok;
    } catch (const std::exception& e) {
        std::fprintf(stderr, "FAIL exception: %s\n", e.what());
        return 1;
    }
    std::puts(ok ? "vision-encoder-cuda: PASS" : "vision-encoder-cuda: FAIL");
    return ok ? 0 : 1;
}
