// Device self-test of the shared vision encoder: synthetic towers on the HIP runner against the host FP32 reference.
// Gate (per row of the hidden state and of every tap): cosine mean >= 0.9995, min >= 0.998, relative L2 <= 2 %.
// Needs a GPU; run it before any family adapter's ticket. Exit 0 = PASS.
#include "strix/vision/embed_cache.hpp"

#include "vision_encoder_toy.hpp"

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
    bool split;
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
    EncoderOptions opt;
    opt.split_activations = k.split;
    opt.mlp_workspace_bytes = std::size_t{1} << 20U;  // several MLP chunks
    Encoder enc(k.cfg, opt);
    enc.load(t.w);
    EncodeStats stats;
    stats.per_layer = true;
    const EncoderOutput got = enc.encode(g, p, stats);
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
    const EncoderPlan pl = plan(k.cfg, g.rows, k.tiles, opt);
    ok = ok && enc.device_bytes() <= pl.weight_bytes + pl.workspace_bytes() + (std::size_t{1} << 20U);
    std::printf("%-22s rows %4u  cos mean %.6f min %.6f  relL2 %.5f  proj min %.6f  %.2f ms  dev %.1f MiB  %s\n", k.name, g.rows,
                worst.cos_mean, worst.cos_min, worst.rel_l2, proj.cos_min, stats.total_ms, static_cast<double>(enc.device_bytes()) / 1048576.0,
                ok ? "PASS" : "FAIL");
    return ok;
}

}  // namespace

int main() {
    using C = EncoderConfig;
    std::vector<Case> cases;
    cases.push_back({"d64-rms-swiglu-rope", tower(128, 2, 352, C::Norm::rmsnorm, C::Mlp::swiglu, C::Pos::rope2d, 14), 13, 13, 1, false});
    cases.push_back({"d64-rms-swiglu-tiles", tower(128, 2, 352, C::Norm::rmsnorm, C::Mlp::swiglu, C::Pos::rope2d, 14), 12, 12, 3, false});
    Case deep{"d72-ln-tanh-both-taps", tower(144, 2, 538, C::Norm::layernorm, C::Mlp::gelu_tanh, C::Pos::rope2d_plus_learned, 16), 10, 14, 1, false};
    deep.cfg.deepstack_taps = {0, 2};
    deep.cfg.final_norm = true;
    cases.push_back(deep);
    cases.push_back({"d80-ln-gelu-learned", tower(160, 2, 640, C::Norm::layernorm, C::Mlp::gelu, C::Pos::learned_interp, 14), 9, 7, 1, false});
    cases.push_back({"d64-split", tower(128, 2, 352, C::Norm::rmsnorm, C::Mlp::swiglu, C::Pos::rope2d, 14), 13, 13, 1, true});
    Case clamp{"d64-swiglu-limit-adj", tower(256, 4, 512, C::Norm::rmsnorm, C::Mlp::swiglu, C::Pos::rope2d, 14), 20, 17, 1, false};
    clamp.cfg.swiglu_limit = 1.0F;
    clamp.cfg.swiglu_gate_first = false;
    clamp.cfg.rope_pairing = C::RopePairing::adjacent;
    cases.push_back(clamp);
    bool ok = true;
    try {
        std::uint64_t seed = 101;
        for (const auto& k : cases) ok = run(k, seed += 7U) && ok;
    } catch (const std::exception& e) {
        std::fprintf(stderr, "FAIL exception: %s\n", e.what());
        return 1;
    }
    std::puts(ok ? "vision-encoder-hip: PASS" : "vision-encoder-hip: FAIL");
    return ok ? 0 : 1;
}
