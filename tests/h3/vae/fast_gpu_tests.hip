// GPU qualification of the FAST-class lever h3-vae-fast24-v1 (distilled 24-layer non-causal decoder).
// usage: strix-h3-fast-gpu-tests [fast weights] [fixture root] [--real]  (defaults H3_VAE_FAST_WEIGHTS, H3_VAE_FIXTURES)
// Gates: frames vs the python fast decoder (fixture fast, 24 x 7 x 16 x 16: 2 temporal blocks, shifted windows across
// the block boundary) PSNR >= 60 dB, finite, deterministic. Reported (lever quality, not gated here): PSNR of the fast
// decoder against the 36-layer reference decoder on the same latent (python fast vs python 36L, ours vs python 36L).
// --real: 37 x 48 x 84 (5 s 768x1344) decode timed.

#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <fstream>
#include <random>
#include <stdexcept>
#include <string>

#include "h3/vae/fast_decoder.hpp"
#include "h3/vae/vae_decoder.hpp"
#include "h3/vae/image.hpp"

using namespace strix::h3::vae;

namespace {
int failures = 0;
void check(bool ok, const std::string& what) {
    std::printf("%s %s\n", ok ? "PASS" : "FAIL", what.c_str());
    std::fflush(stdout);
    if (!ok) ++failures;
}

// This mode uses the pre-existing decoder API so the identical test source can
// also link against the parent revision for a byte-for-byte 2048 regression.
int dump_2048(const std::string& weights, const std::string& output, bool distilled) {
    std::ofstream file(output, std::ios::binary);
    const auto write = [&](const auto& v) {
        file.write(reinterpret_cast<const char*>(v.data()), std::streamsize(v.size() * sizeof(v[0])));
    };
    if (distilled) {
        fast::GpuDecoder dec(weights);
        H16 z(std::size_t(24) * 7 * 16 * 16);
        for (std::size_t i = 0; i < z.size(); ++i) z[i] = f2h(float(int(i % 29) - 14) / 32);
        write(dec.decode(z, 7, 16, 16));
    } else {
        GpuDecoder dec(weights);
        for (const auto [ny, nx] : {std::pair{2, 2}, std::pair{6, 10}, std::pair{8, 8}}) {
            H16 in(std::size_t(24) * ny * nx);
            for (std::size_t i = 0; i < in.size(); ++i) in[i] = f2h(float(int(i % 29) - 14) / 32);
            for (int stop : {0, 1, 36}) {
                H16 residual;
                const auto pixels = dec.decode_call(in, 1, ny, nx, nullptr, &residual, stop);
                write(residual); write(pixels);
            }
        }
    }
    if (!file) throw std::runtime_error("2048 regression byte dump failed");
    return 0;
}

// The image FAST flag uses the shared full-attention decoder, not the
// distilled video's windowed/temporal policy. No video fixture is needed.
int image_case(const std::string& weights) {
    ImageVae host(weights);
    const int width = host.head().x_embed.out, axes = width == 1024 ? 2 : 3;
    GpuDecoder dec(weights, 24, true, true);
    GpuDecoder plain(weights, 24, false, true);
    SafeTensors st(weights);
    const auto head = load_head(st);
    std::vector<LayerH> layers;
    for (int i = 0; i < 24; ++i) layers.push_back(load_layer(st, i));
    check(width == 1024 || width == 2048, "image weight width admitted");
    // Square, rectangular, and Lp padding boundary (P=60 then P=64).
    for (const auto [ny, nx] : {std::pair{2, 2}, std::pair{2, 4}, std::pair{6, 10}, std::pair{8, 8}}) {
        std::vector<float> z(std::size_t(24) * ny * nx);
        for (std::size_t i = 0; i < z.size(); ++i) z[i] = float(int(i % 29) - 14) / 32.0F;
        const auto ref = host.decode(z, ny * 16, nx * 16);
        const auto got = host.decode(z, ny * 16, nx * 16, [&](const auto& pq, int h, int w) {
            H16 in(pq.size());
            for (std::size_t i = 0; i < in.size(); ++i) in[i] = f2h(pq[i]);
            const auto pixels = dec.decode_call(in, 1, h, w);
            check(diff_f16(pixels, dec.decode_call(in, 1, h, w)).bitwise(), "image deterministic rerun");
            check(diff_f16(pixels, plain.decode_call(in, 1, h, w)).bitwise(), "image fused/unfused SwiGLU exact");
            auto x = precision::embed<half_bits>(head, in, 1, h, w);
            const auto rope = image_rope_table(h, w, false, axes);
            check(rope.pairs == axes * 8, "image RoPE uses weight geometry");
            for (const auto& l : layers) precision::layer_forward<half_bits>(l, x, h * w + kSuffix, rope);
            const auto cpu = precision::head_out<half_bits>(head, x, 1, h, w, nullptr, 1);
            const auto d = diff_f16(pixels, cpu);
            std::printf("  image %d-wide %dx%d call vs host FP16: %.2f dB, rel L2 %.4g\n", width, h, w, d.psnr(1), d.rel_l2());
            check(d.valid() && d.psnr(1) >= 52, "image call PSNR >= 52 dB vs host half-policy VAE");
            std::vector<float> out(pixels.size());
            for (std::size_t i = 0; i < out.size(); ++i) out[i] = h2f(pixels[i]);
            return out;
        });
        const auto d = diff_f32(got, ref);
        std::printf("  image %d-wide %dx%d RGB vs host FP32: %.2f dB\n", width, ny * 16, nx * 16, d.psnr(1));
        check(d.valid() && d.psnr(1) >= 35, "image RGB finite, PSNR >= 35 dB vs host FP32 VAE (candidate gate)");
    }
    std::printf("%s (%d failures)\n", failures ? "FAILED" : "ALL PASS", failures);
    return failures ? 1 : 0;
}
}  // namespace

int main(int argc, char** argv) {
    if (argc == 4 && (!std::strcmp(argv[1], "--bytes-video") || !std::strcmp(argv[1], "--bytes-fast"))) {
        try { return dump_2048(argv[2], argv[3], !std::strcmp(argv[1], "--bytes-fast")); }
        catch (const std::exception& e) { std::printf("FAIL 2048 byte dump: %s\n", e.what()); return 1; }
    }
    if (argc == 3 && !std::strcmp(argv[1], "--image")) {
        try { return image_case(argv[2]); }
        catch (const std::exception& e) { std::printf("FAIL image VAE: %s\n", e.what()); return 1; }
    }
    std::string weights = std::getenv("H3_VAE_FAST_WEIGHTS") ? std::getenv("H3_VAE_FAST_WEIGHTS") : "";
    std::string root = std::getenv("H3_VAE_FIXTURES") ? std::getenv("H3_VAE_FIXTURES") : "";
    bool real = false;
    int pos = 0;
    for (int i = 1; i < argc; ++i) {
        if (!std::strcmp(argv[i], "--real")) real = true;
        else if (pos++ == 0) weights = argv[i];
        else root = argv[i];
    }
    if (weights.empty() || root.empty() || !file_exists(weights) || !file_exists(root + "/fast/manifest.json")) {
        std::printf("FAIL qualification needs the fast decoder weights and the fast fixture\nFAILED (1 failures)\n");
        return 1;
    }
    std::printf("lever identity %s (opt-in only)\n", fast::kIdentity);
    fast::GpuDecoder dec(weights);
    {
        bool r = false;
        try { (void)dec.decode(H16(10, 0), 7, 16, 16); } catch (const std::invalid_argument&) { r = true; }
        check(r, "fast decoder rejects a latent of the wrong length");
    }
    const Fixture fx(root + "/fast");
    const auto zf = fx.f32("z_norm_f32");
    H16 z(zf.size());
    for (std::size_t i = 0; i < zf.size(); ++i) z[i] = f2h(zf[i]);
    double sec = 0;
    const auto frames = dec.decode(z, 7, 16, 16, &sec);
    const Diff d = diff_f32(frames, fx.f32("frames_f32"));
    std::printf("  fast frames vs python fast: n=%zu max_abs=%.4g psnr=%.2f dB (%.2f s)\n", d.n, d.max_abs, d.psnr(1.0), sec);
    check(d.valid() && d.psnr(1.0) >= 60.0, "fast decoder frames PSNR >= 60 dB vs the python fast decoder, finite");
    check(diff_f32(dec.decode(z, 7, 16, 16), frames).bitwise(), "fast decoder deterministic (bitwise rerun)");
    if (file_exists(root + "/t7/manifest.json")) {  // same seeded latent through the 36-layer reference decoder
        const Fixture t7(root + "/t7");
        const auto ref36 = t7.f32("frames_f32");
        std::printf("  lever delta: python fast vs python 36L %.2f dB; ours fast vs python 36L %.2f dB\n",
                    diff_f32(fx.f32("frames_f32"), ref36).psnr(1.0), diff_f32(frames, ref36).psnr(1.0));
    }
    if (real) {
        std::mt19937 rng(7);
        std::normal_distribution<float> nd;
        H16 zr(std::size_t(24) * 37 * 48 * 84);
        for (auto& v : zr) v = f2h(nd(rng));
        double s = 0;
        const auto fr = dec.decode(zr, 37, 48, 84, &s);
        bool finite = fr.size() == std::size_t(3) * dec.frames(37) * 768 * 1344;
        for (float f : fr) finite &= std::isfinite(f);
        std::printf("  real 37x48x84 fast decode: %d frames, %.1f s (one node)\n", dec.frames(37), s);
        check(finite, "real shape: finite frames");
    }
    std::printf("%s (%d failures)\n", failures ? "FAILED" : "ALL PASS", failures);
    return failures ? 1 : 0;
}
