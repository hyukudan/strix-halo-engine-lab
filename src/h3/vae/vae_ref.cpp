#include "h3/vae/vae_ref.hpp"
#include "h3/vae/precision.hpp"
#include "h3/vae/vae_threads.hpp"

#include <algorithm>
#include <cmath>
#include <cstring>
#include <stdexcept>
#include <thread>

namespace strix::h3::vae {

namespace {
template <class F> void parallel_rows(int rows, F&& f) {
    const unsigned hw = std::max(1U, std::thread::hardware_concurrency());
    const int nt = int(std::min<unsigned>(hw, unsigned(std::max(1, rows))));
    Workers pool;
    for (int t = 0; t < nt; ++t)
        pool.run([&, t] {
            for (int r = t; r < rows; r += nt) f(r);
        });
    pool.join();
}
H16 to_h16(const std::vector<std::uint8_t>& raw) {
    H16 out(raw.size() / 2);
    std::memcpy(out.data(), raw.data(), raw.size());
    return out;
}
H16 read_f16(const SafeTensors& st, const std::string& name) {
    if (st.info(name).dtype == "f16") return to_h16(st.read(name));
    if (st.info(name).dtype == "f32") {  // the fast decoder ships FP32: cast as the module's .to(fp16)
        const auto raw = st.read(name);
        H16 out(raw.size() / 4);
        for (std::size_t i = 0; i < out.size(); ++i) {
            float f;
            std::memcpy(&f, raw.data() + 4 * i, 4);
            out[i] = f2h(f);
        }
        return out;
    }
    throw std::runtime_error("unsupported dtype for " + name);
}
}  // namespace

LinearH load_linear(const SafeTensors& st, const std::string& prefix) {
    LinearH l;
    const auto& wi = st.info(prefix + ".weight");
    l.out = int(wi.shape[0]);
    l.in = int(wi.numel() / wi.shape[0]);
    l.w = read_f16(st, prefix + ".weight");
    if (st.has(prefix + ".bias")) l.b = read_f16(st, prefix + ".bias");
    else l.b.assign(std::size_t(l.out), 0);
    return l;
}

LayerH load_layer(const SafeTensors& st, int layer, const std::string& block_prefix) {
    const std::string p = block_prefix + std::to_string(layer) + ".";
    LayerH w;
    w.norm1 = read_f16(st, p + "norm1.weight");
    w.norm2 = read_f16(st, p + "norm2.weight");
    w.scale1 = read_f16(st, p + "scale1");
    w.scale2 = read_f16(st, p + "scale2");
    w.qkv = load_linear(st, p + "attn.to_qkv");
    w.out = load_linear(st, p + "attn.to_out");
    w.w1 = load_linear(st, p + "ff.w1");
    w.w2 = load_linear(st, p + "ff.w2");
    return w;
}

HeadH load_head(const SafeTensors& st, const std::string& dec_prefix) {
    HeadH h;
    h.post_quant = load_linear(st, "post_quant_conv");
    h.x_embed = load_linear(st, dec_prefix + "x_embedder");
    h.proj = load_linear(st, dec_prefix + "proj_out");
    h.registers = read_f16(st, dec_prefix + "register_tokens");
    h.norm_out_w = read_f16(st, dec_prefix + "norm_out.weight");
    h.norm_out_b = read_f16(st, dec_prefix + "norm_out.bias");
    if (st.has("latents_mean")) {
        h.latents_mean = read_f16(st, "latents_mean");
        h.latents_std = read_f16(st, "latents_std");
    }
    return h;
}

RopeTable rope_table(int nt, int ny, int nx) {
    RopeTable r;
    r.L = nt * ny * nx + kSuffix;
    r.c.assign(std::size_t(r.L) * kRotPairs, 1.0F);
    r.s.assign(std::size_t(r.L) * kRotPairs, 0.0F);
    // inv_freq = 1 / 100 ** arange(0, 1, 6/48) in FP32, then the module cast to FP16.
    float inv[8];
    for (int f = 0; f < 8; ++f) {
        const float e = float(f) * 0.125F;
        const float p = std::pow(100.0F, e);
        inv[f] = rh(1.0F / p);
    }
    const auto coord = [](int i, int n) {  // FP16 arithmetic: ((i + 0.5) / n) * 2 - 1
        const float a = rh(float(i) + 0.5F);
        const float b = rh(a / float(n));
        const float c = rh(2.0F * b);
        return rh(c - 1.0F);
    };
    const float two_pi = float(2.0 * 3.14159265358979323846);
    for (int t = 0; t < nt; ++t)
        for (int y = 0; y < ny; ++y)
            for (int x = 0; x < nx; ++x) {
                const int tok = (t * ny + y) * nx + x;
                const float ids[3] = {coord(t, nt), coord(y, ny), coord(x, nx)};
                for (int a = 0; a < 3; ++a)
                    for (int f = 0; f < 8; ++f) {
                        const float ang = (two_pi * ids[a]) * inv[f];
                        r.c[std::size_t(tok) * kRotPairs + a * 8 + f] = rh(std::cos(ang));
                        r.s[std::size_t(tok) * kRotPairs + a * 8 + f] = rh(std::sin(ang));
                    }
            }
    return r;
}

H16 linear(const H16& x, int rows, const LinearH& w, bool round_before_bias) { return precision::linear<half_bits>(x,rows,w,round_before_bias); }

void linear_residual(H16& x, const H16& in, int rows, const LinearH& w, const H16& scale) { precision::linear_residual<half_bits>(x,in,rows,w,scale); }

H16 rms_norm(const H16& x, int rows, int width, const H16* weight, float eps) { return precision::rms_norm<half_bits>(x,rows,width,weight,eps); }

H16 layer_norm(const H16& x, int rows, int width, const H16& w, const H16& b, float eps) { return precision::layer_norm<half_bits>(x,rows,width,w,b,eps); }

namespace {
// Per-head RMSNorm (unit weight) + split-half rope on [0, 48): FP16 rounding after each product and add.

}  // namespace

// FP16 sanitization after attention (reference): NaN -> 0, +-Inf -> +-65504.
half_bits nan_to_num16(half_bits b) {
    if ((b & 0x7c00U) != 0x7c00U) return b;
    if (b & 0x03ffU) return 0;
    return half_bits((b & 0x8000U) | 0x7bffU);
}

H16 attention(const H16& qkv, int L, const RopeTable& rope, H16* q_out, H16* k_out) { return precision::attention<half_bits>(qkv,L,rope,q_out,k_out); }

void layer_forward(const LayerH& w, H16& x, int L, const RopeTable& rope) { precision::layer_forward<half_bits>(w,x,L,rope); }

H16 post_quant(const HeadH& h, const H16& z, int voxels) { return precision::post_quant<half_bits>(h,z,voxels); }

H16 embed(const HeadH& h, const H16& dec_in, int nt, int ny, int nx) { return precision::embed<half_bits>(h,dec_in,nt,ny,nx); }

H16 unpatchify(const H16& pr, int nt, int ny, int nx) { return precision::unpatchify<half_bits>(pr,nt,ny,nx); }

H16 head_out(const HeadH& h, const H16& x, int nt, int ny, int nx, H16* proj_rows) { return precision::head_out<half_bits>(h,x,nt,ny,nx,proj_rows); }

// Invalid comparisons (different lengths, empty, any non-finite value on either side) score -inf PSNR and +inf
// error, so every threshold fails on them.
double Diff::psnr(double peak) const {
    if (!valid()) return -INFINITY;
    return rms_err > 0 ? 20.0 * std::log10(peak / rms_err) : INFINITY;
}
double Diff::snr_db() const {
    if (!valid()) return -INFINITY;
    return rms_err > 0 ? 20.0 * std::log10(rms_ref / rms_err) : INFINITY;
}
double Diff::rel_l2() const {
    if (!valid()) return INFINITY;
    return rms_ref > 0 ? rms_err / rms_ref : rms_err;
}

namespace {
template <class T, class ToD, class Bits>
Diff diff_impl(const std::vector<T>& got, const std::vector<T>& ref, ToD to_d, Bits bits) {
    Diff d;
    d.size_mismatch = got.size() != ref.size();
    d.n = std::min(got.size(), ref.size());
    double se = 0, sr = 0;
    for (std::size_t i = 0; i < d.n; ++i) {
        if (bits(got[i]) != bits(ref[i])) ++d.mismatches;
        const double a = to_d(got[i]), b = to_d(ref[i]);
        if (!std::isfinite(a) || !std::isfinite(b)) { ++d.nonfinite; continue; }
        d.max_abs = std::max(d.max_abs, std::fabs(a - b));
        se += (a - b) * (a - b);
        sr += b * b;
    }
    d.rms_err = std::sqrt(se / double(std::max<std::size_t>(1, d.n)));
    d.rms_ref = std::sqrt(sr / double(std::max<std::size_t>(1, d.n)));
    return d;
}
}  // namespace

Diff diff_f16(const H16& got, const H16& ref) {
    return diff_impl(got, ref, [](half_bits h) { return double(h2f(h)); }, [](half_bits h) { return unsigned(h); });
}
Diff diff_f32(const std::vector<float>& got, const std::vector<float>& ref) {
    return diff_impl(got, ref, [](float f) { return double(f); }, [](float f) { return f32_bits(f); });
}

}  // namespace strix::h3::vae
