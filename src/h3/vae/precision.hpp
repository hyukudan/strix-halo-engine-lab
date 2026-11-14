#pragma once
// Shared scalar-policy VAE reference operators. FP16 wrappers retain their rounding points.
#include <algorithm>
#include <cmath>
#include <cstring>
#include <thread>
#include "h3/vae/vae_ref.hpp"
#include "h3/vae/enc_ref.hpp"
#include "h3/vae/vae_threads.hpp"
namespace strix::h3::vae::precision {
template<class F> void parallel_rows(int rows, F&& f) {
    const int n=int(std::max(1U,std::min<unsigned>(std::thread::hardware_concurrency(),unsigned(std::max(1,rows)))));
    Workers workers;
    for(int i=0;i<n;++i) workers.run([&,i] { for(int r=i;r<rows;r+=n) f(r); });
    workers.join();
}
template<class T>
std::vector<T> linear(const std::vector<T>& x, int rows, const LinearT<T>& w, bool round_before_bias = false) {
    std::vector<T> y(std::size_t(rows) * w.out);
    std::vector<float> wf(w.w.size());
    for (std::size_t i = 0; i < wf.size(); ++i) wf[i] = Scalar<T>::get(w.w[i]);
    parallel_rows(rows, [&](int r) {
        std::vector<float> xr(static_cast<std::size_t>(w.in));
        for (int k = 0; k < w.in; ++k) xr[std::size_t(k)] = Scalar<T>::get(x[std::size_t(r) * w.in + k]);
        for (int n = 0; n < w.out; ++n) {
            const float* wr = wf.data() + std::size_t(n) * w.in;
            float acc[8] = {};
            int k = 0;
            for (; k + 8 <= w.in; k += 8)
                for (int u = 0; u < 8; ++u) acc[u] += xr[std::size_t(k + u)] * wr[k + u];
            float s = ((acc[0] + acc[1]) + (acc[2] + acc[3])) + ((acc[4] + acc[5]) + (acc[6] + acc[7]));
            for (; k < w.in; ++k) s += xr[std::size_t(k)] * wr[k];
            y[std::size_t(r) * w.out + n] = round_before_bias ? Scalar<T>::set(Scalar<T>::round(s) + Scalar<T>::get(w.b[std::size_t(n)])) : Scalar<T>::set(s + Scalar<T>::get(w.b[std::size_t(n)]));
        }
    });
    return y;
}

template<class T>
void linear_residual(std::vector<T>& x, const std::vector<T>& in, int rows, const LinearT<T>& w, const std::vector<T>& scale) {
    const std::vector<T> y = linear<T>(in, rows, w);
    for (int r = 0; r < rows; ++r)
        for (int n = 0; n < w.out; ++n) {
            const std::size_t i = std::size_t(r) * w.out + n;
            x[i] = Scalar<T>::set(Scalar<T>::get(x[i]) + Scalar<T>::get(y[i]) * Scalar<T>::get(scale[std::size_t(n)]));
        }
}

template<class T>
std::vector<T> rms_norm(const std::vector<T>& x, int rows, int width, const std::vector<T>* weight, float eps) {
    std::vector<T> y(x.size());
    for (int r = 0; r < rows; ++r) {
        const T* xr = x.data() + std::size_t(r) * width;
        double ss = 0;
        for (int k = 0; k < width; ++k) { const float v = Scalar<T>::get(xr[k]); ss += double(v) * v; }
        const float inv = 1.0F / std::sqrt(float(ss / width) + eps);
        for (int k = 0; k < width; ++k) {
            float v = Scalar<T>::get(xr[k]) * inv;
            if (weight) v *= Scalar<T>::get((*weight)[std::size_t(k)]);
            y[std::size_t(r) * width + k] = Scalar<T>::set(v);
        }
    }
    return y;
}

template<class T>
std::vector<T> layer_norm(const std::vector<T>& x, int rows, int width, const std::vector<T>& w, const std::vector<T>& b, float eps) {
    std::vector<T> y(x.size());
    for (int r = 0; r < rows; ++r) {
        const T* xr = x.data() + std::size_t(r) * width;
        double mu = 0;
        for (int k = 0; k < width; ++k) mu += Scalar<T>::get(xr[k]);
        mu /= width;
        double var = 0;
        for (int k = 0; k < width; ++k) { const double d = Scalar<T>::get(xr[k]) - mu; var += d * d; }
        var /= width;
        const float inv = float(1.0 / std::sqrt(var + eps));
        for (int k = 0; k < width; ++k)
            y[std::size_t(r) * width + k] = Scalar<T>::set(float(Scalar<T>::get(xr[k]) - mu) * inv * Scalar<T>::get(w[std::size_t(k)]) + Scalar<T>::get(b[std::size_t(k)]));
    }
    return y;
}

template<class T>
void rms_rope_head(const T* src, T* dst, const float* c, const float* s, int pairs = kRotPairs) {
    double ss = 0;
    for (int d = 0; d < kHeadDim; ++d) { const float v = Scalar<T>::get(src[d]); ss += double(v) * v; }
    const float inv = 1.0F / std::sqrt(float(ss / kHeadDim) + 1e-5F);
    float n[kHeadDim];
    for (int d = 0; d < kHeadDim; ++d) n[d] = Scalar<T>::round(Scalar<T>::get(src[d]) * inv);  // unit weight: x * inv * 1
    for (int i = 0; i < pairs; ++i) {
        const float x0 = n[i], x1 = n[i + pairs];
        const float o0 = Scalar<T>::round(Scalar<T>::round(x0 * c[i]) + x1 * -s[i]);
        const float o1 = Scalar<T>::round(Scalar<T>::round(x1 * c[i]) + x0 * s[i]);
        dst[i] = Scalar<T>::set(o0);
        dst[i + pairs] = Scalar<T>::set(o1);
    }
    for (int d = 2 * pairs; d < kHeadDim; ++d) dst[d] = Scalar<T>::set(n[d]);
}

template<class T>
std::vector<T> attention(const std::vector<T>& qkv, int L, const RopeTable& rope, std::vector<T>* q_out = nullptr, std::vector<T>* k_out = nullptr,
                         int kDim = vae::kDim) {  // width: 2048 (video, p-h3-image) or 1024 (MiniMax-H3-Image)
    const int kHeads = kDim / kHeadDim, kQkv = 3 * kDim, kRotPairs = rope.pairs;
    std::vector<T> q(std::size_t(L) * kDim), k(std::size_t(L) * kDim), v(std::size_t(L) * kDim);
    for (int r = 0; r < L; ++r)
        for (int h = 0; h < kHeads; ++h) {
            const T* row = qkv.data() + std::size_t(r) * kQkv + h * 3 * kHeadDim;
            const float* c = rope.c.data() + std::size_t(r) * kRotPairs;
            const float* s = rope.s.data() + std::size_t(r) * kRotPairs;
            rms_rope_head<T>(row, q.data() + std::size_t(r) * kDim + h * kHeadDim, c, s, kRotPairs);
            rms_rope_head<T>(row + kHeadDim, k.data() + std::size_t(r) * kDim + h * kHeadDim, c, s, kRotPairs);
            std::memcpy(v.data() + std::size_t(r) * kDim + h * kHeadDim, row + 2 * kHeadDim, kHeadDim * sizeof(T));
        }
    if (q_out) *q_out = q;
    if (k_out) *k_out = k;
    std::vector<T> o(std::size_t(L) * kDim);
    std::vector<float> kf(k.size()), vf(v.size());
    for (std::size_t i = 0; i < k.size(); ++i) { kf[i] = Scalar<T>::get(k[i]); vf[i] = Scalar<T>::get(v[i]); }
    parallel_rows(L * kHeads, [&](int job) {
        const int r = job / kHeads, h = job % kHeads;
        float qr[kHeadDim];
        for (int d = 0; d < kHeadDim; ++d) qr[d] = Scalar<T>::get(q[std::size_t(r) * kDim + h * kHeadDim + d]);
        std::vector<double> sc(static_cast<std::size_t>(L));
        double mx = -1e300;
        for (int j = 0; j < L; ++j) {
            const float* kr = kf.data() + std::size_t(j) * kDim + h * kHeadDim;
            double dot = 0;
            for (int d = 0; d < kHeadDim; ++d) dot += double(qr[d]) * kr[d];
            sc[std::size_t(j)] = dot * 0.125;
            mx = std::max(mx, sc[std::size_t(j)]);
        }
        double sum = 0;
        for (int j = 0; j < L; ++j) { sc[std::size_t(j)] = std::exp(sc[std::size_t(j)] - mx); sum += sc[std::size_t(j)]; }
        double acc[kHeadDim] = {};
        for (int j = 0; j < L; ++j) {
            const float* vr = vf.data() + std::size_t(j) * kDim + h * kHeadDim;
            for (int d = 0; d < kHeadDim; ++d) acc[d] += sc[std::size_t(j)] * vr[d];
        }
        for (int d = 0; d < kHeadDim; ++d) o[std::size_t(r) * kDim + h * kHeadDim + d] = Scalar<T>::sanitize(Scalar<T>::set(float(acc[d] / sum)));
    });
    return o;
}

template<class T>
void layer_forward(const LayerT<T>& w, std::vector<T>& x, int L, const RopeTable& rope) {
    const int kDim = w.qkv.in, kFf = w.w2.in, kFf2 = 2 * kFf;  // geometry from the weights (video: 2048 / 8192)
    const std::vector<T> a = rms_norm<T>(x, L, kDim, &w.norm1, 1e-5F);
    const std::vector<T> qkv = linear<T>(a, L, w.qkv);
    const std::vector<T> o = attention<T>(qkv, L, rope, nullptr, nullptr, kDim);
    linear_residual<T>(x, o, L, w.out, w.scale1);
    const std::vector<T> a2 = rms_norm<T>(x, L, kDim, &w.norm2, 1e-5F);
    const std::vector<T> h = linear<T>(a2, L, w.w1);
    std::vector<T> g(std::size_t(L) * kFf);
    for (int r = 0; r < L; ++r)
        for (int n = 0; n < kFf; ++n) {
            const float gate = Scalar<T>::get(h[std::size_t(r) * kFf2 + n]);
            const float up = Scalar<T>::get(h[std::size_t(r) * kFf2 + kFf + n]);
            const float si = Scalar<T>::round(gate / (1.0F + std::exp(-gate)));
            g[std::size_t(r) * kFf + n] = Scalar<T>::set(si * up);
        }
    linear_residual<T>(x, g, L, w.w2, w.scale2);
}

template<class T>
std::vector<T> post_quant(const HeadT<T>& h, const std::vector<T>& z, int voxels) {
    std::vector<T> out(z.size());
    for (int co = 0; co < kLatentC; ++co)
        for (int p = 0; p < voxels; ++p) {
            float s = 0;
            for (int ci = 0; ci < kLatentC; ++ci) s += Scalar<T>::get(h.post_quant.w[std::size_t(co) * kLatentC + ci]) * Scalar<T>::get(z[std::size_t(ci) * voxels + p]);
            out[std::size_t(co) * voxels + p] = Scalar<T>::set(s + Scalar<T>::get(h.post_quant.b[std::size_t(co)]));
        }
    return out;
}

template<class T>
std::vector<T> embed(const HeadT<T>& h, const std::vector<T>& dec_in, int nt, int ny, int nx) {
    const int P = nt * ny * nx, L = P + kSuffix, kDim = h.x_embed.out;
    std::vector<T> rows(std::size_t(P) * kLatentC);
    for (int p = 0; p < P; ++p)
        for (int c = 0; c < kLatentC; ++c) rows[std::size_t(p) * kLatentC + c] = dec_in[std::size_t(c) * P + p];
    // The transposed view rounds matmul before bias; the scalar policy controls precision.
    const std::vector<T> e = linear<T>(rows, P, h.x_embed, true);
    std::vector<T> x(std::size_t(L) * kDim, 0);
    std::memcpy(x.data(), e.data(), e.size() * sizeof(T));
    std::memcpy(x.data() + std::size_t(P) * kDim, h.registers.data(), std::size_t(kRegisters) * kDim * sizeof(T));
    return x;  // last row: zeros
}

template<class T>
std::vector<T> unpatchify(const std::vector<T>& pr, int nt, int ny, int nx, int patch_t = 4) {
    const int F = patch_t * nt, Hh = 16 * ny, Ww = 16 * nx;
    std::vector<T> out(std::size_t(3) * F * Hh * Ww);
    for (int t = 0; t < nt; ++t)
        for (int y = 0; y < ny; ++y)
            for (int x = 0; x < nx; ++x) {
                const T* row = pr.data() + std::size_t((t * ny + y) * nx + x) * (3 * patch_t * 256);
                for (int c = 0; c < 3; ++c)
                    for (int pt = 0; pt < patch_t; ++pt)
                        for (int py = 0; py < 16; ++py)
                            for (int px = 0; px < 16; ++px)
                                out[((std::size_t(c) * F + t * patch_t + pt) * Hh + y * 16 + py) * Ww + x * 16 + px] =
                                    row[((c * patch_t + pt) * 16 + py) * 16 + px];
            }
    return out;
}

template<class T>
std::vector<T> head_out(const HeadT<T>& h, const std::vector<T>& x, int nt, int ny, int nx, std::vector<T>* proj_rows = nullptr, int patch_t = 4) {
    const int P = nt * ny * nx, kDim = int(h.norm_out_w.size());
    const std::vector<T> xs(x.begin(), x.begin() + std::ptrdiff_t(std::size_t(P) * kDim));
    const std::vector<T> n = layer_norm<T>(xs, P, kDim, h.norm_out_w, h.norm_out_b, 1e-5F);
    const std::vector<T> pr = linear<T>(n, P, h.proj);
    if (proj_rows) *proj_rows = pr;
    return unpatchify<T>(pr, nt, ny, nx, patch_t);
}
inline int reflect(int i,int n) { return i<0 ? -i : (i>=n ? 2*n-2-i : i); }
template<class T>
enc::MapT<T> conv(const enc::Conv2T<T>& c, const enc::MapT<T>& x, int stride, const enc::MapT<T>* res = nullptr) {
    enc::MapT<T> y;
    y.C = c.cout;
    const int pad = stride == 1 ? c.k / 2 : 0;  // downsample: no left/top pad, reflect 1 on the right/bottom
    const int Hp = stride == 1 ? x.H + 2 * pad : x.H + 1, Wp = stride == 1 ? x.W + 2 * pad : x.W + 1;
    y.H = (Hp - c.k) / stride + 1;
    y.W = (Wp - c.k) / stride + 1;
    y.v.resize(std::size_t(y.C) * y.H * y.W);
    std::vector<float> xf(x.v.size()), wf(c.w.size());
    for (std::size_t i = 0; i < xf.size(); ++i) xf[i] = Scalar<T>::get(x.v[i]);
    for (std::size_t i = 0; i < wf.size(); ++i) wf[i] = Scalar<T>::get(c.w[i]);
    parallel_rows(c.cout, [&](int co) {
        std::vector<double> acc(std::size_t(y.H) * y.W, 0.0);
        for (int ci = 0; ci < c.cin; ++ci)
            for (int ky = 0; ky < c.k; ++ky)
                for (int kx = 0; kx < c.k; ++kx) {
                    const double wv = wf[((std::size_t(co) * c.cin + ci) * c.k + ky) * c.k + kx];
                    for (int oy = 0; oy < y.H; ++oy) {
                        const int iy = reflect(oy * stride + ky - pad, x.H);
                        const float* xr = xf.data() + (std::size_t(ci) * x.H + iy) * x.W;
                        double* ar = acc.data() + std::size_t(oy) * y.W;
                        for (int ox = 0; ox < y.W; ++ox) ar[ox] += wv * xr[reflect(ox * stride + kx - pad, x.W)];
                    }
                }
        const float b = Scalar<T>::get(c.b[std::size_t(co)]);
        for (std::size_t q = 0; q < acc.size(); ++q) {
            float v = Scalar<T>::round(float(acc[q]) + b);
            if (res) v = Scalar<T>::round(v + Scalar<T>::get(res->v[std::size_t(co) * acc.size() + q]));
            y.v[std::size_t(co) * acc.size() + q] = Scalar<T>::set(v);
        }
    });
    return y;
}

template<class T>
enc::MapT<T> gn_silu(const enc::NormT<T>& n, const enc::MapT<T>& x) {
    enc::MapT<T> y = x;
    const int G = 32, cg = x.C / G;
    const std::size_t plane = std::size_t(x.H) * x.W;
    for (int g = 0; g < G; ++g) {
        double s = 0, q = 0;
        const std::size_t cnt = std::size_t(cg) * plane;
        for (std::size_t i = 0; i < cnt; ++i) s += Scalar<T>::get(x.v[std::size_t(g) * cnt + i]);
        const double mu = s / double(cnt);
        for (std::size_t i = 0; i < cnt; ++i) { const double d = Scalar<T>::get(x.v[std::size_t(g) * cnt + i]) - mu; q += d * d; }
        const float inv = float(1.0 / std::sqrt(q / double(cnt) + 1e-6));
        for (int c = g * cg; c < (g + 1) * cg; ++c) {
            const float w = Scalar<T>::get(n.w[std::size_t(c)]), b = Scalar<T>::get(n.b[std::size_t(c)]);
            for (std::size_t p = 0; p < plane; ++p) {
                const std::size_t i = std::size_t(c) * plane + p;
                const float nv = Scalar<T>::round(float(Scalar<T>::get(x.v[i]) - mu) * inv * w + b);
                y.v[i] = Scalar<T>::set(nv / (1.0F + std::exp(-nv)));
            }
        }
    }
    return y;
}

template<class T>
enc::MapT<T> resblock(const enc::ResT<T>& r, const enc::MapT<T>& x) {
    const enc::MapT<T> h = conv<T>(r.c1, gn_silu<T>(r.n1, x), 1);
    const enc::MapT<T> s = r.has_nin ? conv<T>(r.nin, x, 1) : x;
    return conv<T>(r.c2, gn_silu<T>(r.n2, h), 1, &s);
}

template<class T>
enc::MapT<T> normalize_pixels(const std::vector<float>& x, int H, int W) {
    const float mean[3] = {Scalar<T>::round(0.485F), Scalar<T>::round(0.456F), Scalar<T>::round(0.406F)}, sd[3] = {Scalar<T>::round(0.229F), Scalar<T>::round(0.224F), Scalar<T>::round(0.225F)};
    enc::MapT<T> m;
    m.C = 3; m.H = H; m.W = W;
    m.v.resize(x.size());
    for (int c = 0; c < 3; ++c)
        for (std::size_t p = 0; p < std::size_t(H) * W; ++p) {
            const std::size_t i = std::size_t(c) * H * W + p;
            float v = Scalar<T>::round(x[i]);
            v = Scalar<T>::round(v + 1.0F);
            v = Scalar<T>::round(v * 0.5F);
            v = Scalar<T>::round(v - mean[c]);
            m.v[i] = Scalar<T>::set(v / sd[c]);
        }
    return m;
}

template<class T>
enc::MapT<T> quant(const enc::WeightsT<T>& w, const enc::MapT<T>& x) { return conv<T>(w.quant, x, 1); }

template<class T>
enc::MapT<T> encode_tile(const enc::WeightsT<T>& w, const enc::MapT<T>& pixels) {
    enc::check_image(pixels.H,pixels.W,pixels.v.size());
    if (pixels.C!=3 || pixels.H>256 || pixels.W>256) throw std::invalid_argument("encoder: invalid tile");
    enc::MapT<T> x = conv<T>(w.conv_in, pixels, 1);
    for (int l = 0; l < 6; ++l) {
        x = resblock<T>(w.blocks[l][0], x);
        x = resblock<T>(w.blocks[l][1], x);
        if (l < 4) x = conv<T>(w.down[l], x, 2);
    }
    return quant<T>(w, conv<T>(w.conv_out, gn_silu<T>(w.norm_out, x), 1));
}
}
